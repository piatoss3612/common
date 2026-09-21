use super::*;

/// A task in one retained term-chunk slot.
#[derive(Clone, Copy, Debug)]
pub struct ChunkRequest<'a> {
    /// Retained slot whose scalar, digit, and result fragments must be leased.
    pub slot: usize,
    /// The bounded kernel and its complete resource requirements.
    pub task: Request<'a>,
}

/// Independent term chunks with a bounded, ordered partial frontier.
///
/// `SLOTS` bounds retained storage; it does not assign workers or change the
/// arithmetic grain. Every slot has independent preparation and window
/// dependencies. A short or joint chunk therefore runs alongside other chunks,
/// and a ready window can use any compatible scratch lease. Completed chunks
/// retire in input order before their slots are reused. At most `SLOTS` chunk
/// results exist, including an explicit reservation for the earliest missing
/// chunk, so later completions cannot prevent its reduction from progressing.
///
/// The caller supplies one retained arena per slot, frontier metadata, and a
/// shared temporary scratch provider. Charge [`MsmPlan::retained_for_slots`],
/// this run's inline state, identities, frontiers, and all provider capacity to
/// admission. Streaming instead uses [`MsmRun`]: it keeps a complete window
/// bucket set across chunks and avoids the independent chunk collapses here.
pub struct ParallelMsmRun<'a, 'i, C: PastaCurve, const SLOTS: usize> {
    runs: [MsmRun<'a, 'i, C>; SLOTS],
    plan: MsmPlan<C>,
    input: Input<'i, C>,
    chunks: usize,
    next: usize,
    retired: usize,
    result: ProjectivePoint<C>,
    failed: bool,
}

impl<'a, 'i, C: PastaCurve, const SLOTS: usize> ParallelMsmRun<'a, 'i, C, SLOTS> {
    /// Binds independent chunks without allocating or writing arithmetic data.
    ///
    /// Returns [`TaskError::Storage`] for zero slots or frontier capacity,
    /// [`TaskError::InvalidRequest`] for mismatched input length or streaming,
    /// and [`TaskError::Overflow`] for unrepresentable retained storage counts
    /// or bytes. Frontiers can contain a single task entry; more entries
    /// allow more windows from each prepared chunk to be detached at once.
    pub fn new<const TASKS: usize>(
        plan: MsmPlan<C>,
        input: Input<'i, C>,
        identities: &'a mut [Identity; SLOTS],
        storage: &'a mut [[TaskStorage; TASKS]; SLOTS],
    ) -> Result<Self, TaskError> {
        Self::validate(plan, input)?;
        if TASKS == 0 {
            return Err(TaskError::Storage);
        }
        let chunks = input.len().div_ceil(plan.cap.max(1));
        let mut metadata = identities.iter_mut().zip(storage);
        let runs = core::array::from_fn(|slot| {
            let (identity, storage) = metadata.next().unwrap();
            let range = Self::range(plan, chunks, slot);
            MsmRun::bind_range(plan, input, range, identity, storage)
                .expect("validated chunk metadata")
        });
        Ok(Self {
            runs,
            plan,
            input,
            chunks,
            next: chunks.min(SLOTS),
            retired: 0,
            result: ProjectivePoint::IDENTITY,
            failed: false,
        })
    }

    fn validate(plan: MsmPlan<C>, input: Input<'i, C>) -> Result<(), TaskError> {
        let slots = NonZeroUsize::new(SLOTS).ok_or(TaskError::Storage)?;
        if input.len() != plan.terms || plan.options.streaming() {
            return Err(TaskError::InvalidRequest);
        }
        plan.retained_for_slots(slots)
            .map_err(|_| TaskError::Overflow)?;
        Ok(())
    }

    fn range(plan: MsmPlan<C>, chunks: usize, chunk: usize) -> core::ops::Range<usize> {
        if chunk >= chunks {
            return 0..0;
        }
        let start = chunk * plan.cap;
        start..start.saturating_add(plan.cap).min(plan.terms)
    }

    /// Exposes a bounded number of tasks, starting with the earliest chunk.
    /// Resource-blocked requests remain pending while the scheduler considers
    /// other slots and operations. No task graph is materialized.
    pub fn ready(&self, output: &mut [Option<ChunkRequest<'a>>]) -> usize {
        if self.failed {
            return 0;
        }
        let mut written = 0;
        let mut requests = [None; 32];
        for chunk in self.retired..self.next {
            if written == output.len() {
                break;
            }
            let slot = chunk % SLOTS;
            let limit = requests.len().min(output.len() - written);
            let count = self.runs[slot].ready(&mut requests[..limit]);
            for &task in requests[..count].iter().flatten() {
                output[written] = Some(ChunkRequest { slot, task });
                written += 1;
            }
        }
        written
    }

    /// Enumerates one slot starting at a task index in its current epoch.
    /// Schedulers with small request buffers must visit every slot and page
    /// within it before concluding that no compatible work is ready. Advance
    /// to one past the last returned task index, and restart after publication.
    pub fn ready_slot_from(
        &self,
        slot: usize,
        start: usize,
        output: &mut [Option<ChunkRequest<'a>>],
    ) -> usize {
        if self.failed || slot >= SLOTS {
            return 0;
        }
        let mut written = 0;
        let mut cursor = start;
        for entry in output {
            let mut request = [None];
            if self.runs[slot].ready_from(cursor, &mut request) == 0 {
                break;
            }
            let task = request[0].unwrap();
            cursor = task.key.index() + 1;
            *entry = Some(ChunkRequest { slot, task });
            written += 1;
        }
        written
    }

    /// Claims one task after acquiring its slot's retained fragments and shared
    /// scratch atomically. The detached task does not borrow this run.
    pub fn try_claim<R: Resources<C>>(
        &mut self,
        request: ChunkRequest<'a>,
        acquire: impl FnOnce() -> Option<R>,
    ) -> Result<Option<Task<'a, MsmKernel<'i, C>, R>>, TaskError> {
        if self.failed {
            return Err(TaskError::Failed);
        }
        self.runs
            .get_mut(request.slot)
            .ok_or(TaskError::Stale)?
            .try_claim(request.task, acquire)
    }

    /// Publishes to the slot named in the original request, then consumes all
    /// consecutive completed chunks in order. A foreign slot returns the
    /// receipt intact. Failed runs accept outstanding receipts for draining.
    ///
    /// Returned resources must be released before leasing the slot again. Its
    /// next chunk becomes ready immediately; publication itself performs only
    /// a bounded reduction of at most `SLOTS` projective points.
    #[expect(
        clippy::result_large_err,
        reason = "return owned receipts without allocation"
    )]
    pub fn complete<R>(
        &mut self,
        slot: usize,
        completion: Completion<'a, R, MsmOutput<C>>,
    ) -> Result<Published<C, R>, crate::exec::run::PublishError<'a, R, MsmOutput<C>>> {
        let Some(run) = self.runs.get_mut(slot) else {
            return Err((TaskError::Stale, completion));
        };
        let mut published = run.complete(completion)?;
        self.failed |= run.is_failed();
        if !self.failed {
            while self.retired < self.next {
                let slot = self.retired % SLOTS;
                let Some(partial) = self.runs[slot].result() else {
                    break;
                };
                self.result = self.result.add(&partial);
                self.retired += 1;
                if self.next < self.chunks {
                    let range = Self::range(self.plan, self.chunks, self.next);
                    if self.runs[slot]
                        .rebind_range(self.plan, self.input, range)
                        .is_err()
                    {
                        self.failed = true;
                        published.error = Some(CurveError::SizeOverflow);
                        break;
                    }
                    self.next += 1;
                }
            }
        }
        published.result = self.result();
        Ok(published)
    }

    /// Final sum after every chunk retires, or identity for an empty input.
    pub fn result(&self) -> Option<ProjectivePoint<C>> {
        (!self.failed && self.retired == self.chunks).then_some(self.result)
    }

    /// Whether arithmetic validation, cancellation, or unwinding failed a task.
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Outstanding detached tasks, including those that must drain on failure.
    pub fn inflight(&self) -> usize {
        self.runs.iter().map(MsmRun::inflight).sum()
    }

    /// Reuses all metadata for another preplanned invocation after completion.
    /// Old keys remain stale. Errors match [`Self::new`], with
    /// [`TaskError::Busy`] before completion and [`TaskError::Failed`] on failure.
    pub fn rebind(&mut self, plan: MsmPlan<C>, input: Input<'i, C>) -> Result<(), TaskError> {
        if self.failed {
            return Err(TaskError::Failed);
        }
        if self.result().is_none() {
            return Err(TaskError::Busy);
        }
        Self::validate(plan, input)?;
        let chunks = input.len().div_ceil(plan.cap.max(1));
        for (slot, run) in self.runs.iter_mut().enumerate() {
            if let Err(error) = run.rebind_range(plan, input, Self::range(plan, chunks, slot)) {
                self.failed = true;
                return Err(error);
            }
        }
        self.plan = plan;
        self.input = input;
        self.chunks = chunks;
        self.next = chunks.min(SLOTS);
        self.retired = 0;
        self.result = ProjectivePoint::IDENTITY;
        Ok(())
    }
}
