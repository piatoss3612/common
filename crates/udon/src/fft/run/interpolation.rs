use super::super::ClassState;
use super::*;

/// A fixed set of inverse transforms and their coefficient-sum dependencies.
///
/// Entry zero is the output class. Every other class may be smaller and use a
/// different coset. Each transform owns its stage barriers. With `consume`,
/// equal-domain evaluations merge before the output inverse; other inverses
/// start independently. Without it, each lift retains its own coefficients.
#[derive(Clone, Copy, Debug)]
pub struct InterpolationPlan<'t, M: PrimeModulus, const CLASSES: usize> {
    transforms: [FftPlan<'t, M>; CLASSES],
    fused: [bool; CLASSES],
    consume: bool,
}

impl<'t, M: PrimeModulus, const CLASSES: usize> InterpolationPlan<'t, M, CLASSES> {
    /// Validates preplanned, in-place, full-support, normalized inverse FFTs
    /// with natural output. All classes must fit the output; their fragment
    /// sizes must equal the output tile or their smaller entire class size.
    /// Returns [`FftError::InvalidExecution`] for other requests or no classes,
    /// and [`FftError::InvalidLayout`] for a lift larger than the output.
    pub fn new(transforms: [FftPlan<'t, M>; CLASSES], consume: bool) -> Result<Self, FftError> {
        if CLASSES == 0 {
            return Err(FftError::InvalidExecution);
        }
        for plan in &transforms {
            if plan.size() > transforms[0].size() {
                return Err(FftError::InvalidLayout);
            }
            if !plan.inverse()
                || plan.separate
                || plan.request.support != InputSupport::Full
                || plan.request.inverse_scale != InverseScale::Normalized
                || plan.request.output_order != ElementOrder::Natural
                || plan.tile != transforms[0].tile.min(plan.size())
            {
                return Err(FftError::InvalidExecution);
            }
        }
        let fused = core::array::from_fn(|i| {
            i != 0
                && consume
                && transforms[i]
                    .plan
                    .domain()
                    .same_domain(transforms[0].plan.domain())
        });
        Ok(Self {
            transforms,
            fused,
            consume,
        })
    }

    /// Snapshot fields for one class. Fused lifts do not need snapshots.
    /// An out-of-range class returns `None`. Working class banks and all idle
    /// provider and metadata capacity must also be charged to admission.
    pub fn snapshot_fields(&self, class: usize) -> Option<usize> {
        self.transforms.get(class).map(|p| {
            if self.fused[class] {
                0
            } else {
                p.retained_fields()
            }
        })
    }
}

/// A bounded addition from a lift into output class zero.
#[derive(Clone, Debug)]
pub struct AdditionRequest<'a> {
    /// Nonforgeable claim identity in this class's addition frontier.
    pub key: TaskKey<'a>,
    /// Exclusive output-class range.
    pub write: Range<usize>,
    /// Shared lift range, relative to the lift's own physical bank.
    pub read: Range<usize>,
    /// Whether this adds evaluations before inversion or coefficients after it.
    pub evaluations: bool,
    /// Arithmetic and memory-traffic estimates for optional scheduling policy.
    pub estimate: WorkEstimate,
}

/// Detached class addition using [`Resources`] like FFT tasks.
/// `values` names the requested output range, `source` the lift read range;
/// `pair` and `factor` are empty. Reordered evaluation reads remain shared.
pub struct AdditionKernel<M: PrimeModulus> {
    marker: core::marker::PhantomData<M>,
    start: usize,
    len: usize,
    size: usize,
    reversed: bool,
}

impl<M: PrimeModulus, R: Resources<M>> Kernel<R> for AdditionKernel<M> {
    type Output = Result<(), FftError>;
    fn execute(&mut self, resources: &mut R) -> Self::Output {
        self.run(resources.buffers())
    }
}

impl<M: PrimeModulus> AdditionKernel<M> {
    fn run(&self, buffers: Buffers<'_, M>) -> Result<(), FftError> {
        let Buffers {
            values,
            pair,
            source,
            factor,
        } = buffers;
        super::super::check_length("addition output", self.len, values.len())?;
        super::super::check_length(
            "addition source",
            if self.reversed { self.size } else { self.len },
            source.len(),
        )?;
        super::super::check_length("addition pair", 0, pair.len())?;
        super::super::check_length("addition factor", 0, factor.len())?;
        for (offset, value) in values.iter_mut().enumerate() {
            let index = if self.reversed {
                reverse(self.start + offset, self.size.ilog2())
            } else {
                offset
            };
            *value = value.add(source.get(index).expect("invalid class source view"));
        }
        Ok(())
    }
}

/// Resources and newly satisfied class-consumer dependencies.
pub struct InterpolationPublished<R> {
    /// Returned resource envelope, to release before dispatching successors.
    pub resources: R,
    /// Resource validation error, if any.
    pub error: Option<FftError>,
    /// Normal return, cancellation, or caught unwind.
    pub outcome: Outcome,
    /// The class whose own normalized coefficients just became available.
    pub coefficients: Option<usize>,
    /// A lift whose last read by this interpolation has completed. Other
    /// application consumers must also finish before its bank is recycled.
    pub released: Option<usize>,
    /// Whether output class zero contains the complete coefficient sum.
    pub complete: bool,
}

/// Independent class transforms, bounded merges, and incremental reductions.
///
/// Addition frontiers reserve their output ranges before dispatch. Exclusive
/// output leases resolve overlap between lifts; dependency-blocked work never
/// occupies a worker. Different lift sizes and completed classes do not impose
/// a batch inverse barrier. Admission must cover all retained banks through
/// their last consumers, plus one compatible task bundle and queue entry.
pub struct InterpolationRun<'a, 't, M: PrimeModulus, const CLASSES: usize> {
    plan: InterpolationPlan<'t, M, CLASSES>,
    runs: [FftRun<'a, 't, M>; CLASSES],
    additions: [Frontier<'a>; CLASSES],
    started: [bool; CLASSES],
    failed: bool,
}

impl<'a, 't, M: PrimeModulus, const CLASSES: usize> InterpolationRun<'a, 't, M, CLASSES> {
    /// Binds two frontiers per class: transform then addition. Zero frontier
    /// capacity returns [`TaskError::Storage`]. Metadata is proportional to
    /// classes times frontier capacity, independent of total stage task count.
    pub fn new<const TASKS: usize>(
        plan: InterpolationPlan<'t, M, CLASSES>,
        identities: &'a mut [[Identity; 2]; CLASSES],
        storage: &'a mut [[[TaskStorage; TASKS]; 2]; CLASSES],
    ) -> Result<Self, TaskError> {
        if TASKS == 0 {
            return Err(TaskError::Storage);
        }
        let mut metadata = identities.iter_mut().zip(storage).enumerate();
        let pairs: [_; CLASSES] = core::array::from_fn(|_| {
            let (class, ([transform_id, add_id], [transform_slots, add_slots])) =
                metadata.next().unwrap();
            let run = if plan.fused[class] {
                FftRun::empty(plan.transforms[class], transform_id, transform_slots)
            } else {
                FftRun::new(plan.transforms[class], false, transform_id, transform_slots)
            }
            .expect("validated metadata");
            let frontier = Frontier::new(
                add_id,
                add_slots,
                if class == 0 {
                    0
                } else {
                    plan.transforms[class].fragments()
                },
            )
            .expect("validated metadata");
            (Some(run), Some(frontier))
        });
        let mut pairs = pairs;
        let runs = core::array::from_fn(|i| pairs[i].0.take().unwrap());
        let additions = core::array::from_fn(|i| pairs[i].1.take().unwrap());
        Ok(Self {
            plan,
            runs,
            additions,
            started: [false; CLASSES],
            failed: false,
        })
    }

    fn merged(&self) -> bool {
        self.plan
            .fused
            .iter()
            .enumerate()
            .all(|(i, &fused)| !fused || self.additions[i].is_complete())
    }

    /// Pages one class's transform tasks. Output inversion waits only for
    /// its same-domain evaluation merges; other inverses start immediately.
    pub fn ready_transform_from(
        &self,
        class: usize,
        start: usize,
        output: &mut [Option<Request<'a>>],
    ) -> usize {
        if self.failed
            || class >= CLASSES
            || self.plan.fused[class]
            || (class == 0 && !self.merged())
        {
            return 0;
        }
        self.runs[class].ready_from(start, output)
    }

    fn addition(&self, class: usize, key: TaskKey<'a>) -> AdditionRequest<'a> {
        let plan = self.plan.transforms[class];
        let start = key.index() * plan.tile;
        let reversed = self.plan.fused[class]
            && plan.request.input_order != self.plan.transforms[0].request.input_order;
        AdditionRequest {
            key,
            write: start..start + plan.tile,
            read: if reversed {
                0..plan.size()
            } else {
                start..start + plan.tile
            },
            evaluations: self.plan.fused[class],
            estimate: WorkEstimate {
                arithmetic: plan.tile,
                traffic_bytes: plan
                    .tile
                    .saturating_mul(size_of::<PastaField<M>>())
                    .saturating_mul(3),
                cache_bytes: plan
                    .tile
                    .saturating_mul(size_of::<PastaField<M>>())
                    .saturating_mul(2),
            },
        }
    }

    /// Pages additions whose inputs are ready. Every completed nonfused class
    /// may reduce as soon as output inversion completes, independently of
    /// unrelated class inverses. Restart cursors after publication.
    pub fn ready_addition_from(
        &self,
        class: usize,
        start: usize,
        output: &mut [Option<AdditionRequest<'a>>],
    ) -> usize {
        if self.failed
            || class == 0
            || class >= CLASSES
            || (!self.plan.fused[class]
                && (!self.runs[0].is_complete() || !self.runs[class].is_complete()))
        {
            return 0;
        }
        let mut written = 0;
        for key in self.additions[class]
            .tasks()
            .filter(|key| key.index() >= start)
            .take(output.len())
        {
            output[written] = Some(self.addition(class, key));
            written += 1;
        }
        written
    }

    /// Claims one transform after its complete resource bundle is available.
    /// Dependency-blocked output tasks return [`TaskError::Busy`].
    pub fn try_claim_transform<R: Resources<M>>(
        &mut self,
        class: usize,
        request: Request<'a>,
        acquire: impl FnOnce() -> Option<R>,
    ) -> Result<Option<Task<'a, FftKernel<'t, M>, R>>, TaskError> {
        if self.failed {
            return Err(TaskError::Failed);
        }
        if class >= CLASSES || self.plan.fused[class] {
            return Err(TaskError::Stale);
        }
        if class == 0 && !self.merged() {
            return Err(TaskError::Busy);
        }
        let result = self.runs[class].try_claim(request, acquire)?;
        self.started[class] |= result.is_some();
        Ok(result)
    }

    /// Claims one bounded addition with the same resource provider as FFT
    /// tasks. Dependencies return [`TaskError::Busy`] before leasing.
    pub fn try_claim_addition<R: Resources<M>>(
        &mut self,
        class: usize,
        request: AdditionRequest<'a>,
        acquire: impl FnOnce() -> Option<R>,
    ) -> Result<Option<Task<'a, AdditionKernel<M>, R>>, TaskError> {
        if self.failed {
            return Err(TaskError::Failed);
        }
        if class == 0 || class >= CLASSES {
            return Err(TaskError::Stale);
        }
        self.additions[class].check_key(request.key)?;
        if !self.plan.fused[class]
            && (!self.runs[0].is_complete() || !self.runs[class].is_complete())
        {
            return Err(TaskError::Busy);
        }
        let actual = self.addition(class, request.key);
        let kernel = AdditionKernel {
            marker: core::marker::PhantomData,
            start: actual.write.start,
            len: actual.write.len(),
            size: self.plan.transforms[class].size(),
            reversed: self.plan.fused[class]
                && self.plan.transforms[class].request.input_order
                    != self.plan.transforms[0].request.input_order,
        };
        let result = self.additions[class].try_claim(request.key, kernel, acquire)?;
        if result.is_some() {
            self.started[class] = true;
            if self.plan.fused[class] {
                self.started[0] = true;
            }
        }
        Ok(result)
    }

    /// Publishes one transform receipt and immediately readies its additions.
    /// A foreign class returns the complete receipt intact.
    pub fn complete_transform<R>(
        &mut self,
        class: usize,
        receipt: Completion<'a, R, Result<(), FftError>>,
    ) -> Result<
        InterpolationPublished<R>,
        crate::exec::run::PublishError<'a, R, Result<(), FftError>>,
    > {
        let Some(run) = self.runs.get_mut(class) else {
            return Err((TaskError::Stale, receipt));
        };
        let task = run.complete(receipt)?;
        self.failed |= run.is_failed();
        Ok(InterpolationPublished {
            resources: task.resources,
            error: task.error,
            outcome: task.outcome,
            coefficients: task.complete.then_some(class),
            released: None,
            complete: self.is_complete(),
        })
    }

    /// Publishes an addition receipt, releasing the lift after its last read.
    /// Failed runs still accept their outstanding receipts for draining.
    pub fn complete_addition<R>(
        &mut self,
        class: usize,
        receipt: Completion<'a, R, Result<(), FftError>>,
    ) -> Result<
        InterpolationPublished<R>,
        crate::exec::run::PublishError<'a, R, Result<(), FftError>>,
    > {
        let Some(frontier) = self.additions.get_mut(class) else {
            return Err((TaskError::Stale, receipt));
        };
        let task = frontier.complete(receipt)?;
        let error = task.output.and_then(Result::err);
        self.failed |= task.outcome != Outcome::Success || error.is_some();
        let released = (frontier.is_complete() && !self.failed).then_some(class);
        Ok(InterpolationPublished {
            resources: task.resources,
            error,
            outcome: task.outcome,
            coefficients: None,
            released,
            complete: self.is_complete(),
        })
    }

    /// Current semantic class state. Consumed lifts do not promise a retained
    /// polynomial. Output becomes coefficients only when its sum is complete.
    /// Unstarted classes remain evaluations; an invalid index returns `None`.
    pub fn state(&self, class: usize) -> Option<ClassState> {
        self.runs.get(class)?;
        Some(if class == 0 && self.is_complete() {
            ClassState::Coefficients
        } else if !self.started[class] {
            ClassState::Evaluations
        } else if class != 0 && !self.plan.consume && self.runs[class].is_complete() && !self.failed
        {
            ClassState::Coefficients
        } else {
            ClassState::Consumed
        })
    }

    /// Whether output contains the successful complete polynomial sum.
    pub fn is_complete(&self) -> bool {
        !self.failed
            && self.runs[0].is_complete()
            && self.additions.iter().all(Frontier::is_complete)
    }

    /// Whether any task failed or returned invalid resources.
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Outstanding transform and addition receipts, including after failure.
    pub fn inflight(&self) -> usize {
        self.runs.iter().map(FftRun::inflight).sum::<usize>()
            + self.additions.iter().map(Frontier::inflight).sum::<usize>()
    }
}
