use super::super::ClassState;
use super::*;

/// A fixed set of inverse transforms and their coefficient-sum dependencies.
///
/// Each class describes evaluations of one polynomial. Entry zero receives the
/// coefficient sum. Other classes, called lifts, may be smaller and use different
/// cosets; their missing higher coefficients contribute zero to the sum.
///
/// With `consume`, equal-domain evaluations merge before their representative
/// inverse, including groups smaller than the output. Other inverses start
/// independently. Without it, each lift retains its own coefficients. Use
/// [`Self::execute`] for contiguous buffers or [`InterpolationRun`] for
/// incremental scheduling.
#[derive(Clone, Copy, Debug)]
pub struct InterpolationPlan<'t, M: PrimeModulus, const CLASSES: usize> {
    transforms: [FftPlan<'t, M>; CLASSES],
    merge_into: [Option<usize>; CLASSES],
    consume: bool,
    budget: crate::exec::TaskBudget,
}

impl<'t, M: PrimeModulus, const CLASSES: usize> InterpolationPlan<'t, M, CLASSES> {
    /// Resolves inverse transforms for classes described by domain, tables, and order.
    ///
    /// Class zero receives the coefficient sum. Smaller classes contribute zero
    /// above their degree bound. `consume` permits merging equal-domain evaluations
    /// before interpolation; otherwise each lift retains its own coefficients.
    /// Every physical fragment layout is interpreted relative to that class size.
    /// The task and workspace limits cover all classes together, including
    /// concurrently retained transform scratch. Tables remain borrowed; no
    /// working storage is bound or written.
    ///
    /// `CLASSES` must be nonzero. Returns [`FftError::InvalidExecution`] for a fragment
    /// length that is not a power of two, [`FftError::InvalidLayout`] for a class
    /// larger than class zero, [`FftError::SizeOverflow`] for unrepresentable
    /// workspace, or [`FftError::MemoryLimit`] if required scratch exceeds the byte
    /// ceiling.
    pub fn new(
        classes: [(Transform<'t, M>, ElementOrder); CLASSES],
        consume: bool,
        layout: super::super::StorageLayout,
        options: crate::exec::ExecutionOptions,
    ) -> Result<Self, FftError> {
        const {
            assert!(CLASSES > 0, "interpolation needs an output class");
        }
        if classes
            .iter()
            .any(|(transform, _)| transform.domain().size() > classes[0].0.domain().size())
        {
            return Err(FftError::InvalidLayout);
        }
        let merged: [bool; CLASSES] = core::array::from_fn(|i| {
            consume
                && classes[..i]
                    .iter()
                    .any(|(transform, _)| transform.domain().same_domain(classes[i].0.domain()))
        });
        let active = merged.iter().filter(|merged| !**merged).count();
        let mandatory: [usize; CLASSES] = core::array::from_fn(|i| match layout {
            super::super::StorageLayout::Fragments {
                length,
                whole_bank: false,
            } if !merged[i]
                && length.get() < classes[i].0.domain().size()
                && classes[i].1 == ElementOrder::Natural =>
            {
                classes[i].0.domain().size()
            }
            _ => 0,
        });
        let required = mandatory
            .iter()
            .try_fold(0usize, |n, fields| n.checked_add(*fields))
            .and_then(|n| n.checked_mul(core::mem::size_of::<PastaField<M>>()))
            .ok_or(FftError::SizeOverflow)?;
        let extra = options
            .memory_limit()
            .map(|limit| {
                limit
                    .checked_sub(required)
                    .ok_or(FftError::MemoryLimit { required, limit })
            })
            .transpose()?;
        let inner = options.task_budget().partition(active).unwrap().1;
        let build = |i: usize, layout| {
            let mut options = crate::exec::ExecutionOptions::default().with_task_budget(inner);
            if let Some(extra) = extra
                && !merged[i]
            {
                options = options.with_memory_limit(
                    mandatory[i] * core::mem::size_of::<PastaField<M>>() + extra / active,
                );
            }
            FftPlan::new(
                classes[i].0,
                TransformRequest {
                    input_order: classes[i].1,
                    ..TransformRequest::new(Direction::Inverse)
                },
                layout,
                options,
            )
        };
        let first = build(0, layout)?;
        // Coefficient additions use a common physical subdivision. Contiguous
        // banks can accommodate whichever subdivision the output selected.
        let layout = match layout {
            super::super::StorageLayout::Contiguous => super::super::StorageLayout::Fragments {
                length: NonZeroUsize::new(first.tile()).unwrap(),
                whole_bank: true,
            },
            layout => layout,
        };
        let mut transforms = [first; CLASSES];
        for (i, target) in transforms.iter_mut().enumerate().skip(1) {
            *target = build(i, layout)?;
        }
        let mut result = Self::with_transforms(transforms, consume);
        result.budget = options.task_budget();
        Ok(result)
    }

    /// Binds compatible class transforms and selects whether lifts may be consumed.
    ///
    /// Every plan must describe an in-place, full-support, normalized inverse
    /// FFT with natural output. All classes must fit the output; their fragment
    /// sizes must equal the output tile or their smaller entire class size.
    ///
    /// `CLASSES` must be nonzero. Incompatible plans panic.
    pub(in crate::fft) fn with_transforms(
        transforms: [FftPlan<'t, M>; CLASSES],
        consume: bool,
    ) -> Self {
        const {
            assert!(CLASSES > 0, "interpolation needs an output class");
        }
        for plan in &transforms {
            assert!(plan.size() <= transforms[0].size(), "class exceeds output");
            assert!(
                plan.inverse()
                    && !plan.separate()
                    && plan.request.support == InputSupport::Full
                    && plan.request.inverse_scale == InverseScale::Normalized
                    && plan.request.output_order == ElementOrder::Natural
                    && plan.tile == transforms[0].tile.min(plan.size()),
                "incompatible class transform"
            );
        }
        let merge_into = core::array::from_fn(|i| {
            if consume {
                (0..i).find(|&target| {
                    transforms[i]
                        .plan
                        .domain()
                        .same_domain(transforms[target].plan.domain())
                })
            } else {
                None
            }
        });
        Self {
            transforms,
            merge_into,
            consume,
            budget: crate::exec::TaskBudget::SERIAL,
        }
    }

    /// Retained scratch field count for one class.
    ///
    /// Classes merged into another's evaluations need no scratch; other
    /// classes use their transform's [`FftPlan::retained_fields`]. An
    /// out-of-range class returns `None`. Admission must also count working
    /// class banks and all provisioned provider and metadata capacity.
    pub fn snapshot_fields(&self, class: usize) -> Option<usize> {
        self.transforms.get(class).map(|p| {
            if self.merge_into[class].is_some() {
                0
            } else {
                p.retained_fields()
            }
        })
    }

    /// Interpolates contiguous class buffers and leaves their sum in class zero.
    ///
    /// Each value slice must contain exactly its transform's domain size, in
    /// that plan's input order. Each scratch slice needs at least
    /// [`Self::snapshot_fields`] entries for its class. Output coefficients are
    /// normalized and in natural order. With `consume = false`, every lift
    /// retains its own natural coefficients; otherwise lift contents are
    /// unspecified. The total task budget covers concurrent classes and their
    /// inner transforms.
    ///
    /// Buffer requirement violations panic before any mutation. No allocation occurs;
    /// an executor panic leaves valid loose fields but requires refilling affected
    /// evaluations before retrying.
    ///
    /// This example adds a constant polynomial to a linear polynomial evaluated
    /// on a different coset, retaining the constant's coefficients:
    ///
    /// ```
    /// use zakura_udon::{
    ///     exec::{ExecutionOptions, SerialExecutor},
    ///     field::Fp,
    ///     fft::{Domain, ElementOrder, StorageLayout, Transform, run::InterpolationPlan},
    /// };
    ///
    /// let output_domain = Domain::new(1).unwrap().coset();
    /// let lift_domain = Domain::new(0).unwrap().subgroup();
    /// let plan = InterpolationPlan::new([
    ///     (Transform::new(output_domain), ElementOrder::Natural),
    ///     (Transform::new(lift_domain), ElementOrder::Natural),
    /// ], false, StorageLayout::Contiguous, ExecutionOptions::default()).unwrap();
    /// // 1 + x at ZETA and -ZETA, plus the constant polynomial 5.
    /// let shift = output_domain.shift();
    /// let mut output = [<Fp>::ONE.add(&shift), <Fp>::ONE.sub(&shift)];
    /// let mut lift = [Fp::from_u64(5)];
    /// plan.execute(
    ///     [&mut output, &mut lift], [&mut [], &mut []],
    ///     &SerialExecutor,
    /// );
    /// assert_eq!(output.map(|value| value.reduce()), [Fp::from_u64(6), Fp::ONE]); // 6 + x
    /// assert_eq!(lift.map(|value| value.reduce()), [Fp::from_u64(5)]);
    /// ```
    pub fn execute<E: crate::exec::Executor>(
        self,
        values: [&mut [PastaField<M>]; CLASSES],
        scratch: [&mut [PastaField<M>]; CLASSES],
        executor: &E,
    ) {
        self.execute_with(
            values,
            scratch,
            NonZeroUsize::new(self.budget.get()).unwrap(),
            executor,
        )
    }

    pub(in crate::fft) fn execute_with<E: crate::exec::Executor>(
        self,
        mut values: [&mut [PastaField<M>]; CLASSES],
        mut scratch: [&mut [PastaField<M>]; CLASSES],
        max_tasks: NonZeroUsize,
        executor: &E,
    ) {
        for i in 0..CLASSES {
            super::super::assert_length("class", self.transforms[i].size(), values[i].len());
            super::super::check_scratch(self.snapshot_fields(i).unwrap(), scratch[i].len());
        }
        // Preserve terminal inverse/add fusion for the contiguous radix-2 path.
        // Other geometries use the same plan through independently scoped runs.
        if max_tasks.get() == 1
            && self.transforms.iter().all(|p| {
                p.codelet == Codelet::Radix2 && p.twiddles.is_none() && p.columns.is_none()
            })
        {
            let mut buffers = values.into_iter();
            let mut classes: [_; CLASSES] = core::array::from_fn(|i| {
                super::super::Class::new(
                    self.transforms[i].plan,
                    buffers.next().unwrap(),
                    self.transforms[i].request.input_order,
                )
            });
            let (output, lifts) = classes.split_first_mut().unwrap();
            let options = super::super::Strategy::SERIAL;
            let result = if self.consume {
                super::super::interpolate_sum(output, lifts, options, executor, &mut [])
            } else {
                super::super::interpolate_classes(output, lifts, options, executor, &mut [])
            };
            result.expect("validated interpolation classes");
            return;
        }
        for i in 1..CLASSES {
            if let Some(target) = self.merge_into[i] {
                let (before, after) = values.split_at_mut(i);
                let different_order = self.transforms[i].request.input_order
                    != self.transforms[target].request.input_order;
                for (index, value) in before[target].iter_mut().enumerate() {
                    let source = if different_order {
                        reverse(index, self.transforms[i].size().ilog2())
                    } else {
                        index
                    };
                    *value = value.add(&after[0][source]);
                }
            }
        }
        fn visit<M: PrimeModulus, E: crate::exec::Executor>(
            plans: &[FftPlan<'_, M>],
            merged: &[Option<usize>],
            values: &mut [&mut [PastaField<M>]],
            scratch: &mut [&mut [PastaField<M>]],
            tasks: NonZeroUsize,
            executor: &E,
        ) {
            if plans.len() == 1 || tasks.get() == 1 {
                for (((plan, merged), values), scratch) in
                    plans.iter().zip(merged).zip(values).zip(scratch)
                {
                    if merged.is_none() {
                        plan.execute_with(None, values, None, scratch, tasks, executor);
                    }
                }
            } else {
                let mid = plans.len() / 2;
                let left_tasks = tasks.get() / 2;
                let (a, b) = values.split_at_mut(mid);
                let (sa, sb) = scratch.split_at_mut(mid);
                executor.join(
                    || {
                        visit(
                            &plans[..mid],
                            &merged[..mid],
                            a,
                            sa,
                            NonZeroUsize::new(left_tasks).unwrap(),
                            executor,
                        )
                    },
                    || {
                        visit(
                            &plans[mid..],
                            &merged[mid..],
                            b,
                            sb,
                            NonZeroUsize::new(tasks.get() - left_tasks).unwrap(),
                            executor,
                        )
                    },
                );
            }
        }
        visit(
            &self.transforms,
            &self.merge_into,
            &mut values,
            &mut scratch,
            max_tasks,
            executor,
        );
        let (output, lifts) = values.split_first_mut().unwrap();
        for (i, lift) in lifts.iter().enumerate() {
            if self.merge_into[i + 1].is_none() {
                for (out, value) in output.iter_mut().zip(lift.iter()) {
                    *out = out.add(value);
                }
            }
        }
    }
}

/// A bounded evaluation merge or addition into the output coefficient sum.
#[derive(Clone, Debug)]
pub struct AdditionRequest<'a> {
    /// Destination class: an equal-domain representative for evaluation merges,
    /// or class zero for the final coefficient sum.
    pub target: usize,
    /// Nonforgeable claim identity in this class's addition frontier.
    pub key: TaskKey<'a>,
    /// Exclusive range in the [`Self::target`] class's physical bank.
    pub write: Range<usize>,
    /// Shared lift range, relative to the lift's own physical bank.
    pub read: Range<usize>,
    /// Whether this adds evaluations before inversion or coefficients after it.
    pub evaluations: bool,
}

/// Detached class addition using [`Resources`] like FFT tasks.
///
/// In [`Buffers`], `values` names the requested target range and `source` the
/// lift read range; `pair` and `factor` are empty. Reordered evaluation reads
/// remain shared.
pub struct AdditionKernel<M: PrimeModulus> {
    marker: core::marker::PhantomData<M>,
    start: usize,
    len: usize,
    size: usize,
    reversed: bool,
}

impl<M: PrimeModulus, R: Resources<M>> Kernel<R> for AdditionKernel<M> {
    type Output = ();
    fn execute(&mut self, resources: &mut R) -> Self::Output {
        self.run(resources.buffers())
    }
}

impl<M: PrimeModulus> AdditionKernel<M> {
    fn run(&self, buffers: Buffers<'_, M>) {
        let Buffers {
            values,
            pair,
            source,
            factor,
        } = buffers;
        super::super::assert_length("addition output", self.len, values.len());
        super::super::assert_length(
            "addition source",
            if self.reversed { self.size } else { self.len },
            source.len(),
        );
        super::super::assert_length("addition pair", 0, pair.len());
        super::super::assert_length("addition factor", 0, factor.len());
        for (offset, value) in values.iter_mut().enumerate() {
            let index = if self.reversed {
                reverse(self.start + offset, self.size.ilog2())
            } else {
                offset
            };
            *value = value.add(source.get(index).expect("invalid class source view"));
        }
    }
}

/// Resources and newly satisfied class-consumer dependencies.
pub struct InterpolationPublished<R> {
    /// Returned resource envelope, to release before dispatching successors.
    pub resources: R,
    /// Phase transition error, if any.
    pub error: Option<TaskError>,
    /// Normal return, cancellation, or caught unwind.
    pub outcome: Outcome,
    /// The class whose inverse just produced normalized coefficients.
    ///
    /// With consumption enabled, these include any classes merged into its
    /// evaluations. Class zero's sum is final only when [`Self::complete`] is true.
    pub coefficients: Option<usize>,
    /// A lift whose last read by this interpolation has completed. Other
    /// application consumers must also finish before its bank is recycled.
    pub released: Option<usize>,
    /// Whether output class zero contains the complete coefficient sum.
    pub complete: bool,
}

/// Independent class transforms, bounded merges, and incremental reductions.
///
/// Addition frontiers reserve their target ranges before dispatch. Exclusive
/// target leases resolve overlap between lifts; dependency-blocked work never
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
    /// Binds two frontiers per class: transform then addition.
    ///
    /// `TASKS` must be nonzero. Metadata is proportional to classes times frontier
    /// capacity, independent of total stage task count.
    pub fn new<const TASKS: usize>(
        plan: InterpolationPlan<'t, M, CLASSES>,
        identities: &'a mut [[Identity; 2]; CLASSES],
        storage: &'a mut [[[TaskStorage; TASKS]; 2]; CLASSES],
    ) -> Self {
        const { assert!(TASKS > 0, "task capacity must be nonzero") };
        let mut metadata = identities.iter_mut().zip(storage).enumerate();
        let pairs: [_; CLASSES] = core::array::from_fn(|_| {
            let (class, ([transform_id, add_id], [transform_slots, add_slots])) =
                metadata.next().unwrap();
            let run = if plan.merge_into[class].is_some() {
                FftRun::empty(plan.transforms[class], transform_id, transform_slots)
            } else {
                FftRun::new(plan.transforms[class], false, transform_id, transform_slots)
            };
            let frontier = Frontier::new(
                add_id,
                add_slots,
                if class == 0 {
                    0
                } else {
                    plan.transforms[class].fragments()
                },
            );
            (Some(run), Some(frontier))
        });
        let mut pairs = pairs;
        let runs = core::array::from_fn(|i| pairs[i].0.take().unwrap());
        let additions = core::array::from_fn(|i| pairs[i].1.take().unwrap());
        Self {
            plan,
            runs,
            additions,
            started: [false; CLASSES],
            failed: false,
        }
    }

    fn merged(&self, class: usize) -> bool {
        self.plan
            .merge_into
            .iter()
            .enumerate()
            .all(|(i, &target)| target != Some(class) || self.additions[i].is_complete())
    }

    /// Pages one class's transform tasks.
    ///
    /// Each representative waits for its same-domain evaluation merges; other
    /// inverses start immediately. Classes merged into a representative have
    /// no transform tasks. An invalid class index produces no requests.
    pub fn ready_transform_from(
        &self,
        class: usize,
        start: usize,
        output: &mut [Option<Request<'a>>],
    ) -> usize {
        if self.failed
            || class >= CLASSES
            || self.plan.merge_into[class].is_some()
            || !self.merged(class)
        {
            return 0;
        }
        self.runs[class].ready_from(start, output)
    }

    fn addition(&self, class: usize, key: TaskKey<'a>) -> AdditionRequest<'a> {
        let plan = self.plan.transforms[class];
        let start = key.index() * plan.tile;
        let reversed = self.plan.merge_into[class].is_some()
            && plan.request.input_order
                != self.plan.transforms[self.plan.merge_into[class].unwrap_or(0)]
                    .request
                    .input_order;
        AdditionRequest {
            target: self.plan.merge_into[class].unwrap_or(0),
            key,
            write: start..start + plan.tile,
            read: if reversed {
                0..plan.size()
            } else {
                start..start + plan.tile
            },
            evaluations: self.plan.merge_into[class].is_some(),
        }
    }

    /// Pages additions whose inputs are ready.
    ///
    /// Evaluation merges are ready immediately. Each representative's
    /// coefficients may join the output sum once both its inverse and the
    /// output inverse complete, independently of unrelated inverses. Class
    /// zero and invalid class indices produce no requests. Restart cursors
    /// after publication.
    pub fn ready_addition_from(
        &self,
        class: usize,
        start: usize,
        output: &mut [Option<AdditionRequest<'a>>],
    ) -> usize {
        if self.failed
            || class == 0
            || class >= CLASSES
            || (self.plan.merge_into[class].is_none()
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
    ///
    /// A transform waiting for evaluation merges returns [`TaskError::Busy`].
    pub fn try_claim_transform<R: Resources<M>>(
        &mut self,
        class: usize,
        request: Request<'a>,
        acquire: impl FnOnce() -> Option<R>,
    ) -> Result<Option<Task<'a, FftKernel<'t, M>, R>>, TaskError> {
        if self.failed {
            return Err(TaskError::Failed);
        }
        if class >= CLASSES || self.plan.merge_into[class].is_some() {
            return Err(TaskError::Stale);
        }
        if !self.merged(class) {
            return Err(TaskError::Busy);
        }
        let result = self.runs[class].try_claim(request, acquire)?;
        self.started[class] |= result.is_some();
        Ok(result)
    }

    /// Claims one bounded addition with the same resource provider as FFT tasks.
    ///
    /// Dependencies return [`TaskError::Busy`] before leasing.
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
        if self.plan.merge_into[class].is_none()
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
            reversed: self.plan.merge_into[class].is_some()
                && self.plan.transforms[class].request.input_order
                    != self.plan.transforms[self.plan.merge_into[class].unwrap_or(0)]
                        .request
                        .input_order,
        };
        let result = self.additions[class].try_claim(request.key, kernel, acquire)?;
        if result.is_some() {
            self.started[class] = true;
            if self.plan.merge_into[class].is_some() {
                self.started[actual.target] = true;
            }
        }
        Ok(result)
    }

    /// Publishes one transform receipt and updates class dependencies.
    ///
    /// A foreign class returns the complete receipt intact.
    pub fn complete_transform<R>(
        &mut self,
        class: usize,
        receipt: Completion<'a, R, ()>,
    ) -> Result<InterpolationPublished<R>, crate::exec::run::PublishError<'a, R, ()>> {
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
    ///
    /// Failed runs still accept their outstanding receipts for draining.
    pub fn complete_addition<R>(
        &mut self,
        class: usize,
        receipt: Completion<'a, R, ()>,
    ) -> Result<InterpolationPublished<R>, crate::exec::run::PublishError<'a, R, ()>> {
        let Some(frontier) = self.additions.get_mut(class) else {
            return Err((TaskError::Stale, receipt));
        };
        let task = frontier.complete(receipt)?;
        self.failed |= task.outcome != Outcome::Success;
        let released = (frontier.is_complete() && !self.failed).then_some(class);
        Ok(InterpolationPublished {
            resources: task.resources,
            error: None,
            outcome: task.outcome,
            coefficients: None,
            released,
            complete: self.is_complete(),
        })
    }

    /// Current semantic class state.
    ///
    /// Consumed lifts do not promise a retained polynomial. Output becomes
    /// coefficients only when its sum is complete.
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

    /// Whether any task was cancelled, panicked, or failed to advance its phase.
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Outstanding transform and addition receipts, including after failure.
    pub fn inflight(&self) -> usize {
        self.runs.iter().map(FftRun::inflight).sum::<usize>()
            + self.additions.iter().map(Frontier::inflight).sum::<usize>()
    }
}
