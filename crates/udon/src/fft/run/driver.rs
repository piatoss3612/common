use super::*;
use crate::exec::Executor;

struct Lease<'a, M: PrimeModulus> {
    values: &'a mut [PastaField<M>],
    pair: &'a mut [PastaField<M>],
    source: &'a [PastaField<M>],
    factor: &'a [PastaField<M>],
}
impl<M: PrimeModulus> Resources<M> for Lease<'_, M> {
    fn buffers(&mut self) -> Buffers<'_, M> {
        Buffers {
            values: self.values,
            pair: self.pair,
            source: &self.source,
            factor: &self.factor,
        }
    }
}

fn dispatch<'a, 't, 'b, M: PrimeModulus, E: Executor>(
    run: &mut FftRun<'a, 't, M>,
    requests: &mut [Option<Request<'a>>],
    leases: impl Iterator<Item = Lease<'b, M>>,
    executor: &E,
) {
    crate::exec::run::dispatch!(
        run,
        requests,
        leases,
        executor,
        "validated transform phases"
    );
}

impl<M: PrimeModulus> FftPlan<'_, M> {
    /// Transforms contiguous buffers using the caller's scoped executor.
    ///
    /// `values` must contain exactly [`Self::size`] fields. `input` is `Some`
    /// exactly when the request selects [`InputStorage::Preserve`]; its length must
    /// match the request's full domain or prefix. Otherwise `values` initially
    /// holds the in-place input. Input uses the request's input order;
    /// positions beyond a declared prefix are treated as zero.
    ///
    /// The result uses the requested output order and inverse scale. `factor`,
    /// when present, must have exactly [`Self::size`] entries in that output
    /// order and is multiplied pointwise after the transform's scaling.
    /// `scratch` needs at least [`Self::retained_fields`] entries.
    ///
    /// Panics if the buffers do not meet these requirements. These checks precede
    /// mutation. This driver does not allocate.
    ///
    /// The task limit bounds detached envelopes in each structured join. It
    /// neither changes arithmetic grain nor assigns workers to this operation.
    /// All kernels use the same run protocol as application scheduling, but
    /// structured joins return their receipts together. A panic leaves working
    /// values canonical and waits for all joined tasks before propagating;
    /// refill the input before retrying an in-place transform.
    pub fn execute<E: Executor>(
        self,
        input: Option<&[PastaField<M>]>,
        values: &mut [PastaField<M>],
        factor: Option<&[PastaField<M>]>,
        scratch: &mut [PastaField<M>],
        executor: &E,
    ) {
        self.execute_with(
            input,
            values,
            factor,
            scratch,
            NonZeroUsize::new(self.budget.get()).unwrap(),
            executor,
        )
    }

    pub(crate) fn execute_with<E: Executor>(
        self,
        input: Option<&[PastaField<M>]>,
        values: &mut [PastaField<M>],
        factor: Option<&[PastaField<M>]>,
        scratch: &mut [PastaField<M>],
        max_tasks: NonZeroUsize,
        executor: &E,
    ) {
        assert_eq!(
            self.separate(),
            input.is_some(),
            "input storage must match the plan"
        );
        super::super::assert_length("output", self.size(), values.len());
        if let Some(input) = input {
            super::super::assert_length("input", self.request.input_len(self.size()), input.len());
        }
        if let Some(factor) = factor {
            super::super::assert_length("factor", self.size(), factor.len());
        }
        super::super::check_scratch(self.retained_fields(), scratch.len());
        if self.fragments() == 1 {
            let mut identity = Identity::new();
            let mut slot = [TaskStorage::EMPTY];
            let mut run = FftRun::new(self, factor.is_some(), &mut identity, &mut slot);
            let mut request = [None];
            run.ready(&mut request);
            let lease = Lease {
                values,
                pair: &mut [],
                source: input.unwrap_or(&[]),
                factor: factor.unwrap_or(&[]),
            };
            let mut task = run
                .try_claim(request[0].take().unwrap(), || Some(lease))
                .expect("local claim")
                .unwrap();
            task.execute().expect("fresh local task");
            let published = run.complete(task.finish()).expect("local receipt");
            assert!(published.error.is_none(), "validated transform phases");
            return;
        }
        let max_tasks = if self.first() > self.size()
            || self.separate() && self.request.input_len(self.size()) <= 1
        {
            NonZeroUsize::MIN
        } else {
            max_tasks
        };
        self.execute_tiles(input, values, factor, scratch, max_tasks, executor)
    }

    #[inline(never)]
    fn execute_tiles<E: Executor>(
        self,
        input: Option<&[PastaField<M>]>,
        values: &mut [PastaField<M>],
        factor: Option<&[PastaField<M>]>,
        scratch: &mut [PastaField<M>],
        max_tasks: NonZeroUsize,
        executor: &E,
    ) {
        let mut identity = Identity::new();
        let mut slots = [const { TaskStorage::EMPTY }; 32];
        let mut run = FftRun::new(self, factor.is_some(), &mut identity, &mut slots);
        let mut requests: [Option<Request<'_>>; 32] = core::array::from_fn(|_| None);
        while !run.is_complete() {
            let count = run.ready(
                &mut requests[..if matches!(
                    run.kind,
                    WorkKind::Permute | WorkKind::InitializeScatter
                ) {
                    1
                } else {
                    max_tasks.get().min(32)
                }],
            );
            let first = requests[0]
                .as_ref()
                .expect("structured run progress")
                .key
                .index();
            let empty = |values| Lease {
                values,
                pair: &mut [],
                source: &[],
                factor: &[],
            };
            match run.kind {
                WorkKind::InitializeScatter => dispatch(
                    &mut run,
                    &mut requests[..count],
                    core::iter::once(Lease {
                        values,
                        pair: &mut [],
                        source: input.unwrap(),
                        factor: &[],
                    }),
                    executor,
                ),
                WorkKind::Permute => dispatch(
                    &mut run,
                    &mut requests[..count],
                    core::iter::once(empty(values)),
                    executor,
                ),
                WorkKind::Snapshot => {
                    let leases = scratch[..self.size()]
                        .chunks_exact_mut(self.tile)
                        .enumerate()
                        .skip(first)
                        .take(count)
                        .map(|(i, output)| Lease {
                            values: output,
                            pair: &mut [],
                            source: &values[i * self.tile..(i + 1) * self.tile],
                            factor: &[],
                        });
                    dispatch(&mut run, &mut requests[..count], leases, executor);
                }
                WorkKind::Pair => {
                    let leases = values
                        .chunks_exact_mut(run.block)
                        .flat_map(|group| {
                            let (left, right) = group.split_at_mut(group.len() / 2);
                            left.chunks_exact_mut(self.tile)
                                .zip(right.chunks_exact_mut(self.tile))
                        })
                        .skip(first)
                        .take(count)
                        .map(|(values, pair)| Lease {
                            values,
                            pair,
                            source: &[],
                            factor: &[],
                        });
                    dispatch(&mut run, &mut requests[..count], leases, executor);
                }
                WorkKind::Column => {
                    let fields = self.columns.unwrap().0 * self.fragments();
                    let leases = scratch[..run.band() * self.fragments()]
                        .chunks_mut(fields)
                        .skip(first)
                        .take(count)
                        .map(|output| Lease {
                            values: output,
                            pair: &mut [],
                            source: values,
                            factor: &[],
                        });
                    dispatch(&mut run, &mut requests[..count], leases, executor);
                }
                WorkKind::Scatter => {
                    let column = run.column;
                    let band = run.band();
                    let leases = values
                        .chunks_exact_mut(self.tile)
                        .skip(first)
                        .take(count)
                        .map(|output| Lease {
                            values: &mut output[column..column + band],
                            pair: &mut [],
                            source: &scratch[..band * self.fragments()],
                            factor: &[],
                        });
                    dispatch(&mut run, &mut requests[..count], leases, executor);
                }
                _ => {
                    let source = match run.kind {
                        WorkKind::Reorder => &scratch[..self.size()],
                        WorkKind::Initialize | WorkKind::Fused if self.separate() => input.unwrap(),
                        _ => &[],
                    };
                    let product =
                        matches!(run.kind, WorkKind::Finish | WorkKind::Fused) && factor.is_some();
                    let leases = values
                        .chunks_exact_mut(self.tile)
                        .enumerate()
                        .skip(first)
                        .take(count)
                        .map(|(i, output)| Lease {
                            values: output,
                            pair: &mut [],
                            source,
                            factor: if product {
                                &factor.unwrap()[i * self.tile..(i + 1) * self.tile]
                            } else {
                                &[]
                            },
                        });
                    dispatch(&mut run, &mut requests[..count], leases, executor);
                }
            }
        }
    }
}

impl<M: PrimeModulus> FftPlan<'_, M> {
    /// Scratch fields for a polynomial-major batch of full in-place transforms.
    ///
    /// Partitions the task budget across polynomials and within each transform.
    /// Separate inputs, preserved inputs, and prefixes return
    /// [`FftError::InvalidExecution`], including for an empty batch.
    /// An empty batch otherwise needs no scratch. Returns
    /// [`FftError::SizeOverflow`] if the scratch count overflows `usize` or its
    /// field slice would exceed `isize::MAX` bytes.
    pub fn batch_fields(self, count: usize) -> Result<usize, FftError> {
        self.batch_fields_with(count, NonZeroUsize::new(self.budget.get()).unwrap())
    }

    pub(crate) fn batch_fields_with(
        self,
        count: usize,
        max_tasks: NonZeroUsize,
    ) -> Result<usize, FftError> {
        let (plan, jobs, _) = self.batch_geometry(count, max_tasks)?;
        super::super::check_field_count(
            plan.retained_fields()
                .checked_mul(jobs)
                .ok_or(FftError::SizeOverflow)?,
        )
    }

    fn batch_geometry(
        mut self,
        count: usize,
        max_tasks: NonZeroUsize,
    ) -> Result<(Self, usize, NonZeroUsize), FftError> {
        if self.separate() || self.request.support != InputSupport::Full {
            return Err(FftError::InvalidExecution);
        }
        let mut jobs = count.min(max_tasks.get());
        let inner = loop {
            let inner = NonZeroUsize::new(max_tasks.get() / jobs.max(1)).unwrap();
            if let Some(limit) = self.memory_limit {
                let options = crate::exec::ExecutionOptions::DEFAULT
                    .with_task_budget(crate::exec::TaskBudget::new(inner.get()).unwrap())
                    .with_memory_limit(limit / jobs.max(1));
                self.columns = super::super::Strategy::columns(
                    self.tile,
                    self.fragments(),
                    options,
                    usize::MAX,
                );
                if self
                    .retained_fields()
                    .saturating_mul(jobs)
                    .saturating_mul(core::mem::size_of::<PastaField<M>>())
                    > limit
                    && jobs > 1
                {
                    jobs = jobs.div_ceil(2);
                    continue;
                }
            } else if let Some((columns, tasks)) = self.columns {
                self.columns = Some((columns, tasks.min(inner.get())));
            }
            break inner;
        };
        Ok((self, jobs, inner))
    }

    /// Executes consecutive full-domain polynomials in place without allocating.
    ///
    /// Each consecutive [`Self::size`]-element block holds one polynomial and
    /// follows the order and scaling contract of [`Self::execute`]. Scratch
    /// must meet [`Self::batch_fields`] for this polynomial count. Empty batches
    /// are accepted when the plan supports batching.
    ///
    /// The plan must select full in-place transforms. Panics if the plan or buffers do
    /// not meet these requirements, before writing any output. Executor panic behavior
    /// follows [`Self::execute`].
    pub fn execute_batch<E: Executor>(
        self,
        values: &mut [PastaField<M>],
        scratch: &mut [PastaField<M>],
        executor: &E,
    ) {
        self.execute_batch_with(
            values,
            scratch,
            NonZeroUsize::new(self.budget.get()).unwrap(),
            executor,
        )
    }

    pub(crate) fn execute_batch_with<E: Executor>(
        self,
        values: &mut [PastaField<M>],
        scratch: &mut [PastaField<M>],
        max_tasks: NonZeroUsize,
        executor: &E,
    ) {
        assert!(
            values.len().is_multiple_of(self.size()),
            "batch must contain complete polynomials"
        );
        assert!(
            !self.separate() && self.request.support == InputSupport::Full,
            "batching requires full in-place transforms"
        );
        let count = values.len() / self.size();
        let fields = self
            .batch_fields_with(count, max_tasks)
            .expect("scratch bounded by batch storage");
        super::super::check_scratch(fields, scratch.len());
        let (plan, jobs, inner) = self
            .batch_geometry(count, max_tasks)
            .expect("batch-compatible plan");
        fn visit<M: PrimeModulus, E: Executor>(
            plan: FftPlan<'_, M>,
            values: &mut [PastaField<M>],
            scratch: &mut [PastaField<M>],
            jobs: usize,
            inner: NonZeroUsize,
            executor: &E,
        ) {
            if jobs <= 1 {
                for values in values.chunks_exact_mut(plan.size()) {
                    plan.execute_with(None, values, None, scratch, inner, executor);
                }
            } else {
                let left_jobs = jobs / 2;
                let count = values.len() / plan.size();
                let (left, right) = values.split_at_mut(count / 2 * plan.size());
                let (a, b) = scratch.split_at_mut(left_jobs * plan.retained_fields());
                executor.join(
                    || visit(plan, left, a, left_jobs, inner, executor),
                    || visit(plan, right, b, jobs - left_jobs, inner, executor),
                );
            }
        }
        visit(plan, values, &mut scratch[..fields], jobs, inner, executor);
    }
}
