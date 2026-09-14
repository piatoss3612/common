use super::*;
use crate::exec::{Executor, for_each_task_mut};

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
    mut leases: impl Iterator<Item = Lease<'b, M>>,
    executor: &E,
) -> Result<(), FftError> {
    if requests.len() == 1 {
        let mut task = run
            .try_claim(requests[0].take().unwrap(), || leases.next())
            .expect("structured claim")
            .expect("complete resource iterator");
        task.execute().expect("fresh task");
        return run
            .complete(task.finish())
            .expect("structured receipt")
            .error
            .map_or(Ok(()), Err);
    }
    let mut tasks: [Option<Task<'a, FftKernel<'t, M>, Lease<'b, M>>>; 32] =
        core::array::from_fn(|_| None);
    let count = requests.len();
    for ((slot, request), lease) in tasks.iter_mut().zip(requests).zip(leases) {
        *slot = run
            .try_claim(request.take().expect("ready request"), || Some(lease))
            .expect("structured task claim");
    }
    for_each_task_mut(&mut tasks[..count], executor, |_, task| {
        task.as_mut()
            .expect("complete resource iterator")
            .execute()
            .expect("fresh task")
    });
    let mut error = None;
    for task in &mut tasks[..count] {
        let published = run
            .complete(task.take().unwrap().finish())
            .expect("structured receipt");
        error = error.or(published.error);
    }
    error.map_or(Ok(()), Err)
}

impl<M: PrimeModulus> FftPlan<'_, M> {
    /// Executes this run with contiguous buffers and the caller's scoped pool.
    ///
    /// `input` is `Some` exactly for a separate-output plan; otherwise values
    /// contain the disposable input. `factor`, when present, contains the full
    /// output-order product input. `scratch` must contain
    /// [`Self::retained_fields`] fields. Length and scratch
    /// errors are returned before mutation. This driver does not allocate.
    ///
    /// The task limit bounds detached envelopes in each structured join. It
    /// neither changes arithmetic grain nor assigns workers to this operation.
    /// All kernels use the same run protocol as application scheduling, but
    /// structured joins return their receipts together. A panic leaves working
    /// values canonical and waits for all joined tasks before propagating.
    pub fn execute<E: Executor>(
        self,
        input: Option<&[PastaField<M>]>,
        values: &mut [PastaField<M>],
        factor: Option<&[PastaField<M>]>,
        scratch: &mut [PastaField<M>],
        max_tasks: NonZeroUsize,
        executor: &E,
    ) -> Result<(), FftError> {
        if self.separate != input.is_some() {
            return Err(FftError::InvalidExecution);
        }
        super::super::check_length("output", self.size(), values.len())?;
        if let Some(input) = input {
            super::super::check_length("input", self.request.input_len(self.size()), input.len())?;
        }
        if let Some(factor) = factor {
            super::super::check_length("factor", self.size(), factor.len())?;
        }
        super::super::ScratchRequirements {
            field_elements: self.retained_fields(),
        }
        .check(scratch.len())?;
        if self.fragments() == 1 {
            let mut identity = Identity::new();
            let mut slot = [TaskStorage::EMPTY];
            let mut run = FftRun::new(self, factor.is_some(), &mut identity, &mut slot)
                .expect("one local task");
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
            return run
                .complete(task.finish())
                .expect("local receipt")
                .error
                .map_or(Ok(()), Err);
        }
        let max_tasks = if self.first() > self.size()
            || self.separate && self.request.input_len(self.size()) <= 1
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
    ) -> Result<(), FftError> {
        let mut identity = Identity::new();
        let mut slots = [const { TaskStorage::EMPTY }; 32];
        let mut run =
            FftRun::new(self, factor.is_some(), &mut identity, &mut slots).expect("fixed frontier");
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
                )?,
                WorkKind::Permute => dispatch(
                    &mut run,
                    &mut requests[..count],
                    core::iter::once(empty(values)),
                    executor,
                )?,
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
                    dispatch(&mut run, &mut requests[..count], leases, executor)?;
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
                    dispatch(&mut run, &mut requests[..count], leases, executor)?;
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
                    dispatch(&mut run, &mut requests[..count], leases, executor)?;
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
                    dispatch(&mut run, &mut requests[..count], leases, executor)?;
                }
                _ => {
                    let source = match run.kind {
                        WorkKind::Reorder => &scratch[..self.size()],
                        WorkKind::Initialize | WorkKind::Fused if self.separate => input.unwrap(),
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
                    dispatch(&mut run, &mut requests[..count], leases, executor)?;
                }
            }
        }
        Ok(())
    }
}
