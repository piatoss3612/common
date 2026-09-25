//! Batch schedules over contiguous inputs with caller-owned metadata.

use super::super::{
    ArithmeticOptions, BatchOptions, CurveError, Input, PastaCurve, ProjectivePoint, Requirements,
    Scratch, assert_length, assert_scratch, checked_count, min,
    schedule::{JobStorage, Options, Plan, job, requirements, split},
};
use super::MsmPlan;
use crate::exec::{ExecutionOptions, Executor, TaskBudget};

/// Initialized opaque metadata for a scheduled contiguous job range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkerStorage {
    begin: usize,
    end: usize,
    requirements: Requirements,
}
impl WorkerStorage {
    /// Initializer; populated by [`BatchPlan::new`].
    pub const EMPTY: Self = Self {
        begin: 0,
        end: 0,
        requirements: Requirements::ZERO,
    };
}
pub(in crate::curve::msm) fn execute<C: PastaCurve, X: Executor>(
    plan: &Plan,
    inputs: &[Input<'_, C>],
    output: &mut [ProjectivePoint<C>],
    executor: &X,
    scratch: Scratch<'_, C>,
) {
    execute_inputs(inputs, output, plan.options, executor, scratch);
}
fn execute_inputs<C: PastaCurve, X: Executor>(
    inputs: &[Input<'_, C>],
    output: &mut [ProjectivePoint<C>],
    options: Options,
    executor: &X,
    mut scratch: Scratch<'_, C>,
) {
    if let Some((mid, left)) = split(inputs, options.task_budget.get()) {
        let a = options.with_task_budget(TaskBudget::new(left).unwrap());
        let b =
            options.with_task_budget(TaskBudget::new(options.task_budget.get() - left).unwrap());
        let (sa, sb) = scratch.split(requirements(&inputs[..mid], a).unwrap());
        let (oa, ob) = output.split_at_mut(mid);
        executor.join(
            || execute_inputs(&inputs[..mid], oa, a, executor, sa),
            || execute_inputs(&inputs[mid..], ob, b, executor, sb),
        );
    } else {
        for (input, result) in inputs.iter().zip(output) {
            *result = execute_job(
                input,
                job(input, options).unwrap(),
                options,
                executor,
                scratch.reborrow(),
            );
        }
    }
}
fn execute_job<C: PastaCurve, X: Executor>(
    input: &Input<'_, C>,
    job: JobStorage,
    options: Options,
    executor: &X,
    scratch: Scratch<'_, C>,
) -> ProjectivePoint<C> {
    MsmPlan::from_job(
        input.len(),
        options.arithmetic,
        job,
        options.specializes_short(),
        options.memory_limit(),
    )
    .execute(*input, executor, scratch)
}

/// Borrowed reusable plan over immutable inputs and caller-owned metadata.
///
/// The plan borrows both bases and scalar rows through its inputs. To execute
/// changing scalar rows over retained bases, rebind a
/// [`Selection`](crate::curve::msm::Selection) and
/// build a new plan for the rebound inputs.
///
/// Different output buffers and previously used scratch may be supplied on every
/// execution, including concurrent calls with separate buffers. The
/// [memory ceiling](ExecutionOptions::with_memory_limit) covers scratch from
/// [`Self::requirements`]. Metadata sized by [`Self::storage_len`] is separate.
pub struct BatchPlan<'a, 'i, C: PastaCurve> {
    inputs: &'a [Input<'i, C>],
    jobs: &'a [JobStorage],
    workers: &'a [WorkerStorage],
    plan: Plan,
    temporary_bytes: usize,
}
impl<'a, 'i, C: PastaCurve> BatchPlan<'a, 'i, C> {
    /// Returns required metadata counts as `(job entries, worker entries)`.
    ///
    /// Supply initialized [`JobStorage`] and [`WorkerStorage`] slices with at
    /// least these lengths to [`Self::new`], using the same options and input
    /// count. The counts conservatively reserve storage even if planning uses
    /// fewer worker ranges. Returns [`CurveError::SizeOverflow`] if either slice
    /// would be too large.
    pub const fn storage_len(
        inputs: usize,
        options: ExecutionOptions,
    ) -> Result<(usize, usize), CurveError> {
        Self::storage_len_with(
            inputs,
            BatchOptions::new(ArithmeticOptions::DEFAULT).with_task_budget(options.task_budget()),
        )
    }
    pub(in crate::curve::msm) const fn storage_len_with(
        inputs: usize,
        options: BatchOptions,
    ) -> Result<(usize, usize), CurveError> {
        Ok((
            size!(checked_count::<JobStorage>(inputs, 1)),
            size!(checked_count::<WorkerStorage>(
                min(inputs, options.task_budget.get()),
                1
            )),
        ))
    }
    /// Builds a reusable schedule into initialized metadata.
    ///
    /// Size metadata with [`Self::storage_len`] and initialize it with
    /// [`JobStorage::EMPTY`] and [`WorkerStorage::EMPTY`]. Short buffers panic before
    /// writes. Returns [`CurveError::SizeOverflow`] for unrepresentable sizes, or
    /// [`CurveError::MemoryLimit`] under the [workspace
    /// ceiling](ExecutionOptions::with_memory_limit). All returned errors precede
    /// writes; tails beyond the required metadata prefixes remain untouched.
    pub fn new(
        inputs: &'a [Input<'i, C>],
        options: ExecutionOptions,
        jobs: &'a mut [JobStorage],
        workers: &'a mut [WorkerStorage],
    ) -> Result<Self, CurveError> {
        Self::new_with(inputs, options.into(), jobs, workers)
    }
    pub(in crate::curve::msm) fn new_with(
        inputs: &'a [Input<'i, C>],
        options: BatchOptions,
        jobs: &'a mut [JobStorage],
        workers: &'a mut [WorkerStorage],
    ) -> Result<Self, CurveError> {
        let (j, w) = Self::storage_len_with(inputs.len(), options)?;
        assert_scratch("jobs", j, jobs.len());
        assert_scratch("workers", w, workers.len());
        let plan = Plan::new(inputs, options)?;
        let temporary_bytes = plan.requirements.bytes::<C>()?;
        let jobs = &mut jobs[..j];
        let workers = &mut workers[..w];
        let used = fill_metadata(inputs, plan.options, jobs, workers, 0);
        Ok(Self {
            inputs,
            jobs,
            workers: &workers[..used],
            plan,
            temporary_bytes,
        })
    }
    /// Execution scratch counts, excluding the separately borrowed metadata.
    ///
    /// These are the required prefixes for [`Self::execute`]'s chosen schedule.
    /// A prefix may cover several jobs that reuse its storage; it is not a
    /// minimum over all possible schedules or input-specific arithmetic.
    pub const fn requirements(&self) -> Requirements {
        self.plan.requirements
    }
    /// Arithmetic workspace bytes, excluding metadata and unused buffer tails.
    pub const fn temporary_bytes(&self) -> usize {
        self.temporary_bytes
    }
    /// Number of job ranges in the retained schedule.
    ///
    /// A range may share scoped tasks across jobs or use tasks within each job.
    /// This count does not measure task concurrency or executor threads.
    #[cfg(test)]
    pub(in crate::curve::msm) const fn worker_ranges(&self) -> usize {
        self.workers.len()
    }
    /// Executes the retained plan, writing one result per input in input order.
    ///
    /// Panics before writes unless the output length equals the input count and scratch
    /// meets [`Self::requirements`]. Unused scratch tails remain untouched. An executor
    /// panic may partially write output and scratch; reuse after unwinding requires all
    /// scoped work to finish unwinding before buffers are reused.
    pub fn execute<X: Executor>(
        &self,
        output: &mut [ProjectivePoint<C>],
        executor: &X,
        scratch: Scratch<'_, C>,
    ) {
        assert_length("output", self.inputs.len(), output.len());
        let scratch = scratch.checked(self.plan.requirements);
        execute_workers(
            self.inputs,
            self.jobs,
            self.workers,
            output,
            0,
            self.plan.options,
            executor,
            scratch,
        );
    }
}
fn fill_metadata<C: PastaCurve>(
    inputs: &[Input<'_, C>],
    options: Options,
    jobs: &mut [JobStorage],
    workers: &mut [WorkerStorage],
    offset: usize,
) -> usize {
    if let Some((mid, left)) = split(inputs, options.task_budget.get()) {
        let (a, b) = jobs.split_at_mut(mid);
        let used = fill_metadata(
            &inputs[..mid],
            options.with_task_budget(TaskBudget::new(left).unwrap()),
            a,
            workers,
            offset,
        );
        used + fill_metadata(
            &inputs[mid..],
            options.with_task_budget(TaskBudget::new(options.task_budget.get() - left).unwrap()),
            b,
            &mut workers[used..],
            offset + mid,
        )
    } else if inputs.is_empty() {
        0
    } else {
        let mut requirements = Requirements::ZERO;
        for (input, storage) in inputs.iter().zip(jobs) {
            *storage = job(input, options).unwrap();
            requirements = requirements.include(storage.requirements);
        }
        workers[0] = WorkerStorage {
            begin: offset,
            end: offset + inputs.len(),
            requirements,
        };
        1
    }
}
#[expect(
    clippy::too_many_arguments,
    reason = "Scoped worker ranges carry disjoint output and scratch borrows."
)]
fn execute_workers<C: PastaCurve, X: Executor>(
    inputs: &[Input<'_, C>],
    jobs: &[JobStorage],
    workers: &[WorkerStorage],
    output: &mut [ProjectivePoint<C>],
    offset: usize,
    options: Options,
    executor: &X,
    mut scratch: Scratch<'_, C>,
) {
    if workers.len() > 1 {
        let mid = workers.len() / 2;
        let left = workers[..mid].iter().fold(Requirements::ZERO, |sum, w| {
            sum.plus(w.requirements).unwrap()
        });
        let (sa, sb) = scratch.split(left);
        let split = workers[mid].begin - offset;
        let (oa, ob) = output.split_at_mut(split);
        executor.join(
            || {
                execute_workers(
                    inputs,
                    jobs,
                    &workers[..mid],
                    oa,
                    offset,
                    options,
                    executor,
                    sa,
                )
            },
            || {
                execute_workers(
                    inputs,
                    jobs,
                    &workers[mid..],
                    ob,
                    offset + split,
                    options,
                    executor,
                    sb,
                )
            },
        );
    } else if let Some(worker) = workers.first() {
        for i in worker.begin..worker.end {
            let budget = jobs[i].budget;
            output[i - offset] = execute_job(
                &inputs[i],
                jobs[i],
                options.with_task_budget(budget),
                executor,
                scratch.reborrow(),
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        curve::{
            AffinePoint, Pallas,
            msm::{ArithmeticOptions, Bases, tests::Buffers},
        },
        exec::{SerialExecutor, TaskBudget},
        field::Fq,
    };
    use core::num::NonZeroUsize;
    use std::{vec, vec::Vec};

    #[test]
    fn weighted_ranges_preserve_dominant_and_odd_budgets() {
        let bases = vec![AffinePoint::<Pallas>::GENERATOR; 1024];
        let scalars = vec![Fq::ONE; 1024];
        let input = |n| Input::new(Bases::Affine(&bases[..n]), &scalars[..n]);
        let inputs = [input(2), input(1024), input(3), input(17)];
        for budget in [2, 3, 5, 17] {
            let options = BatchOptions::default()
                .with_task_budget(TaskBudget::new(budget).unwrap())
                .with_memory_limit(usize::MAX);
            let mut jobs = [JobStorage::EMPTY; 4];
            let mut workers = [WorkerStorage::EMPTY; 4];
            let plan = BatchPlan::new_with(&inputs, options, &mut jobs, &mut workers).unwrap();
            assert_eq!(plan.worker_ranges(), 1);
            assert_eq!(plan.jobs[1].budget.get(), budget);
        }
        let inputs = [input(2); 30];
        for budget in [3, 5] {
            let options = BatchOptions::default()
                .with_task_budget(TaskBudget::new(budget).unwrap())
                .with_memory_limit(usize::MAX);
            let mut jobs = [JobStorage::EMPTY; 30];
            let mut workers = [WorkerStorage::EMPTY; 5];
            let plan = BatchPlan::new_with(&inputs, options, &mut jobs, &mut workers).unwrap();
            assert_eq!(plan.worker_ranges(), budget);
            for worker in plan.workers {
                assert_eq!(worker.end - worker.begin, inputs.len() / budget);
            }
        }
    }

    #[test]
    fn metadata_partitions_cover_jobs_with_disjoint_scratch() {
        let bases = vec![AffinePoint::<Pallas>::GENERATOR; 513];
        let scalars = vec![Fq::ONE; 513];
        for sizes in [
            vec![],
            vec![0],
            vec![513, 2, 0, 1, 128],
            vec![2; 31],
            vec![0, 0, 0],
            vec![1, 513],
        ] {
            let inputs: Vec<_> = sizes
                .iter()
                .map(|&n| Input::new(Bases::Affine(&bases[..n]), &scalars[..n]))
                .collect();
            for tasks in [1, 2, 3, 5, 17, 65] {
                for pass in [1, 2, 17, 128, 513] {
                    for limit in [8192, 32768, 1048576] {
                        let options = BatchOptions::new(
                            ArithmeticOptions::DEFAULT
                                .with_max_terms_per_pass(NonZeroUsize::new(pass)),
                        )
                        .with_task_budget(TaskBudget::new(tasks).unwrap())
                        .with_memory_limit(limit);
                        let (j, w) =
                            BatchPlan::<Pallas>::storage_len_with(inputs.len(), options).unwrap();
                        let mut jobs = vec![JobStorage::EMPTY; j];
                        let mut workers = vec![WorkerStorage::EMPTY; w];
                        let plan =
                            match BatchPlan::new_with(&inputs, options, &mut jobs, &mut workers) {
                                Ok(p) => p,
                                Err(CurveError::MemoryLimit { .. }) => {
                                    assert!(jobs.iter().all(|j| *j == JobStorage::EMPTY));
                                    assert!(workers.iter().all(|w| *w == WorkerStorage::EMPTY));
                                    continue;
                                }
                                Err(e) => panic!("{e:?}"),
                            };
                        let mut visits = vec![0; inputs.len()];
                        let mut live_budget = 0;
                        let mut scratch = Buffers::new(plan.requirements());
                        let mut remaining = scratch.borrow();
                        for worker in plan.workers {
                            live_budget += plan.jobs[worker.begin..worker.end]
                                .iter()
                                .map(|job| job.budget.get())
                                .max()
                                .unwrap();
                            let (mut owned, rest) = remaining.split(worker.requirements);
                            remaining = rest;
                            for (i, visited) in visits
                                .iter_mut()
                                .enumerate()
                                .take(worker.end)
                                .skip(worker.begin)
                            {
                                *visited += 1;
                                let job = plan.jobs[i];
                                owned.reborrow().checked(job.requirements);
                                assert!(job.workers <= job.budget.get());
                                if !inputs[i].is_empty() {
                                    assert!(job.pass > 0 && job.pass <= job.cap);
                                }
                            }
                        }
                        assert!(visits.iter().all(|&n| n == 1));
                        assert!(live_budget <= tasks);
                        assert!(plan.temporary_bytes() <= limit);
                        // Different output values expose range/order mistakes.
                        let mut output = vec![ProjectivePoint::IDENTITY; inputs.len()];
                        plan.execute(&mut output, &SerialExecutor, scratch.borrow());
                        for (p, &n) in output.iter().zip(&sizes) {
                            assert_eq!(*p, bases[0].mul_projective(&Fq::from_u64(n as u64)));
                        }
                    }
                }
            }
        }
    }
}
