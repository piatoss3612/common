//! A shared window queue with disjoint, reusable worker scratch.
//!
//! Workers borrow prepared scalar records and digits for the whole batch. Each
//! task covers one input's recoding window; workers claim tasks from a shared
//! counter so they can move to another input after finishing a window.
//!
//! Each worker owns arithmetic scratch sized to cover every job and a result
//! row with one slot per batch window. Only the worker claiming a window writes
//! its result; other rows hold identity in that slot. This uses more result
//! storage to avoid shared mutable arithmetic buffers. All workers finish before
//! reduction reads their rows and folds each input's windows in order.

use super::*;
use core::sync::atomic::{AtomicUsize, Ordering};

pub(super) struct Layout {
    pub requirements: Requirements,
    work: Requirements,
    tasks: usize,
    workers: usize,
}

struct Batch<'a, 'i, C: PastaCurve> {
    inputs: &'a [Input<'i, C>],
    jobs: Option<&'a [JobStorage]>,
    options: ExecutionOptions,
}

impl<C: PastaCurve> Batch<'_, '_, C> {
    fn job(&self, index: usize) -> Result<JobStorage, CurveError> {
        self.jobs.map_or_else(
            || job(&self.inputs[index], self.options),
            |jobs| Ok(jobs[index]),
        )
    }

    fn range(&self, range: core::ops::Range<usize>) -> Batch<'_, '_, C> {
        Batch {
            inputs: &self.inputs[range.clone()],
            jobs: self.jobs.map(|jobs| &jobs[range]),
            options: self.options,
        }
    }

    fn preparation(&self) -> Result<Requirements, CurveError> {
        (0..self.inputs.len()).try_fold(ZERO, |sum, index| {
            let required = self.job(index)?.requirements;
            sum.plus(Requirements {
                scalars: required.scalars,
                digits: required.digits,
                ..ZERO
            })
        })
    }

    fn layout(&self) -> Result<Layout, CurveError> {
        let mut work = ZERO;
        let mut tasks = 0;
        for (index, input) in self.inputs.iter().enumerate() {
            if input.is_empty() {
                continue;
            }
            let job = self.job(index)?;
            work = work.include(job.work);
            tasks = add(tasks, job.geometry.windows())?;
        }
        let workers = min(tasks, self.options.task_budget.get());
        // Each worker has a result slot for every window; slots for windows
        // claimed by other workers stay at identity for reduction.
        work.projective = add(work.projective, tasks)?;
        let requirements = self
            .preparation()?
            .plus(work.times::<C>(workers)?)?
            .times::<C>(1)?;
        Ok(Layout {
            requirements,
            work,
            tasks,
            workers,
        })
    }
}

pub(super) fn layout<C: PastaCurve>(
    inputs: &[Input<'_, C>],
    options: ExecutionOptions,
) -> Result<Layout, CurveError> {
    Batch {
        inputs,
        jobs: None,
        options,
    }
    .layout()
}

fn prepare_inputs<C: PastaCurve, X: Executor>(
    batch: &Batch<'_, '_, C>,
    executor: &X,
    scalars: &mut [ScalarStorage<C>],
    digits: &mut [u8],
    budget: TaskBudget,
) {
    if batch.inputs.len() > 1 {
        let mid = batch.inputs.len() / 2;
        let left = batch.range(0..mid);
        let right = batch.range(mid..batch.inputs.len());
        let required = left.preparation().unwrap();
        let (sa, sb) = scalars.split_at_mut(required.scalars);
        let (da, db) = digits.split_at_mut(required.digits);
        // Preparation owns disjoint input slices, but all inputs still share
        // the batch's task budget, including nested preparation and recoding.
        if let Some((a, b)) = budget.split_at(budget.get() / 2) {
            executor.join(
                || prepare_inputs(&left, executor, sa, da, a),
                || prepare_inputs(&right, executor, sb, db, b),
            );
        } else {
            prepare_inputs(&left, executor, sa, da, budget);
            prepare_inputs(&right, executor, sb, db, budget);
        }
    } else if let Some(input) = batch.inputs.first().filter(|input| !input.is_empty()) {
        let job = batch.job(0).unwrap();
        let records = match input.scalars {
            Scalars::Prepared(source) => source.records,
            source => {
                prepared::prepare(source, scalars, budget, executor);
                scalars
            }
        };
        if !digits.is_empty() {
            recode::write_parallel(records, job.geometry, digits, budget, executor);
        }
    }
}

pub(super) fn execute<C: PastaCurve, X: Executor>(
    inputs: &[Input<'_, C>],
    jobs: Option<&[JobStorage]>,
    output: &mut [ProjectivePoint<C>],
    options: ExecutionOptions,
    executor: &X,
    scratch: Scratch<'_, C>,
) {
    let batch = Batch {
        inputs,
        jobs,
        options,
    };
    let layout = batch.layout().unwrap();
    output.fill(ProjectivePoint::IDENTITY);
    if layout.tasks == 0 {
        return;
    }
    prepare_inputs(
        &batch,
        executor,
        scratch.scalars,
        scratch.digits,
        options.task_budget,
    );
    let next = AtomicUsize::new(0);
    run_workers(
        &batch,
        scratch.scalars,
        scratch.digits,
        &next,
        Work {
            affine: scratch.affine,
            projective: scratch.projective,
            field: scratch.field,
            indices: scratch.indices,
        },
        &layout,
        layout.workers,
        executor,
    );
    reduce(
        &batch,
        output,
        scratch.projective,
        layout.work.projective,
        0,
        options.task_budget,
        executor,
    );
}

// Outputs are independent once all windows have completed. Preserve each
// output's window order while sharing the reduction phase across the pool.
fn reduce<C: PastaCurve, X: Executor>(
    batch: &Batch<'_, '_, C>,
    output: &mut [ProjectivePoint<C>],
    rows: &[ProjectivePoint<C>],
    stride: usize,
    mut first: usize,
    budget: TaskBudget,
    executor: &X,
) {
    if output.len() > 1
        && let Some((a, b)) = budget.split_at(budget.get() / 2)
    {
        let mid = output.len() / 2;
        let left = batch.range(0..mid);
        let right = batch.range(mid..output.len());
        let mut boundary = first;
        for (index, input) in left.inputs.iter().enumerate() {
            if !input.is_empty() {
                boundary += left.job(index).unwrap().geometry.windows();
            }
        }
        let (oa, ob) = output.split_at_mut(mid);
        executor.join(
            || reduce(&left, oa, rows, stride, first, a, executor),
            || reduce(&right, ob, rows, stride, boundary, b, executor),
        );
        return;
    }
    for (index, (input, result)) in batch.inputs.iter().zip(output).enumerate() {
        if input.is_empty() {
            continue;
        }
        let geometry = batch.job(index).unwrap().geometry;
        for window in (0..geometry.windows()).rev() {
            for _ in 0..geometry.width() {
                *result = result.double();
            }
            for row in rows.chunks_exact(stride) {
                *result = result.add(&row[first + window]);
            }
        }
        first += geometry.windows();
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Workers borrow shared records and disjoint arithmetic scratch."
)]
fn run_workers<C: PastaCurve, X: Executor>(
    batch: &Batch<'_, '_, C>,
    scalars: &[ScalarStorage<C>],
    digits: &[u8],
    next: &AtomicUsize,
    mut work: Work<'_, C>,
    layout: &Layout,
    workers: usize,
    executor: &X,
) {
    if workers > 1 {
        let left = workers / 2;
        let (aa, ab) = work.affine.split_at_mut(layout.work.affine * left);
        let (pa, pb) = work.projective.split_at_mut(layout.work.projective * left);
        let (fa, fb) = work.field.split_at_mut(layout.work.field * left);
        let (ia, ib) = work.indices.split_at_mut(layout.work.indices * left);
        executor.join(
            || {
                run_workers(
                    batch,
                    scalars,
                    digits,
                    next,
                    Work {
                        affine: aa,
                        projective: pa,
                        field: fa,
                        indices: ia,
                    },
                    layout,
                    left,
                    executor,
                )
            },
            || {
                run_workers(
                    batch,
                    scalars,
                    digits,
                    next,
                    Work {
                        affine: ab,
                        projective: pb,
                        field: fb,
                        indices: ib,
                    },
                    layout,
                    workers - left,
                    executor,
                )
            },
        );
        return;
    }
    let (results, projective) = work.projective.split_at_mut(layout.tasks);
    work.projective = projective;
    // Every row participates in reduction, including rows of workers that claim
    // no tasks. Clear old results so those slots contribute the identity.
    results.fill(ProjectivePoint::IDENTITY);
    loop {
        // The counter only assigns unique windows; it publishes no data.
        // Preparation precedes the joins, and reduction waits for their return,
        // so result visibility does not depend on the counter's ordering.
        let index = next.fetch_add(1, Ordering::Relaxed);
        if index >= layout.tasks {
            break;
        }
        let mut window = index;
        let mut scalars = scalars;
        let mut digits = digits;
        for (job_index, input) in batch.inputs.iter().enumerate() {
            if input.is_empty() {
                continue;
            }
            let job = batch.job(job_index).unwrap();
            let (records, remaining_scalars) = scalars.split_at(job.requirements.scalars);
            let (recoded, remaining_digits) = digits.split_at(job.requirements.digits);
            if window < job.geometry.windows() {
                let (records, recoded) = match input.scalars {
                    Scalars::Prepared(source) => (
                        source.records,
                        source
                            .cached
                            .filter(|c| c.geometry == job.geometry)
                            .map_or(recoded, |cached| cached.digits),
                    ),
                    _ => (records, recoded),
                };
                results[index] = kernels::run(
                    input,
                    records,
                    recoded,
                    Task {
                        offset: 0,
                        window,
                        pass: job.pass,
                        geometry: job.geometry,
                        accumulation: job.accumulation,
                    },
                    &mut work,
                );
                break;
            }
            window -= job.geometry.windows();
            scalars = remaining_scalars;
            digits = remaining_digits;
        }
    }
}
