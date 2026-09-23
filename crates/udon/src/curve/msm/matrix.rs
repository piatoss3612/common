//! Shared scalar recoding with bounded output tiles.

use super::{
    Bases, CurveError, Input, PastaCurve, PreparedScalars, ProjectivePoint, Requirements, Scalars,
    Scratch, assert_length, checked_count, kernels,
    recode::{self, Geometry},
    schedule::{self, JobStorage, Options},
};
use crate::exec::{ExecutionOptions, Executor};

/// A borrowed matrix of bases multiplied by one prepared scalar vector.
///
/// Output `j` is `sum(scalars[i] * bases[j * output_stride + i * term_stride])`.
/// Strides count bases, including when [`Bases`] holds compact tables. Both
/// strides may be zero and immutable rows may overlap. The scalar count defines
/// the number of terms in every output. Either zero dimension addresses no bases;
/// zero terms produce identity outputs, and zero outputs do no arithmetic.
///
/// Construction checks the address extent. Base values retain [`Bases`]' point
/// and table invariants. The scalar preparation and base storage remain borrowed;
/// no matrix copy, index array, or allocation is required. Arithmetic is
/// variable-time, with no constant-time guarantee for secret inputs.
///
/// Multiply two base rows by the same signed coefficients:
///
/// ```
/// use zakura_udon::{
///     curve::{AffinePoint, Pallas, ProjectivePoint, msm::*},
///     exec::{ExecutionOptions, SerialExecutor, TaskBudget},
///     field::PastaField,
/// };
/// let g = AffinePoint::<Pallas>::GENERATOR;
/// let bases = [g, g, g, g.neg()];
/// let mut records = [ScalarStorage::ZERO; 2];
/// let scalars = PreparedScalars::signed(
///     &[2, -1], &mut records, TaskBudget::SERIAL, &SerialExecutor,
/// );
/// let matrix = SharedScalarInput::new(Bases::Affine(&bases), scalars, 2, 2, 1)?;
/// let options = ExecutionOptions::DEFAULT;
/// let r = matrix.requirements(options)?;
/// let mut digits = vec![0; r.digits()];
/// let mut affine = vec![g; r.affine()];
/// let mut projective = vec![ProjectivePoint::IDENTITY; r.projective()];
/// let mut field = vec![PastaField::ZERO; r.field()];
/// let mut indices = vec![0; r.indices()];
/// let scratch = Scratch::new(&mut [], &mut digits, &mut affine,
///     &mut projective, &mut field, &mut indices);
/// let mut output = [ProjectivePoint::IDENTITY; 2];
/// matrix.execute(&mut output, options, &SerialExecutor, scratch)?;
/// assert_eq!(output[0], g.to_projective());
/// assert_eq!(output[1], g.mul_projective(&PastaField::from_u64(3)));
/// # Ok::<(), zakura_udon::curve::CurveError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SharedScalarInput<'a, C: PastaCurve> {
    bases: Bases<'a, C>,
    scalars: PreparedScalars<'a, C>,
    outputs: usize,
    output_stride: usize,
    term_stride: usize,
}

impl<'a, C: PastaCurve> SharedScalarInput<'a, C> {
    /// Checks a matrix whose term count is `scalars.len()`.
    ///
    /// Row-major storage uses `(output_stride, term_stride) = (terms, 1)`;
    /// term-major storage uses `(1, outputs)`. Padding and repeated bases are
    /// allowed. Returns [`CurveError::SizeOverflow`] if the output slice cannot
    /// be represented or base-index arithmetic overflows, and [`CurveError::MatrixTooSmall`]
    /// if that address exceeds `bases.len()`. Empty shapes ignore both strides.
    pub fn new(
        bases: Bases<'a, C>,
        scalars: PreparedScalars<'a, C>,
        outputs: usize,
        output_stride: usize,
        term_stride: usize,
    ) -> Result<Self, CurveError> {
        checked_count::<ProjectivePoint<C>>(outputs, 1)?;
        let required = extent(outputs, scalars.len(), output_stride, term_stride)?;
        if required > bases.len() {
            return Err(CurveError::MatrixTooSmall {
                required,
                provided: bases.len(),
            });
        }
        Ok(Self {
            bases,
            scalars,
            outputs,
            output_stride,
            term_stride,
        })
    }

    /// Number of output sums.
    pub const fn outputs(&self) -> usize {
        self.outputs
    }

    /// Number of shared scalar coefficients.
    pub const fn terms(&self) -> usize {
        self.scalars.len()
    }

    /// Scratch counts under the supplied task and workspace ceilings.
    ///
    /// Counts include shared temporary recoding and bounded result tiles. They
    /// exclude borrowed scalar/base preparation, output, and stack frames, as in
    /// [`ExecutionOptions`]. Execution can adapt to smaller supplied capacities.
    /// Returns [`CurveError::SizeOverflow`] for unrepresentable storage or
    /// [`CurveError::MemoryLimit`] if the planner finds no fitting layout.
    pub fn requirements(&self, options: ExecutionOptions) -> Result<Requirements, CurveError> {
        Ok(self.plan(options, None)?.requirements)
    }

    /// Writes one sum per output, sharing scalar recoding across base rows.
    ///
    /// The output length must equal [`Self::outputs`], or this panics before
    /// writes. Resource errors match [`Self::requirements`], with
    /// [`CurveError::ScratchTooSmall`] when supplied capacities cannot fit the
    /// selected arithmetic. The planner's stopping point is not a proven minimum
    /// memory requirement. All returned errors precede output and scratch writes;
    /// unused scratch tails remain untouched.
    ///
    /// The executor may work across outputs and within each output under one task
    /// budget. An executor panic may leave partial output and scratch. Reuse is
    /// valid after all scoped jobs finish unwinding; there is no whole-call rollback.
    pub fn execute<X: Executor>(
        &self,
        output: &mut [ProjectivePoint<C>],
        options: ExecutionOptions,
        executor: &X,
        scratch: Scratch<'_, C>,
    ) -> Result<(), CurveError> {
        assert_length("output", self.outputs, output.len());
        let plan = self.plan(options, Some(scratch.capacity()))?;
        let scratch = scratch.checked(plan.requirements);
        output.fill(ProjectivePoint::IDENTITY);
        if self.outputs == 0 || self.terms() == 0 {
            return Ok(());
        }
        let windows = plan.job.geometry.windows();
        let (results, projective) = scratch.projective.split_at_mut(plan.results);
        let mut work = Scratch::new(
            scratch.scalars,
            &mut [],
            scratch.affine,
            projective,
            scratch.field,
            scratch.indices,
        );
        for first in (0..self.terms()).step_by(plan.job.cap) {
            let end = self.terms().min(first + plan.job.cap);
            let records = &self.scalars.records[first..end];
            let digits = if let Some(cache) = self
                .scalars
                .cached
                .filter(|c| c.geometry == plan.job.geometry && plan.job.cap == self.terms())
            {
                cache.digits
            } else {
                let len = plan.job.geometry.storage_len(records.len()).unwrap();
                let digits = &mut scratch.digits[..len];
                recode::write(records, plan.job.geometry, digits);
                &*digits
            };
            for (tile, output) in output.chunks_mut(plan.lanes * plan.workers).enumerate() {
                let start = tile * plan.lanes * plan.workers;
                let tasks = output.len().div_ceil(plan.lanes) * windows;
                let results = &mut results[..tasks * plan.lanes];
                let task = kernels::Task {
                    offset: first,
                    window: 0,
                    pass: plan.job.pass,
                    geometry: plan.job.geometry,
                    accumulation: plan.job.accumulation,
                };
                run_tasks(
                    self,
                    records,
                    digits,
                    task,
                    start,
                    plan.lanes,
                    results,
                    0,
                    plan.workers.min(tasks),
                    executor,
                    work.reborrow(),
                    plan.work,
                );
                for (lane, output) in output.iter_mut().enumerate() {
                    let group = lane / plan.lanes;
                    let lane = lane % plan.lanes;
                    let mut sum = ProjectivePoint::IDENTITY;
                    for window in (0..windows).rev() {
                        for _ in 0..plan.job.geometry.width() {
                            sum = sum.double();
                        }
                        sum = sum.add(&results[(group * windows + window) * plan.lanes + lane]);
                    }
                    *output = output.add(&sum);
                }
            }
        }
        Ok(())
    }

    fn plan(
        &self,
        requested: ExecutionOptions,
        capacity: Option<Requirements>,
    ) -> Result<Plan, CurveError> {
        if self.outputs == 0 || self.terms() == 0 {
            return Ok(Plan {
                job: JobStorage::EMPTY,
                lanes: 1,
                workers: 1,
                results: 0,
                work: JobStorage::EMPTY.work,
                requirements: JobStorage::EMPTY.requirements,
            });
        }
        // Only scalar/base facts are queried; this synthetic input is never
        // executed as a dense row over the matrix's physical storage.
        let facts = Input {
            bases: self.bases,
            scalars: Scalars::Prepared(self.scalars),
            indices: None,
        };
        let mut options = Options::new(requested.into());
        // Bound bucket multiplication independently of the matrix dimensions.
        let mut max_lanes = 4;
        loop {
            let job = schedule::job(&facts, options)?;
            let lanes = if matches!(job.geometry, Geometry::Short(_)) {
                self.outputs.min(max_lanes)
            } else {
                1
            };
            let groups = self.outputs.div_ceil(lanes);
            let workers = options
                .task_budget
                .get()
                .min(groups.saturating_mul(job.geometry.windows()));
            let work = job.work.times::<C>(lanes)?;
            let results = checked_count::<ProjectivePoint<C>>(
                groups.min(workers),
                job.geometry.windows() * lanes,
            )?;
            let requirements = work
                .times::<C>(workers)?
                .plus(Requirements {
                    digits: job.requirements.digits,
                    projective: results,
                    ..JobStorage::EMPTY.requirements
                })?
                .times::<C>(1)?;
            let bytes = requirements.bytes::<C>()?;
            let memory_ok = requested.memory_limit().is_none_or(|limit| bytes <= limit);
            if memory_ok && capacity.is_none_or(|c| requirements.fits(c)) {
                return Ok(Plan {
                    job,
                    lanes,
                    workers,
                    results,
                    work,
                    requirements,
                });
            }
            if lanes > 1 {
                max_lanes = lanes.div_ceil(2);
            } else if let Some(smaller) = schedule::smaller(options, job.cap) {
                options = smaller;
            } else if !memory_ok {
                return Err(CurveError::MemoryLimit {
                    limit: requested.memory_limit().unwrap(),
                    required: bytes,
                });
            } else {
                return Err(requirements.capacity_error(capacity.unwrap()));
            }
        }
    }
}

fn extent(
    outputs: usize,
    terms: usize,
    output_stride: usize,
    term_stride: usize,
) -> Result<usize, CurveError> {
    if outputs == 0 || terms == 0 {
        return Ok(0);
    }
    (outputs - 1)
        .checked_mul(output_stride)
        .and_then(|last| {
            (terms - 1)
                .checked_mul(term_stride)
                .and_then(|term| last.checked_add(term))
        })
        .and_then(|last| last.checked_add(1))
        .ok_or(CurveError::SizeOverflow)
}

struct Plan {
    job: JobStorage,
    lanes: usize,
    workers: usize,
    results: usize,
    work: Requirements,
    requirements: Requirements,
}

#[expect(
    clippy::too_many_arguments,
    reason = "Scoped tiles share scalar data and own disjoint results and scratch."
)]
fn run_tasks<C: PastaCurve, X: Executor>(
    input: &SharedScalarInput<'_, C>,
    records: &[super::ScalarStorage<C>],
    digits: &[u8],
    task: kernels::Task,
    output_start: usize,
    lanes: usize,
    results: &mut [ProjectivePoint<C>],
    task_start: usize,
    workers: usize,
    executor: &X,
    mut scratch: Scratch<'_, C>,
    work: Requirements,
) {
    if workers > 1 {
        let left_workers = workers / 2;
        let left_tasks = (results.len() / lanes) / 2;
        let (left, right) = results.split_at_mut(left_tasks * lanes);
        let (a, b) = schedule::split_scratch(scratch, work.times::<C>(left_workers).unwrap());
        executor.join(
            || {
                run_tasks(
                    input,
                    records,
                    digits,
                    task,
                    output_start,
                    lanes,
                    left,
                    task_start,
                    left_workers,
                    executor,
                    a,
                    work,
                )
            },
            || {
                run_tasks(
                    input,
                    records,
                    digits,
                    task,
                    output_start,
                    lanes,
                    right,
                    task_start + left_tasks,
                    workers - left_workers,
                    executor,
                    b,
                    work,
                )
            },
        );
    } else {
        let scratch = scratch.reborrow().checked(work);
        let mut work = kernels::Work {
            affine: scratch.affine,
            projective: scratch.projective,
            field: scratch.field,
            indices: scratch.indices,
        };
        for (index, results) in results.chunks_mut(lanes).enumerate() {
            let index = task_start + index;
            let output = output_start + index / task.geometry.windows() * lanes;
            let task = kernels::Task {
                window: index % task.geometry.windows(),
                ..task
            };
            let used = lanes.min(input.outputs - output);
            kernels::shared(
                input.bases,
                output * input.output_stride,
                input.output_stride,
                input.term_stride,
                records,
                digits,
                task,
                &mut work,
                &mut results[..used],
            );
        }
    }
}

#[cfg(test)]
mod tests;
