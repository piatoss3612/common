//! Contiguous compatibility provisioning over bounded operation runs.

use super::{
    Accumulation, Bases, CurveError, ExecutionOptions, Input, PastaCurve, ProjectivePoint,
    Requirements, ScalarStorage, Scalars, Scratch, check_length, check_scratch, checked_count,
    recode::Geometry,
};
use crate::exec::{Executor, TaskBudget};
use core::num::NonZeroUsize;

macro_rules! size {
    ($e:expr) => {
        match $e {
            Ok(v) => v,
            Err(e) => return Err(e),
        }
    };
}
const fn min(a: usize, b: usize) -> usize {
    if a < b { a } else { b }
}
const fn max(a: usize, b: usize) -> usize {
    if a > b { a } else { b }
}
const fn add(a: usize, b: usize) -> Result<usize, CurveError> {
    match a.checked_add(b) {
        Some(n) => Ok(n),
        None => Err(CurveError::SizeOverflow),
    }
}
const ZERO: Requirements = Requirements {
    scalars: 0,
    digits: 0,
    affine: 0,
    projective: 0,
    field: 0,
    indices: 0,
};
impl Requirements {
    const fn include(self, b: Self) -> Self {
        Self {
            scalars: max(self.scalars, b.scalars),
            digits: max(self.digits, b.digits),
            affine: max(self.affine, b.affine),
            projective: max(self.projective, b.projective),
            field: max(self.field, b.field),
            indices: max(self.indices, b.indices),
        }
    }
    pub(super) const fn plus(self, b: Self) -> Result<Self, CurveError> {
        Ok(Self {
            scalars: size!(add(self.scalars, b.scalars)),
            digits: size!(add(self.digits, b.digits)),
            affine: size!(add(self.affine, b.affine)),
            projective: size!(add(self.projective, b.projective)),
            field: size!(add(self.field, b.field)),
            indices: size!(add(self.indices, b.indices)),
        })
    }
    pub(super) const fn times<C: PastaCurve>(self, workers: usize) -> Result<Self, CurveError> {
        Ok(Self {
            scalars: size!(checked_count::<ScalarStorage<C>>(self.scalars, workers)),
            digits: size!(checked_count::<u8>(self.digits, workers)),
            affine: size!(checked_count::<super::AffinePoint<C>>(self.affine, workers)),
            projective: size!(checked_count::<ProjectivePoint<C>>(
                self.projective,
                workers
            )),
            field: size!(checked_count::<crate::field::PastaField<C::Base>>(
                self.field, workers
            )),
            indices: size!(checked_count::<usize>(self.indices, workers)),
        })
    }
}

/// Initialized opaque metadata for one input in an [`ExecutionPlan`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JobStorage {
    pub(super) geometry: Geometry,
    pub(super) cap: usize,
    pub(super) pass: usize,
    pub(super) workers: usize,
    budget: TaskBudget,
    streaming: bool,
    pub(super) accumulation: Accumulation,
    pub(super) work: Requirements,
    pub(super) requirements: Requirements,
}
impl JobStorage {
    /// Initializer; populated by [`ExecutionPlan::new`].
    pub const EMPTY: Self = Self {
        geometry: Geometry::Joint,
        cap: 0,
        pass: 0,
        workers: 0,
        budget: TaskBudget::SERIAL,
        streaming: false,
        accumulation: Accumulation::Affine,
        work: ZERO,
        requirements: ZERO,
    };
}
/// Initialized opaque metadata for a scheduled contiguous job range.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WorkerStorage {
    begin: usize,
    end: usize,
    requirements: Requirements,
}
impl WorkerStorage {
    /// Initializer; populated by [`ExecutionPlan::new`].
    pub const EMPTY: Self = Self {
        begin: 0,
        end: 0,
        requirements: ZERO,
    };
}

pub(super) const fn layout<C: PastaCurve>(
    terms: usize,
    geometry: Geometry,
    retained: bool,
    cached: bool,
    compact: bool,
    options: ExecutionOptions,
) -> Result<JobStorage, CurveError> {
    size!(checked_count::<ScalarStorage<C>>(terms, 1));
    if terms == 0 {
        return Ok(JobStorage::EMPTY);
    }
    let cap = cap(terms, options);
    let pass = match options.max_terms_per_pass {
        Some(c) => min(cap, c.get()),
        None => cap,
    };
    let windows = geometry.windows();
    let workers = if options.streaming {
        1
    } else {
        min(windows, options.task_budget.get())
    };
    let accumulation = match options.accumulation {
        // Tiny affine passes repeatedly invert sparse bucket levels. The
        // projective backend retains its buckets across all passes.
        Accumulation::Auto if pass < 128 => Accumulation::Projective,
        Accumulation::Auto => Accumulation::Affine,
        a => a,
    };
    let work = if options.streaming {
        Requirements {
            projective: size!(checked_count::<ProjectivePoint<C>>(
                windows,
                geometry.buckets()
            )),
            ..ZERO
        }
    } else {
        match geometry {
            Geometry::Short(bits) => Requirements {
                projective: if bits > 1 && cap >= 32 { 16 } else { 0 },
                ..ZERO
            },
            Geometry::Joint if compact => Requirements {
                projective: if !retained && cap >= 32 { 16 } else { 0 },
                ..ZERO
            },
            Geometry::Joint => Requirements {
                affine: size!(checked_count::<super::AffinePoint<C>>(pass, 9)),
                // Active-term compaction can produce any tail shorter than eight.
                projective: max(16, 8 * min(pass, 7)),
                field: max(
                    size!(checked_count::<crate::field::PastaField<C::Base>>(pass, 4)),
                    8 * min(pass, 7),
                ),
                indices: pass,
                ..ZERO
            },
            Geometry::Booth(_) => {
                let buckets = geometry.buckets();
                if matches!(accumulation, Accumulation::Projective) {
                    Requirements {
                        projective: max(buckets, 16),
                        ..ZERO
                    }
                } else {
                    let deposits = size!(add(
                        size!(checked_count::<super::AffinePoint<C>>(pass, 2)),
                        buckets
                    ));
                    Requirements {
                        affine: size!(add(deposits, buckets)),
                        projective: if matches!(accumulation, Accumulation::Hybrid) {
                            max(buckets, 16)
                        } else {
                            16
                        },
                        field: 2 * (deposits / 2),
                        indices: 3 * buckets,
                        ..ZERO
                    }
                }
            }
        }
    };
    let mut requirements = size!(work.times::<C>(workers));
    requirements.projective = size!(add(requirements.projective, windows));
    requirements.scalars = if retained { 0 } else { cap };
    requirements.digits = if cached && cap == terms && !options.streaming {
        0
    } else {
        size!(geometry.storage_len(cap))
    };
    // Validate final slice counts, including intermediate-result prefixes.
    requirements = size!(requirements.times::<C>(1));
    Ok(JobStorage {
        geometry,
        cap,
        pass,
        workers,
        budget: options.task_budget,
        streaming: options.streaming,
        accumulation,
        work,
        requirements,
    })
}
const fn cap(terms: usize, options: ExecutionOptions) -> usize {
    min(
        8192,
        match options.chunk_size {
            Some(c) => min(terms, c.get()),
            None => terms,
        },
    )
}
const fn arithmetic_options(terms: usize, mut options: ExecutionOptions) -> ExecutionOptions {
    options.task_budget = TaskBudget::SERIAL;
    if terms >= 4096 && options.window_bits.is_none() && !options.joint_tables {
        options.window_bits = Some(11);
    }
    options
}
const fn conservative<C: PastaCurve>(
    terms: usize,
    options: ExecutionOptions,
) -> Result<JobStorage, CurveError> {
    layout::<C>(
        terms,
        Geometry::for_len(
            cap(terms, options),
            arithmetic_options(cap(terms, options), options),
        ),
        false,
        false,
        false,
        options,
    )
}

// Shared deterministic memory search for const and runtime planning. Reduce
// affine staging first, then concurrency, then retain projective buckets, and
// finally shorten complete chunks (including records and digits).
const fn smaller(mut options: ExecutionOptions, n: usize) -> Option<ExecutionOptions> {
    let pass = match options.max_terms_per_pass {
        Some(p) => min(n, p.get()),
        None => n,
    };
    if pass > 128 {
        options.max_terms_per_pass = NonZeroUsize::new(128);
    } else if options.task_budget.get() > 1 {
        options.task_budget = match TaskBudget::new(options.task_budget.get().div_ceil(2)) {
            Some(b) => b,
            None => unreachable!(),
        };
    } else if matches!(options.accumulation, Accumulation::Auto)
        && matches!(Geometry::for_len(n, options), Geometry::Booth(_))
    {
        options.accumulation = Accumulation::Projective;
    } else if n > 1 {
        let chunk = n.div_ceil(2);
        options.chunk_size = NonZeroUsize::new(chunk);
        if chunk < super::BOOTH_MIN
            && !options.joint_tables
            && options.window_bits.is_none()
            && matches!(
                options.accumulation,
                Accumulation::Auto | Accumulation::Projective
            )
        {
            options.window_bits = Some(4);
            options.accumulation = Accumulation::Projective;
        }
    } else {
        return None;
    }
    Some(options)
}

pub(super) const fn single_requirements<C: PastaCurve>(
    terms: usize,
    mut options: ExecutionOptions,
) -> Result<Requirements, CurveError> {
    let mut r = size!(conservative::<C>(terms, options)).requirements;
    if let Some(limit) = options.memory_limit {
        while size!(r.bytes::<C>()) > limit {
            options = match smaller(options, cap(terms, options)) {
                Some(o) => o,
                None => {
                    return Err(CurveError::MemoryLimit {
                        limit,
                        required: size!(r.bytes::<C>()),
                    });
                }
            };
            r = size!(conservative::<C>(terms, options)).requirements;
        }
    }
    Ok(r)
}

fn job<C: PastaCurve>(
    input: &Input<'_, C>,
    options: ExecutionOptions,
) -> Result<JobStorage, CurveError> {
    let retained = match input.scalars {
        Scalars::Prepared(s) => Some(s),
        _ => None,
    };
    let geometry = retained.map_or_else(
        || {
            Geometry::for_len(
                cap(input.len(), options),
                arithmetic_options(cap(input.len(), options), options),
            )
        },
        |s| {
            Geometry::for_shape(
                cap(input.len(), options),
                s.shape,
                arithmetic_options(cap(input.len(), options), options),
            )
        },
    );
    let cached = retained
        .and_then(|s| s.cached)
        .is_some_and(|c| c.geometry == geometry);
    layout::<C>(
        input.len(),
        geometry,
        retained.is_some(),
        cached,
        matches!(input.bases, Bases::Compact(_) | Bases::CompactPrepared(_)),
        options,
    )
}
fn weight<C: PastaCurve>(input: &Input<'_, C>) -> u128 {
    let windows = match input.scalars {
        Scalars::Prepared(s) => {
            match Geometry::for_shape(input.len(), s.shape, ExecutionOptions::SERIAL) {
                Geometry::Short(b) => usize::from(b).max(1),
                g => g.windows() * 8,
            }
        }
        _ => 128,
    };
    (input.len() as u128).max(1) * windows as u128
}
fn split<C: PastaCurve>(inputs: &[Input<'_, C>], budget: usize) -> Option<(usize, usize)> {
    if inputs.len() < 2 || budget < 2 {
        return None;
    }
    let total: u128 = inputs.iter().map(weight).sum();
    // Odd budgets need unequal work ranges. A half-by-half split would leave
    // one worker processing half the jobs while two process the other half.
    let target = total * (budget / 2) as u128 / budget as u128;
    let mut sum = 0;
    let mut mid = 1;
    let mut best = u128::MAX;
    let mut left_weight = 0;
    for (i, input) in inputs[..inputs.len() - 1].iter().enumerate() {
        sum += weight(input);
        let distance = sum.abs_diff(target);
        if distance < best {
            best = distance;
            mid = i + 1;
            left_weight = sum;
        }
    }
    let left = ((budget as u128 * left_weight + total / 2) / total) as usize;
    // Keep a dominant job's budget when a neighboring range cannot justify
    // even one worker. That range runs sequentially around the parallel job.
    (left != 0 && left != budget).then_some((mid, left))
}
fn requirements<C: PastaCurve>(
    inputs: &[Input<'_, C>],
    options: ExecutionOptions,
) -> Result<Requirements, CurveError> {
    if let Some((mid, left)) = split(inputs, options.task_budget.get()) {
        let a = requirements(
            &inputs[..mid],
            options.with_task_budget(TaskBudget::new(left).unwrap()),
        )?;
        let b = requirements(
            &inputs[mid..],
            options.with_task_budget(TaskBudget::new(options.task_budget.get() - left).unwrap()),
        )?;
        a.plus(b)?.times::<C>(1)
    } else {
        let mut r = ZERO;
        for input in inputs {
            r = r.include(job(input, options)?.requirements);
        }
        Ok(r)
    }
}
pub(super) struct Plan {
    pub requirements: Requirements,
    options: ExecutionOptions,
}
impl Plan {
    pub fn new<C: PastaCurve>(
        inputs: &[Input<'_, C>],
        mut options: ExecutionOptions,
    ) -> Result<Self, CurveError> {
        let mut r = requirements(inputs, options)?;
        if let Some(limit) = options.memory_limit {
            while r.bytes::<C>()? > limit {
                let n = inputs
                    .iter()
                    .map(|i| cap(i.len(), options))
                    .max()
                    .unwrap_or(0);
                options = smaller(options, n).ok_or(CurveError::MemoryLimit {
                    limit,
                    required: r.bytes::<C>()?,
                })?;
                r = requirements(inputs, options)?;
            }
        }
        Ok(Self {
            requirements: r,
            options,
        })
    }
}

pub(super) fn split_scratch<C: PastaCurve>(
    scratch: Scratch<'_, C>,
    left: Requirements,
) -> (Scratch<'_, C>, Scratch<'_, C>) {
    let (sa, sb) = scratch.scalars.split_at_mut(left.scalars);
    let (da, db) = scratch.digits.split_at_mut(left.digits);
    let (aa, ab) = scratch.affine.split_at_mut(left.affine);
    let (pa, pb) = scratch.projective.split_at_mut(left.projective);
    let (fa, fb) = scratch.field.split_at_mut(left.field);
    let (ia, ib) = scratch.indices.split_at_mut(left.indices);
    (
        Scratch::new(sa, da, aa, pa, fa, ia),
        Scratch::new(sb, db, ab, pb, fb, ib),
    )
}
pub(super) fn execute<C: PastaCurve, X: Executor>(
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
    options: ExecutionOptions,
    executor: &X,
    mut scratch: Scratch<'_, C>,
) {
    if let Some((mid, left)) = split(inputs, options.task_budget.get()) {
        let a = options.with_task_budget(TaskBudget::new(left).unwrap());
        let b =
            options.with_task_budget(TaskBudget::new(options.task_budget.get() - left).unwrap());
        let (sa, sb) = split_scratch(scratch, requirements(&inputs[..mid], a).unwrap());
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
    options: ExecutionOptions,
    executor: &X,
    scratch: Scratch<'_, C>,
) -> ProjectivePoint<C> {
    super::run::MsmPlan::from_job(input.len(), options, job)
        .execute(
            *input,
            // Preparation and streaming deposits use retained storage, so their
            // concurrency need not stop at the number of window scratch bundles.
            // The driver clamps temporary leases to the available windows.
            NonZeroUsize::new(job.budget.get()).unwrap(),
            executor,
            scratch,
        )
        .expect("validated synchronous MSM storage")
}

/// Borrowed reusable plan over immutable inputs and caller-owned metadata.
///
/// The plan borrows both bases and scalar rows through its inputs. To execute
/// changing scalar rows over retained bases, rebind a [`super::Selection`] and
/// use [`super::Input::execute`] or [`super::execute_batch`].
///
/// Different output buffers and previously used scratch may be supplied on every
/// execution, including concurrent calls with separate buffers. The
/// [memory ceiling](ExecutionOptions::with_memory_limit) includes metadata
/// prefixes reserved by [`Self::storage_len`] and scratch from
/// [`Self::requirements`].
pub struct ExecutionPlan<'a, 'i, C: PastaCurve> {
    inputs: &'a [Input<'i, C>],
    jobs: &'a [JobStorage],
    workers: &'a [WorkerStorage],
    plan: Plan,
    temporary_bytes: usize,
}
impl<'a, 'i, C: PastaCurve> ExecutionPlan<'a, 'i, C> {
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
    /// [`JobStorage::EMPTY`] and [`WorkerStorage::EMPTY`]. Returns
    /// [`CurveError::ScratchTooSmall`] for short buffers,
    /// [`CurveError::SizeOverflow`] for unrepresentable sizes, or
    /// [`CurveError::MemoryLimit`] under the
    /// [memory policy](ExecutionOptions::with_memory_limit).
    /// All returned errors precede writes; tails beyond the required metadata
    /// prefixes remain untouched.
    pub fn new(
        inputs: &'a [Input<'i, C>],
        options: ExecutionOptions,
        jobs: &'a mut [JobStorage],
        workers: &'a mut [WorkerStorage],
    ) -> Result<Self, CurveError> {
        let (j, w) = Self::storage_len(inputs.len(), options)?;
        check_scratch("jobs", j, jobs.len())?;
        check_scratch("workers", w, workers.len())?;
        let metadata = j
            .checked_mul(core::mem::size_of::<JobStorage>())
            .and_then(|n| {
                w.checked_mul(core::mem::size_of::<WorkerStorage>())
                    .and_then(|m| n.checked_add(m))
            })
            .ok_or(CurveError::SizeOverflow)?;
        let mut adjusted = options;
        if let Some(limit) = options.memory_limit {
            adjusted.memory_limit =
                Some(limit.checked_sub(metadata).ok_or(CurveError::MemoryLimit {
                    limit,
                    required: metadata,
                })?);
        }
        let plan = Plan::new(inputs, adjusted).map_err(|error| match error {
            CurveError::MemoryLimit { required, .. } => CurveError::MemoryLimit {
                limit: options.memory_limit.unwrap(),
                required: required.saturating_add(metadata),
            },
            e => e,
        })?;
        let temporary_bytes = plan
            .requirements
            .bytes::<C>()?
            .checked_add(metadata)
            .ok_or(CurveError::SizeOverflow)?;
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
    /// Use these counts for [`Self::execute`]. The metadata's share of the memory
    /// ceiling can make this layout differ from [`super::batch_requirements`].
    pub const fn requirements(&self) -> Requirements {
        self.plan.requirements
    }
    /// Total temporary bytes, including the reserved metadata prefixes.
    ///
    /// Uses the accounting of [`ExecutionOptions::with_memory_limit`], even if
    /// the plan uses fewer worker entries than [`Self::storage_len`] reserves.
    pub const fn temporary_bytes(&self) -> usize {
        self.temporary_bytes
    }
    /// Number of job ranges in the retained schedule.
    ///
    /// A range may share scoped tasks across jobs or use tasks within each job.
    /// This count does not measure task concurrency or executor threads.
    pub const fn worker_ranges(&self) -> usize {
        self.workers.len()
    }
    /// Executes the retained plan, writing one result per input in input order.
    ///
    /// Returns [`CurveError::LengthMismatch`] unless the output length equals the
    /// input count, or [`CurveError::ScratchTooSmall`] if scratch is shorter than
    /// [`Self::requirements`]. All returned errors precede writes. An executor
    /// panic may partially write output and scratch; reuse after unwinding
    /// follows [`Input::execute`].
    pub fn execute<X: Executor>(
        &self,
        output: &mut [ProjectivePoint<C>],
        executor: &X,
        scratch: Scratch<'_, C>,
    ) -> Result<(), CurveError> {
        check_length("output", self.inputs.len(), output.len())?;
        let scratch = scratch.checked(self.plan.requirements)?;
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
        Ok(())
    }
}
fn fill_metadata<C: PastaCurve>(
    inputs: &[Input<'_, C>],
    options: ExecutionOptions,
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
        let mut requirements = ZERO;
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
    options: ExecutionOptions,
    executor: &X,
    mut scratch: Scratch<'_, C>,
) {
    if workers.len() > 1 {
        let mid = workers.len() / 2;
        let left = workers[..mid]
            .iter()
            .fold(ZERO, |sum, w| sum.plus(w.requirements).unwrap());
        let (sa, sb) = split_scratch(scratch, left);
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
        curve::{AffinePoint, Pallas},
        exec::SerialExecutor,
        field::Fq,
    };
    use std::{vec, vec::Vec};

    #[test]
    fn weighted_ranges_preserve_dominant_and_odd_budgets() {
        let bases = vec![AffinePoint::<Pallas>::GENERATOR; 1024];
        let scalars = vec![Fq::ONE; 1024];
        let input = |n| Input::new(Bases::Affine(&bases[..n]), &scalars[..n]).unwrap();
        let inputs = [input(2), input(1024), input(3), input(17)];
        for budget in [2, 3, 5, 17] {
            let options = ExecutionOptions::SERIAL
                .with_task_budget(TaskBudget::new(budget).unwrap())
                .with_memory_limit(usize::MAX);
            let mut jobs = [JobStorage::EMPTY; 4];
            let mut workers = [WorkerStorage::EMPTY; 4];
            let plan = ExecutionPlan::new(&inputs, options, &mut jobs, &mut workers).unwrap();
            assert_eq!(plan.worker_ranges(), 1);
            assert_eq!(plan.jobs[1].budget.get(), budget);
        }
        let inputs = [input(2); 30];
        for budget in [3, 5] {
            let options = ExecutionOptions::SERIAL
                .with_task_budget(TaskBudget::new(budget).unwrap())
                .with_memory_limit(usize::MAX);
            let mut jobs = [JobStorage::EMPTY; 30];
            let mut workers = [WorkerStorage::EMPTY; 5];
            let plan = ExecutionPlan::new(&inputs, options, &mut jobs, &mut workers).unwrap();
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
                .map(|&n| Input::new(Bases::Affine(&bases[..n]), &scalars[..n]).unwrap())
                .collect();
            for tasks in [1, 2, 3, 5, 17, 65] {
                for pass in [1, 2, 17, 128, 513] {
                    for limit in [8192, 32768, 1048576] {
                        let options = ExecutionOptions::SERIAL
                            .with_task_budget(TaskBudget::new(tasks).unwrap())
                            .with_max_terms_per_pass(NonZeroUsize::new(pass))
                            .with_memory_limit(limit);
                        let (j, w) =
                            ExecutionPlan::<Pallas>::storage_len(inputs.len(), options).unwrap();
                        let mut jobs = vec![JobStorage::EMPTY; j];
                        let mut workers = vec![WorkerStorage::EMPTY; w];
                        let plan =
                            match ExecutionPlan::new(&inputs, options, &mut jobs, &mut workers) {
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
                        let mut scratch = super::super::tests::Buffers::new(plan.requirements());
                        let mut remaining = scratch.borrow();
                        for worker in plan.workers {
                            live_budget += plan.jobs[worker.begin..worker.end]
                                .iter()
                                .map(|job| job.budget.get())
                                .max()
                                .unwrap();
                            let (mut owned, rest) = split_scratch(remaining, worker.requirements);
                            remaining = rest;
                            for (i, visited) in visits
                                .iter_mut()
                                .enumerate()
                                .take(worker.end)
                                .skip(worker.begin)
                            {
                                *visited += 1;
                                let job = plan.jobs[i];
                                owned.reborrow().checked(job.requirements).unwrap();
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
                        plan.execute(&mut output, &SerialExecutor, scratch.borrow())
                            .unwrap();
                        for (p, &n) in output.iter().zip(&sizes) {
                            assert_eq!(*p, bases[0].mul_projective(&Fq::from_u64(n as u64)));
                        }
                    }
                }
            }
        }
    }
}
