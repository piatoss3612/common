use super::transform::Run;
use super::{
    Class, ClassState, ExecutionOptions, Executor, FftError, InputOrder, PastaField, PrimeModulus,
    ScratchRequirements, check_field_count, interpolation_scratch, min,
};
use crate::exec::TaskBudget;

/// Optional concurrency across interpolation classes with caller-owned scratch.
///
/// Classes, including the output class, can run independently using separate
/// scratch partitions. [`super::interpolate_classes`] reuses one scratch region
/// across classes and permits parallel work within each transform.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterpolationOptions {
    /// Transform geometry; its task count is capped by each class's share.
    pub transform: ExecutionOptions,
    /// Nonzero maximum number of classes executing concurrently.
    pub max_class_tasks: usize,
    /// Nonzero total task ceiling, across and within class transforms.
    pub max_tasks: usize,
}

/// Equal-capacity scratch partitions for concurrent classes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterpolationRequirements {
    /// Total fields in the caller's scratch slice.
    pub scratch_fields: usize,
    /// Capacity reserved per class worker, reused for its assigned transforms.
    pub per_worker_scratch_fields: usize,
    /// Concurrent class workers, including the output class.
    pub scratch_partitions: usize,
    /// Maximum tasks within any one transform.
    pub transform_tasks: usize,
}

impl InterpolationOptions {
    const fn geometry(self, count: usize) -> Result<(ExecutionOptions, usize), FftError> {
        let budget = match TaskBudget::new(self.max_tasks) {
            Some(budget) => budget,
            None => return Err(FftError::InvalidExecution),
        };
        let (jobs, inner) = match budget.partition(min(count, self.max_class_tasks)) {
            Some(partition) => partition,
            None => return Err(FftError::InvalidExecution),
        };
        let mut options = self.transform;
        options.max_tasks = min(options.max_tasks, inner.get());
        Ok((options, jobs))
    }

    /// Const sizing for the output plus all lifts.
    ///
    /// A lift can have any supported size up to the output size. Each partition
    /// reserves the largest transform requirement; no additional coefficient
    /// buffer is needed. An empty lift slice is accepted.
    ///
    /// Returns [`FftError::InvalidExecution`] for zero class or total task limits.
    /// Size and transform-option errors follow
    /// [`ExecutionOptions::interpolation_requirements`]. Overflow of the total
    /// scratch capacity or class count returns [`FftError::SizeOverflow`].
    pub const fn requirements(
        self,
        output_size: usize,
        lift_sizes: &[usize],
    ) -> Result<InterpolationRequirements, FftError> {
        let count = match lift_sizes.len().checked_add(1) {
            Some(n) => n,
            None => return Err(FftError::SizeOverflow),
        };
        let (options, jobs) = match self.geometry(count) {
            Ok(g) => g,
            Err(e) => return Err(e),
        };
        let per_worker = match options.interpolation_requirements(output_size, lift_sizes) {
            Ok(r) => r.field_elements,
            Err(e) => return Err(e),
        };
        self.finish_requirements(options, jobs, per_worker)
    }

    const fn finish_requirements(
        self,
        options: ExecutionOptions,
        jobs: usize,
        per_worker: usize,
    ) -> Result<InterpolationRequirements, FftError> {
        let count = match per_worker.checked_mul(jobs) {
            Some(n) => n,
            None => return Err(FftError::SizeOverflow),
        };
        match check_field_count(count) {
            Ok(scratch_fields) => Ok(InterpolationRequirements {
                scratch_fields,
                per_worker_scratch_fields: per_worker,
                scratch_partitions: jobs,
                transform_tasks: options.max_tasks,
            }),
            Err(e) => Err(e),
        }
    }

    /// Checks class phases and computes the same partitions as the const query.
    ///
    /// Sizing errors follow [`Self::requirements`]. Classes must still contain
    /// evaluations, as checked by [`interpolation_scratch`], otherwise this
    /// returns [`FftError::InvalidClassState`].
    pub const fn scratch<M: PrimeModulus>(
        self,
        output: &Class<'_, M>,
        lifts: &[Class<'_, M>],
    ) -> Result<InterpolationRequirements, FftError> {
        let count = match lifts.len().checked_add(1) {
            Some(n) => n,
            None => return Err(FftError::SizeOverflow),
        };
        let (options, jobs) = match self.geometry(count) {
            Ok(g) => g,
            Err(e) => return Err(e),
        };
        let per_worker = match interpolation_scratch(output, lifts, options) {
            Ok(r) => r.field_elements,
            Err(e) => return Err(e),
        };
        self.finish_requirements(options, jobs, per_worker)
    }
}

fn inverse<M: PrimeModulus, E: Executor>(
    class: &mut Class<'_, M>,
    options: ExecutionOptions,
    executor: &E,
    scratch: &mut [PastaField<M>],
) {
    class.state = ClassState::Consumed;
    if class.order == InputOrder::Natural {
        class.plan.permute(class.values);
    }
    let fields = class
        .plan
        .scratch_requirements(options)
        .unwrap()
        .field_elements;
    class.plan.run(
        class.values,
        options,
        executor,
        &mut scratch[..fields],
        Run::inverse(&[]),
    );
    class.order = InputOrder::Natural;
    class.state = ClassState::Coefficients;
}

fn lifts_parallel<M: PrimeModulus, E: Executor>(
    lifts: &mut [Class<'_, M>],
    jobs: usize,
    options: ExecutionOptions,
    executor: &E,
    scratch: &mut [PastaField<M>],
) {
    if jobs <= 1 {
        for lift in lifts {
            inverse(lift, options, executor, scratch);
        }
    } else {
        let count = lifts.len() / 2;
        let left_jobs = jobs / 2;
        let (left, right) = lifts.split_at_mut(count);
        let (ls, rs) = scratch.split_at_mut(scratch.len() / jobs * left_jobs);
        executor.join(
            || lifts_parallel(left, left_jobs, options, executor, ls),
            || lifts_parallel(right, jobs - left_jobs, options, executor, rs),
        );
    }
}

/// Interpolates classes concurrently and adds their coefficients to `output`.
///
/// The polynomial sum, accepted class sizes and shifts, and output ordering
/// follow [`super::interpolate_classes`]. Every lift retains its own normalized,
/// natural-order coefficients. All classes enter [`ClassState::Coefficients`]
/// on success and cannot be scattered into or interpolated again.
///
/// Scratch must contain at least the fields reported by
/// [`InterpolationOptions::scratch`], otherwise this returns
/// [`FftError::ScratchTooSmall`]. Other errors follow that query. All checks
/// precede mutation. The module's [working-storage rules](super) apply to panics;
/// any class whose inverse began has left its evaluation phase.
pub fn interpolate_classes_parallel<M: PrimeModulus, E: Executor>(
    output: &mut Class<'_, M>,
    lifts: &mut [Class<'_, M>],
    options: InterpolationOptions,
    executor: &E,
    scratch: &mut [PastaField<M>],
) -> Result<(), FftError> {
    let required = options.scratch(output, lifts)?;
    ScratchRequirements {
        field_elements: required.scratch_fields,
    }
    .check(scratch.len())?;
    let mut transform = options.transform;
    transform.max_tasks = required.transform_tasks;
    let scratch = &mut scratch[..required.scratch_fields];
    if required.scratch_partitions == 1 {
        inverse(output, transform, executor, scratch);
        lifts_parallel(lifts, 1, transform, executor, scratch);
    } else {
        let (os, ls) = scratch.split_at_mut(required.per_worker_scratch_fields);
        executor.join(
            || inverse(output, transform, executor, os),
            || {
                lifts_parallel(
                    lifts,
                    required.scratch_partitions - 1,
                    transform,
                    executor,
                    ls,
                )
            },
        );
    }
    add_coefficients(output, lifts);
    Ok(())
}

fn add_coefficients<M: PrimeModulus>(output: &mut Class<'_, M>, lifts: &[Class<'_, M>]) {
    for lift in lifts
        .iter()
        .filter(|lift| lift.state == ClassState::Coefficients)
    {
        for (value, coefficient) in output.values.iter_mut().zip(lift.values.iter()) {
            *value = value.add(coefficient);
        }
    }
}

fn merge_evaluations<M: PrimeModulus>(output: &mut Class<'_, M>, lift: &mut Class<'_, M>) {
    output.state = ClassState::Consumed;
    lift.state = ClassState::Consumed;
    for (index, value) in output.values.iter_mut().enumerate() {
        let source = if output.order == lift.order {
            index
        } else {
            lift.plan.reversed(index)
        };
        *value = value.add(&lift.values[source]);
    }
}

/// Interpolates a coefficient sum while consuming every lift's working storage.
///
/// The polynomial sum and accepted classes follow [`super::interpolate_classes`].
/// On success `output` contains natural-order coefficients in
/// [`ClassState::Coefficients`], and every lift enters [`ClassState::Consumed`]
/// without a promised polynomial result. No class can be scattered into or
/// interpolated again. Consuming lifts permits combining equal-domain evaluations
/// before interpolation.
///
/// Scratch and errors follow [`interpolation_scratch`], with
/// [`FftError::ScratchTooSmall`] for a shorter scratch slice. All checks precede
/// mutation. The module's [working-storage rules](super) cover panics; classes
/// whose evaluation storage was consumed cannot be retried.
pub fn interpolate_sum<M: PrimeModulus, E: Executor>(
    output: &mut Class<'_, M>,
    lifts: &mut [Class<'_, M>],
    options: ExecutionOptions,
    executor: &E,
    scratch: &mut [PastaField<M>],
) -> Result<(), FftError> {
    interpolation_scratch(output, lifts, options)?.check(scratch.len())?;
    output.state = ClassState::Consumed;
    for lift in lifts.iter_mut() {
        if lift.plan.domain().same_domain(output.plan.domain()) {
            merge_evaluations(output, lift);
        }
    }
    for index in 0..lifts.len() {
        let (before, after) = lifts.split_at_mut(index + 1);
        let class = &mut before[index];
        if class.state != ClassState::Evaluations {
            continue;
        }
        for lift in after.iter_mut() {
            if lift.state == ClassState::Evaluations
                && class.plan.domain().same_domain(lift.plan.domain())
            {
                merge_evaluations(class, lift);
            }
        }
        inverse(class, options, executor, scratch);
    }
    inverse(output, options, executor, scratch);
    add_coefficients(output, lifts);
    for lift in lifts {
        lift.state = ClassState::Consumed;
    }
    Ok(())
}
