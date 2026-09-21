use super::transform::Run;
use super::{
    Class, ClassState, ElementOrder, ExecutionOptions, Executor, FftError, PastaField,
    PrimeModulus, interpolation_scratch,
};
#[cfg(test)]
use super::{ScratchRequirements, check_field_count, min};
#[cfg(test)]
use crate::exec::TaskBudget;

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterpolationOptions {
    pub transform: ExecutionOptions,
    pub max_class_tasks: usize,
    pub max_tasks: usize,
}

#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct InterpolationRequirements {
    pub scratch_fields: usize,
    pub per_worker_scratch_fields: usize,
    pub scratch_partitions: usize,
    pub transform_tasks: usize,
}

#[cfg(test)]
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
    if class.order == ElementOrder::Natural {
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
    class.order = ElementOrder::Natural;
    class.state = ClassState::Coefficients;
}

#[cfg(test)]
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

#[cfg(test)]
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
