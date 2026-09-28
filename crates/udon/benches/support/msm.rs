//! Benchmark-owned metadata for short batches; planning remains in timed calls.
use zakura_udon::{
    curve::{CurveError, PastaCurve, ProjectivePoint},
    exec::{ExecutionOptions, Executor},
    msm::{
        Input, Requirements, Scratch,
        execution::{BatchPlan, JobStorage, WorkerStorage},
    },
};

fn planned<C: PastaCurve, T>(
    inputs: &[Input<'_, C>],
    options: ExecutionOptions,
    f: impl FnOnce(BatchPlan<'_, '_, C>) -> Result<T, CurveError>,
) -> Result<T, CurveError> {
    let (j, w) = BatchPlan::<C>::storage_len(inputs.len(), options)?;
    let mut jobs = [JobStorage::EMPTY; 16];
    let mut workers = [WorkerStorage::EMPTY; 16];
    assert!(j <= jobs.len() && w <= workers.len());
    f(BatchPlan::new(
        inputs,
        options,
        &mut jobs[..j],
        &mut workers[..w],
    )?)
}

pub fn batch_requirements<C: PastaCurve>(
    inputs: &[Input<'_, C>],
    options: ExecutionOptions,
) -> Result<Requirements, CurveError> {
    planned(inputs, options, |plan| Ok(plan.requirements()))
}

pub fn execute_batch<C: PastaCurve, E: Executor>(
    inputs: &[Input<'_, C>],
    output: &mut [ProjectivePoint<C>],
    options: ExecutionOptions,
    executor: &E,
    scratch: Scratch<'_, C>,
) -> Result<(), CurveError> {
    planned(inputs, options, |plan| {
        plan.execute(output, executor, scratch);
        Ok(())
    })
}
