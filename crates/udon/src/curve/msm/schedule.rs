//! Two scoped phases: write digit chunks, then evaluate independent MSM partitions.
//!
//! Each worker borrows exclusive working slices and reuses them across tasks.
//! Only digits and task results need storage spanning all inputs. Sizing uses
//! input lengths, whether scalars are prepared, and options, covering every
//! scalar-dependent kernel choice, including the short-scalar paths.

use super::{
    CurveError, ExecutionOptions, Input, PastaCurve, ProjectivePoint, Requirements, Scalars,
    Scratch, checked_count,
    kernels::{self, Task, Work},
    recode,
};
use crate::{
    curve::EisensteinTableBatch,
    exec::{Executor, TaskBudget},
    field::PastaField,
};

macro_rules! size {
    ($value:expr) => {
        match $value {
            Ok(value) => value,
            Err(error) => return Err(error),
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

const fn windows(n: usize) -> usize {
    if n == 0 {
        0
    } else if n < super::BOOTH_MIN {
        1
    } else {
        super::WINDOWS
    }
}

const fn parts(n: usize, natural_tasks: usize, budget: usize) -> usize {
    // First expose jobs and whole windows. Term partitions add bucket-collapse
    // work, so use them only to fill otherwise unused concurrency, up to four.
    if n < 8 || natural_tasks >= budget {
        1
    } else {
        min(
            4,
            min(
                if n < super::BOOTH_MIN {
                    n / 8
                } else {
                    n.div_ceil(recode::CHUNK)
                },
                budget.div_ceil(natural_tasks),
            ),
        )
    }
}

const fn pass(n: usize, parts: usize, options: ExecutionOptions) -> usize {
    let n = n.div_ceil(parts);
    match options.max_terms_per_pass {
        Some(cap) => min(n, cap.get()),
        None => n,
    }
}

#[derive(Clone, Copy, Default)]
struct Workspace {
    affine: usize,
    projective: usize,
    field: usize,
    indices: usize,
}

impl Workspace {
    const fn new<C: PastaCurve>(
        n: usize,
        parts: usize,
        options: ExecutionOptions,
    ) -> Result<Self, CurveError> {
        let cap = pass(n, parts, options);
        if n == 0 {
            return Ok(Self {
                affine: 0,
                projective: 0,
                field: 0,
                indices: 0,
            });
        }
        if n < super::BOOTH_MIN {
            // Full passes use affine table preparation at eight bases. Only
            // shorter passes or tails need projective staging. Balanced term
            // partitions have at most these two lengths.
            let full = size!(EisensteinTableBatch::<C>::requirements(cap));
            let a = size!(EisensteinTableBatch::<C>::requirements(n / parts % cap));
            let b = size!(EisensteinTableBatch::<C>::requirements(
                n.div_ceil(parts) % cap
            ));
            Ok(Self {
                affine: size!(checked_count::<super::AffinePoint<C>>(cap, 9)),
                projective: max(
                    if n >= 32 { 16 } else { 0 },
                    max(
                        full.projective_scratch,
                        max(a.projective_scratch, b.projective_scratch),
                    ),
                ),
                field: max(full.field_scratch, max(a.field_scratch, b.field_scratch)),
                indices: cap,
            })
        } else {
            let buckets = super::BUCKETS;
            let halves = 2;
            let deposits = size!(add(
                size!(checked_count::<super::AffinePoint<C>>(cap, halves)),
                buckets
            ));
            let pairs = deposits / 2;
            Ok(Self {
                affine: size!(checked_count::<super::AffinePoint<C>>(
                    size!(add(deposits, buckets)),
                    1
                )),
                projective: 0,
                field: size!(checked_count::<PastaField<C::Base>>(pairs, 6)),
                indices: size!(checked_count::<usize>(size!(add(3 * buckets, pairs)), 1)),
            })
        }
    }

    fn include(&mut self, other: Self) {
        self.affine = self.affine.max(other.affine);
        self.projective = self.projective.max(other.projective);
        self.field = self.field.max(other.field);
        self.indices = self.indices.max(other.indices);
    }

    const fn requirements<C: PastaCurve>(
        self,
        digits: usize,
        tasks: usize,
        workers: usize,
    ) -> Result<Requirements, CurveError> {
        Ok(Requirements {
            digits: size!(checked_count::<u8>(digits, 1)),
            affine: size!(checked_count::<super::AffinePoint<C>>(self.affine, workers)),
            projective: size!(checked_count::<ProjectivePoint<C>>(
                size!(add(
                    tasks,
                    size!(checked_count::<ProjectivePoint<C>>(
                        self.projective,
                        workers
                    ))
                )),
                1
            )),
            field: size!(checked_count::<PastaField<C::Base>>(self.field, workers)),
            indices: size!(checked_count::<usize>(self.indices, workers)),
        })
    }
}

pub(super) const fn single_requirements<C: PastaCurve>(
    terms: usize,
    options: ExecutionOptions,
) -> Result<Requirements, CurveError> {
    size!(checked_count::<PastaField<C::Scalar>>(terms, 1));
    let natural = windows(terms);
    let parts = parts(terms, natural, options.task_budget.get());
    let tasks = natural * parts;
    let workers = min(tasks, options.task_budget.get());
    let digits = size!(recode::storage_len(terms));
    size!(Workspace::new::<C>(terms, parts, options)).requirements::<C>(digits, tasks, workers)
}

pub(super) struct Plan {
    pub requirements: Requirements,
    options: ExecutionOptions,
    natural: usize,
    tasks: usize,
    workers: usize,
    work: Workspace,
}

impl Plan {
    pub fn new<C: PastaCurve>(
        inputs: &[Input<'_, C>],
        options: ExecutionOptions,
    ) -> Result<Self, CurveError> {
        let mut natural = 0;
        let mut digits = 0;
        for input in inputs {
            natural = add(natural, windows(input.len()))?;
            digits = add(digits, input.digit_scratch_len()?)?;
        }
        let mut work = Workspace::default();
        let mut tasks = 0;
        for input in inputs {
            let parts = parts(input.len(), natural, options.task_budget.get());
            tasks = add(tasks, windows(input.len()) * parts)?;
            work.include(Workspace::new::<C>(input.len(), parts, options)?);
        }
        let workers = min(tasks, options.task_budget.get());
        let requirements = work.requirements::<C>(digits, tasks, workers)?;
        Ok(Self {
            requirements,
            options,
            natural,
            tasks,
            workers,
            work,
        })
    }

    fn parts(&self, terms: usize) -> usize {
        parts(terms, self.natural, self.options.task_budget.get())
    }
}

pub(super) fn execute<C: PastaCurve, X: Executor>(
    plan: &Plan,
    inputs: &[Input<'_, C>],
    output: &mut [ProjectivePoint<C>],
    executor: &X,
    scratch: Scratch<'_, C>,
) {
    prepare(inputs, scratch.digits, plan.options.task_budget, executor);
    let (results, projective) = scratch.projective.split_at_mut(plan.tasks);
    let work = Work {
        affine: scratch.affine,
        projective,
        field: scratch.field,
        indices: scratch.indices,
    };
    if plan.tasks != 0 {
        visit(
            plan,
            inputs,
            scratch.digits,
            0,
            results,
            work,
            plan.workers,
            executor,
        );
    }
    let mut offset = 0;
    for (input, result) in inputs.iter().zip(output) {
        let parts = plan.parts(input.len());
        let windows = windows(input.len());
        *result = ProjectivePoint::IDENTITY;
        if input.len() < super::BOOTH_MIN {
            if windows != 0 {
                for part in 0..parts {
                    *result = result.add(&results[offset + part]);
                }
            }
        } else {
            for window in (0..windows).rev() {
                for _ in 0..super::WINDOW_BITS {
                    *result = result.double();
                }
                for part in 0..parts {
                    *result = result.add(&results[offset + window * parts + part]);
                }
            }
        }
        offset += windows * parts;
    }
}

fn prepare<C: PastaCurve, X: Executor>(
    inputs: &[Input<'_, C>],
    mut digits: &mut [u8],
    budget: TaskBudget,
    executor: &X,
) {
    if inputs.len() > 1 && budget.get() > 1 {
        let mid = inputs.len() / 2;
        let left_len = inputs[..mid]
            .iter()
            .map(|i| i.digit_scratch_len().unwrap())
            .sum();
        let (a, b) = digits.split_at_mut(left_len);
        let (left, right) = budget.split_at(budget.get() / 2).unwrap();
        executor.join(
            || prepare(&inputs[..mid], a, left, executor),
            || prepare(&inputs[mid..], b, right, executor),
        );
    } else {
        for input in inputs {
            let (head, tail) = digits.split_at_mut(input.digit_scratch_len().unwrap());
            if let Scalars::Raw(scalars) = input.scalars {
                recode::prepare::<C, X>(scalars, head, budget, executor);
            }
            digits = tail;
        }
    }
}

#[expect(
    clippy::too_many_arguments,
    reason = "Scoped recursion splits task results and exclusive workspace together."
)]
fn visit<C: PastaCurve, X: Executor>(
    plan: &Plan,
    inputs: &[Input<'_, C>],
    digits: &[u8],
    offset: usize,
    results: &mut [ProjectivePoint<C>],
    mut work: Work<'_, C>,
    workers: usize,
    executor: &X,
) {
    if workers > 1 {
        let left = workers / 2;
        let mid = results.len() / workers * left + min(left, results.len() % workers);
        let (ra, rb) = results.split_at_mut(mid);
        let (aa, ab) = work.affine.split_at_mut(plan.work.affine * left);
        let (pa, pb) = work.projective.split_at_mut(plan.work.projective * left);
        let (fa, fb) = work.field.split_at_mut(plan.work.field * left);
        let (ia, ib) = work.indices.split_at_mut(plan.work.indices * left);
        executor.join(
            || {
                visit(
                    plan,
                    inputs,
                    digits,
                    offset,
                    ra,
                    Work {
                        affine: aa,
                        projective: pa,
                        field: fa,
                        indices: ia,
                    },
                    left,
                    executor,
                )
            },
            || {
                visit(
                    plan,
                    inputs,
                    digits,
                    offset + mid,
                    rb,
                    Work {
                        affine: ab,
                        projective: pb,
                        field: fb,
                        indices: ib,
                    },
                    workers - left,
                    executor,
                )
            },
        );
        return;
    }
    let mut task_offset = 0;
    let mut digit_offset = 0;
    for input in inputs {
        let parts = plan.parts(input.len());
        let count = windows(input.len()) * parts;
        let begin = max(offset, task_offset);
        let end = min(offset + results.len(), task_offset + count);
        let len = input.digit_scratch_len().unwrap();
        let prepared = match input.scalars {
            Scalars::Raw(_) => &digits[digit_offset..digit_offset + len],
            Scalars::Prepared(s) => s.digits,
        };
        for task in begin..end {
            let relative = task - task_offset;
            results[task - offset] = kernels::run(
                input,
                prepared,
                Task {
                    window: relative / parts,
                    part: relative % parts,
                    parts,
                    pass: pass(input.len(), parts, plan.options),
                },
                &mut work,
            );
        }
        digit_offset += len;
        task_offset += count;
        if task_offset >= offset + results.len() {
            break;
        }
    }
}
