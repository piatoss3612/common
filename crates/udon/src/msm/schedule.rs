//! Job layout, memory adaptation, and batch requirement planning for MSM inputs.

use super::{
    Accumulation, Algorithm, ArithmeticOptions, Bases, BatchOptions, CurveError, Input, PastaCurve,
    ProjectivePoint, Requirements, ScalarStorage, Scalars, add, checked_count, max, min,
    recode::Geometry,
};
use crate::exec::{ExecutionOptions, TaskBudget};
use core::num::NonZeroUsize;

/// Initialized opaque metadata for one input in a [`BatchPlan`](super::execution::BatchPlan).
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct JobStorage {
    pub(super) geometry: Geometry,
    pub(super) cap: usize,
    pub(super) pass: usize,
    pub(super) workers: usize,
    pub(super) budget: TaskBudget,
    streaming: bool,
    pub(super) accumulation: Accumulation,
    pub(super) work: Requirements,
    pub(super) requirements: Requirements,
}
impl JobStorage {
    /// Initializer; populated by [`BatchPlan::new`](super::execution::BatchPlan::new).
    pub const EMPTY: Self = Self {
        geometry: Geometry::Joint,
        cap: 0,
        pass: 0,
        workers: 0,
        budget: TaskBudget::SERIAL,
        streaming: false,
        accumulation: Accumulation::Affine,
        work: Requirements::ZERO,
        requirements: Requirements::ZERO,
    };
}
// Adaptation is private: resolving an automatic width or accumulator must not
// turn the requested kernel into an explicit selection.
#[derive(Clone, Copy)]
pub(super) struct Options {
    pub(super) arithmetic: ArithmeticOptions,
    pub(super) task_budget: TaskBudget,
    memory_limit: Option<usize>,
    width: Option<u8>,
    accumulation: Accumulation,
    pub(super) alpha: bool,
}
impl Options {
    pub(super) const fn new(options: BatchOptions) -> Self {
        Self {
            arithmetic: options.arithmetic,
            task_budget: options.task_budget,
            memory_limit: options.memory_limit,
            width: None,
            accumulation: options.arithmetic.accumulation(),
            alpha: true,
        }
    }
    pub(super) const fn with_task_budget(mut self, budget: TaskBudget) -> Self {
        self.task_budget = budget;
        self
    }
    /// Whether batch execution may specialize to a short geometry after preparation.
    pub(super) const fn specializes_short(self) -> bool {
        self.width.is_none()
    }
    pub(super) const fn memory_limit(self) -> Option<usize> {
        self.memory_limit
    }
    const fn geometry(self, n: usize) -> Geometry {
        match self.width {
            Some(width) => Geometry::Booth(width),
            None => Geometry::select(
                n,
                super::recode::Shape {
                    bits: 255,
                    weight: 0,
                },
                self.arithmetic,
                self.task_budget,
            ),
        }
    }
}

const fn layout<C: PastaCurve>(
    terms: usize,
    geometry: Geometry,
    retained: bool,
    cached: bool,
    compact: bool,
    options: Options,
) -> Result<JobStorage, CurveError> {
    size!(checked_count::<ScalarStorage<C>>(terms, 1));
    if terms == 0 {
        return Ok(JobStorage::EMPTY);
    }
    let cap = cap(terms, options.arithmetic);
    let pass = match options.arithmetic.max_terms_per_pass {
        Some(c) => min(cap, c.get()),
        None => cap,
    };
    let windows = geometry.windows();
    let workers = if options.arithmetic.streaming() {
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
    let work = if options.arithmetic.streaming() {
        Requirements {
            projective: size!(checked_count::<ProjectivePoint<C>>(
                windows,
                geometry.buckets()
            )),
            ..Requirements::ZERO
        }
    } else {
        match geometry {
            Geometry::Short(bits) => Requirements {
                projective: if bits > 1 && cap >= 32 { 16 } else { 0 },
                ..Requirements::ZERO
            },
            Geometry::Joint if compact => Requirements {
                projective: if !retained && cap >= 32 { 16 } else { 0 },
                ..Requirements::ZERO
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
                ..Requirements::ZERO
            },
            Geometry::Booth(_) | Geometry::Alpha(_) => {
                let buckets = geometry.buckets();
                if matches!(accumulation, Accumulation::Projective) {
                    Requirements {
                        projective: max(buckets, 16),
                        ..Requirements::ZERO
                    }
                } else {
                    let deposits = size!(add(
                        size!(checked_count::<super::AffinePoint<C>>(
                            pass,
                            if matches!(geometry, Geometry::Alpha(_)) {
                                1
                            } else {
                                2
                            }
                        )),
                        buckets
                    ));
                    // Denominators and suffix products for every pair, plus
                    // loose coordinates for every sum and odd survivor.
                    let pairs = deposits / 2;
                    let loose = size!(checked_count::<crate::field::PastaField<C::Base>>(
                        size!(add(pairs, buckets)),
                        2
                    ));
                    Requirements {
                        affine: size!(add(deposits, buckets)),
                        projective: if matches!(accumulation, Accumulation::Hybrid) {
                            max(buckets, 16)
                        } else {
                            16
                        },
                        field: size!(add(2 * pairs, loose)),
                        indices: 3 * buckets,
                        ..Requirements::ZERO
                    }
                }
            }
        }
    };
    let mut requirements = size!(work.times::<C>(workers));
    requirements.projective = size!(add(requirements.projective, windows));
    requirements.scalars = if retained { 0 } else { cap };
    requirements.digits = if cached && cap == terms && !options.arithmetic.streaming() {
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
        streaming: options.arithmetic.streaming(),
        accumulation,
        work,
        requirements,
    })
}
/// Lays out one unprepared, uncached chunk with a fixed geometry.
#[cfg(test)]
pub(super) const fn fixed_geometry<C: PastaCurve>(
    cap: usize,
    geometry: Geometry,
    arithmetic: ArithmeticOptions,
) -> Result<JobStorage, CurveError> {
    layout::<C>(
        cap,
        geometry,
        false,
        false,
        false,
        Options::new(BatchOptions::new(arithmetic)),
    )
}
const fn cap(terms: usize, options: ArithmeticOptions) -> usize {
    min(8192, min(terms, options.chunk_cap()))
}
const fn conservative<C: PastaCurve>(
    terms: usize,
    options: Options,
) -> Result<JobStorage, CurveError> {
    layout::<C>(
        terms,
        options.geometry(cap(terms, options.arithmetic)),
        false,
        false,
        false,
        options,
    )
}

// Shared deterministic memory search for const and runtime planning. Reduce
// affine staging first, then concurrency, then retain projective buckets, and
// finally shorten complete chunks (including records and digits).
pub(super) const fn smaller(mut options: Options, n: usize) -> Option<Options> {
    let pass = match options.arithmetic.max_terms_per_pass {
        Some(p) => min(n, p.get()),
        None => n,
    };
    if options.arithmetic.streaming() {
        options.arithmetic.algorithm = Algorithm::Auto;
    } else if pass > 128 {
        options.arithmetic.max_terms_per_pass = NonZeroUsize::new(128);
    } else if options.task_budget.get() > 1 {
        options.task_budget = match TaskBudget::new(options.task_budget.get().div_ceil(2)) {
            Some(b) => b,
            None => unreachable!(),
        };
    } else if matches!(options.accumulation, Accumulation::Auto)
        && matches!(options.geometry(n), Geometry::Booth(_))
    {
        options.accumulation = Accumulation::Projective;
    } else if n > 1 {
        let chunk = n.div_ceil(2);
        options.arithmetic.chunk_size = NonZeroUsize::new(chunk);
        if chunk < super::BOOTH_MIN
            && matches!(
                options.arithmetic.algorithm,
                Algorithm::Auto
                    | Algorithm::Booth { width: None, .. }
                    | Algorithm::StreamingBooth { width: None }
            )
        {
            options.width = Some(4);
            if matches!(options.accumulation, Accumulation::Auto) {
                options.accumulation = Accumulation::Projective;
            }
        }
    } else {
        return None;
    }
    Some(options)
}

#[cfg(test)]
pub(super) const fn single_requirements<C: PastaCurve>(
    terms: usize,
    options: BatchOptions,
) -> Result<Requirements, CurveError> {
    let mut options = Options::new(options);
    let mut r = size!(conservative::<C>(terms, options)).requirements;
    if let Some(limit) = options.memory_limit {
        while size!(r.bytes::<C>()) > limit {
            options = match smaller(options, cap(terms, options.arithmetic)) {
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

pub(super) fn unbound<C: PastaCurve>(
    terms: usize,
    options: ExecutionOptions,
    source_fragment: Option<NonZeroUsize>,
    alpha: Option<super::AlphaDescription>,
) -> Result<(JobStorage, ArithmeticOptions), CurveError> {
    let execution = options;
    let mut requested = BatchOptions::from(options);
    requested.arithmetic.chunk_size = source_fragment;
    if alpha.is_none()
        && terms >= 4096
        && source_fragment.is_some_and(|fragment| fragment.get() < 1024)
    {
        requested.arithmetic.algorithm = Algorithm::StreamingBooth { width: None };
    }
    let mut options = Options::new(requested);
    loop {
        let job = match alpha {
            Some(description) => layout::<C>(
                terms,
                Geometry::Alpha(description.window_bits()),
                false,
                false,
                false,
                options,
            )?,
            None => conservative::<C>(terms, options)?,
        };
        if options.memory_limit.is_none_or(|limit| {
            job.requirements
                .bytes::<C>()
                .is_ok_and(|bytes| bytes <= limit)
        }) {
            return Ok((job, options.arithmetic));
        }
        if let Some(smaller) = smaller(options, cap(terms, options.arithmetic)) {
            options = smaller;
        } else if alpha.is_some() {
            // Retained originals remain useful even when the expanded buckets
            // cannot fit. Restart with the caller's original resource ceilings.
            return unbound::<C>(terms, execution, source_fragment, None);
        } else {
            return Err(CurveError::MemoryLimit {
                limit: options.memory_limit.unwrap(),
                required: job.requirements.bytes::<C>()?,
            });
        }
    }
}

pub(super) fn job<C: PastaCurve>(
    input: &Input<'_, C>,
    options: Options,
) -> Result<JobStorage, CurveError> {
    let retained = match input.scalars {
        Scalars::Prepared(s) => Some(s),
        _ => None,
    };
    let n = cap(input.len(), options.arithmetic);
    let geometry = retained.map_or_else(
        || options.geometry(n),
        |s| {
            let shape = Geometry::select(n, s.shape, options.arithmetic, options.task_budget);
            if matches!(shape, Geometry::Short(_)) {
                shape
            } else {
                options.geometry(n)
            }
        },
    );
    // A prepared bank determines its recoding width even with a test-forced
    // ordinary algorithm; accumulation and pass choices remain independent.
    let geometry = if !options.alpha
        || matches!(geometry, Geometry::Short(_))
        || options.arithmetic.streaming()
    {
        geometry
    } else {
        input
            .bases
            .alpha()
            .map_or(geometry, |c| Geometry::Alpha(c.description().window_bits()))
    };
    let cached = retained.is_some_and(|s| s.cached_digits(geometry).is_some());
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
            match Geometry::for_shape(input.len(), s.shape, ArithmeticOptions::DEFAULT) {
                Geometry::Short(b) => usize::from(b).max(1),
                g => g.windows() * 8,
            }
        }
        _ => 128,
    };
    (input.len() as u128).max(1) * windows as u128
}
pub(super) fn split<C: PastaCurve>(
    inputs: &[Input<'_, C>],
    budget: usize,
) -> Option<(usize, usize)> {
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
pub(super) fn requirements<C: PastaCurve>(
    inputs: &[Input<'_, C>],
    options: Options,
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
        let mut r = Requirements::ZERO;
        for input in inputs {
            r = r.include(job(input, options)?.requirements);
        }
        Ok(r)
    }
}
pub(super) struct Plan {
    pub(super) requirements: Requirements,
    pub(super) options: Options,
}
impl Plan {
    pub(super) fn with_capacity<C: PastaCurve>(
        inputs: &[Input<'_, C>],
        options: BatchOptions,
        capacity: Requirements,
    ) -> Result<Self, CurveError> {
        Self::resolve(inputs, options, Some(capacity))
    }
    pub(super) fn new<C: PastaCurve>(
        inputs: &[Input<'_, C>],
        options: BatchOptions,
    ) -> Result<Self, CurveError> {
        Self::resolve(inputs, options, None)
    }
    fn resolve<C: PastaCurve>(
        inputs: &[Input<'_, C>],
        requested: BatchOptions,
        capacity: Option<Requirements>,
    ) -> Result<Self, CurveError> {
        let mut options = Options::new(requested);
        loop {
            let r = requirements(inputs, options)?;
            let bytes = r.bytes::<C>()?;
            let memory_ok = options.memory_limit.is_none_or(|limit| bytes <= limit);
            if memory_ok && capacity.is_none_or(|c| r.fits(c)) {
                return Ok(Self {
                    requirements: r,
                    options,
                });
            }
            let n = inputs
                .iter()
                .map(|i| cap(i.len(), options.arithmetic))
                .max()
                .unwrap_or(0);
            if let Some(smaller) = smaller(options, n) {
                options = smaller;
            } else if options.alpha && inputs.iter().any(|i| i.bases.alpha().is_some()) {
                options = Options {
                    alpha: false,
                    ..Options::new(requested)
                };
            } else if !memory_ok {
                return Err(CurveError::MemoryLimit {
                    limit: options.memory_limit.unwrap(),
                    required: bytes,
                });
            } else {
                return Err(r.capacity_error(capacity.unwrap()));
            }
        }
    }
}
