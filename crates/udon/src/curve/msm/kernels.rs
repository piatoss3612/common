//! Monomorphic base access and arithmetic shared by serial and grouped execution.
//!
//! Select the base representation and indexed access once per task, so inner
//! loops specialize without repeating enum dispatch for every term.

use super::{Bases, Input, buckets, recode};
use crate::{
    curve::{
        AffinePoint, CurveTableEntry, EisensteinTableBatch, PastaCurve, Point, PreparedAffinePoint,
        ProjectivePoint, eisenstein, eisenstein_batch,
    },
    exec::SerialExecutor,
    field::PastaField,
};

pub(super) struct Work<'a, C: PastaCurve> {
    pub affine: &'a mut [AffinePoint<C>],
    pub projective: &'a mut [ProjectivePoint<C>],
    pub field: &'a mut [PastaField<C::Base>],
    pub indices: &'a mut [usize],
}

#[derive(Clone, Copy)]
pub(super) struct Task {
    pub window: usize,
    pub part: usize,
    pub parts: usize,
    pub pass: usize,
}

trait Base<C: PastaCurve>: Copy + Sync {
    fn point(self, rotation: usize) -> Option<AffinePoint<C>>;
    fn is_identity(self) -> bool {
        false
    }
}

impl<C: PastaCurve> Base<C> for AffinePoint<C> {
    fn point(self, rotation: usize) -> Option<AffinePoint<C>> {
        Some(self.rotated(rotation))
    }
}
impl<C: PastaCurve> Base<C> for PreparedAffinePoint<C> {
    fn point(self, rotation: usize) -> Option<AffinePoint<C>> {
        Some(self.rotated(rotation))
    }
}
impl<C: PastaCurve> Base<C> for Point<C> {
    fn point(self, rotation: usize) -> Option<AffinePoint<C>> {
        self.as_affine().map(|p| p.rotated(rotation))
    }
    fn is_identity(self) -> bool {
        Point::is_identity(&self)
    }
}

struct View<'a, B, const INDEXED: bool> {
    bases: &'a [B],
    indices: &'a [u32],
}

impl<B: Copy, const INDEXED: bool> View<'_, B, INDEXED> {
    #[inline]
    fn at(&self, term: usize) -> B {
        self.bases[if INDEXED {
            self.indices[term] as usize
        } else {
            term
        }]
    }
}

pub(super) fn run<C: PastaCurve>(
    input: &Input<'_, C>,
    digits: &[u8],
    task: Task,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    match input.bases {
        Bases::Affine(bases) => access(input, bases, digits, task, work),
        Bases::Prepared(bases) => access(input, bases, digits, task, work),
        Bases::Points(bases) => access(input, bases, digits, task, work),
    }
}

fn access<C: PastaCurve, B: Base<C>>(
    input: &Input<'_, C>,
    bases: &[B],
    digits: &[u8],
    task: Task,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    match input.indices {
        Some(indices) => execute(
            input,
            View::<_, true> { bases, indices },
            digits,
            task,
            work,
        ),
        None => execute(
            input,
            View::<_, false> {
                bases,
                indices: &[],
            },
            digits,
            task,
            work,
        ),
    }
}

fn execute<C: PastaCurve, B: Base<C>, const INDEXED: bool>(
    input: &Input<'_, C>,
    view: View<'_, B, INDEXED>,
    digits: &[u8],
    task: Task,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    if input.len() < super::BOOTH_MIN {
        small(input, view, digits, task, work)
    } else {
        window(input.len(), view, digits, task, work)
    }
}

fn small<C: PastaCurve, B: Base<C>, const INDEXED: bool>(
    input: &Input<'_, C>,
    view: View<'_, B, INDEXED>,
    digits: &[u8],
    task: Task,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    let range = term_range(input.len(), task.part, task.parts);
    let pass = task.pass;
    let mut sum = ProjectivePoint::IDENTITY;
    let bits = usize::from(digits[0]);
    let digits = &digits[1..];
    if bits != 255 {
        // Short scalars avoid GLV setup. At 32 terms, four-bit projective
        // buckets amortize their reduction better than bit interleaving.
        if input.len() >= 32 {
            let buckets = &mut work.projective[..16];
            for window in (0..bits.div_ceil(4)).rev() {
                for _ in 0..4 {
                    sum = sum.double();
                }
                buckets.fill(ProjectivePoint::IDENTITY);
                for i in range.clone() {
                    let digit = (digits[i * eisenstein::MAX_DIGITS + window / 2]
                        >> (4 * (window % 2)))
                        & 15;
                    if digit != 0
                        && let Some(p) = view.at(i).point(0)
                    {
                        buckets[usize::from(digit)] = buckets[usize::from(digit)].add_mixed(&p);
                    }
                }
                let mut running = ProjectivePoint::IDENTITY;
                for bucket in buckets[1..].iter().rev() {
                    running = running.add(bucket);
                    sum = sum.add(&running);
                }
            }
            return sum;
        }
        for bit in (0..bits).rev() {
            sum = sum.double();
            for i in range.clone() {
                if digits[i * eisenstein::MAX_DIGITS + bit / 8] & (1 << (bit % 8)) != 0
                    && let Some(p) = view.at(i).point(0)
                {
                    sum = sum.add_mixed(&p);
                }
            }
        }
        return sum;
    }
    for start in (range.start..range.end).step_by(pass) {
        let n = pass.min(range.end - start);
        let (tables, bases) = work.affine.split_at_mut(8 * n);
        let bases = &mut bases[..n];
        // Compact preparation requires nonidentity bases. Substitute the generator
        // for identities, then suppress those terms using the recorded flags.
        for (i, base) in bases.iter_mut().enumerate() {
            let p = view.at(start + i).point(0);
            work.indices[i] = usize::from(p.is_some());
            *base = p.unwrap_or(AffinePoint::GENERATOR);
        }
        let r = EisensteinTableBatch::<C>::requirements(n).unwrap();
        eisenstein_batch::prepare_inner(
            bases,
            tables,
            &mut work.projective[..r.projective_scratch],
            &mut work.field[..r.field_scratch],
            1,
            &SerialExecutor,
        );
        let mut partial = ProjectivePoint::IDENTITY;
        for column in (0..eisenstein::MAX_DIGITS).rev() {
            partial = partial.double();
            for i in 0..n {
                let code = digits[(start + i) * eisenstein::MAX_DIGITS + column];
                if code != 0 && work.indices[i] != 0 {
                    partial = partial
                        .add_mixed(&eisenstein::digit_point(&tables[8 * i..8 * i + 8], code));
                }
            }
        }
        sum = sum.add(&partial);
    }
    sum
}

pub(super) fn term_range(n: usize, part: usize, parts: usize) -> core::ops::Range<usize> {
    let start = (n / parts) * part + part.min(n % parts);
    start..start + n / parts + usize::from(part < n % parts)
}

fn window<C: PastaCurve, B: Base<C>, const INDEXED: bool>(
    terms: usize,
    view: View<'_, B, INDEXED>,
    digits: &[u8],
    task: Task,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    let halves = 2;
    let buckets = super::BUCKETS;
    let range = term_range(terms, task.part, task.parts);
    let pass = task.pass.min(range.len());
    let capacity = halves * pass + buckets;
    let (points, survivors) = work.affine.split_at_mut(capacity);
    let survivors = &mut survivors[..buckets];
    let (starts, indices) = work.indices.split_at_mut(buckets);
    let (lens, indices) = indices.split_at_mut(buckets);
    let (cursors, writes) = indices.split_at_mut(buckets);
    lens.fill(0);
    for first in (range.start..range.end).step_by(pass) {
        let end = range.end.min(first + pass);
        // Reserve space for both prior survivors and this pass's deposits before
        // writing either, keeping every bucket's segment disjoint.
        cursors.copy_from_slice(lens);
        for term in first..end {
            if view.at(term).is_identity() {
                continue;
            }
            for half in 0..halves {
                let digit = recode::digit(digits, terms, term, task.window * halves + half);
                if digit != 0 {
                    cursors[usize::from(digit.unsigned_abs()) - 1] += 1;
                }
            }
        }
        let mut total = 0;
        for i in 0..buckets {
            starts[i] = total;
            total += cursors[i];
            if lens[i] != 0 {
                points[starts[i]] = survivors[i];
            }
            let cursor = starts[i] + lens[i];
            lens[i] = cursors[i];
            cursors[i] = cursor;
        }
        for term in first..end {
            let base = view.at(term);
            for half in 0..halves {
                let digit = recode::digit(digits, terms, term, task.window * halves + half);
                if digit != 0
                    && let Some(point) = base.point(half)
                {
                    let bucket = usize::from(digit.unsigned_abs()) - 1;
                    points[cursors[bucket]] = if digit < 0 { point.neg() } else { point };
                    cursors[bucket] += 1;
                }
            }
        }
        // All levels share the same field staging storage. Restricting to the
        // actual deposits also shortens each structure-of-arrays stride.
        buckets::reduce(&mut points[..total], starts, lens, work.field, writes);
        for i in 0..buckets {
            if lens[i] != 0 {
                survivors[i] = points[starts[i]];
            }
        }
    }
    // Carry affine survivors across passes; pay weighted projective collapse
    // only once per window partition.
    buckets::collapse(survivors, lens)
}
