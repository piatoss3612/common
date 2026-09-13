//! Monomorphic base access and arithmetic for one complete chunk or window.

use super::{
    Accumulation, Bases, Input, ScalarStorage, buckets,
    recode::{self, Geometry},
};
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
    pub offset: usize,
    pub window: usize,
    pub pass: usize,
    pub geometry: Geometry,
    pub accumulation: Accumulation,
}
trait Base<C: PastaCurve>: Copy + Sync {
    fn point(self, rotation: usize) -> Option<AffinePoint<C>>;
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
}
struct View<'a, B, const INDEXED: bool> {
    bases: &'a [B],
    indices: &'a [u32],
    offset: usize,
    stride: usize,
}
impl<B: Copy, const INDEXED: bool> View<'_, B, INDEXED> {
    #[inline]
    fn index(&self, term: usize) -> usize {
        let i = self.offset + term;
        if INDEXED { self.indices[i] as usize } else { i }
    }
    #[inline]
    fn at(&self, term: usize) -> B {
        self.bases[self.index(term) * self.stride]
    }
}

pub(super) fn run<C: PastaCurve>(
    input: &Input<'_, C>,
    records: &[ScalarStorage<C>],
    digits: &[u8],
    task: Task,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    match input.bases {
        Bases::Affine(b) => access(input, b, records, digits, task, work, 1),
        Bases::Prepared(b) => access(input, b, records, digits, task, work, 1),
        Bases::Points(b) => access(input, b, records, digits, task, work, 1),
        Bases::Compact(b) => compact(input, b.as_slice(), records, digits, task, work),
        Bases::CompactPrepared(b) => compact(input, b.as_slice(), records, digits, task, work),
    }
}

fn compact<C: PastaCurve, B: Base<C> + CurveTableEntry<C>>(
    input: &Input<'_, C>,
    bases: &[B],
    records: &[ScalarStorage<C>],
    digits: &[u8],
    task: Task,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    if task.geometry != Geometry::Joint {
        return access(input, bases, records, digits, task, work, 8);
    }
    let top = digits
        .chunks_exact(recode::JOINT_STRIDE)
        .map(|r| usize::from(r[eisenstein::MAX_DIGITS]))
        .max()
        .unwrap_or(0);
    let mut sum = ProjectivePoint::IDENTITY;
    for column in (0..top).rev() {
        sum = sum.double();
        for (i, row) in digits.chunks_exact(recode::JOINT_STRIDE).enumerate() {
            let code = row[column];
            if code != 0 {
                let j = input
                    .indices
                    .map_or(task.offset + i, |indices| indices[task.offset + i] as usize);
                sum = sum.add_mixed(&eisenstein::digit_point(&bases[8 * j..8 * j + 8], code));
            }
        }
    }
    sum
}

fn access<C: PastaCurve, B: Base<C>>(
    input: &Input<'_, C>,
    bases: &[B],
    records: &[ScalarStorage<C>],
    digits: &[u8],
    task: Task,
    work: &mut Work<'_, C>,
    stride: usize,
) -> ProjectivePoint<C> {
    match input.indices {
        Some(indices) => execute(
            View::<_, true> {
                bases,
                indices,
                offset: task.offset,
                stride,
            },
            records,
            digits,
            task,
            work,
        ),
        None => execute(
            View::<_, false> {
                bases,
                indices: &[],
                offset: task.offset,
                stride,
            },
            records,
            digits,
            task,
            work,
        ),
    }
}
fn execute<C: PastaCurve, B: Base<C>, const INDEXED: bool>(
    view: View<'_, B, INDEXED>,
    records: &[ScalarStorage<C>],
    digits: &[u8],
    task: Task,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    match task.geometry {
        Geometry::Short(bits) => short(view, records, bits, work),
        Geometry::Joint => joint(view, records, digits, task.pass, work),
        Geometry::Booth(_) if digits.is_empty() => {
            window::<C, B, INDEXED, true>(view, records, digits, task, work)
        }
        Geometry::Booth(_) => window::<C, B, INDEXED, false>(view, records, digits, task, work),
    }
}
fn short<C: PastaCurve, B: Base<C>, const INDEXED: bool>(
    view: View<'_, B, INDEXED>,
    records: &[ScalarStorage<C>],
    bits: u8,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    let mut sum = ProjectivePoint::IDENTITY;
    // Zero, Boolean, and signed-unit rows execute at every input length without
    // GLV tables, inversions, or a full-width doubling ladder.
    if bits <= 1 || records.len() < 32 {
        for bit in (0..bits).rev() {
            sum = sum.double();
            for (i, record) in records.iter().enumerate() {
                if record.magnitude & (1 << bit) != 0
                    && let Some(p) = view.at(i).point(0)
                {
                    sum = sum.add_mixed(&if record.negative { p.neg() } else { p });
                }
            }
        }
    } else {
        let buckets = &mut work.projective[..16];
        for window in (0..usize::from(bits).div_ceil(4)).rev() {
            for _ in 0..4 {
                sum = sum.double();
            }
            buckets.fill(ProjectivePoint::IDENTITY);
            for (i, record) in records.iter().enumerate() {
                let digit = ((record.magnitude >> (4 * window)) & 15) as usize;
                if digit != 0
                    && let Some(p) = view.at(i).point(0)
                {
                    buckets[digit] =
                        buckets[digit].add_mixed(&if record.negative { p.neg() } else { p });
                }
            }
            let mut running = ProjectivePoint::IDENTITY;
            for bucket in buckets[1..].iter().rev() {
                running = running.add(bucket);
                sum = sum.add(&running);
            }
        }
    }
    sum
}
fn joint<C: PastaCurve, B: Base<C>, const INDEXED: bool>(
    view: View<'_, B, INDEXED>,
    records: &[ScalarStorage<C>],
    digits: &[u8],
    pass: usize,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    let mut sum = ProjectivePoint::IDENTITY;
    for first in (0..records.len()).step_by(pass) {
        let end = records.len().min(first + pass);
        let (tables, bases) = work.affine.split_at_mut(8 * (end - first));
        let mut active = 0;
        let mut top = 0;
        for i in first..end {
            let len = usize::from(digits[i * recode::JOINT_STRIDE + eisenstein::MAX_DIGITS]);
            if len != 0
                && let Some(p) = view.at(i).point(0)
            {
                bases[active] = p;
                work.indices[active] = i;
                active += 1;
                top = top.max(len);
            }
        }
        if active == 0 {
            continue;
        }
        let r = EisensteinTableBatch::<C>::requirements(active).unwrap();
        eisenstein_batch::prepare_inner(
            &bases[..active],
            &mut tables[..8 * active],
            &mut work.projective[..r.projective_scratch],
            &mut work.field[..r.field_scratch],
            1,
            &SerialExecutor,
        );
        let mut partial = ProjectivePoint::IDENTITY;
        for column in (0..top).rev() {
            partial = partial.double();
            for (i, &term) in work.indices[..active].iter().enumerate() {
                let code = digits[term * recode::JOINT_STRIDE + column];
                if code != 0 {
                    partial = partial
                        .add_mixed(&eisenstein::digit_point(&tables[8 * i..8 * i + 8], code));
                }
            }
        }
        sum = sum.add(&partial);
    }
    sum
}
pub(super) fn collapse_projective<C: PastaCurve>(
    buckets: &[ProjectivePoint<C>],
) -> ProjectivePoint<C> {
    let mut running = ProjectivePoint::IDENTITY;
    let mut sum = ProjectivePoint::IDENTITY;
    for bucket in buckets.iter().rev() {
        running = running.add(bucket);
        sum = sum.add(&running);
    }
    sum
}
fn window<C: PastaCurve, B: Base<C>, const INDEXED: bool, const DIRECT: bool>(
    view: View<'_, B, INDEXED>,
    records: &[ScalarStorage<C>],
    digits: &[u8],
    task: Task,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    let terms = records.len();
    let buckets = task.geometry.buckets();
    if task.accumulation == Accumulation::Projective {
        let sums = &mut work.projective[..buckets];
        sums.fill(ProjectivePoint::IDENTITY);
        recode::window_rows::<C, DIRECT>(
            records,
            digits,
            0..terms,
            task.geometry,
            task.window,
            &mut |term, a: i16, b: i16| {
                for (half, digit) in [a, b].into_iter().enumerate() {
                    if digit != 0
                        && let Some(p) = view.at(term).point(half)
                    {
                        let i = usize::from(digit.unsigned_abs()) - 1;
                        sums[i] = sums[i].add_mixed(&if digit < 0 { p.neg() } else { p });
                    }
                }
            },
        );
        return collapse_projective(sums);
    }
    let pass = task.pass.min(terms);
    let (points, survivors) = work.affine.split_at_mut(2 * pass + buckets);
    let survivors = &mut survivors[..buckets];
    let (starts, indices) = work.indices.split_at_mut(buckets);
    let (lens, cursors) = indices.split_at_mut(buckets);
    let cursors = &mut cursors[..buckets];
    lens.fill(0);
    for first in (0..terms).step_by(pass) {
        let end = terms.min(first + pass);
        if task.accumulation == Accumulation::Hybrid && end == terms {
            let sums = &mut work.projective[..buckets];
            for i in 0..buckets {
                sums[i] = if lens[i] == 0 {
                    ProjectivePoint::IDENTITY
                } else {
                    survivors[i].to_projective()
                };
            }
            recode::window_rows::<C, DIRECT>(
                records,
                digits,
                first..end,
                task.geometry,
                task.window,
                &mut |term, a: i16, b: i16| {
                    for (half, digit) in [a, b].into_iter().enumerate() {
                        if digit != 0
                            && let Some(p) = view.at(term).point(half)
                        {
                            let i = usize::from(digit.unsigned_abs()) - 1;
                            sums[i] = sums[i].add_mixed(&if digit < 0 { p.neg() } else { p });
                        }
                    }
                },
            );
            return collapse_projective(sums);
        }
        cursors.copy_from_slice(lens);
        recode::window_rows::<C, DIRECT>(
            records,
            digits,
            first..end,
            task.geometry,
            task.window,
            &mut |term, a: i16, b: i16| {
                if view.at(term).point(0).is_some() {
                    for digit in [a, b] {
                        if digit != 0 {
                            cursors[usize::from(digit.unsigned_abs()) - 1] += 1;
                        }
                    }
                }
            },
        );
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
        recode::window_rows::<C, DIRECT>(
            records,
            digits,
            first..end,
            task.geometry,
            task.window,
            &mut |term, a: i16, b: i16| {
                for (half, digit) in [a, b].into_iter().enumerate() {
                    if digit != 0
                        && let Some(p) = view.at(term).point(half)
                    {
                        let i = usize::from(digit.unsigned_abs()) - 1;
                        points[cursors[i]] = if digit < 0 { p.neg() } else { p };
                        cursors[i] += 1;
                    }
                }
            },
        );
        buckets::reduce(&mut points[..total], starts, lens, work.field);
        for i in 0..buckets {
            if lens[i] != 0 {
                survivors[i] = points[starts[i]];
            }
        }
    }
    buckets::collapse(survivors, lens)
}

pub(super) fn stream<C: PastaCurve>(
    input: &Input<'_, C>,
    terms: usize,
    digits: &[u8],
    task: Task,
    sums: &mut [ProjectivePoint<C>],
) {
    fn deposit<C: PastaCurve, B: Base<C>, const INDEXED: bool>(
        view: View<'_, B, INDEXED>,
        terms: usize,
        digits: &[u8],
        task: Task,
        sums: &mut [ProjectivePoint<C>],
    ) {
        recode::rows(
            digits,
            terms,
            0..terms,
            task.geometry,
            task.window,
            |term, a, b| {
                for (half, digit) in [a, b].into_iter().enumerate() {
                    if digit != 0
                        && let Some(p) = view.at(term).point(half)
                    {
                        let i = usize::from(digit.unsigned_abs()) - 1;
                        sums[i] = sums[i].add_mixed(&if digit < 0 { p.neg() } else { p });
                    }
                }
            },
        );
    }
    macro_rules! access {
        ($bases:expr, $stride:expr) => {
            match input.indices {
                Some(indices) => deposit(
                    View::<_, true> {
                        bases: $bases,
                        indices,
                        offset: task.offset,
                        stride: $stride,
                    },
                    terms,
                    digits,
                    task,
                    sums,
                ),
                None => deposit(
                    View::<_, false> {
                        bases: $bases,
                        indices: &[],
                        offset: task.offset,
                        stride: $stride,
                    },
                    terms,
                    digits,
                    task,
                    sums,
                ),
            }
        };
    }
    match input.bases {
        Bases::Affine(b) => access!(b, 1),
        Bases::Prepared(b) => access!(b, 1),
        Bases::Points(b) => access!(b, 1),
        Bases::Compact(b) => access!(b.as_slice(), 8),
        Bases::CompactPrepared(b) => access!(b.as_slice(), 8),
    }
}
