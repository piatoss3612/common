//! Monomorphic base access and arithmetic for one complete chunk or window.

use super::storage::Storage;
use crate::curve::reduce::{collapse, reduce};
use crate::exec::run::ReadView;

use super::{
    Accumulation, Bases, Input, ScalarStorage,
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
    pub(super) affine: &'a mut [AffinePoint<C>],
    pub(super) projective: &'a mut [ProjectivePoint<C>],
    pub(super) field: &'a mut [PastaField<C::Base>],
    pub(super) indices: &'a mut [usize],
}
#[derive(Clone, Copy)]
pub(super) struct Task {
    pub(super) offset: usize,
    pub(super) window: usize,
    pub(super) pass: usize,
    pub(super) geometry: Geometry,
    pub(super) accumulation: Accumulation,
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
#[derive(Clone, Copy)]
pub(super) enum Indices<'a> {
    Slice(&'a [u32]),
    Strided {
        offset: usize,
        stride: usize,
    },
    Fragment {
        view: &'a dyn ReadView<u32>,
        offset: usize,
    },
}
impl Indices<'_> {
    #[inline]
    fn get(self, index: usize) -> usize {
        match self {
            Self::Slice(values) => values[index] as usize,
            Self::Strided { offset, stride } => offset + index * stride,
            Self::Fragment { view, offset } => {
                *view.get(index - offset).expect("validated index fragment") as usize
            }
        }
    }
}
pub(super) struct Selection<'a, C: PastaCurve> {
    pub(super) bases: Bases<'a, C>,
    pub(super) indices: Option<Indices<'a>>,
}
struct View<'a, B, const INDEXED: bool> {
    bases: &'a [B],
    indices: Indices<'a>,
    offset: usize,
    stride: usize,
}
impl<B: Copy, const INDEXED: bool> View<'_, B, INDEXED> {
    #[inline]
    fn index(&self, term: usize) -> usize {
        let i = self.offset + term;
        if INDEXED { self.indices.get(i) } else { i }
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
    run_view(input, records, digits, task, work)
}

pub(super) fn run_view<C: PastaCurve>(
    input: &Input<'_, C>,
    records: impl Storage<ScalarStorage<C>>,
    digits: impl Storage<u8>,
    task: Task,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    run_selected(
        &Selection {
            bases: input.bases,
            indices: input.indices.map(Indices::Slice),
        },
        records,
        digits,
        task,
        work,
    )
}

pub(super) fn run_selected<C: PastaCurve>(
    input: &Selection<'_, C>,
    records: impl Storage<ScalarStorage<C>>,
    digits: impl Storage<u8>,
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
    input: &Selection<'_, C>,
    bases: &[B],
    records: impl Storage<ScalarStorage<C>>,
    digits: impl Storage<u8>,
    task: Task,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    if task.geometry != Geometry::Joint {
        return access(input, bases, records, digits, task, work, 8);
    }
    let top = digits
        .chunks_exact(recode::JOINT_STRIDE)
        .map(|r| usize::from(r.get(eisenstein::MAX_DIGITS)))
        .max()
        .unwrap_or(0);
    let mut sum = ProjectivePoint::IDENTITY;
    for column in (0..top).rev() {
        sum = sum.double();
        for (i, row) in digits.chunks_exact(recode::JOINT_STRIDE).enumerate() {
            let code = row.get(column);
            if code != 0 {
                let j = input
                    .indices
                    .map_or(task.offset + i, |indices| indices.get(task.offset + i));
                sum = sum.add_mixed(&eisenstein::digit_point(&bases[8 * j..8 * j + 8], code));
            }
        }
    }
    sum
}

fn access<C: PastaCurve, B: Base<C>>(
    input: &Selection<'_, C>,
    bases: &[B],
    records: impl Storage<ScalarStorage<C>>,
    digits: impl Storage<u8>,
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
                indices: Indices::Slice(&[]),
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
    records: impl Storage<ScalarStorage<C>>,
    digits: impl Storage<u8>,
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
    records: impl Storage<ScalarStorage<C>>,
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
    records: impl Storage<ScalarStorage<C>>,
    digits: impl Storage<u8>,
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
            let len = usize::from(digits.get(i * recode::JOINT_STRIDE + eisenstein::MAX_DIGITS));
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
                let code = digits.get(term * recode::JOINT_STRIDE + column);
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
#[inline]
fn accumulate_projective<C: PastaCurve, B: Base<C>, const INDEXED: bool, const DIRECT: bool>(
    view: &View<'_, B, INDEXED>,
    records: impl Storage<ScalarStorage<C>>,
    digits: impl Storage<u8>,
    terms: core::ops::Range<usize>,
    task: Task,
    sums: &mut [ProjectivePoint<C>],
) {
    recode::window_views::<C, DIRECT>(
        records,
        digits,
        terms,
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
}

fn window<C: PastaCurve, B: Base<C>, const INDEXED: bool, const DIRECT: bool>(
    view: View<'_, B, INDEXED>,
    records: impl Storage<ScalarStorage<C>>,
    digits: impl Storage<u8>,
    task: Task,
    work: &mut Work<'_, C>,
) -> ProjectivePoint<C> {
    let terms = records.len();
    let buckets = task.geometry.buckets();
    if task.accumulation == Accumulation::Projective {
        let sums = &mut work.projective[..buckets];
        sums.fill(ProjectivePoint::IDENTITY);
        accumulate_projective::<C, B, INDEXED, DIRECT>(
            &view,
            records,
            digits,
            0..terms,
            task,
            sums,
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
            accumulate_projective::<C, B, INDEXED, DIRECT>(
                &view,
                records,
                digits,
                first..end,
                task,
                sums,
            );
            return collapse_projective(sums);
        }
        cursors.copy_from_slice(lens);
        recode::window_views::<C, DIRECT>(
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
        recode::window_views::<C, DIRECT>(
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
        reduce(&mut points[..total], starts, lens, work.field);
        for i in 0..buckets {
            if lens[i] != 0 {
                survivors[i] = points[starts[i]];
            }
        }
    }
    collapse(survivors, lens)
}

pub(super) fn stream<C: PastaCurve>(
    input: &Input<'_, C>,
    terms: usize,
    digits: &[u8],
    task: Task,
    sums: &mut [ProjectivePoint<C>],
) {
    stream_view(input, terms, digits, task, sums)
}

pub(super) fn stream_view<C: PastaCurve>(
    input: &Input<'_, C>,
    terms: usize,
    digits: impl Storage<u8>,
    task: Task,
    sums: &mut [ProjectivePoint<C>],
) {
    stream_selected(
        &Selection {
            bases: input.bases,
            indices: input.indices.map(Indices::Slice),
        },
        terms,
        digits,
        task,
        sums,
    )
}

pub(super) fn stream_selected<C: PastaCurve>(
    input: &Selection<'_, C>,
    terms: usize,
    digits: impl Storage<u8>,
    task: Task,
    sums: &mut [ProjectivePoint<C>],
) {
    fn deposit<C: PastaCurve, B: Base<C>, const INDEXED: bool>(
        view: View<'_, B, INDEXED>,
        terms: usize,
        digits: impl Storage<u8>,
        task: Task,
        sums: &mut [ProjectivePoint<C>],
    ) {
        recode::rows_view(
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
                        indices: Indices::Slice(&[]),
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

/// Visits short scalar digits across adjacent output lanes; full-width tiles
/// reuse the selected single-output kernel.
#[expect(
    clippy::too_many_arguments,
    reason = "A checked matrix supplies two strides and a bounded output tile."
)]
pub(super) fn shared<C: PastaCurve>(
    bases: Bases<'_, C>,
    output_offset: usize,
    output_stride: usize,
    term_stride: usize,
    records: &[ScalarStorage<C>],
    digits: &[u8],
    task: Task,
    work: &mut Work<'_, C>,
    output: &mut [ProjectivePoint<C>],
) {
    if output.len() == 1 {
        output[0] = run_selected(
            &Selection {
                bases,
                indices: Some(Indices::Strided {
                    offset: output_offset,
                    stride: term_stride,
                }),
            },
            records,
            digits,
            task,
            work,
        );
        return;
    }
    #[expect(
        clippy::too_many_arguments,
        reason = "Monomorphic matrix access preserves the borrowed base layout."
    )]
    fn interleaved<C: PastaCurve, B: Base<C>>(
        bases: &[B],
        entry_stride: usize,
        output_offset: usize,
        output_stride: usize,
        term_stride: usize,
        records: &[ScalarStorage<C>],
        task: Task,
        work: &mut Work<'_, C>,
        output: &mut [ProjectivePoint<C>],
    ) {
        let lanes = output.len();
        let point = |term: usize, lane: usize, rotation: usize| {
            bases[(output_offset + lane * output_stride + (task.offset + term) * term_stride)
                * entry_stride]
                .point(rotation)
        };
        output.fill(ProjectivePoint::IDENTITY);
        if let Geometry::Short(bits) = task.geometry {
            if bits <= 1 || records.len() < 32 {
                for bit in (0..bits).rev() {
                    for sum in output.iter_mut() {
                        *sum = sum.double();
                    }
                    for (term, record) in records.iter().enumerate() {
                        if record.magnitude & (1 << bit) != 0 {
                            for (lane, sum) in output.iter_mut().enumerate() {
                                if let Some(p) = point(term, lane, 0) {
                                    *sum =
                                        sum.add_mixed(&if record.negative { p.neg() } else { p });
                                }
                            }
                        }
                    }
                }
            } else {
                let buckets = &mut work.projective[..16 * lanes];
                for window in (0..usize::from(bits).div_ceil(4)).rev() {
                    for sum in output.iter_mut() {
                        for _ in 0..4 {
                            *sum = sum.double();
                        }
                    }
                    buckets.fill(ProjectivePoint::IDENTITY);
                    for (term, record) in records.iter().enumerate() {
                        let digit = ((record.magnitude >> (4 * window)) & 15) as usize;
                        if digit != 0 {
                            for lane in 0..lanes {
                                if let Some(p) = point(term, lane, 0) {
                                    let sum = &mut buckets[digit * lanes + lane];
                                    *sum =
                                        sum.add_mixed(&if record.negative { p.neg() } else { p });
                                }
                            }
                        }
                    }
                    for (lane, sum) in output.iter_mut().enumerate() {
                        let mut running = ProjectivePoint::IDENTITY;
                        for bucket in (1..16).rev() {
                            running = running.add(&buckets[bucket * lanes + lane]);
                            *sum = sum.add(&running);
                        }
                    }
                }
            }
        } else {
            unreachable!("multiple output lanes require short scalars");
        }
    }
    macro_rules! access {
        ($bases:expr, $stride:expr) => {
            interleaved(
                $bases,
                $stride,
                output_offset,
                output_stride,
                term_stride,
                records,
                task,
                work,
                output,
            )
        };
    }
    match bases {
        Bases::Affine(b) => access!(b, 1),
        Bases::Prepared(b) => access!(b, 1),
        Bases::Points(b) => access!(b, 1),
        Bases::Compact(b) => access!(b.as_slice(), 8),
        Bases::CompactPrepared(b) => access!(b.as_slice(), 8),
    }
}
