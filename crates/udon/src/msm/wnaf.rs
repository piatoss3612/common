//! Shared GLV wNAF ladders over strided odd-multiple banks.
use super::{PreparedScalars, Requirements, Scratch};
use crate::{
    checks::{assert_length, assert_scratch},
    curve::{
        AffinePoint, CurveError, CurveTableEntry, CurveTableRequirements, PastaCurve,
        ProjectivePoint,
        pasta::{batch, checked_count},
    },
    exec::{ExecutionOptions, Executor, TaskBudget, for_each_chunk_mut},
    field::PastaField,
};
use core::{marker::PhantomData, ops::Range};

/// Borrowed odd multiples `(2j+1)P` in strided layers.
///
/// Every entry must be nonidentity and satisfy its type's mathematical
/// invariants. Use [`super::Bases::Odd`] or [`super::Bases::OddPrepared`] to
/// supply this storage to MSMs. [`super::SharedScalarInput`] can use the odd
/// layers to share GLV wNAF recoding across outputs. Execution can also use
/// the original bases; kernel selection is private and may change.
///
/// Entries remain owned by the caller, separate from execution workspace
/// ceilings. Preparation and arithmetic allocate nothing and are variable-time.
#[derive(Clone, Copy, Debug)]
pub struct OddTable<'a, C: PastaCurve, E: CurveTableEntry<C> = AffinePoint<C>> {
    entries: &'a [E],
    width: u8,
    stride: usize,
    start: usize,
    len: usize,
    marker: PhantomData<C>,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        curve::{Pallas, Vesta},
        exec::SerialExecutor,
    };

    #[test]
    fn affine_ladders_visit_only_active_columns() {
        fn check<C: PastaCurve>() {
            let g = AffinePoint::<C>::GENERATOR;
            let entries = [g];
            let table = OddTable::bind(2, &entries, 1, 1).unwrap();
            for (values, inversions, expected) in [
                ([0, 0], 0, ProjectivePoint::IDENTITY),
                ([1, 0], 0, g.to_projective()),
                ([1, -1], 0, ProjectivePoint::IDENTITY),
                ([2, 0], 1, g.to_projective().double()),
            ] {
                let mut records = [super::super::ScalarStorage::ZERO; 2];
                let scalars = PreparedScalars::signed(
                    &values,
                    &mut records,
                    TaskBudget::SERIAL,
                    &SerialExecutor,
                );
                let ladder = Ladder {
                    table,
                    scalars,
                    outputs: 3,
                    output_stride: 0,
                    term_stride: 0,
                };
                let (r, lanes, workers) = ladder.plan(ExecutionOptions::DEFAULT, None).unwrap();
                let mut scratch = super::super::test_support::Buffers::new(r);
                let mut output = [ProjectivePoint::GENERATOR; 3];
                assert_eq!(
                    crate::field::count_inversions(|| ladder.execute(
                        &mut output,
                        lanes,
                        workers,
                        &SerialExecutor,
                        scratch.borrow().checked(r)
                    )),
                    inversions
                );
                assert_eq!(output, [expected; 3]);
                scratch.tails(r);
            }
        }
        check::<Pallas>();
        check::<Vesta>();
    }

    #[test]
    fn affine_steps_mix_active_and_exceptional_lanes() {
        fn check<C: PastaCurve>() {
            let g = AffinePoint::<C>::GENERATOR;
            let p = g.to_projective();
            let two = p.double();
            let three = two.add(&p);
            let addends = [g, g, g.neg(), *two.to_point().as_affine().unwrap()];
            let mut points = [ProjectivePoint::IDENTITY, p, p, p];
            let mut scratch = [PastaField::ONE; 9];
            assert_eq!(
                crate::field::count_inversions(|| batch::affine_step(
                    &mut points,
                    &mut scratch,
                    Some(|i| addends[i])
                )),
                1
            );
            assert_eq!(points, [p, two, ProjectivePoint::IDENTITY, three]);
            assert!(scratch[8].is_one());
            assert_eq!(
                crate::field::count_inversions(|| batch::affine_step(
                    &mut points,
                    &mut scratch,
                    None::<fn(usize) -> AffinePoint<C>>
                )),
                1
            );
            assert_eq!(
                points,
                [
                    p.double(),
                    two.double(),
                    ProjectivePoint::IDENTITY,
                    three.double()
                ]
            );
            assert!(scratch[8].is_one());
        }
        check::<Pallas>();
        check::<Vesta>();
    }
}
impl<'a, C: PastaCurve, E: CurveTableEntry<C>> OddTable<'a, C, E> {
    const fn layers(width: u8) -> Result<usize, CurveError> {
        if width < 2 || width > 8 {
            return Err(CurveError::InvalidWindowBits { bits: width as u32 });
        }
        Ok(1 << (width - 2))
    }
    /// Counts for preparing odd multiples at a width in `2..=8`.
    ///
    /// Returns [`CurveError::InvalidWindowBits`] for other widths, or
    /// [`CurveError::SizeOverflow`] if a required slice cannot be represented.
    /// Scratch holds one layer and is reused during preparation.
    pub const fn requirements(
        width: u8,
        bases: usize,
    ) -> Result<CurveTableRequirements, CurveError> {
        Ok(CurveTableRequirements {
            table_entries: size!(checked_count::<E>(bases, size!(Self::layers(width)))),
            projective_scratch: size!(checked_count::<ProjectivePoint<C>>(bases, 1)),
            field_scratch: size!(checked_count::<PastaField<C::Base>>(bases, 1)),
        })
    }
    /// Prepares contiguous layer-major odd multiples.
    ///
    /// Size buffers with [`Self::requirements`]. Entries have exactly the
    /// reported count; scratch has at least the reported counts. Source bases
    /// must satisfy their entry type's invariants. Returns the same width and
    /// size errors as `requirements`; mismatched buffer lengths panic. Both
    /// checks precede writes and executor work. Surplus scratch tails remain
    /// untouched. Only entries remain borrowed, not bases, scratch, or executor.
    ///
    /// An executor panic may leave partial entries and scratch. Storage is
    /// reusable after all scoped jobs finish unwinding; there is no rollback.
    pub fn prepare<B: CurveTableEntry<C>, X: Executor>(
        width: u8,
        bases: &[B],
        entries: &'a mut [E],
        projective: &mut [ProjectivePoint<C>],
        field: &mut [PastaField<C::Base>],
        budget: TaskBudget,
        executor: &X,
    ) -> Result<Self, CurveError> {
        let count = Self::requirements(width, bases.len())?.table_entries;
        assert_length("entries", count, entries.len());
        assert_scratch("projective", bases.len(), projective.len());
        assert_scratch("field", bases.len(), field.len());
        for layer in 0..Self::layers(width)? {
            let (previous, next) = entries.split_at_mut(layer * bases.len());
            for_each_chunk_mut(
                &mut projective[..bases.len()],
                64,
                budget,
                executor,
                |chunk, out, _| {
                    for (i, p) in out.iter_mut().enumerate() {
                        let index = chunk * 64 + i;
                        let base = bases[index].affine().to_projective();
                        *p = if layer == 0 {
                            base
                        } else {
                            previous[(layer - 1) * bases.len() + index]
                                .affine()
                                .to_projective()
                                .add(&base.double())
                        };
                    }
                },
            );
            batch::normalize(
                &projective[..bases.len()],
                &mut field[..bases.len()],
                |i, p| {
                    next[i] = E::from_affine(p.as_affine().expect("nonidentity odd multiple"));
                },
            );
        }
        Self::bind(width, entries, bases.len(), bases.len())
    }
    /// Binds trusted odd layers without writing or copying storage.
    ///
    /// Layer `j`, base `i` lives at `j * layer_stride + i`. Returns
    /// [`CurveError::InvalidWindowBits`] for widths outside `2..=8`, or
    /// [`CurveError::SizeOverflow`] for an unrepresentable extent. Panics if
    /// entries do not cover that extent; surplus entries are allowed. An empty
    /// bank addresses no entries regardless of its stride.
    ///
    /// Only dimensions are checked. Entries must satisfy their type's invariants
    /// and the indicated odd-multiple relations, including when layers overlap.
    /// Invalid mathematical data can panic during arithmetic or produce wrong
    /// results. This layout accepts every second layer of an α bank.
    pub fn bind(
        width: u8,
        entries: &'a [E],
        bases: usize,
        layer_stride: usize,
    ) -> Result<Self, CurveError> {
        let layers = Self::layers(width)?;
        let extent = if bases == 0 {
            0
        } else {
            (layers - 1)
                .checked_mul(layer_stride)
                .and_then(|n| n.checked_add(bases))
                .ok_or(CurveError::SizeOverflow)?
        };
        checked_count::<E>(extent, 1)?;
        assert!(extent <= entries.len(), "incomplete odd table");
        Ok(Self {
            entries,
            width,
            stride: layer_stride,
            start: 0,
            len: bases,
            marker: PhantomData,
        })
    }
    /// Borrows a consecutive sub-bank, retaining the original layer stride.
    ///
    /// Panics for reversed or out-of-range bounds.
    pub fn range(self, range: Range<usize>) -> Self {
        assert!(range.start <= range.end && range.end <= self.len);
        Self {
            start: self.start + range.start,
            len: range.end - range.start,
            ..self
        }
    }
    /// Number of bases.
    pub const fn len(self) -> usize {
        self.len
    }
    /// Whether the bank is empty.
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
    pub(super) fn originals(self) -> &'a [E] {
        &self.entries[self.start..self.start + self.len]
    }
    fn entry(self, layer: usize, base: usize) -> E {
        self.entries[layer * self.stride + self.start + base]
    }
}

// The matrix planner owns validation and fallback. This descriptor is only a
// monomorphic view for one admitted, complete scalar row.
#[derive(Clone, Copy)]
pub(super) struct Ladder<'a, C: PastaCurve, E: CurveTableEntry<C>> {
    pub(super) table: OddTable<'a, C, E>,
    pub(super) scalars: PreparedScalars<'a, C>,
    pub(super) outputs: usize,
    pub(super) output_stride: usize,
    pub(super) term_stride: usize,
}
impl<C: PastaCurve, E: CurveTableEntry<C>> Ladder<'_, C, E> {
    pub(super) fn plan(
        &self,
        options: ExecutionOptions,
        capacity: Option<Requirements>,
    ) -> Result<(Requirements, usize, usize), CurveError> {
        if self.outputs == 0 || self.scalars.is_empty() {
            return Ok((Requirements::default(), 1, 1));
        }
        let digits = checked_count::<u8>(self.scalars.len(), 256)?;
        let mut lanes = self.outputs.min(256);
        let mut workers = options
            .task_budget()
            .get()
            .min(self.outputs.div_ceil(lanes));
        loop {
            let r = Requirements {
                digits,
                field: checked_count::<PastaField<C::Base>>(2 * lanes, workers)?,
                ..Requirements::default()
            };
            let bytes = r.bytes::<C>()?;
            let memory = options.memory_limit().is_none_or(|limit| bytes <= limit);
            if memory && capacity.is_none_or(|c| r.fits(c)) {
                return Ok((r, lanes, workers));
            }
            if workers > 1 {
                workers = workers.div_ceil(2);
            } else if lanes > 1 {
                lanes = lanes.div_ceil(2);
            } else if !memory {
                return Err(CurveError::MemoryLimit {
                    limit: options.memory_limit().unwrap(),
                    required: bytes,
                });
            } else {
                return Err(r.capacity_error(capacity.unwrap()));
            }
        }
    }
    pub(super) fn execute<X: Executor>(
        &self,
        output: &mut [ProjectivePoint<C>],
        lanes: usize,
        workers: usize,
        executor: &X,
        scratch: Scratch<'_, C>,
    ) {
        let mut columns = 0;
        for (record, row) in self
            .scalars
            .records
            .iter()
            .zip(scratch.digits.chunks_exact_mut(256))
        {
            for (half, mut value) in record.halves.into_iter().enumerate() {
                for bit in 0..128 {
                    let digit = if value & 1 != 0 {
                        let residue = (value & ((1 << self.table.width) - 1)) as i16;
                        if residue >= 1 << (self.table.width - 1) {
                            residue - (1 << self.table.width)
                        } else {
                            residue
                        }
                    } else {
                        0
                    };
                    row[half * 128 + bit] = digit as i8 as u8;
                    if digit != 0 {
                        columns = columns.max(bit + 1);
                    }
                    // Avoid overflow near the signed bound when subtracting a negative digit.
                    value = (value >> 1) - (i128::from(digit) >> 1);
                }
                assert_eq!(value, 0, "bounded GLV wNAF");
            }
        }
        self.run(
            output,
            0,
            scratch.digits,
            scratch.field,
            columns,
            lanes,
            workers,
            executor,
        );
    }
    #[allow(clippy::too_many_arguments)] // Separate borrowed output and worker-local scratch.
    fn run<X: Executor>(
        &self,
        output: &mut [ProjectivePoint<C>],
        offset: usize,
        digits: &[u8],
        fields: &mut [PastaField<C::Base>],
        columns: usize,
        lanes: usize,
        workers: usize,
        executor: &X,
    ) {
        if workers > 1 {
            let left_workers = workers / 2;
            let mid = output.len() * left_workers / workers;
            let (a, b) = output.split_at_mut(mid);
            let (fa, fb) = fields.split_at_mut(2 * lanes * left_workers);
            executor.join(
                || {
                    self.run(
                        a,
                        offset,
                        digits,
                        fa,
                        columns,
                        lanes,
                        left_workers,
                        executor,
                    )
                },
                || {
                    self.run(
                        b,
                        offset + mid,
                        digits,
                        fb,
                        columns,
                        lanes,
                        workers - left_workers,
                        executor,
                    )
                },
            );
            return;
        }
        for (chunk, output) in output.chunks_mut(lanes).enumerate() {
            let start = offset + chunk * lanes;
            output.fill(ProjectivePoint::IDENTITY);
            for bit in (0..columns).rev() {
                batch::affine_step(output, fields, None::<fn(usize) -> AffinePoint<C>>);
                for term in 0..self.scalars.len() {
                    for half in 0..2 {
                        let digit = digits[term * 256 + half * 128 + bit] as i8;
                        if digit == 0 {
                            continue;
                        }
                        batch::affine_step(
                            output,
                            fields,
                            Some(|lane| {
                                let base =
                                    (start + lane) * self.output_stride + term * self.term_stride;
                                let point = self
                                    .table
                                    .entry(usize::from(digit.unsigned_abs() / 2), base)
                                    .rotated(half);
                                if digit < 0 { point.neg() } else { point }
                            }),
                        );
                    }
                }
            }
        }
    }
}
