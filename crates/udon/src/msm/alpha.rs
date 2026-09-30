//! Prepared α-only recoding: α = 1 − ω in the Eisenstein integers.
//!
//! The residue codebook factors each nonzero pair modulo 2^w as u·η·δ.
//! Point layers store ηP; buckets accumulate u·ηP and are integrated with δ.
//! Each residue uses four bytes, including both signed carry corrections.

use crate::{
    checks::{assert_length, assert_scratch},
    curve::{
        AffinePoint, CurveError, CurveTableEntry, CurveTableRequirements, PastaCurve,
        ProjectivePoint,
        pasta::{batch::normalize, checked_count},
    },
    exec::{Executor, TaskBudget, for_each_chunk_mut},
    field::PastaField,
};
use core::{marker::PhantomData, ops::Range};

/// Description of an α-only prepared MSM bank, independent of its entry type.
///
/// Here `ω` is the curve's GLV endomorphism and `α = 1 − ω`. Larger widths
/// retain more point layers to reduce online bucket work. Use
/// [`Self::requirements`] to compare retained table and preparation workspace
/// costs for the selected [`CurveTableEntry`] representation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AlphaDescription {
    width: u8,
}
impl AlphaDescription {
    /// Selects width five, six, or seven.
    ///
    /// Returns [`CurveError::InvalidWindowBits`] for other widths.
    pub const fn new(width: u8) -> Result<Self, CurveError> {
        if width < 5 || width > 7 {
            return Err(CurveError::InvalidWindowBits { bits: width as u32 });
        }
        Ok(Self { width })
    }
    /// Bits consumed from each GLV component per main window.
    pub const fn window_bits(self) -> u8 {
        self.width
    }
    /// Number of prepared layers and integration buckets.
    pub const fn layers(self) -> usize {
        1 << (self.width - 1)
    }
    /// Number of four-byte residue entries.
    pub const fn codes(self) -> usize {
        1 << (2 * self.width)
    }
    /// Number of scratch `u16`s for codebook construction.
    pub const fn codebook_scratch(self) -> usize {
        let side = (1 << self.width) + 1;
        2 * self.codes() + side * side
    }
    /// Main windows; an additional window retains the exact signed residual.
    pub const fn main_windows(self) -> usize {
        127_usize.div_ceil(self.width as usize)
    }
    /// Counts for preparing a bank, including an empty bank.
    ///
    /// Returns [`CurveError::SizeOverflow`] if any required slice cannot be
    /// represented. Scratch is one layer, reused across layers; only table
    /// entries remain borrowed after preparation. Codebook storage is separate.
    pub const fn requirements<C: PastaCurve, E: CurveTableEntry<C>>(
        self,
        bases: usize,
    ) -> Result<CurveTableRequirements, CurveError> {
        Ok(CurveTableRequirements {
            table_entries: size!(checked_count::<E>(bases, self.layers())),
            projective_scratch: size!(checked_count::<ProjectivePoint<C>>(bases, 1)),
            field_scratch: size!(checked_count::<PastaField<C::Base>>(bases, 1)),
        })
    }
}

// Private cache representation shared by the codebook writer and kernels.
// Keeping these masks together lets the layout change without public promises.
#[derive(Clone, Copy)]
pub(super) struct Code(u32);
impl Code {
    const PRESENT: u32 = 1 << 15;
    const INDEX_MASK: u32 = 63;
    fn pack(bucket: usize, layer: usize, unit: usize, carries: [i8; 2]) -> Self {
        Self(
            Self::PRESENT
                | bucket as u32
                | ((layer as u32) << 6)
                | ((unit as u32) << 12)
                | ((carries[0] as u8 as u32) << 16)
                | ((carries[1] as u8 as u32) << 24),
        )
    }
    pub(super) fn from_bits(bits: u32) -> Self {
        Self(bits)
    }
    pub(super) fn present(self) -> bool {
        self.0 & Self::PRESENT != 0
    }
    pub(super) fn bucket(self) -> usize {
        (self.0 & Self::INDEX_MASK) as usize
    }
    pub(super) fn layer(self) -> usize {
        ((self.0 >> 6) & Self::INDEX_MASK) as usize
    }
    pub(super) fn rotation(self) -> usize {
        ((self.0 >> 13) & 3) as usize
    }
    pub(super) fn negative(self) -> bool {
        self.0 & (1 << 12) != 0
    }
    fn carries(self) -> [i8; 2] {
        [(self.0 >> 16) as u8 as i8, (self.0 >> 24) as u8 as i8]
    }
}

/// One coefficient's minimal binary unit-digit integration program.
///
/// Initialize with [`Self::ZERO`]. Trusted bindings must use the exact output
/// of [`AlphaCodebook::prepare`] for their description.
/// The POD representation supports owner-controlled Bento artifacts, not a
/// portable or version-independent recoding format. Owners must regenerate
/// recoding data when changing the implementation that consumes it.
#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, bento::Pod)]
pub struct AlphaCoefficient {
    a: u16,
    b: u16,
    // Each byte is zero or 1 + unit in [+1,-1,+ω,-ω,+ω²,-ω²].
    pub(super) digits: [u8; 8],
}
impl AlphaCoefficient {
    /// Initializer overwritten by preparation.
    pub const ZERO: Self = Self {
        a: 0,
        b: 0,
        digits: [0; 8],
    };
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
struct Eis {
    a: i32,
    b: i32,
}
impl Eis {
    const ZERO: Self = Self { a: 0, b: 0 };
    fn mul(self, rhs: Self) -> Self {
        Self {
            a: self.a * rhs.a - self.b * rhs.b,
            b: self.a * rhs.b + self.b * rhs.a - self.b * rhs.b,
        }
    }
    fn unit(self, unit: usize) -> Self {
        let rotated = match unit / 2 {
            0 => self,
            1 => Self {
                a: -self.b,
                b: self.a - self.b,
            },
            2 => Self {
                a: self.b - self.a,
                b: -self.a,
            },
            _ => unreachable!(),
        };
        if unit & 1 == 0 {
            rotated
        } else {
            Self {
                a: -rotated.a,
                b: -rotated.b,
            }
        }
    }
    fn norm(self) -> i32 {
        self.a.abs().max(self.b.abs())
    }
    fn euclid(self) -> i32 {
        self.a * self.a - self.a * self.b + self.b * self.b
    }
    fn residue(self, width: u8) -> usize {
        let mask = (1 << width) - 1;
        (((self.a & mask) << width) | (self.b & mask)) as usize
    }
    fn from_residue(index: usize, width: u8) -> Self {
        Self {
            a: (index >> width) as i32,
            b: (index & ((1 << width) - 1)) as i32,
        }
    }
}
// Interleaving odd multiples of P and αP makes layer zero P and lets a
// wNAF consumer borrow every second layer without another point bank.
fn variant(layer: usize) -> Eis {
    let n = 2 * (layer / 2) as i32 + 1;
    Eis {
        a: n,
        b: if layer & 1 == 0 { 0 } else { -n },
    }
}

/// Borrowed deterministic recoding data, shared by any number of point banks.
///
/// The data depends on the description, not the curve or point-entry type.
/// Storage belongs to the caller; this handle neither allocates nor owns it.
/// Generated data can be embedded with Bento under an artifact owner's schema.
/// The recoding representation is internal to this implementation and is not a
/// stable wire format. All recoding and table arithmetic is variable-time.
#[derive(Clone, Copy, Debug)]
pub struct AlphaCodebook<'a> {
    description: AlphaDescription,
    codes: &'a [u32],
    coefficients: &'a [AlphaCoefficient],
}
impl<'a> AlphaCodebook<'a> {
    /// Builds a codebook without allocation.
    ///
    /// `codes` and `coefficients` have exactly [`AlphaDescription::codes`] and
    /// [`AlphaDescription::layers`] entries; scratch has at least
    /// [`AlphaDescription::codebook_scratch`] entries. Length errors panic before
    /// any writes. Only the required scratch prefix is touched, and scratch is
    /// not retained. Codebook construction is intended for reuse across MSMs.
    pub fn prepare(
        description: AlphaDescription,
        codes: &'a mut [u32],
        coefficients: &'a mut [AlphaCoefficient],
        scratch: &mut [u16],
    ) -> Self {
        assert_length("codes", description.codes(), codes.len());
        assert_length("coefficients", description.layers(), coefficients.len());
        assert_scratch("codebook", description.codebook_scratch(), scratch.len());
        let width = description.width;
        let radix = 1_i32 << width;
        let radius = radix / 2;
        let side = (radix + 1) as usize;
        let (classes, rest) = scratch.split_at_mut(description.codes());
        let (best, weights) = rest.split_at_mut(description.codes());
        let weights = &mut weights[..side * side];
        classes.fill(u16::MAX);
        let mut count = 0;
        // The 6 * layers units of the α subgroup enumerate each orbit. Even
        // residues can have stabilizers; repeated assignments retain the class.
        for index in 1..classes.len() {
            if classes[index] != u16::MAX {
                continue;
            }
            let value = Eis::from_residue(index, width);
            for layer in 0..description.layers() {
                for unit in 0..6 {
                    let j = value.mul(variant(layer).unit(unit)).residue(width);
                    assert!(classes[j] == u16::MAX || classes[j] == count);
                    classes[j] = count;
                }
            }
            count += 1;
        }
        assert_eq!(usize::from(count), description.layers());
        let index = |v: Eis| (v.a + radius) as usize * side + (v.b + radius) as usize;
        weights.fill(u16::MAX);
        weights[index(Eis::ZERO)] = 0;
        // Bounded relaxation of zero-cost doubling and unit-cost predecessors.
        // This is the same shortest-path problem as 0/1 BFS, without a queue
        // whose transient capacity would become part of the caller contract.
        loop {
            let mut changed = false;
            for a in -radius..=radius {
                for b in -radius..=radius {
                    let value = Eis { a, b };
                    let weight = weights[index(value)];
                    if weight == u16::MAX {
                        continue;
                    }
                    for digit in 0..7 {
                        let unit = if digit == 0 {
                            Eis::ZERO
                        } else {
                            Eis { a: 1, b: 0 }.unit(digit - 1)
                        };
                        let pred = Eis {
                            a: 2 * a + unit.a,
                            b: 2 * b + unit.b,
                        };
                        if pred.norm() <= radius {
                            let candidate = weight + u16::from(digit != 0);
                            if candidate < weights[index(pred)] {
                                weights[index(pred)] = candidate;
                                changed = true;
                            }
                        }
                    }
                }
            }
            if !changed {
                break;
            }
        }
        coefficients.fill(AlphaCoefficient::ZERO);
        let key = |v: Eis| (weights[index(v)], v.norm(), v.euclid(), v.a, v.b);
        for a in -radius..=radius {
            for b in -radius..=radius {
                let value = Eis { a, b };
                if value == Eis::ZERO {
                    continue;
                }
                let coefficient = &mut coefficients[usize::from(classes[value.residue(width)])];
                let old = Eis {
                    a: i32::from(coefficient.a as i16),
                    b: i32::from(coefficient.b as i16),
                };
                if old == Eis::ZERO || key(value) < key(old) {
                    coefficient.a = a as i16 as u16;
                    coefficient.b = b as i16 as u16;
                }
            }
        }
        for coefficient in coefficients.iter_mut() {
            let mut value = Eis {
                a: i32::from(coefficient.a as i16),
                b: i32::from(coefficient.b as i16),
            };
            let mut position = 0;
            while value != Eis::ZERO {
                assert!(position < coefficient.digits.len());
                if (value.a | value.b) & 1 == 0 {
                    value = Eis {
                        a: value.a / 2,
                        b: value.b / 2,
                    };
                } else {
                    let target = weights[index(value)] - 1;
                    let (unit, next) = (0..6)
                        .find_map(|unit| {
                            let digit = Eis { a: 1, b: 0 }.unit(unit);
                            if ((value.a - digit.a) | (value.b - digit.b)) & 1 != 0 {
                                return None;
                            }
                            let next = Eis {
                                a: (value.a - digit.a) / 2,
                                b: (value.b - digit.b) / 2,
                            };
                            (weights[index(next)] == target).then_some((unit, next))
                        })
                        .expect("shortest unit-digit path");
                    coefficient.digits[position] = 1 + unit as u8;
                    value = next;
                }
                position += 1;
            }
        }
        codes.fill(0);
        best.fill(u16::MAX);
        for layer in 0..description.layers() {
            for unit in 0..6 {
                for (bucket, coefficient) in coefficients.iter().enumerate() {
                    let digit = variant(layer).unit(unit).mul(Eis {
                        a: i32::from(coefficient.a as i16),
                        b: i32::from(coefficient.b as i16),
                    });
                    let index = digit.residue(width);
                    assert_ne!(index, 0);
                    if digit.norm() < i32::from(best[index]) {
                        best[index] = digit.norm() as u16;
                        let a = (digit.a.rem_euclid(radix) - digit.a) >> width;
                        let b = (digit.b.rem_euclid(radix) - digit.b) >> width;
                        let a = i8::try_from(a).expect("α carry fits one byte");
                        let b = i8::try_from(b).expect("α carry fits one byte");
                        codes[index] = Code::pack(bucket, layer, unit, [a, b]).0;
                    }
                }
            }
        }
        assert!(codes[1..].iter().all(|&code| code != 0));
        Self::bind(description, codes, coefficients)
    }
    /// Borrows trusted output of [`Self::prepare`] for this description.
    ///
    /// Panics unless both slice lengths exactly match the description. This
    /// does not validate mathematical contents: the owner must guarantee the
    /// generated residue and integration data. Invalid data can panic during
    /// arithmetic or produce incorrect results. Binding writes no storage.
    pub fn bind(
        description: AlphaDescription,
        codes: &'a [u32],
        coefficients: &'a [AlphaCoefficient],
    ) -> Self {
        assert_length("codes", description.codes(), codes.len());
        assert_length("coefficients", description.layers(), coefficients.len());
        Self {
            description,
            codes,
            coefficients,
        }
    }
    /// Returns the selected description.
    pub const fn description(self) -> AlphaDescription {
        self.description
    }
    pub(super) fn coefficients(self) -> &'a [AlphaCoefficient] {
        self.coefficients
    }
    pub(super) fn write(self, halves: [i128; 2], row: &mut [u8]) {
        let mut pair = halves;
        let width = self.description.width;
        let mask = (1_i128 << width) - 1;
        for window in 0..self.description.main_windows() {
            let index = (((pair[0] & mask) << width) | (pair[1] & mask)) as usize;
            let code = self.codes[index];
            row[window * 4..window * 4 + 4].copy_from_slice(&code.to_le_bytes());
            for (value, carry) in pair.iter_mut().zip(Code(code).carries()) {
                *value = (*value >> width) + i128::from(carry);
            }
        }
        let tail = self.description.main_windows() * 4;
        for (half, value) in pair.into_iter().enumerate() {
            let value = i16::try_from(value).expect("bounded α recoding residual");
            row[tail + half * 2..tail + half * 2 + 2].copy_from_slice(&value.to_le_bytes());
        }
    }
}

/// Borrowed layer-major point table for α recoding.
///
/// Affine storage is suitable for embedding; [`crate::curve::PreparedAffinePoint`]
/// entries trade more retained storage for cheaper endomorphism rotations.
/// For base `P`, layers `2j` and `2j+1` contain `(2j+1)P` and
/// `(2j+1)(P−ωP)`, respectively. Every entry is nonidentity and must satisfy
/// its entry type's mathematical invariants.
///
/// [`super::Bases`] lets execution use the expanded layers or their original
/// bases as resources permit. Kernel selection is private and can change.
/// Tables and codebooks are retained caller-owned storage, separate from MSM
/// workspace ceilings. Preparation and execution allocate nothing and are
/// variable-time.
///
/// Prepare one bank and cache a full scalar row for its resolved plan:
///
/// ```
/// use zakura_udon::{
///     curve::{AffinePoint, Pallas, PastaCurve, ProjectivePoint},
///     exec::{ExecutionOptions, SerialExecutor, TaskBudget},
///     field::PastaField,
///     msm::{*, execution::MsmPlan},
/// };
/// let description = AlphaDescription::new(5)?;
/// let mut codes = vec![0; description.codes()];
/// let mut coefficients = vec![AlphaCoefficient::ZERO; description.layers()];
/// let mut work = vec![0; description.codebook_scratch()];
/// let book = AlphaCodebook::prepare(description, &mut codes, &mut coefficients, &mut work);
/// let g = AffinePoint::<Pallas>::GENERATOR;
/// let r = description.requirements::<Pallas, AffinePoint<Pallas>>(1)?;
/// let mut entries = vec![g; r.table_entries];
/// let mut projective = vec![ProjectivePoint::IDENTITY; r.projective_scratch];
/// let mut field = vec![PastaField::ZERO; r.field_scratch];
/// let table = AlphaTable::prepare(book, &[g], &mut entries, &mut projective,
///     &mut field, TaskBudget::SERIAL, &SerialExecutor);
/// assert_eq!(table.odd_multiples(3)?.len(), 1);
/// let values = [PastaField::<<Pallas as PastaCurve>::Scalar>::from_u64(7).invert().unwrap()];
/// let mut records = [ScalarStorage::ZERO];
/// let prepared = PreparedScalars::prepare(&values, &mut records,
///     TaskBudget::SERIAL, &SerialExecutor);
/// let input = Input::new_prepared(Bases::Alpha(table), prepared);
/// let options = ExecutionOptions::DEFAULT;
/// let plan = MsmPlan::for_input(&input, options)?;
/// let bytes = prepared.alpha_cache_len(&plan);
/// assert!(bytes > 0);
/// let mut digits = vec![0; bytes];
/// let cached = prepared.cache_alpha(&plan, book, &mut digits,
///     TaskBudget::SERIAL, &SerialExecutor);
/// let input = Input::new_prepared(Bases::Alpha(table), cached);
/// assert_eq!(input.requirements(options)?.digits(), 0);
/// assert_eq!(cached.retained_bytes(), prepared.retained_bytes() + bytes);
/// # Ok::<(), zakura_udon::curve::CurveError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct AlphaTable<'a, C: PastaCurve, E: CurveTableEntry<C> = AffinePoint<C>> {
    codebook: AlphaCodebook<'a>,
    entries: &'a [E],
    stride: usize,
    start: usize,
    len: usize,
    marker: PhantomData<C>,
}
impl<'a, C: PastaCurve, E: CurveTableEntry<C>> AlphaTable<'a, C, E> {
    /// Prepares all layers using the supplied execution budget.
    ///
    /// Buffers follow [`AlphaDescription::requirements`]: entries have the exact
    /// length and scratch has at least the indicated counts. Unrepresentable
    /// sizes or incorrect lengths panic before writing or invoking the executor.
    /// Source bases and the codebook must satisfy their mathematical invariants.
    /// Surplus scratch tails remain untouched; scratch and source bases are not
    /// retained. The result borrows only entries and the codebook.
    ///
    /// Each layer uses at most one inversion. An executor panic may leave
    /// partial entries and scratch; storage can be reused after all scoped jobs
    /// finish unwinding. There is no whole-call rollback.
    pub fn prepare<B: CurveTableEntry<C>, X: Executor>(
        codebook: AlphaCodebook<'a>,
        bases: &[B],
        entries: &'a mut [E],
        projective: &mut [ProjectivePoint<C>],
        field: &mut [PastaField<C::Base>],
        budget: TaskBudget,
        executor: &X,
    ) -> Self {
        let r = codebook
            .description
            .requirements::<C, E>(bases.len())
            .expect("table size");
        assert_length("entries", r.table_entries, entries.len());
        assert_scratch("projective", r.projective_scratch, projective.len());
        assert_scratch("field", r.field_scratch, field.len());
        if bases.is_empty() {
            return Self::bind(codebook, entries);
        }
        let projective = &mut projective[..bases.len()];
        for layer in 0..codebook.description.layers() {
            let (previous, next) = entries.split_at_mut(layer * bases.len());
            for_each_chunk_mut(projective, 64, budget, executor, |chunk, output, _| {
                for (i, point) in output.iter_mut().enumerate() {
                    let index = chunk * 64 + i;
                    let p = bases[index].affine();
                    let p = if layer & 1 == 0 {
                        p.to_projective()
                    } else {
                        p.to_projective().add_mixed(&p.endomorphism().neg())
                    };
                    *point = if layer < 2 {
                        p
                    } else {
                        previous[(layer - 2) * bases.len() + index]
                            .affine()
                            .to_projective()
                            .add(&p.double())
                    };
                }
            });
            normalize(projective, field, |i, p| {
                next[i] = E::from_affine(p.as_affine().expect("nonidentity α representative"));
            });
        }
        Self::bind(codebook, entries)
    }
    /// Borrows trusted output of [`Self::prepare`], with complete layers.
    ///
    /// Panics unless the entry count is a multiple of the layer count, including
    /// zero. Checks dimensions only: the codebook and every entry must satisfy
    /// their mathematical invariants and the layer relations in [`Self`]. Invalid
    /// data can panic during arithmetic or produce incorrect results. No storage
    /// is written or allocated.
    pub const fn bind(codebook: AlphaCodebook<'a>, entries: &'a [E]) -> Self {
        let layers = codebook.description.layers();
        assert!(entries.len().is_multiple_of(layers), "incomplete α table");
        let len = entries.len() / layers;
        Self {
            codebook,
            entries,
            stride: len,
            start: 0,
            len,
            marker: PhantomData,
        }
    }
    /// Number of available bases.
    pub const fn len(self) -> usize {
        self.len
    }
    /// Whether there are no bases.
    pub const fn is_empty(self) -> bool {
        self.len == 0
    }
    /// Returns the shared codebook.
    pub const fn codebook(self) -> AlphaCodebook<'a> {
        self.codebook
    }
    /// Selects consecutive bases while preserving the original layer stride.
    /// Panics for reversed or out-of-range bounds.
    pub fn range(self, range: Range<usize>) -> Self {
        assert!(range.start <= range.end && range.end <= self.len);
        Self {
            start: self.start + range.start,
            len: range.end - range.start,
            ..self
        }
    }
    /// Borrows odd layers without copying points or allocating.
    ///
    /// Returns [`CurveError::InvalidWindowBits`] unless the width is in
    /// `2..=codebook().description().window_bits()`, or [`CurveError::SizeOverflow`]
    /// if the stride cannot be represented. The view retains this range's bases
    /// and physical layer stride. Use it through [`super::Bases::Odd`] or
    /// [`super::Bases::OddPrepared`] with [`super::SharedScalarInput`].
    pub fn odd_multiples(self, width: u8) -> Result<super::OddTable<'a, C, E>, CurveError> {
        if width < 2 || width > self.codebook.description.width {
            return Err(CurveError::InvalidWindowBits {
                bits: u32::from(width),
            });
        }
        super::OddTable::bind(
            width,
            &self.entries[self.start..],
            self.len,
            self.stride.checked_mul(2).ok_or(CurveError::SizeOverflow)?,
        )
    }
    pub(super) fn entry(self, layer: usize, base: usize) -> E {
        assert!(base < self.len && layer < self.codebook.description.layers());
        self.entries[layer * self.stride + self.start + base]
    }
    pub(super) fn originals(self) -> &'a [E] {
        &self.entries[self.start..self.start + self.len]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use num_bigint::BigInt;
    use std::{vec, vec::Vec};

    #[test]
    fn all_residues_carries_and_signed_tails_reconstruct_integers() {
        for width in 5..=7 {
            let d = AlphaDescription::new(width).unwrap();
            let mut codes = vec![0; d.codes()];
            let mut coefficients = vec![AlphaCoefficient::ZERO; d.layers()];
            let book = AlphaCodebook::prepare(
                d,
                &mut codes,
                &mut coefficients,
                &mut vec![0; d.codebook_scratch()],
            );
            let coefficient = |code: u32| {
                let c = book.coefficients[(code & 63) as usize];
                // Independently reconstruct the binary unit program.
                let mut a = 0;
                let mut b = 0;
                for (i, digit) in c.digits.into_iter().enumerate() {
                    let (x, y) = match digit {
                        0 => (0, 0),
                        1 => (1, 0),
                        2 => (-1, 0),
                        3 => (0, 1),
                        4 => (0, -1),
                        5 => (-1, -1),
                        6 => (1, 1),
                        _ => panic!(),
                    };
                    a += x << i;
                    b += y << i;
                }
                assert_eq!((a, b), (c.a as i16 as i32, c.b as i16 as i32));
                // Multiply by the independently decoded layer and unit.
                let layer = (code >> 6) & 63;
                let n = (2 * (layer / 2) + 1) as i32;
                let (mut a, mut b) = if layer & 1 == 0 {
                    (a * n, b * n)
                } else {
                    ((a + b) * n, (2 * b - a) * n)
                };
                for _ in 0..((code >> 13) & 3) {
                    (a, b) = (-b, a - b);
                }
                if code & 4096 != 0 {
                    a = -a;
                    b = -b;
                }
                (a, b)
            };
            assert_eq!(book.codes[0], 0);
            for (i, &code) in book.codes.iter().enumerate().skip(1) {
                assert_ne!(code & 0x8000, 0);
                let (a, b) = coefficient(code);
                let ca = (code >> 16) as u8 as i8 as i32;
                let cb = (code >> 24) as u8 as i8 as i32;
                assert_eq!(a + (ca << width), (i >> width) as i32);
                assert_eq!(b + (cb << width), (i & ((1 << width) - 1)) as i32);
            }
            let extremes = [
                i128::MIN,
                i128::MIN + 1,
                -(1 << 126),
                -1,
                0,
                1,
                (1 << 126) - 1,
                i128::MAX,
            ];
            for a in extremes {
                for b in extremes {
                    let mut row = vec![0; (d.main_windows() + 1) * 4];
                    book.write([a, b], &mut row);
                    let words: Vec<_> = row
                        .chunks_exact(4)
                        .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                        .collect();
                    let tail = words[d.main_windows()];
                    let mut x = BigInt::from(tail as u16 as i16);
                    let mut y = BigInt::from((tail >> 16) as u16 as i16);
                    for &code in words[..d.main_windows()].iter().rev() {
                        let (a, b) = if code == 0 { (0, 0) } else { coefficient(code) };
                        x = (x << width) + a;
                        y = (y << width) + b;
                    }
                    assert_eq!((x, y), (BigInt::from(a), BigInt::from(b)));
                }
            }
        }
    }
}
