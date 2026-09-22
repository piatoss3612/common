//! Expanded signed-window tables, borrowed from caller-owned storage.

use super::{
    AffinePoint, CurveError, CurveTableEntry, CurveTableRequirements, PastaCurve, ProjectivePoint,
    assert_scratch, batch, glv_decompose,
};
use crate::field::PastaField;

/// The layout of an expanded fixed-base multiplication table.
///
/// [`Default`] selects width 4. For width `w` in `2..=8`, let
/// `n = ceil(128 / w)` and `h = 2^(w - 1)`, where `ceil` rounds up. Write
/// `[k] base` for integer multiplication of a nonidentity point `base` by `k`.
/// Entry `window * h + (m - 1)` holds `[m * 2^(w * window)] base` for
/// `window` in `0..n` and `m` in `1..=h`. Only width 2 has a final entry at
/// `n * h`, holding `[2^(w * n)] base` for the last signed-digit carry.
/// Pasta's GLV bounds exclude this carry at widths 3 through 8.
///
/// The two halves from [`glv_decompose`] share this table; the second applies
/// [`AffinePoint::endomorphism`].
/// Multiplication uses signed digits in `[-h, h - 1]` and zero-pads the last
/// partial window. Storing each window's shifted multiples avoids doublings
/// during multiplication. Larger windows use more storage for fewer additions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedBaseDescription {
    /// Bits per signed window; supported widths are `2..=8`.
    pub window_bits: u32,
}

impl Default for FixedBaseDescription {
    fn default() -> Self {
        Self { window_bits: 4 }
    }
}

impl FixedBaseDescription {
    /// Computes the exact entry count and minimum preparation scratch lengths.
    ///
    /// Returns [`CurveError::InvalidWindowBits`] for widths outside `2..=8`.
    /// Width 4 needs 256 entries and 8 elements of each scratch type;
    /// width 8 needs 2048 entries and 128 elements of each scratch type.
    /// Larger scratch buffers let [`FixedBaseTable::prepare_with`] share
    /// inversions across windows without changing the layout.
    pub const fn requirements(self) -> Result<CurveTableRequirements, CurveError> {
        if self.window_bits < 2 || self.window_bits > 8 {
            return Err(CurveError::InvalidWindowBits {
                bits: self.window_bits,
            });
        }
        let h = 1 << (self.window_bits - 1);
        let n = 128_usize.div_ceil(self.window_bits as usize);
        Ok(CurveTableRequirements {
            table_entries: n * h + (self.window_bits == 2) as usize,
            projective_scratch: h,
            field_scratch: h,
        })
    }
}

/// A borrowed expanded table for repeated multiplication of one nonidentity base.
///
/// The default entry type is [`AffinePoint`]. Select
/// [`PreparedAffinePoint`](super::PreparedAffinePoint) to cache endomorphism
/// coordinates, using 96 bytes per entry instead of 64.
///
/// [`prepare_with`](Self::prepare_with) fills caller buffers for an explicit
/// description; [`prepare`](Self::prepare) selects a description automatically.
/// [`bind`](Self::bind) borrows trusted stored entries and checks their
/// description and length. Multiplication is variable-time, performs no
/// doublings, and uses no caller scratch or allocation. Preparation and
/// multiplication provide no constant-time guarantee for secret inputs.
#[derive(Clone, Copy)]
pub struct FixedBaseTable<'a, C: PastaCurve, E: CurveTableEntry<C> = AffinePoint<C>> {
    description: FixedBaseDescription,
    base: AffinePoint<C>,
    entries: &'a [E],
}

impl<C: PastaCurve, E: CurveTableEntry<C>> core::fmt::Debug for FixedBaseTable<'_, C, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FixedBaseTable")
            .field("description", &self.description)
            .field("base", &self.base)
            .field("entries", &self.entries)
            .finish()
    }
}

impl<'a, C: PastaCurve, E: CurveTableEntry<C>> FixedBaseTable<'a, C, E> {
    /// Prepares an expanded table within the supplied buffer capacities.
    ///
    /// Udon selects the window from retained entry capacity and both scratch
    /// capacities. At least 129 entries and two elements of each scratch type
    /// are required. The returned view borrows only the used entry prefix and
    /// reports its format through [`Self::description`]. Preparation then uses
    /// the scratch allowance as described by [`Self::prepare_with`]. Unused
    /// buffer tails are untouched, and scratch can be reused immediately.
    /// Use [`Self::prepare_with`] to select the layout independently of scratch.
    ///
    /// Insufficient capacity panics before writes. Preparation is variable-time
    /// and performs no allocation.
    pub fn prepare(
        base: &AffinePoint<C>,
        entries: &'a mut [E],
        projective_scratch: &mut [ProjectivePoint<C>],
        field_scratch: &mut [PastaField<C::Base>],
    ) -> Result<Self, CurveError> {
        let mut description = FixedBaseDescription { window_bits: 2 };
        for window_bits in 3..=8 {
            let candidate = FixedBaseDescription { window_bits };
            let required = candidate.requirements()?;
            if entries.len() < required.table_entries
                || projective_scratch.len() < required.projective_scratch
                || field_scratch.len() < required.field_scratch
            {
                break;
            }
            description = candidate;
        }
        Self::prepare_with(
            description,
            base,
            entries,
            projective_scratch,
            field_scratch,
        )
    }

    /// Prepares the specified layout within the supplied scratch allowance.
    ///
    /// Uses the entry count from [`FixedBaseDescription::requirements`], leaving
    /// any remaining entry tail untouched. The description is independent of
    /// all buffer capacities, and the returned table borrows only its used
    /// entries. Scratch can be reused immediately.
    ///
    /// The shorter scratch buffer bounds each normalization batch. Preparation
    /// accumulates as many whole windows as fit, then normalizes them with one
    /// inversion. The width-2 carry joins the last batch if it fits, or uses a
    /// separate inversion. Minimum scratch holds one window; scratch of at least
    /// `table_entries` elements of each type normalizes the entire table with
    /// one inversion. Unused scratch tails are untouched.
    ///
    /// Returns [`CurveError::InvalidWindowBits`] for unsupported widths. Panics
    /// if entries or either scratch buffer are shorter than the reported
    /// requirements. These checks precede all writes. Preparation is
    /// variable-time and performs no allocation.
    ///
    /// ```
    /// use zakura_udon::{
    ///     curve::{FixedBaseDescription, FixedBaseTable, Pallas, PallasAffine,
    ///             PallasProjective},
    ///     field::{Fp, Fq},
    /// };
    ///
    /// let description = FixedBaseDescription { window_bits: 4 };
    /// let base = PallasAffine::GENERATOR;
    /// let mut entries = [base; 256];
    /// // Eight elements of each scratch type suffice; 256 permit one inversion.
    /// let mut projective = [PallasProjective::IDENTITY; 256];
    /// let mut field = [Fp::ZERO; 256];
    /// let table = FixedBaseTable::<Pallas>::prepare_with(
    ///     description, &base, &mut entries, &mut projective, &mut field,
    /// ).unwrap();
    /// let scalar = Fq::from_u64(42);
    /// assert_eq!(table.mul(&scalar), base.mul_projective(&scalar));
    /// let rebound = FixedBaseTable::bind(
    ///     description, &base, table.as_slice(),
    /// ).unwrap();
    /// assert_eq!(rebound.mul(&scalar), table.mul(&scalar));
    /// ```
    pub fn prepare_with(
        description: FixedBaseDescription,
        base: &AffinePoint<C>,
        entries: &'a mut [E],
        projective_scratch: &mut [ProjectivePoint<C>],
        field_scratch: &mut [PastaField<C::Base>],
    ) -> Result<Self, CurveError> {
        let requirements = description.requirements()?;
        assert_scratch("entries", requirements.table_entries, entries.len());
        assert_scratch(
            "projective",
            requirements.projective_scratch,
            projective_scratch.len(),
        );
        assert_scratch("field", requirements.field_scratch, field_scratch.len());
        let entries = &mut entries[..requirements.table_entries];
        let h = requirements.projective_scratch;
        let capacity = projective_scratch.len().min(field_scratch.len());
        let window_capacity = capacity / h * h;
        let mut window_base = base.to_projective();
        let mut offset = 0;
        while offset < entries.len() {
            let remaining = entries.len() - offset;
            // Keep windows intact, but include the carry in the final batch
            // whenever both scratch buffers have room for it.
            let count = if remaining <= capacity {
                remaining
            } else {
                window_capacity
            };
            let points = &mut projective_scratch[..count];
            for window in points.chunks_exact_mut(h) {
                window[0] = window_base;
                for i in 1..h {
                    // Index i holds (i + 1) times the base. Every even multiple
                    // has its half at i / 2, so it needs only a doubling.
                    window[i] = if i % 2 == 1 {
                        window[i / 2].double()
                    } else {
                        window[i - 1].add(&window_base)
                    };
                }
                // The last multiple is 2^(window_bits - 1) times this window's
                // base, so one doubling advances to the next window.
                window_base = window[h - 1].double();
            }
            if !count.is_multiple_of(h) {
                // Only width 2 has an entry beyond its complete windows.
                points[count - 1] = window_base;
            }
            batch::normalize(points, &mut field_scratch[..count], |index, point| {
                // A nonzero base in a prime-order group stays nonzero for these
                // small multiples and powers of two.
                entries[offset + index] = E::from_affine(
                    point
                        .as_affine()
                        .expect("fixed-base entries are nonidentity"),
                );
            });
            offset += count;
        }
        Ok(Self {
            description,
            base: *base,
            entries,
        })
    }

    /// Borrows trusted entries in [`FixedBaseDescription`]'s storage order.
    ///
    /// Entries must have been prepared for this description and `base`. Binding
    /// checks the description and storage length without inspecting entries.
    /// Returns [`CurveError::InvalidWindowBits`] for unsupported widths. Panics
    /// unless the entry count matches [`FixedBaseDescription::requirements`].
    pub const fn bind(
        description: FixedBaseDescription,
        base: &AffinePoint<C>,
        entries: &'a [E],
    ) -> Result<Self, CurveError> {
        let requirements = match description.requirements() {
            Ok(requirements) => requirements,
            Err(error) => return Err(error),
        };
        assert!(
            entries.len() == requirements.table_entries,
            "fixed-base table length mismatch"
        );
        Ok(Self {
            description,
            base: *base,
            entries,
        })
    }

    /// Returns the table layout.
    pub const fn description(&self) -> FixedBaseDescription {
        self.description
    }

    /// Borrows the nonidentity base supplied when constructing this table.
    pub const fn base(&self) -> &AffinePoint<C> {
        &self.base
    }

    /// Borrows entries in [`FixedBaseDescription`]'s storage order.
    pub const fn as_slice(&self) -> &'a [E] {
        self.entries
    }

    /// Multiplies the base by a scalar using signed digits and mixed additions.
    ///
    /// Processes the full canonical scalar; zero returns identity.
    /// Execution is variable-time and uses bounded stack storage without
    /// doubling, allocation, or caller scratch.
    pub fn mul(&self, scalar: &PastaField<C::Scalar>) -> ProjectivePoint<C> {
        let (a, b) = glv_decompose::<C>(scalar);
        let w = self.description.window_bits as usize;
        let h = 1 << (w - 1);
        let n = 128_usize.div_ceil(w);
        let mut result = ProjectivePoint::IDENTITY;
        for (rotation, half) in [a, b].into_iter().enumerate() {
            let (digits, carry) = signed_window_digits(half.unsigned_abs(), w);
            for (window, &digit) in digits[..n].iter().enumerate() {
                if digit != 0 {
                    let entry = self.entries[window * h + digit.unsigned_abs() as usize - 1]
                        .rotated(rotation);
                    let negative = (digit < 0) ^ (half < 0);
                    result = result.add_mixed(&if negative { entry.neg() } else { entry });
                }
            }
            if carry {
                let entry = self.entries[n * h].rotated(rotation);
                result = result.add_mixed(&if half < 0 { entry.neg() } else { entry });
            }
        }
        result
    }
}

/// Encodes a magnitude in signed width-`w` digits and a final carry.
///
/// Requires `w` in `2..=8`; digits are in `[-2^(w - 1), 2^(w - 1) - 1]`.
/// Only the first `ceil(128 / w)` digits are used; the remaining digits are zero.
pub(super) fn signed_window_digits(mut magnitude: u128, w: usize) -> ([i16; 64], bool) {
    // Zero-pad the last partial window. Pasta's lattice bounds are
    // |k1| < (a + b)/2 + 1 and |k2| < (b + d)/2 + 1, where d = a + b.
    // Only k2 at width 2 can carry, so only that layout stores a carry entry.
    let mut digits = [0; 64];
    let mut carry = 0;
    for digit in &mut digits[..128_usize.div_ceil(w)] {
        *digit = super::scalar::centered_digit(
            (magnitude & ((1 << w) - 1)) as u16,
            false,
            &mut carry,
            w as u32,
        );
        magnitude >>= w;
    }
    (digits, carry != 0)
}
