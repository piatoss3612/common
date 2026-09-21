//! Expanded signed-window tables, borrowed from caller-owned storage.

use super::{
    AffinePoint, CurveError, CurveTableEntry, CurveTableRequirements, PastaCurve, ProjectivePoint,
    batch, check_length, check_scratch, glv_decompose, table_entry::check_entry,
};
use crate::field::PastaField;

/// The layout of an expanded fixed-base multiplication table.
///
/// [`Default`] selects width 4. For width `w` in `2..=8`, let
/// `n = ceil(128 / w)` and `h = 2^(w - 1)`, where `ceil` rounds up. Write
/// `[k] base` for integer multiplication of a nonidentity point `base` by `k`.
/// Entry `window * h + (m - 1)` holds `[m * 2^(w * window)] base` for
/// `window` in `0..n` and `m` in `1..=h`. The final entry at `n * h` is
/// `[2^(w * n)] base`, used for the last signed-digit carry.
/// Pasta's GLV bounds allow this carry only at width 2; widths 3 through 8
/// retain and validate the final entry for a uniform layout.
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
    /// Computes storage requirements for this table layout.
    ///
    /// Returns [`CurveError::InvalidWindowBits`] for widths outside `2..=8`.
    /// Width 4 needs 257 entries and 8 elements of each scratch type;
    /// width 8 needs 2049 entries and 128 elements of each scratch type.
    pub const fn requirements(self) -> Result<CurveTableRequirements, CurveError> {
        if self.window_bits < 2 || self.window_bits > 8 {
            return Err(CurveError::InvalidWindowBits {
                bits: self.window_bits,
            });
        }
        let h = 1 << (self.window_bits - 1);
        let n = 128_usize.div_ceil(self.window_bits as usize);
        Ok(CurveTableRequirements {
            table_entries: n * h + 1,
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
/// [`prepare`](Self::prepare) fills caller buffers. [`bind`](Self::bind) checks
/// stored entries; [`bind_trusted`](Self::bind_trusted) skips entry validation
/// when the caller has already established their mathematical validity.
/// All constructors check the base, description, and exact table length.
/// Multiplication is variable-time, performs no doublings, and uses no caller
/// scratch or allocation. Table preparation, validation, and multiplication
/// provide no constant-time guarantee for secret inputs.
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
    /// Fills an expanded table and returns a borrowed view of it.
    ///
    /// `entries` must have exactly [`CurveTableRequirements::table_entries`]
    /// elements, as reported by [`FixedBaseDescription::requirements`]. Scratch
    /// slices must meet the reported minimum lengths. Initial buffer contents
    /// do not matter, and scratch tails beyond those lengths are untouched.
    /// The returned table borrows only `entries`; both scratch buffers can be
    /// reused immediately.
    ///
    /// Returns [`CurveError::InvalidWindowBits`] for an unsupported width,
    /// [`CurveError::InvalidBase`] for unreduced or off-curve base coordinates,
    /// [`CurveError::LengthMismatch`] for an incorrect table length, or
    /// [`CurveError::ScratchTooSmall`] for either short scratch buffer. Every
    /// error leaves all buffers unchanged.
    ///
    /// ```
    /// use zakura_udon::{
    ///     curve::{
    ///         FixedBaseDescription, CurveTableRequirements, PallasAffine,
    ///         FixedBaseTable, Pallas, PallasProjective,
    ///     },
    ///     field::{Fp, Fq},
    /// };
    /// const DESCRIPTION: FixedBaseDescription = FixedBaseDescription { window_bits: 4 };
    /// const REQUIRED: CurveTableRequirements = match DESCRIPTION.requirements() {
    ///     Ok(required) => required,
    ///     Err(_) => panic!("invalid table description"),
    /// };
    /// let base = PallasAffine::GENERATOR;
    /// let mut entries = [base; REQUIRED.table_entries];
    /// let mut projective = [PallasProjective::IDENTITY; REQUIRED.projective_scratch];
    /// let mut field = [Fp::ZERO; REQUIRED.field_scratch];
    /// let table = FixedBaseTable::<Pallas>::prepare(
    ///     DESCRIPTION, &base, &mut entries, &mut projective, &mut field,
    /// ).unwrap();
    /// let scalar = Fq::from_u64(42);
    /// assert_eq!(table.mul(&scalar), base.mul_projective(&scalar));
    /// ```
    pub fn prepare(
        description: FixedBaseDescription,
        base: &AffinePoint<C>,
        entries: &'a mut [E],
        projective_scratch: &mut [ProjectivePoint<C>],
        field_scratch: &mut [PastaField<C::Base>],
    ) -> Result<Self, CurveError> {
        let requirements = check_inputs(description, base, entries.len())?;
        check_scratch(
            "projective",
            requirements.projective_scratch,
            projective_scratch.len(),
        )?;
        check_scratch("field", requirements.field_scratch, field_scratch.len())?;
        let h = requirements.projective_scratch;
        let projective_scratch = &mut projective_scratch[..h];
        let field_scratch = &mut field_scratch[..h];
        // Reuse one window of projective multiples at a time. Normalizing each
        // window shares an inversion across h entries; the final carry needs
        // one more. This bounds scratch independently of the number of windows.
        let mut window_base = base.to_projective();
        for window in entries[..requirements.table_entries - 1].chunks_exact_mut(h) {
            projective_scratch[0] = window_base;
            for i in 1..h {
                projective_scratch[i] = projective_scratch[i - 1].add(&window_base);
            }
            batch::normalize(projective_scratch, field_scratch, |index, point| {
                // A nonzero base in a prime-order group stays nonzero for these
                // small multiples and powers of two.
                window[index] = E::from_affine(
                    point
                        .as_affine()
                        .expect("fixed-base entries are nonidentity"),
                );
            });
            // The last multiple is 2^(window_bits - 1) times this window's
            // base, so one doubling advances to the next window.
            window_base = projective_scratch[h - 1].double();
        }
        entries[requirements.table_entries - 1] = E::from_affine(
            window_base
                .to_point()
                .as_affine()
                .expect("a power of two times a nonidentity base is nonidentity"),
        );
        Ok(Self {
            description,
            base: *base,
            entries,
        })
    }

    /// Binds stored entries after checking every specified multiple of `base`.
    ///
    /// Uses [`FixedBaseDescription`]'s layout. Returns the description, base,
    /// and length errors from [`Self::bind_trusted`], or
    /// [`CurveError::InvalidTable`] if an entry has unreduced coordinates,
    /// differs from its specified multiple, or has an inconsistent cache.
    /// Validation uses no scratch, allocation, or inversion.
    pub fn bind(
        description: FixedBaseDescription,
        base: &AffinePoint<C>,
        entries: &'a [E],
    ) -> Result<Self, CurveError> {
        let table = Self::bind_trusted(description, base, entries)?;
        table.validate()?;
        Ok(table)
    }

    /// Binds a table whose entries have already been validated by its owner.
    ///
    /// Checks the description, base, and exact length, but does not inspect table
    /// entries. They must be the reduced, on-curve multiples in
    /// [`FixedBaseDescription`]'s layout, including the final carry entry.
    /// Cached coordinates must satisfy
    /// [`PreparedAffinePoint`](super::PreparedAffinePoint)'s invariants.
    /// Incorrect entries can make [`Self::mul`] panic or give incorrect results
    /// but remain memory-safe. Use [`Self::bind`] for unvalidated data, or call
    /// [`Self::validate`] before multiplication.
    ///
    /// Returns [`CurveError::InvalidWindowBits`] for an unsupported width,
    /// [`CurveError::InvalidBase`] for unreduced or off-curve base coordinates,
    /// or [`CurveError::LengthMismatch`] unless `entries` has exactly the length
    /// reported by [`FixedBaseDescription::requirements`].
    pub fn bind_trusted(
        description: FixedBaseDescription,
        base: &AffinePoint<C>,
        entries: &'a [E],
    ) -> Result<Self, CurveError> {
        check_inputs(description, base, entries.len())?;
        Ok(Self {
            description,
            base: *base,
            entries,
        })
    }

    /// Checks every entry against its specified multiple without inversion.
    ///
    /// Returns [`CurveError::InvalidTable`] for unreduced or incorrect entries,
    /// including inconsistent cached coordinates. This also checks views
    /// created by [`Self::bind_trusted`] and requires neither scratch nor
    /// allocation.
    pub fn validate(&self) -> Result<(), CurveError> {
        let h = self.description.requirements()?.projective_scratch;
        let mut window_base = self.base.to_projective();
        for window in self.entries[..self.entries.len() - 1].chunks_exact(h) {
            let mut expected = window_base;
            for (i, entry) in window.iter().enumerate() {
                check_entry(&expected, entry)?;
                if i + 1 < h {
                    expected = expected.add(&window_base);
                }
            }
            window_base = expected.double();
        }
        check_entry(&window_base, &self.entries[self.entries.len() - 1])
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
    /// Processes the full canonical scalar; zero returns identity. The scalar
    /// must satisfy [`PastaField`]'s reduced-residue invariant, and a table
    /// created with [`Self::bind_trusted`] must satisfy its entry requirements.
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

fn check_inputs<C: PastaCurve>(
    description: FixedBaseDescription,
    base: &AffinePoint<C>,
    length: usize,
) -> Result<CurveTableRequirements, CurveError> {
    let requirements = description.requirements()?;
    check_length("entries", requirements.table_entries, length)?;
    if AffinePoint::<C>::from_xy(base.x, base.y).is_none() {
        return Err(CurveError::InvalidBase);
    }
    Ok(requirements)
}

/// Encodes a magnitude in signed width-`w` digits and a final carry.
///
/// Requires `w` in `2..=8`; digits are in `[-2^(w - 1), 2^(w - 1) - 1]`.
/// Only the first `ceil(128 / w)` digits are used; the remaining digits are zero.
pub(super) fn signed_window_digits(mut magnitude: u128, w: usize) -> ([i16; 64], bool) {
    // Zero-pad the last partial window. Pasta's lattice bounds are
    // |k1| < (a + b)/2 + 1 and |k2| < (b + d)/2 + 1, where d = a + b.
    // Only k2 at width 2 can carry; other layouts keep the uniform final slot.
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
