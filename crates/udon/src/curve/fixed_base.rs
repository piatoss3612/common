//! Expanded signed-window tables, borrowed from caller-owned storage.

use super::{
    AffinePoint, CurveError, PastaCurve, ProjectivePoint, batch, check_length, check_scratch,
    is_reduced,
};
use crate::field::PastaField;

/// The layout of an expanded fixed-base multiplication table.
///
/// [`Default`] selects width 4. For width `w` in `2..=8`, let
/// `n = ceil(255 / w)` and `h = 2^(w - 1)`, where `ceil` rounds up. Write
/// `[k] base` for integer multiplication of a nonidentity point `base` by `k`.
/// Entry `window * h + (m - 1)` holds `[m * 2^(w * window)] base` for
/// `window` in `0..n` and `m` in `1..=h`. The final entry at `n * h` is
/// `[2^(w * n)] base`, used for the last signed-digit carry.
///
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

/// Exact table length and minimum scratch lengths for fixed-base preparation.
///
/// All lengths count elements, not bytes. Multiplication and binding need no
/// scratch. Scratch tails beyond these lengths are left untouched by preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FixedBaseRequirements {
    /// Number of nonidentity affine entries in the expanded table.
    pub affine_points: usize,
    /// Minimum number of projective scratch elements.
    pub projective_scratch: usize,
    /// Minimum number of base-field scratch elements.
    pub field_scratch: usize,
}

impl FixedBaseDescription {
    /// Computes storage requirements for this table layout.
    ///
    /// Returns [`CurveError::InvalidWindowBits`] for widths outside `2..=8`.
    /// Width 4 needs 513 affine entries and 8 elements of each scratch type;
    /// width 8 needs 4097 entries and 128 elements of each scratch type.
    pub const fn requirements(self) -> Result<FixedBaseRequirements, CurveError> {
        if self.window_bits < 2 || self.window_bits > 8 {
            return Err(CurveError::InvalidWindowBits {
                bits: self.window_bits,
            });
        }
        let h = 1 << (self.window_bits - 1);
        let n = 255_usize.div_ceil(self.window_bits as usize);
        Ok(FixedBaseRequirements {
            affine_points: n * h + 1,
            projective_scratch: h,
            field_scratch: h,
        })
    }
}

/// A borrowed expanded table for repeated multiplication of one nonidentity base.
///
/// [`prepare`](Self::prepare) fills caller buffers. [`bind`](Self::bind) checks
/// stored entries; [`bind_trusted`](Self::bind_trusted) skips entry validation
/// when the caller has already established their mathematical validity.
/// All constructors check the base, description, and exact table length.
/// Multiplication is variable-time, performs no doublings, and uses no scratch
/// or allocation. Table preparation, validation, and multiplication provide no
/// constant-time guarantee for secret inputs.
#[derive(Clone, Copy)]
pub struct FixedBaseTable<'a, C: PastaCurve> {
    description: FixedBaseDescription,
    base: AffinePoint<C>,
    entries: &'a [AffinePoint<C>],
}

impl<C: PastaCurve> core::fmt::Debug for FixedBaseTable<'_, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("FixedBaseTable")
            .field("description", &self.description)
            .field("base", &self.base)
            .field("entries", &self.entries)
            .finish()
    }
}

impl<'a, C: PastaCurve> FixedBaseTable<'a, C> {
    /// Fills an expanded table and returns a borrowed view of it.
    ///
    /// `entries` must have exactly [`FixedBaseRequirements::affine_points`]
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
    ///         FixedBaseDescription, FixedBaseRequirements, PallasAffine,
    ///         PallasFixedBase, PallasProjective,
    ///     },
    ///     field::{Fp, Fq},
    /// };
    /// const DESCRIPTION: FixedBaseDescription = FixedBaseDescription { window_bits: 4 };
    /// const REQUIRED: FixedBaseRequirements = match DESCRIPTION.requirements() {
    ///     Ok(required) => required,
    ///     Err(_) => panic!("invalid table description"),
    /// };
    /// let base = PallasAffine::GENERATOR;
    /// let mut entries = [base; REQUIRED.affine_points];
    /// let mut projective = [PallasProjective::IDENTITY; REQUIRED.projective_scratch];
    /// let mut field = [Fp::ZERO; REQUIRED.field_scratch];
    /// let table = PallasFixedBase::prepare(
    ///     DESCRIPTION, &base, &mut entries, &mut projective, &mut field,
    /// ).unwrap();
    /// let scalar = Fq::from_u64(42);
    /// assert_eq!(table.mul(&scalar), base.mul_projective(&scalar));
    /// ```
    pub fn prepare(
        description: FixedBaseDescription,
        base: &AffinePoint<C>,
        entries: &'a mut [AffinePoint<C>],
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
        for window in entries[..requirements.affine_points - 1].chunks_exact_mut(h) {
            let mut multiple = window_base;
            for point in projective_scratch.iter_mut() {
                *point = multiple;
                multiple = multiple.add(&window_base);
            }
            batch::normalize(projective_scratch, field_scratch, |index, point| {
                // A nonzero base in a prime-order group stays nonzero for these
                // small multiples and powers of two.
                window[index] = *point
                    .as_affine()
                    .expect("fixed-base entries are nonidentity");
            });
            for _ in 0..description.window_bits {
                window_base = window_base.double();
            }
        }
        entries[requirements.affine_points - 1] = *window_base
            .to_point()
            .as_affine()
            .expect("a power of two times a nonidentity base is nonidentity");
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
    /// [`CurveError::InvalidTable`] if an entry has unreduced coordinates or
    /// differs from its specified multiple. Validation uses no scratch,
    /// allocation, or inversion.
    pub fn bind(
        description: FixedBaseDescription,
        base: &AffinePoint<C>,
        entries: &'a [AffinePoint<C>],
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
        entries: &'a [AffinePoint<C>],
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
    /// Returns [`CurveError::InvalidTable`] for unreduced or incorrect entries.
    /// This also checks views created by [`Self::bind_trusted`] and requires
    /// neither scratch nor allocation.
    pub fn validate(&self) -> Result<(), CurveError> {
        let h = self.description.requirements()?.projective_scratch;
        let mut window_base = self.base.to_projective();
        for window in self.entries[..self.entries.len() - 1].chunks_exact(h) {
            let mut expected = window_base;
            for entry in window {
                check_entry(&expected, entry)?;
                expected = expected.add(&window_base);
            }
            for _ in 0..self.description.window_bits {
                window_base = window_base.double();
            }
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
    pub const fn as_slice(&self) -> &'a [AffinePoint<C>] {
        self.entries
    }

    /// Multiplies the base by a scalar using signed digits and mixed additions.
    ///
    /// Processes the full canonical scalar; zero returns identity. The scalar
    /// must satisfy [`PastaField`]'s reduced-residue invariant, and a table
    /// created with [`Self::bind_trusted`] must satisfy its entry requirements.
    /// Execution is variable-time and requires no doubling, allocation, or
    /// scratch.
    pub fn mul(&self, scalar: &PastaField<C::Scalar>) -> ProjectivePoint<C> {
        let scalar = scalar.to_canonical_uint();
        let w = self.description.window_bits as usize;
        let h = 1 << (w - 1);
        let n = 255_usize.div_ceil(w);
        let mut carry = 0;
        let mut result = ProjectivePoint::IDENTITY;
        for window in 0..n {
            let offset = window * w;
            // The last window is zero-padded; asking for all w bits can run
            // past CanonicalUint's 256-bit boundary for some supported widths.
            let value = scalar.window(offset, w.min(255 - offset)).unwrap() as usize + carry;
            // value = digit + 2^w * carry, with digit in [-h, h - 1]. The
            // positive table stores magnitudes 1..=h; negation handles -h too.
            carry = usize::from(value >= h);
            let digit = value as isize - ((carry << w) as isize);
            if digit != 0 {
                let entry = &self.entries[window * h + digit.unsigned_abs() - 1];
                result = result.add_mixed(&if digit < 0 { entry.neg() } else { *entry });
            }
        }
        if carry != 0 {
            result = result.add_mixed(&self.entries[n * h]);
        }
        result
    }
}

fn check_inputs<C: PastaCurve>(
    description: FixedBaseDescription,
    base: &AffinePoint<C>,
    length: usize,
) -> Result<FixedBaseRequirements, CurveError> {
    let requirements = description.requirements()?;
    check_length("entries", requirements.affine_points, length)?;
    if AffinePoint::<C>::from_xy(base.x, base.y).is_none() {
        return Err(CurveError::InvalidBase);
    }
    Ok(requirements)
}

fn check_entry<C: PastaCurve>(
    expected: &ProjectivePoint<C>,
    entry: &AffinePoint<C>,
) -> Result<(), CurveError> {
    // Check raw residues before any field operation on potentially stored data.
    // Equality with a valid projective multiple also establishes curve membership.
    if !is_reduced(&entry.x) || !is_reduced(&entry.y) || *expected != entry.to_projective() {
        return Err(CurveError::InvalidTable);
    }
    Ok(())
}
