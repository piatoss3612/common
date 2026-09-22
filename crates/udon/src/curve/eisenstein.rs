//! Joint width-three recoding in the Eisenstein integers.
//!
//! A pair `(a, b)` represents `a + b*lambda`, with `lambda² + lambda + 1 = 0`.
//! Multiplication by `lambda` rotates a pair to `(-b, a - b)`; signs and the
//! three rotations give six units. Evaluating `lambda` at the scalar field's
//! cube root of unity maps these integer pairs to scalars.

use super::{
    AffinePoint, CurveError, CurveTableEntry, CurveTableRequirements, PastaCurve,
    PreparedAffinePoint, ProjectivePoint, assert_scratch, batch, table_entry::check_entry,
};
use crate::field::{CanonicalUint, PastaField};
use core::marker::PhantomData;

pub(super) const MAX_DIGITS: usize = 132;

/// A scalar decomposed and recoded for joint Eisenstein multiplication.
///
/// Prepare once for [`EisensteinTable::mul_prepared`] or
/// [`EisensteinTableBatch::mul_prepared`](super::EisensteinTableBatch::mul_prepared)
/// when the same scalar acts on several bases. This fixed-size, allocation-free
/// value is specific to its curve and borrows neither the scalar nor a table.
/// Preparation and multiplication are variable-time.
#[derive(Clone, Copy, Debug)]
pub struct EisensteinScalar<C: PastaCurve> {
    digits: [u8; MAX_DIGITS],
    len: usize,
    batch_safe: Option<bool>,
    marker: PhantomData<C>,
}

impl<C: PastaCurve> EisensteinScalar<C> {
    /// Records joint doubling-ladder digits and eligibility for affine batch ladders.
    ///
    /// Eligibility depends only on the scalar and can be reused across any
    /// nonidentity bases on this curve. Scalars unsuitable for affine batch
    /// arithmetic remain valid for multiplication. Preparing this reusable
    /// value includes that eligibility check; individual table
    /// multiplication need not perform it. Preparation is variable-time and
    /// allocates no storage.
    ///
    /// The scalar must satisfy [`PastaField`]'s reduced-residue invariant.
    /// Violations remain memory-safe but can cause panics or incorrect results.
    pub fn new(scalar: &PastaField<C::Scalar>) -> Self {
        Self::for_single(scalar).certify_batch()
    }

    /// Recodes a canonical scalar integer into signed Eisenstein digits.
    ///
    /// The caller must establish that `scalar` is below `C::Scalar`'s modulus;
    /// [`CanonicalUint`] alone does not establish that bound.
    fn from_canonical(scalar: CanonicalUint) -> Self {
        let (a, b) = super::glv::decompose_canonical::<C>(scalar);
        let (digits, len) = recode(a, b);
        Self {
            digits,
            len,
            batch_safe: None,
            marker: PhantomData,
        }
    }

    pub(super) fn for_single(scalar: &PastaField<C::Scalar>) -> Self {
        Self::from_canonical(scalar.to_canonical_uint())
    }

    fn certify_batch(mut self) -> Self {
        self.batch_safe = Some(super::eisenstein_batch::ladder_safe::<C>(self.digits()));
        self
    }

    pub(super) fn batch_safe(&self) -> bool {
        self.batch_safe
            .unwrap_or_else(|| super::eisenstein_batch::ladder_safe::<C>(self.digits()))
    }

    pub(super) fn digits(&self) -> &[u8] {
        &self.digits[..self.len]
    }
}

/// Coefficients `a + b*lambda`, in retained table order.
pub(super) const REPRESENTATIVES: [(i8, i8); 8] = [
    (1, 0),
    (1, -1),
    (2, -1),
    (1, -2),
    (3, 0),
    (3, -1),
    (1, -3),
    (2, -3),
];

// All 48 signed unit rotations cover precisely the residue pairs modulo 8
// that are not both even. Subtracting the selected digit makes both residual
// coordinates divisible by 8, so the next two ladder digits are zero.
// Generating the selector keeps its codes tied to the retained table order.
const SELECTOR: [(i8, i8, u8); 64] = {
    let mut table = [(0, 0, 0); 64];
    let mut representative = 0;
    while representative < 8 {
        let (mut a, mut b) = REPRESENTATIVES[representative];
        let mut rotation = 0;
        while rotation < 3 {
            let mut negative = 0;
            while negative < 2 {
                let (a, b) = if negative == 0 { (a, b) } else { (-a, -b) };
                let index = (((a & 7) << 3) | (b & 7)) as usize;
                assert!(table[index].2 == 0);
                table[index] = (
                    a,
                    b,
                    (representative * 6 + rotation * 2 + negative + 1) as u8,
                );
                negative += 1;
            }
            (a, b) = (-b, a - b);
            rotation += 1;
        }
        representative += 1;
    }
    let mut index = 0;
    while index < 64 {
        assert!((table[index].2 == 0) == (index & 9 == 0));
        index += 1;
    }
    table
};

// Each digit coordinate has magnitude at most 5. Dividing by two after
// subtraction takes 127-bit inputs to magnitude <= 5 in 127 steps; the
// remaining small pairs terminate within five further steps.
pub(super) fn recode(mut a: i128, mut b: i128) -> ([u8; MAX_DIGITS], usize) {
    debug_assert!(a != i128::MIN && b != i128::MIN);
    let mut digits = [0; MAX_DIGITS];
    let mut len = 0;
    while a != 0 || b != 0 {
        let (digit_a, digit_b, code) = SELECTOR[(((a & 7) << 3) | (b & 7)) as usize];
        digits[len] = code;
        // Equal parity makes this exact, without overflowing on a - digit_a.
        a = (a >> 1) - (i128::from(digit_a) >> 1);
        b = (b >> 1) - (i128::from(digit_b) >> 1);
        len += 1;
    }
    (digits, len)
}

pub(super) fn representatives<C: PastaCurve>(base: &ProjectivePoint<C>) -> [ProjectivePoint<C>; 8] {
    // Write phi(P) for P.endomorphism(). For difference = base - phi(base),
    // difference - phi(difference) = [-3] phi(base). Two endomorphisms then
    // give [-3] base without doublings; sums with phi(base) and further
    // rotations produce REPRESENTATIVES in the required order.
    let phi = base.endomorphism();
    let difference = base.sub(&phi);
    let b = difference.sub(&difference.endomorphism());
    let b_phi = b.endomorphism();
    let minus_three = b_phi.endomorphism();
    let three_a = phi.add(&minus_three);
    let three_b = phi.sub(&minus_three);
    let four_a = phi.sub(&b_phi);
    let four_b = phi.add(&b_phi);
    let nineteen = phi.add(&four_b);
    [
        *base,
        difference,
        four_a.endomorphism(),
        three_b.endomorphism().neg(),
        minus_three.neg(),
        three_a.neg(),
        four_b.endomorphism().endomorphism(),
        nineteen.endomorphism().endomorphism(),
    ]
}

/// Eight borrowed representatives for repeated multiplication of one base.
///
/// Write `[a] P` for multiplication of point `P` by a signed integer `a`.
/// Entry order is `[a] base + [b] base.endomorphism()` for coefficient pairs
/// `(1,0), (1,-1), (2,-1), (1,-2), (3,0), (3,-1), (1,-3), (2,-3)`.
/// Each entry supplies six points by applying [`AffinePoint::endomorphism`]
/// zero, one, or two times, with either sign. These 48 points serve as digits
/// in a doubling ladder over the two halves from
/// [`glv_decompose`](super::glv_decompose).
///
/// The default [`AffinePoint`] entries occupy 512 bytes; choose
/// [`PreparedAffinePoint`] entries for 768 bytes and cheaper rotations.
/// These sizes exclude the base and table handle. Preparation, validation,
/// and multiplication are allocation-free and provide no constant-time
/// guarantee for secret bases, scalars, or table contents.
/// All constructors check the base and exact entry count.
#[derive(Clone, Copy)]
pub struct EisensteinTable<'a, C: PastaCurve, E: CurveTableEntry<C> = AffinePoint<C>> {
    pub(super) base: AffinePoint<C>,
    pub(super) entries: &'a [E; 8],
}

impl<C: PastaCurve, E: CurveTableEntry<C>> core::fmt::Debug for EisensteinTable<'_, C, E> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EisensteinTable")
            .field("base", &self.base)
            .field("entries", &self.entries)
            .finish()
    }
}

impl<'a, C: PastaCurve, E: CurveTableEntry<C>> EisensteinTable<'a, C, E> {
    /// Exact entry count and minimum preparation scratch lengths.
    pub const REQUIREMENTS: CurveTableRequirements = CurveTableRequirements {
        table_entries: 8,
        projective_scratch: 8,
        field_scratch: 8,
    };

    /// Prepares eight entries in caller-owned storage using one inversion.
    ///
    /// Both scratch buffers must have at least eight elements; shorter buffers panic
    /// before any writes. Initial contents do not matter. Scratch tails are untouched
    /// and scratch can be reused as soon as this returns.
    ///
    /// Returns [`CurveError::InvalidBase`] for an unreduced or off-curve base, leaving
    /// all buffers unchanged. The returned view borrows only `entries`.
    ///
    /// ```
    /// use zakura_udon::{
    ///     curve::{
    ///         EisensteinTable, Pallas, PallasAffine, PallasProjective,
    ///         PreparedAffinePoint,
    ///     },
    ///     field::{Fp, Fq},
    /// };
    /// let base = PallasAffine::GENERATOR;
    /// let mut entries = [PreparedAffinePoint::from_affine(&base); 8];
    /// let mut projective = [PallasProjective::IDENTITY; 8];
    /// let mut field = [Fp::ZERO; 8];
    /// let table = EisensteinTable::<Pallas, PreparedAffinePoint<Pallas>>::prepare(
    ///     &base, &mut entries, &mut projective, &mut field,
    /// ).unwrap();
    /// let scalar = Fq::from_u64(42);
    /// assert_eq!(table.mul(&scalar), base.mul_projective(&scalar));
    /// ```
    pub fn prepare(
        base: &AffinePoint<C>,
        entries: &'a mut [E; 8],
        projective_scratch: &mut [ProjectivePoint<C>],
        field_scratch: &mut [PastaField<C::Base>],
    ) -> Result<Self, CurveError> {
        check_base(base)?;
        assert_scratch("projective", 8, projective_scratch.len());
        assert_scratch("field", 8, field_scratch.len());
        projective_scratch[..8].copy_from_slice(&representatives(&base.to_projective()));
        normalize(&projective_scratch[..8], &mut field_scratch[..8], entries);
        Ok(Self {
            base: *base,
            entries,
        })
    }

    /// Binds stored entries after checking the specified multiples and caches.
    ///
    /// Returns the base error from [`Self::bind_trusted`], or
    /// [`CurveError::InvalidTable`] for unreduced, incorrect, or inconsistent entries.
    /// Validation needs no inversion, allocation, or scratch.
    pub fn bind(base: &AffinePoint<C>, entries: &'a [E; 8]) -> Result<Self, CurveError> {
        let table = Self::bind_trusted(base, entries)?;
        table.validate()?;
        Ok(table)
    }

    /// Binds entries whose mathematical validity the owner has established.
    ///
    /// Returns [`CurveError::InvalidBase`] for unreduced or off-curve base coordinates.
    /// Entries must be the reduced, on-curve multiples in this type's order, with
    /// consistent cached coordinates. This method does not inspect them; invalid
    /// entries may panic or give incorrect results during arithmetic, while remaining
    /// memory-safe. Use [`Self::bind`] to validate stored data.
    pub fn bind_trusted(base: &AffinePoint<C>, entries: &'a [E; 8]) -> Result<Self, CurveError> {
        check_base(base)?;
        Ok(Self {
            base: *base,
            entries,
        })
    }

    /// Checks every specified multiple and cached coordinate without inversion.
    ///
    /// Returns [`CurveError::InvalidTable`] for invalid entries. Requires no
    /// allocation or scratch, including for views from [`Self::bind_trusted`].
    pub fn validate(&self) -> Result<(), CurveError> {
        for (expected, entry) in representatives(&self.base.to_projective())
            .iter()
            .zip(self.entries)
        {
            check_entry(expected, entry)?;
        }
        Ok(())
    }

    /// Borrows the nonidentity base.
    pub const fn base(&self) -> &AffinePoint<C> {
        &self.base
    }

    /// Borrows entries in the representative order documented on this type.
    pub const fn as_slice(&self) -> &'a [E] {
        self.entries
    }

    /// Borrows the eight entries in the representative order documented on this type.
    pub const fn as_array(&self) -> &'a [E; 8] {
        self.entries
    }

    /// Multiplies by a reduced scalar; zero returns identity.
    ///
    /// The scalar must satisfy [`PastaField`]'s reduced-residue invariant,
    /// and a table created with [`Self::bind_trusted`] must satisfy its entry
    /// requirements. Uses bounded stack storage without caller scratch or
    /// allocation. Execution is variable-time.
    pub fn mul(&self, scalar: &PastaField<C::Scalar>) -> ProjectivePoint<C> {
        self.mul_prepared(&EisensteinScalar::for_single(scalar))
    }

    /// Multiplies using digits that can be reused across tables and batches.
    ///
    /// Has the same entry requirements as [`Self::mul`], and requires no
    /// scratch or allocation. A zero scalar returns identity.
    pub fn mul_prepared(&self, scalar: &EisensteinScalar<C>) -> ProjectivePoint<C> {
        multiply(self.entries, scalar.digits())
    }
}

fn check_base<C: PastaCurve>(base: &AffinePoint<C>) -> Result<(), CurveError> {
    if AffinePoint::<C>::from_xy(base.x, base.y).is_none() {
        return Err(CurveError::InvalidBase);
    }
    Ok(())
}

pub(super) fn normalize<C: PastaCurve, E: CurveTableEntry<C>>(
    points: &[ProjectivePoint<C>],
    field: &mut [PastaField<C::Base>],
    entries: &mut [E],
) {
    batch::normalize(points, field, |index, point| {
        // None of the eight small Eisenstein representatives vanishes modulo
        // the prime scalar modulus, so their nonidentity multiples stay so.
        entries[index] = E::from_affine(point.as_affine().expect("nonidentity representative"));
    });
}

pub(super) fn digit_point<C: PastaCurve, E: CurveTableEntry<C>>(
    entries: &[E],
    code: u8,
) -> AffinePoint<C> {
    let value = usize::from(code - 1);
    let entry = entries[value / 6].rotated((value % 6) >> 1);
    if value & 1 == 1 { entry.neg() } else { entry }
}

pub(super) fn multiply<C: PastaCurve, E: CurveTableEntry<C>>(
    entries: &[E],
    digits: &[u8],
) -> ProjectivePoint<C> {
    let mut result = ProjectivePoint::IDENTITY;
    for &code in digits.iter().rev() {
        result = result.double();
        if code != 0 {
            result = result.add_mixed(&digit_point(entries, code));
        }
    }
    result
}

/// Multiplies a base using a temporary compact table.
///
/// `base` must be nonidentity, and `scalar` must be below `C::Scalar`'s modulus.
pub(super) fn multiply_once<C: PastaCurve>(
    base: &ProjectivePoint<C>,
    scalar: CanonicalUint,
) -> ProjectivePoint<C> {
    // Keep the input's projective scaling until batch normalization, sharing
    // one inversion across all eight representatives and avoiding a separate
    // base inversion. Cached coordinates make the joint ladder's rotations
    // require only copies and additions.
    let points = representatives(base);
    let mut entries = [PreparedAffinePoint::from_affine(&AffinePoint::GENERATOR); 8];
    let mut field = [PastaField::ZERO; 8];
    normalize(&points, &mut field, &mut entries);
    multiply(
        &entries,
        EisensteinScalar::<C>::from_canonical(scalar).digits(),
    )
}
