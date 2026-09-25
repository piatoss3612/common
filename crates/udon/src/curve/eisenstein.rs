//! Joint width-three recoding in the Eisenstein integers.
//!
//! A pair `(a, b)` represents `a + b*lambda`, with `lambda² + lambda + 1 = 0`.
//! Multiplication by `lambda` rotates a pair to `(-b, a - b)`; signs and the
//! three rotations give six units. Evaluating `lambda` at the scalar field's
//! cube root of unity maps these integer pairs to scalars.

use super::{
    AffinePoint, CurveTableEntry, CurveTableRequirements, PastaCurve, ProjectivePoint,
    assert_scratch, batch,
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
    /// The scalar uses [`PastaField`]'s loose representation.
    pub fn new(scalar: &PastaField<C::Scalar>) -> Self {
        Self::for_single(scalar).certify_batch()
    }

    /// Recodes a canonical scalar integer into signed Eisenstein digits.
    ///
    /// The caller must establish that `scalar` is below `C::Scalar`'s modulus;
    /// [`CanonicalUint`] alone does not establish that bound.
    pub(super) fn from_canonical(scalar: CanonicalUint) -> Self {
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
        self.batch_safe = Some(ladder_safe::<C>(self.digits()));
        self
    }

    pub(super) fn batch_safe(&self) -> bool {
        self.batch_safe
            .unwrap_or_else(|| ladder_safe::<C>(self.digits()))
    }

    pub(super) fn digits(&self) -> &[u8] {
        &self.digits[..self.len]
    }

    /// Wraps recoded digits directly, bypassing scalar decomposition.
    #[cfg(test)]
    pub(super) fn from_digits(digits: &[u8]) -> Self {
        let mut stored = [0; MAX_DIGITS];
        stored[..digits.len()].copy_from_slice(digits);
        Self {
            digits: stored,
            len: digits.len(),
            batch_safe: None,
            marker: PhantomData,
        }
    }
}

/// The scalar-field value of one nonzero digit code.
pub(super) fn digit_scalar<C: PastaCurve>(code: u8) -> PastaField<C::Scalar> {
    let value = usize::from(code - 1);
    let (mut a, mut b) = REPRESENTATIVES[value / 6];
    for _ in 0..(value % 6) / 2 {
        (a, b) = (-b, a - b);
    }
    let signed = |x: i8| {
        let f = PastaField::<C::Scalar>::from_u64(u64::from(x.unsigned_abs()));
        if x < 0 { f.neg() } else { f }
    };
    let d = signed(a).add(&signed(b).mul(&PastaField::<C::Scalar>::ZETA));
    if value & 1 == 1 { d.neg() } else { d }
}

fn ladder_safe<C: PastaCurve>(digits: &[u8]) -> bool {
    let Some((&top, rest)) = digits.split_last() else {
        return false;
    };
    let mut s = digit_scalar::<C>(top);
    // All nonidentity bases have the same prime order. Checking the schedule
    // in the scalar field is therefore exact for every base in the batch,
    // including schedules whose intermediate integer coefficients wrap.
    for &code in rest.iter().rev() {
        let twice = s.double();
        if code == 0 {
            s = twice;
        } else {
            let d = digit_scalar::<C>(code);
            if d.reduce() == s.reduce() || d.reduce() == twice.neg().reduce() {
                return false;
            }
            s = twice.add(&d);
        }
    }
    true
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

pub(super) fn representatives_affine<C: PastaCurve>(
    base: &AffinePoint<C>,
) -> [ProjectivePoint<C>; 8] {
    // Use the same coefficient identities as representatives, keeping phi(base)
    // affine. Commuting sums and negating the projective operand in differences
    // makes six of the seven additions mixed; intermediates need no inversion.
    let phi = base.endomorphism();
    let difference = base.to_projective().add_mixed(&phi.neg());
    let b = difference.sub(&difference.endomorphism());
    let b_phi = b.endomorphism();
    let minus_three = b_phi.endomorphism();
    let three_a = minus_three.add_mixed(&phi);
    let three_b = minus_three.neg().add_mixed(&phi);
    let four_a = b_phi.neg().add_mixed(&phi);
    let four_b = b_phi.add_mixed(&phi);
    let nineteen = four_b.add_mixed(&phi);
    [
        base.to_projective(),
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
/// [`PreparedAffinePoint`](super::PreparedAffinePoint) entries for 768 bytes and
/// cheaper rotations.
/// These sizes exclude the base and table handle. Preparation
/// and multiplication are allocation-free and provide no constant-time
/// guarantee for secret bases, scalars, or table contents.
/// The entry count is encoded in the borrowed array type.
#[derive(Clone, Copy)]
pub struct EisensteinTable<'a, C: PastaCurve, E: CurveTableEntry<C> = AffinePoint<C>> {
    base: AffinePoint<C>,
    entries: &'a [E; 8],
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
    /// The returned view borrows only `entries`.
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
    /// );
    /// let scalar = Fq::from_u64(42);
    /// assert_eq!(table.mul(&scalar), base.mul_projective(&scalar));
    /// ```
    pub fn prepare(
        base: &AffinePoint<C>,
        entries: &'a mut [E; 8],
        projective_scratch: &mut [ProjectivePoint<C>],
        field_scratch: &mut [PastaField<C::Base>],
    ) -> Self {
        assert_scratch("projective", 8, projective_scratch.len());
        assert_scratch("field", 8, field_scratch.len());
        projective_scratch[..8].copy_from_slice(&representatives_affine(base));
        normalize(&projective_scratch[..8], &mut field_scratch[..8], entries);
        Self {
            base: *base,
            entries,
        }
    }

    /// Borrows trusted entries in the representative order documented on this type.
    ///
    /// `entries` must have been prepared for `base`. Binding preserves the stored
    /// representation and performs no field or curve arithmetic.
    pub const fn bind(base: &AffinePoint<C>, entries: &'a [E; 8]) -> Self {
        Self {
            base: *base,
            entries,
        }
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

    /// Multiplies by a scalar; zero returns identity.
    ///
    /// Uses bounded stack storage without caller scratch or
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
