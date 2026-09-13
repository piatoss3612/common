//! Integer Babai rounding with compile-time-checked Pasta lattice parameters.

use super::{
    PastaCurve,
    parameters::{GlvBasis, GlvParameters},
};
use crate::field::{
    PastaField,
    word::{mac, subtract_limbs},
};

/// Splits a scalar into signed halves `(k1, k2)` with magnitudes below `2^127`.
///
/// This GLV decomposition satisfies `k = k1 + lambda * k2` modulo the scalar
/// modulus, where `k` is the canonical integer represented by `scalar` and
/// `lambda` is the scalar field's [`PastaField::zeta`] value. The corresponding
/// point map is [`AffinePoint::endomorphism`](super::AffinePoint::endomorphism).
/// The scalar must satisfy [`PastaField`]'s reduced-residue invariant.
/// This operation is variable-time and needs neither allocation nor caller
/// scratch.
///
/// ```
/// use zakura_udon::{curve::{glv_decompose, Pallas}, field::Fq};
///
/// let scalar = Fq::from_u64(42).sub(&Fq::zeta());
/// let (a, b) = glv_decompose::<Pallas>(&scalar);
/// assert!(a.unsigned_abs() < 1_u128 << 127);
/// assert!(b.unsigned_abs() < 1_u128 << 127);
/// let signed_field = |value: i128| {
///     let magnitude = Fq::from_bytes_reduced(&value.unsigned_abs().to_le_bytes());
///     if value < 0 { magnitude.neg() } else { magnitude }
/// };
/// assert_eq!(signed_field(a).add(&Fq::zeta().mul(&signed_field(b))), scalar);
/// ```
pub fn glv_decompose<C: PastaCurve>(scalar: &PastaField<C::Scalar>) -> (i128, i128) {
    decompose(
        scalar.to_canonical_uint().limbs(),
        &GlvParameters::<C>::BASIS,
    )
}

/// Multiplies two little-endian limb strings into a zeroed result.
fn schoolbook_multiply(lhs: &[u64], rhs: &[u64], result: &mut [u64]) {
    debug_assert_eq!(result.len(), lhs.len() + rhs.len());
    for (lhs_index, &lhs_limb) in lhs.iter().enumerate() {
        let mut carry = 0;
        for (rhs_index, &rhs_limb) in rhs.iter().enumerate() {
            (result[lhs_index + rhs_index], carry) =
                mac(result[lhs_index + rhs_index], lhs_limb, rhs_limb, carry);
        }
        result[lhs_index + rhs.len()] = carry;
    }
}

/// Computes `round((coefficient * scalar) / 2^384)` for Babai rounding.
fn rounded_high_product(coefficient: &[u64; 5], scalar: &[u64; 4]) -> u128 {
    let mut product = [0; 9];
    schoolbook_multiply(coefficient, scalar, &mut product);
    debug_assert_eq!(product[8], 0);
    let round = product[5] >> 63;
    (u128::from(product[6]) | (u128::from(product[7]) << 64)).wrapping_add(u128::from(round))
}

fn multiply_u128(lhs: u128, rhs: u128) -> [u64; 4] {
    let mut product = [0; 4];
    schoolbook_multiply(
        &[lhs as u64, (lhs >> 64) as u64],
        &[rhs as u64, (rhs >> 64) as u64],
        &mut product,
    );
    product
}

fn subtract_256(lhs: [u64; 4], rhs: [u64; 4]) -> [u64; 4] {
    subtract_limbs(&lhs, &rhs).0
}

/// Decodes a two's-complement value whose magnitude is below `2^127`.
fn signed_glv_half(value: [u64; 4]) -> i128 {
    let sign_extension = if value[1] >> 63 == 0 { 0 } else { u64::MAX };
    debug_assert_eq!(value[2], sign_extension);
    debug_assert_eq!(value[3], sign_extension);
    let low = u128::from(value[0]) | (u128::from(value[1]) << 64);
    let half = low as i128;
    debug_assert_ne!(half, i128::MIN);
    half
}

#[inline]
fn decompose(limbs: [u64; 4], lattice: &GlvBasis) -> (i128, i128) {
    // For input (k, 0), the inverse basis gives coefficients (k*d/n, k*b/n).
    // Subtracting rounded multiples of the kernel columns preserves the
    // scalar. Here n is the scalar modulus and d is GlvBasis::d.
    // GlvParameters checks that fixed-point rounding leaves both residual
    // coordinates within the signed 127-bit bound.
    let coefficient_1 = rounded_high_product(&lattice.g1, &limbs);
    let coefficient_2 = rounded_high_product(&lattice.g2, &limbs);
    let half_1 = subtract_256(
        subtract_256(limbs, multiply_u128(coefficient_1, lattice.a)),
        multiply_u128(coefficient_2, lattice.b),
    );
    let half_2 = subtract_256(
        multiply_u128(coefficient_1, lattice.b),
        multiply_u128(coefficient_2, lattice.d),
    );
    (signed_glv_half(half_1), signed_glv_half(half_2))
}
