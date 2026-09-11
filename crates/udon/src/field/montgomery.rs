//! Montgomery multiplication and reduction for the Pasta primes.
//!
//! Multiplication and reduction exploit the sealed Pasta modulus shape.

use super::PrimeModulus;

use super::word::{adc, mac, subtract_limbs};

/// Subtracts `p` if the integer is at least `p`.
///
/// Inputs below `2p` produce a reduced residue.
#[inline]
pub(super) fn reduce_once<M: PrimeModulus>(limbs: [u64; 4]) -> [u64; 4] {
    let (reduced, borrow) = subtract_limbs(&limbs, &M::MODULUS);
    if borrow == 0 { reduced } else { limbs }
}

/// Computes the reduced residue `lhs * rhs * R^-1 mod p`, with `R = 2^256`.
///
/// The input product must be below `p * R`; field arithmetic and conversion
/// from a 256-bit integer both satisfy this bound.
#[inline(always)]
pub(super) fn montgomery_multiply<M: PrimeModulus>(lhs: &[u64; 4], rhs: &[u64; 4]) -> [u64; 4] {
    debug_assert_eq!(M::MODULUS[2], 0);
    debug_assert_eq!(M::MODULUS[3], 1 << 62);

    // Coarsely integrated operand scanning keeps only the live five-limb
    // accumulator. Each round adds one schoolbook row and immediately cancels
    // its low limb. The two upper Pasta modulus limbs are zero and 2^62, so
    // their products require only carry propagation and shifts.
    let mut accumulator = [0; 5];
    for rhs_limb in rhs {
        let mut carry = 0;
        for index in 0..4 {
            (accumulator[index], carry) = mac(accumulator[index], lhs[index], *rhs_limb, carry);
        }
        let (upper, product_overflow) = adc(accumulator[4], carry, 0);
        accumulator[4] = upper;

        let multiplier = accumulator[0].wrapping_mul(M::MONTGOMERY_INV);
        let (cancelled, carry) = mac(accumulator[0], multiplier, M::MODULUS[0], 0);
        debug_assert_eq!(cancelled, 0);
        let (r0, carry) = mac(accumulator[1], multiplier, M::MODULUS[1], carry);
        let (r1, carry) = adc(accumulator[2], 0, carry);
        let (r2, carry) = adc(accumulator[3], multiplier << 62, carry);
        let (r3, reduction_overflow) = adc(accumulator[4], multiplier >> 2, carry);

        accumulator = [r0, r1, r2, r3, product_overflow + reduction_overflow];
    }
    debug_assert_eq!(accumulator[4], 0);
    reduce_once::<M>(accumulator[..4].try_into().unwrap())
}

/// Squares a reduced Montgomery residue, retaining Montgomery form.
#[inline(always)]
pub(super) fn montgomery_square<M: PrimeModulus>(value: &[u64; 4]) -> [u64; 4] {
    montgomery_reduce::<M>(crate::field::word::square_wide(value))
}

/// Montgomery REDC: maps an eight-limb integer below `p * R` to its
/// reduced residue after multiplication by `R^-1`, where `R = 2^256`.
#[inline(always)]
pub(super) fn montgomery_reduce<M: PrimeModulus>(limbs: [u64; 8]) -> [u64; 4] {
    reduce_once::<M>(montgomery_reduce_unreduced::<M>(limbs))
}

/// Computes REDC without its final conditional subtraction.
///
/// Requires `limbs < pR + p²`, where `R = 2^256`.
/// The result is below `2p + p²/R < 3p`.
/// For Pasta, `p < R/3`, so `limbs + (R - 1)p < 2pR + p² < R²`;
/// cancellation therefore fits in eight limbs throughout. A caller with
/// input below `pR` needs one subtraction; signed product differences need two.
#[inline(always)]
pub(super) fn montgomery_reduce_unreduced<M: PrimeModulus>(mut limbs: [u64; 8]) -> [u64; 4] {
    debug_assert_eq!(M::MODULUS[2], 0);
    debug_assert_eq!(M::MODULUS[3], 1 << 62);

    let multiplier = limbs[0].wrapping_mul(M::MONTGOMERY_INV);
    let (cancelled, carry) = mac(limbs[0], multiplier, M::MODULUS[0], 0);
    debug_assert_eq!(cancelled, 0);
    let (r1, carry) = mac(limbs[1], multiplier, M::MODULUS[1], carry);
    let (r2, carry) = adc(limbs[2], 0, carry);
    let (r3, carry) = adc(limbs[3], multiplier << 62, carry);
    let (r4, overflow) = adc(limbs[4], multiplier >> 2, carry);
    let (r5, overflow) = adc(limbs[5], 0, overflow);
    let (r6, overflow) = adc(limbs[6], 0, overflow);
    let (r7, overflow) = adc(limbs[7], 0, overflow);
    debug_assert_eq!(overflow, 0);
    limbs = [0, r1, r2, r3, r4, r5, r6, r7];

    let multiplier = limbs[1].wrapping_mul(M::MONTGOMERY_INV);
    let (cancelled, carry) = mac(limbs[1], multiplier, M::MODULUS[0], 0);
    debug_assert_eq!(cancelled, 0);
    let (r2, carry) = mac(limbs[2], multiplier, M::MODULUS[1], carry);
    let (r3, carry) = adc(limbs[3], 0, carry);
    let (r4, carry) = adc(limbs[4], multiplier << 62, carry);
    let (r5, overflow) = adc(limbs[5], multiplier >> 2, carry);
    let (r6, overflow) = adc(limbs[6], 0, overflow);
    let (r7, overflow) = adc(limbs[7], 0, overflow);
    debug_assert_eq!(overflow, 0);
    limbs = [0, 0, r2, r3, r4, r5, r6, r7];

    let multiplier = limbs[2].wrapping_mul(M::MONTGOMERY_INV);
    let (cancelled, carry) = mac(limbs[2], multiplier, M::MODULUS[0], 0);
    debug_assert_eq!(cancelled, 0);
    let (r3, carry) = mac(limbs[3], multiplier, M::MODULUS[1], carry);
    let (r4, carry) = adc(limbs[4], 0, carry);
    let (r5, carry) = adc(limbs[5], multiplier << 62, carry);
    let (r6, overflow) = adc(limbs[6], multiplier >> 2, carry);
    let (r7, overflow) = adc(limbs[7], 0, overflow);
    debug_assert_eq!(overflow, 0);
    limbs = [0, 0, 0, r3, r4, r5, r6, r7];

    let multiplier = limbs[3].wrapping_mul(M::MONTGOMERY_INV);
    let (cancelled, carry) = mac(limbs[3], multiplier, M::MODULUS[0], 0);
    debug_assert_eq!(cancelled, 0);
    let (r4, carry) = mac(limbs[4], multiplier, M::MODULUS[1], carry);
    let (r5, carry) = adc(limbs[5], 0, carry);
    let (r6, carry) = adc(limbs[6], multiplier << 62, carry);
    let (r7, overflow) = adc(limbs[7], multiplier >> 2, carry);
    debug_assert_eq!(overflow, 0);
    limbs = [0, 0, 0, 0, r4, r5, r6, r7];

    limbs[4..].try_into().unwrap()
}
