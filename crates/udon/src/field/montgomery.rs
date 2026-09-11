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
/// input below `pR` needs one subtraction; the wider bound requires two.
#[inline(always)]
pub(super) fn montgomery_reduce_unreduced<M: PrimeModulus>(limbs: [u64; 8]) -> [u64; 4] {
    // Cancel only the low half, then add the untouched high half once.
    // This is the same REDC integer as full-width cancellation. Under the
    // documented bound the final sum is below 3p < R, so no carry is lost.
    let [mut r0, mut r1, mut r2, mut r3, t4, t5, t6, t7] = limbs;
    for _ in 0..4 {
        let k = r0.wrapping_mul(M::MONTGOMERY_INV);
        let (cancelled, carry) = mac(r0, k, M::MODULUS[0], 0);
        debug_assert_eq!(cancelled, 0);
        let (s0, carry) = mac(r1, k, M::MODULUS[1], carry);
        let (s1, carry) = adc(r2, 0, carry);
        let (s2, carry) = adc(r3, k << 62, carry);
        let s3 = (k >> 2) + carry;
        (r0, r1, r2, r3) = (s0, s1, s2, s3);
    }
    let (r0, carry) = adc(r0, t4, 0);
    let (r1, carry) = adc(r1, t5, carry);
    let (r2, carry) = adc(r2, t6, carry);
    let (r3, carry) = adc(r3, t7, carry);
    debug_assert_eq!(carry, 0);
    [r0, r1, r2, r3]
}

/// Repeated squaring with raw, unreduced intermediates, followed by an
/// optional multiplication by a reduced residue. No field value crosses the
/// canonical representation boundary until the final reduction.
///
/// Requires a reduced input and at most 256 squares. The parameter bundle
/// checks the exact REDC recurrence for every permitted run length, including
/// the final product bound. Larger runs are split by the caller.
#[inline]
pub(super) fn square_run<M: PrimeModulus>(
    value: &[u64; 4],
    count: usize,
    factor: Option<&[u64; 4]>,
) -> [u64; 4] {
    debug_assert!(count <= 256);
    let mut value = *value;
    for _ in 0..count {
        value = montgomery_reduce_unreduced::<M>(super::word::square_wide(&value));
    }
    match factor {
        Some(factor) => montgomery_multiply::<M>(&value, factor),
        None => reduce_once::<M>(value),
    }
}
