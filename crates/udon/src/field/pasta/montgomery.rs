//! Montgomery multiplication and reduction for the Pasta primes.
//!
//! Multiplication and reduction exploit the sealed Pasta modulus shape.

use super::PrimeModulus;

use super::word::{adc, mac, subtract_limbs};

/// Subtracts `p` if the integer is at least `p`.
///
/// Inputs below `2p` produce a reduced residue.
#[inline]
pub(super) const fn reduce_once<M: PrimeModulus>(limbs: [u64; 4]) -> [u64; 4] {
    let (reduced, borrow) = subtract_limbs(&limbs, &M::MODULUS);
    if borrow == 0 { reduced } else { limbs }
}

/// Reduces a five-limb integer below `4p` modulo `2p`.
#[inline]
pub(super) fn reduce_twice_modulus<M: PrimeModulus>(limbs: [u64; 4], carry: u64) -> [u64; 4] {
    let (reduced, borrow) = subtract_limbs(&limbs, &M::TWICE_MODULUS);
    if carry != 0 || borrow == 0 {
        reduced
    } else {
        limbs
    }
}

/// Computes `lhs * rhs * R^-1 mod p` in `[0, 2p)`, with `R = 2^256`.
///
/// Both inputs may lie in `[0, 2p)`; see the closure proof below. Conversion
/// also uses this kernel with `lhs < R` and `rhs = R2 < p`, whose product is
/// below `pR`. The live CIOS accumulator fits because `lhs + p < 3p < R`
/// for field arithmetic; conversion's integrated result is below `2p`.
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
    accumulator[..4].try_into().unwrap()
}

/// Squares a loose Montgomery residue, retaining the `[0, 2p)` bound.
#[inline(always)]
pub(super) fn montgomery_square<M: PrimeModulus>(value: &[u64; 4]) -> [u64; 4] {
    montgomery_reduce_unreduced::<M>(crate::field::pasta::word::square_wide(value))
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
/// input below `pR` produces a loose result below `2p`. Producing a reduced
/// result takes one subtraction, or two for the wider input bound.
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

/// Repeated squaring and an optional product, all in `[0, 2p)`.
///
/// Closure for the Pasta primes is stronger than the generic REDC bound.
/// Write `p = R/4 + c`; the parameters assert `16c² < R` and `3p < R`.
/// Suppose `a,b < 2p` but `u = (ab + mp)/R >= 2p`, with `0 <= m < R`.
/// Then `ab >= pR + p`. Set `A = 2p-a`, `B = 2p-b`, and `S = A+B`.
/// If `S >= 2c+1`, AM-GM gives
/// `ab <= (R/2+c-1/2)² < pR`, a contradiction. Hence `S <= 2c`,
/// `0 < AB <= c² < p`, and, with `L = 4c-2S`, `ab = pR + pL + AB`.
/// Write `u = 2p+k` and `j = L+m-R`; then `kR = AB+jp`, so
/// `0 <= j <= L-1` and `AB+jc = (4k-j)R/4`. But
/// `0 < AB+jc < 4c² < R/4`, impossible for a multiple of `R/4`.
/// Thus arbitrary chains of loose products and squares remain below `2p`.
#[inline]
pub(super) fn square_run<M: PrimeModulus>(
    value: &[u64; 4],
    count: usize,
    factor: Option<&[u64; 4]>,
) -> [u64; 4] {
    let mut value = *value;
    for _ in 0..count {
        value = montgomery_reduce_unreduced::<M>(super::word::square_wide(&value));
    }
    match factor {
        Some(factor) => montgomery_multiply::<M>(&value, factor),
        None => value,
    }
}
