//! Limb arithmetic shared by the runtime field kernels.

use bento::const_arithmetic::u256;

/// Adds two limbs and a carry, returning the low limb and high carry.
#[inline(always)]
pub(super) fn adc(lhs: u64, rhs: u64, carry: u64) -> (u64, u64) {
    let value = u128::from(lhs) + u128::from(rhs) + u128::from(carry);
    (value as u64, (value >> 64) as u64)
}

/// Subtracts a limb and a borrow bit, returning the low limb and borrow bit.
#[inline(always)]
pub(super) fn sbb(lhs: u64, rhs: u64, borrow: u64) -> (u64, u64) {
    let (value, first_borrow) = lhs.overflowing_sub(rhs);
    let (value, second_borrow) = value.overflowing_sub(borrow);
    (value, u64::from(first_borrow | second_borrow))
}

/// Accumulates one limb product plus an accumulator limb and a carry limb.
#[inline(always)]
pub(super) fn mac(accumulator: u64, lhs: u64, rhs: u64, carry: u64) -> (u64, u64) {
    let value = u128::from(lhs) * u128::from(rhs) + u128::from(accumulator) + u128::from(carry);
    (value as u64, (value >> 64) as u64)
}

/// Compares two unsigned 256-bit integers.
#[inline]
pub(super) fn compare_limbs(lhs: &[u64; 4], rhs: &[u64; 4]) -> core::cmp::Ordering {
    for index in (0..4).rev() {
        match lhs[index].cmp(&rhs[index]) {
            core::cmp::Ordering::Equal => {}
            ordering => return ordering,
        }
    }
    core::cmp::Ordering::Equal
}

/// Subtracts two unsigned 256-bit integers, returning the borrow bit.
#[inline]
pub(super) fn subtract_limbs(lhs: &[u64; 4], rhs: &[u64; 4]) -> ([u64; 4], u64) {
    let mut result = [0; 4];
    let mut borrow = 0;
    for index in 0..4 {
        (result[index], borrow) = sbb(lhs[index], rhs[index], borrow);
    }
    (result, borrow)
}

/// Returns the exact eight-limb product, without modular reduction.
#[inline]
pub(super) fn multiply_wide(lhs: &[u64; 4], rhs: &[u64; 4]) -> [u64; 8] {
    u256::mul_wide(lhs, rhs)
}

/// Returns the exact eight-limb square of a 256-bit integer.
#[inline(always)]
pub(super) fn square_wide(value: &[u64; 4]) -> [u64; 8] {
    // Accumulate the six off-diagonal products, double them, then add the
    // four diagonal products. This uses ten limb multiplications instead of
    // the sixteen needed by general multiplication.
    let (r1, carry) = mac(0, value[0], value[1], 0);
    let (r2, carry) = mac(0, value[0], value[2], carry);
    let (r3, r4) = mac(0, value[0], value[3], carry);

    let (r3, carry) = mac(r3, value[1], value[2], 0);
    let (r4, r5) = mac(r4, value[1], value[3], carry);

    let (r5, r6) = mac(r5, value[2], value[3], 0);

    let r7 = r6 >> 63;
    let r6 = (r6 << 1) | (r5 >> 63);
    let r5 = (r5 << 1) | (r4 >> 63);
    let r4 = (r4 << 1) | (r3 >> 63);
    let r3 = (r3 << 1) | (r2 >> 63);
    let r2 = (r2 << 1) | (r1 >> 63);
    let r1 = r1 << 1;

    let (r0, carry) = mac(0, value[0], value[0], 0);
    let (r1, carry) = adc(r1, 0, carry);
    let (r2, carry) = mac(r2, value[1], value[1], carry);
    let (r3, carry) = adc(r3, 0, carry);
    let (r4, carry) = mac(r4, value[2], value[2], carry);
    let (r5, carry) = adc(r5, 0, carry);
    let (r6, carry) = mac(r6, value[3], value[3], carry);
    let (r7, carry) = adc(r7, 0, carry);
    debug_assert_eq!(carry, 0);

    [r0, r1, r2, r3, r4, r5, r6, r7]
}
