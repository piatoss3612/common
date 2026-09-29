//! Limb arithmetic shared by field kernels, GLV, and compile-time bound checks.

/// Adds two limbs and a carry, returning the low limb and high carry.
#[inline(always)]
pub(super) const fn adc(lhs: u64, rhs: u64, carry: u64) -> (u64, u64) {
    let value = lhs as u128 + rhs as u128 + carry as u128;
    (value as u64, (value >> 64) as u64)
}

/// Subtracts a limb and a borrow bit, returning the low limb and borrow bit.
#[inline(always)]
pub(super) const fn sbb(lhs: u64, rhs: u64, borrow: u64) -> (u64, u64) {
    let (value, first_borrow) = lhs.overflowing_sub(rhs);
    let (value, second_borrow) = value.overflowing_sub(borrow);
    (value, (first_borrow | second_borrow) as u64)
}

/// Adds two limbs and a carry bit, returning the sum and its carry bit.
///
/// Chains of these lower to flag-carrying additions; the `u128` helpers
/// materialize each carry as a register value instead.
#[inline(always)]
pub(super) fn carry_add(lhs: u64, rhs: u64, carry: bool) -> (u64, bool) {
    lhs.carrying_add(rhs, carry)
}

/// Subtracts two unsigned 256-bit integers with a flag-carried borrow chain,
/// returning the wrapped difference and borrow bit.
#[inline(always)]
pub(super) fn borrow_sub_limbs(lhs: &[u64; 4], rhs: &[u64; 4]) -> ([u64; 4], bool) {
    let (r0, borrow) = lhs[0].borrowing_sub(rhs[0], false);
    let (r1, borrow) = lhs[1].borrowing_sub(rhs[1], borrow);
    let (r2, borrow) = lhs[2].borrowing_sub(rhs[2], borrow);
    let (r3, borrow) = lhs[3].borrowing_sub(rhs[3], borrow);
    ([r0, r1, r2, r3], borrow)
}

/// Adds two unsigned 256-bit integers with a flag-carried carry chain,
/// returning the wrapped sum and carry bit.
#[inline(always)]
pub(super) fn carry_add_limbs(lhs: &[u64; 4], rhs: &[u64; 4]) -> ([u64; 4], bool) {
    let (r0, carry) = carry_add(lhs[0], rhs[0], false);
    let (r1, carry) = carry_add(lhs[1], rhs[1], carry);
    let (r2, carry) = carry_add(lhs[2], rhs[2], carry);
    let (r3, carry) = carry_add(lhs[3], rhs[3], carry);
    ([r0, r1, r2, r3], carry)
}

/// Returns the low and high limbs of `lhs * rhs`.
#[inline(always)]
pub(super) const fn wide_mul(lhs: u64, rhs: u64) -> (u64, u64) {
    let value = lhs as u128 * rhs as u128;
    (value as u64, (value >> 64) as u64)
}

/// Accumulates one limb product plus an accumulator limb and a carry limb.
#[inline(always)]
pub(crate) const fn mac(accumulator: u64, lhs: u64, rhs: u64, carry: u64) -> (u64, u64) {
    let value = lhs as u128 * rhs as u128 + accumulator as u128 + carry as u128;
    (value as u64, (value >> 64) as u64)
}

/// Compares two unsigned 256-bit integers.
#[inline]
pub(super) const fn compare_limbs(lhs: &[u64; 4], rhs: &[u64; 4]) -> core::cmp::Ordering {
    let mut index = 4;
    while index > 0 {
        index -= 1;
        if lhs[index] < rhs[index] {
            return core::cmp::Ordering::Less;
        }
        if lhs[index] > rhs[index] {
            return core::cmp::Ordering::Greater;
        }
    }
    core::cmp::Ordering::Equal
}

/// Adds two unsigned 256-bit integers, returning the wrapped sum and carry bit.
#[inline]
pub(super) const fn add_limbs(lhs: &[u64; 4], rhs: &[u64; 4]) -> ([u64; 4], u64) {
    let mut result = [0; 4];
    let mut carry = 0;
    let mut index = 0;
    while index < 4 {
        (result[index], carry) = adc(lhs[index], rhs[index], carry);
        index += 1;
    }
    (result, carry)
}

/// Subtracts two unsigned 256-bit integers, returning the wrapped difference
/// and borrow bit.
#[inline]
pub(crate) const fn subtract_limbs(lhs: &[u64; 4], rhs: &[u64; 4]) -> ([u64; 4], u64) {
    let mut result = [0; 4];
    let mut borrow = 0;
    let mut index = 0;
    while index < 4 {
        (result[index], borrow) = sbb(lhs[index], rhs[index], borrow);
        index += 1;
    }
    (result, borrow)
}

/// Returns the exact eight-limb product, without modular reduction.
#[inline]
pub(super) const fn multiply_wide(lhs: &[u64; 4], rhs: &[u64; 4]) -> [u64; 8] {
    let mut product = [0; 8];
    let mut i = 0;
    while i < 4 {
        let mut carry = 0;
        let mut j = 0;
        while j < 4 {
            (product[i + j], carry) = mac(product[i + j], lhs[i], rhs[j], carry);
            j += 1;
        }
        product[i + 4] = carry;
        i += 1;
    }
    product
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
