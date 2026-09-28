//! Unsigned integer arithmetic on four little-endian 64-bit words.
//!
//! Operations use the full width of [`U256`], with no attached modulus or
//! Montgomery factor. Individual operations document overflow behavior and
//! input bounds. Cofactors, exponents, and rounded ratios are ordinary integer
//! calculations even when their inputs come from field or curve parameters.

use super::word::{adc, sbb};
use super::{U256, U512};

mod exponents;
mod ratio;

pub use exponents::{odd_cofactor, tonelli_shanks_exponent};
pub use ratio::round_shifted_ratio;

const fn hex_nibble(nibble: u8) -> u8 {
    match nibble {
        b'0'..=b'9' => nibble - b'0',
        b'a'..=b'f' => nibble - b'a' + 10,
        b'A'..=b'F' => nibble - b'A' + 10,
        _ => panic!("invalid hex digit in a 256-bit constant"),
    }
}

/// Parses a fixed-width hexadecimal integer into little-endian limbs.
///
/// `value` must start with `0x` followed by exactly 64 ASCII hexadecimal digits,
/// most significant first. Digits may use either case. The result is an
/// ordinary integer; parsing does not reduce it modulo a field modulus.
///
/// # Panics
///
/// Panics if the prefix, length, or digits are invalid. Whitespace, underscores,
/// and the uppercase prefix `0X` are not accepted.
///
/// # Examples
///
/// ```
/// # use zakura_bento_core as bento;
/// use bento::const_arithmetic::{U256, u256};
///
/// const VALUE: U256 = u256::from_hex(
///     "0x0000000000000004000000000000000300000000000000020000000000000001",
/// );
/// assert_eq!(VALUE, [1, 2, 3, 4]);
/// ```
pub const fn from_hex(value: &str) -> U256 {
    let encoded = value.as_bytes();
    assert!(
        encoded.len() == 2 + 64 && encoded[0] == b'0' && encoded[1] == b'x',
        "256-bit constants must be 0x-prefixed 64-digit hex strings"
    );
    let mut limbs = [0u64; 4];
    let mut digit = 0;
    while digit < 64 {
        // The first digit after the prefix is the most significant nibble.
        let nibble = hex_nibble(encoded[2 + digit]) as u64;
        let bit = (63 - digit) * 4;
        limbs[bit / 64] |= nibble << (bit % 64);
        digit += 1;
    }
    limbs
}

/// Returns whether `a >= b` as unsigned 256-bit integers.
pub const fn ge(a: &U256, b: &U256) -> bool {
    let mut index = 4;
    while index > 0 {
        index -= 1;
        if a[index] > b[index] {
            return true;
        }
        if a[index] < b[index] {
            return false;
        }
    }
    true
}

/// Adds two unsigned integers, returning the wrapped sum and a carry bit.
///
/// The limbs wrap modulo `2^256`; the returned carry is zero or one.
#[inline]
pub const fn add_with_carry(a: &U256, b: &U256) -> (U256, u64) {
    let mut sum = [0; 4];
    let mut carry = 0;
    let mut index = 0;
    while index < 4 {
        (sum[index], carry) = adc(a[index], b[index], carry);
        index += 1;
    }
    (sum, carry)
}

/// Subtracts `b` from `a`, returning the wrapped difference and a borrow bit.
///
/// The limbs wrap modulo `2^256`; the returned borrow is one exactly when
/// `a < b`.
#[inline]
pub const fn sub_with_borrow(a: &U256, b: &U256) -> (U256, u64) {
    let mut difference = [0; 4];
    let mut borrow = 0;
    let mut index = 0;
    while index < 4 {
        (difference[index], borrow) = sbb(a[index], b[index], borrow);
        index += 1;
    }
    (difference, borrow)
}

/// Subtracts a single-word integer from `value` without wrapping.
///
/// # Panics
///
/// Panics if `value < small`.
pub const fn sub_u64(value: &U256, small: u64) -> U256 {
    let (difference, borrow) = sub_with_borrow(value, &[small, 0, 0, 0]);
    assert!(borrow == 0, "sub_u64 underflow");
    difference
}

/// Shifts an unsigned integer right, discarding its lowest `shift` bits.
///
/// # Panics
///
/// Panics if `shift >= 256`.
#[inline]
pub const fn shr(value: &U256, shift: u32) -> U256 {
    assert!(shift < 256);
    let limb_shift = (shift / 64) as usize;
    let bit_shift = shift % 64;
    let mut result = [0; 4];
    let mut index = 0;
    while index + limb_shift < 4 {
        let mut limb = value[index + limb_shift] >> bit_shift;
        if bit_shift > 0 && index + limb_shift + 1 < 4 {
            limb |= value[index + limb_shift + 1] << (64 - bit_shift);
        }
        result[index] = limb;
        index += 1;
    }
    result
}

/// Divides an unsigned integer exactly by a single-word divisor.
///
/// # Panics
///
/// Panics if `divisor` is zero or the division has a nonzero remainder.
pub const fn div_exact_u64(value: &U256, divisor: u64) -> U256 {
    assert!(divisor != 0);
    let mut quotient = [0; 4];
    let mut remainder: u128 = 0;
    let mut index = 4;
    while index > 0 {
        index -= 1;
        let current = (remainder << 64) | value[index] as u128;
        quotient[index] = (current / divisor as u128) as u64;
        remainder = current % divisor as u128;
    }
    assert!(remainder == 0, "div_exact_u64: inexact division");
    quotient
}

/// Returns the exact 512-bit product of two unsigned 256-bit integers.
#[inline]
pub const fn mul_wide(a: &U256, b: &U256) -> U512 {
    let mut product = [0; 8];
    let mut i = 0;
    while i < 4 {
        let mut carry = 0u64;
        let mut j = 0;
        while j < 4 {
            let term = a[i] as u128 * b[j] as u128 + product[i + j] as u128 + carry as u128;
            product[i + j] = term as u64;
            carry = (term >> 64) as u64;
            j += 1;
        }
        product[i + 4] = carry;
        i += 1;
    }
    product
}

#[cfg(test)]
mod tests;
