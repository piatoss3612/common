//! Canonical modular addition and powers of two by repeated doubling.
//!
//! The modulus bound lets two reduced operands add without overflowing the
//! four-limb representation. Montgomery constants build on these operations.

use super::super::U256;
use super::super::u256::{add_with_carry, ge, sub_with_borrow};

/// Checks the modulus domain shared by this module's arithmetic.
///
/// The modulus must be odd and satisfy `2 < modulus < 2^255`. Primality is not
/// checked or required by this function.
///
/// # Panics
///
/// Panics if any of these conditions is violated.
pub const fn assert_modulus(modulus: &U256) {
    // Oddness makes the Montgomery radix invertible. The upper bound lets
    // reduced sums fit 256 bits and Montgomery intermediates fit 512 bits.
    assert!(modulus[0] & 1 == 1, "modulus must be odd");
    assert!(ge(modulus, &[3, 0, 0, 0]), "modulus must exceed 2");
    assert!(modulus[3] >> 63 == 0, "modulus must be below 2^255");
}

/// Adds two reduced residues modulo `modulus`.
///
/// Both operands and the result are strictly below `modulus`. Addition also
/// preserves Montgomery form when both operands use that representation.
///
/// # Panics
///
/// Panics if [`assert_modulus`] rejects `modulus`, or either operand
/// is greater than or equal to `modulus`.
pub const fn add(modulus: &U256, a: &U256, b: &U256) -> U256 {
    assert_modulus(modulus);
    assert!(
        !ge(a, modulus) && !ge(b, modulus),
        "operands must be reduced"
    );
    // With the modulus below 2^255 and both operands reduced, the sum stays
    // below 2^256 (no carry) and one conditional subtraction reduces it.
    let (sum, carry) = add_with_carry(a, b);
    assert!(carry == 0);
    if ge(&sum, modulus) {
        let (reduced, _) = sub_with_borrow(&sum, modulus);
        reduced
    } else {
        sum
    }
}

/// Returns `2^exponent mod modulus` as a reduced ordinary integer.
///
/// For exponent zero, the result is the ordinary integer one. Use
/// [`super::one`] for the Montgomery representation of one.
///
/// # Panics
///
/// Panics if [`assert_modulus`] rejects `modulus`.
pub const fn pow2_mod(modulus: &U256, exponent: u32) -> U256 {
    assert_modulus(modulus);
    let mut value = [1, 0, 0, 0];
    let mut count = 0;
    while count < exponent {
        value = add(modulus, &value, &value);
        count += 1;
    }
    value
}
