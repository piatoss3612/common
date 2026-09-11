//! Montgomery reduction, conversions, exponentiation, and prime inversion.

use super::super::u256::{ge, mul_wide, sub_u64, sub_with_borrow};
use super::super::word::adc;
use super::super::{U256, U512};
use super::modular::{assert_modulus, pow2_mod};

/// Returns the Montgomery representation of one: `2^256 mod modulus`.
///
/// # Panics
///
/// Panics if [`assert_modulus`] rejects `modulus`.
pub const fn one(modulus: &U256) -> U256 {
    pow2_mod(modulus, 256)
}

/// Returns the conversion factor `2^512 mod modulus`.
///
/// This is the reduced square of the Montgomery radix `R = 2^256`.
/// Multiplying an ordinary integer by this factor with [`mul`]
/// converts it to Montgomery form, as [`from_u256`] does.
///
/// # Panics
///
/// Panics if [`assert_modulus`] rejects `modulus`.
pub const fn r2(modulus: &U256) -> U256 {
    pow2_mod(modulus, 512)
}

/// Returns the coefficient that cancels a low word during Montgomery reduction.
///
/// Pass the modulus's least significant word as `low`. The result is
/// `-low^-1 mod 2^64`. Any odd `u64` is accepted, including one; this word
/// operation does not validate a full modulus. For inversion of a field element
/// in Montgomery form, use [`invert_prime`].
///
/// # Panics
///
/// Panics if `low` is even.
pub const fn reduction_coefficient(low: u64) -> u64 {
    assert!(low & 1 == 1, "low word must be odd");
    // An odd n is its own inverse mod 8; each iteration doubles the number of
    // correct low bits: 3 -> 6 -> 12 -> 24 -> 48 -> 96 >= 64.
    let mut inverse = low;
    let mut iteration = 0;
    while iteration < 5 {
        inverse = inverse.wrapping_mul(2u64.wrapping_sub(low.wrapping_mul(inverse)));
        iteration += 1;
    }
    assert!(low.wrapping_mul(inverse) == 1);
    inverse.wrapping_neg()
}

/// Removes one Montgomery factor from a 512-bit integer.
///
/// Returns `value * R^-1 mod modulus`, where `R = 2^256` and `R^-1` is its
/// modular inverse. The input must satisfy `value < modulus * R`; the result
/// is strictly below `modulus`.
///
/// Use [`to_u256`] to decode a reduced Montgomery residue into an ordinary
/// integer. This wider operation also accepts products and other bounded
/// intermediates.
///
/// # Panics
///
/// Panics if [`assert_modulus`] rejects `modulus`, or
/// `value >= modulus * R`.
pub const fn reduce_wide(modulus: &U256, value: &U512) -> U256 {
    // Below 2^255, `value + m·modulus < 2·modulus·R < 2^512` for in-range
    // input, so the top-carry assertion below never fires spuriously.
    assert_modulus(modulus);
    // Comparing the upper half checks value < modulus * R exactly, including
    // inputs with every bit of the lower half set.
    assert!(
        !ge(&[value[4], value[5], value[6], value[7]], modulus),
        "wide input must be below modulus * R"
    );
    let coefficient = reduction_coefficient(modulus[0]);
    let mut t = *value;
    let mut carry_top = 0u64;
    let mut i = 0;
    while i < 4 {
        let m = t[i].wrapping_mul(coefficient);
        let mut carry = 0u64;
        let mut j = 0;
        while j < 4 {
            let term = t[i + j] as u128 + m as u128 * modulus[j] as u128 + carry as u128;
            t[i + j] = term as u64;
            carry = (term >> 64) as u64;
            j += 1;
        }
        // The preceding iteration's top carry belongs at this word. Defer
        // propagation until here instead of traversing the remaining words.
        (t[i + 4], carry_top) = adc(t[i + 4], carry, carry_top);
        i += 1;
    }
    assert!(carry_top == 0, "reduce_wide input out of range");
    let reduced = [t[4], t[5], t[6], t[7]];
    if ge(&reduced, modulus) {
        let (canonical, _) = sub_with_borrow(&reduced, modulus);
        canonical
    } else {
        reduced
    }
}

/// Multiplies two integers and removes one Montgomery factor.
///
/// Returns `a * b * R^-1 mod modulus`, where `R = 2^256` and `R^-1` is its
/// modular inverse. When `a` and `b` are reduced Montgomery residues, the
/// result is the reduced Montgomery representation of their product.
///
/// More generally, the result is reduced whenever `a * b < modulus * R`.
/// Reduced operands satisfy this bound, but neither operand needs to be
/// reduced if their product meets it. [`reduce_wide`] checks this bound.
///
/// # Panics
///
/// Panics if [`assert_modulus`] rejects `modulus`, or
/// `a * b >= modulus * R`.
pub const fn mul(modulus: &U256, a: &U256, b: &U256) -> U256 {
    reduce_wide(modulus, &mul_wide(a, b))
}

/// Converts an unsigned 256-bit integer to a reduced Montgomery residue.
///
/// Returns `value * 2^256 mod modulus`. Any [`U256`] value is accepted;
/// `value` need not be below `modulus`.
///
/// # Panics
///
/// Panics if [`assert_modulus`] rejects `modulus`.
pub const fn from_u256(modulus: &U256, value: &U256) -> U256 {
    // value < R and R² mod modulus < modulus, so their product meets the
    // reduction bound even when value is not reduced.
    mul(modulus, value, &r2(modulus))
}

/// Converts an unsigned 64-bit integer to a reduced Montgomery residue.
///
/// This is [`from_u256`] for a single-word input. Any `u64` value is
/// accepted, including values greater than or equal to `modulus`.
///
/// # Panics
///
/// Panics if [`assert_modulus`] rejects `modulus`.
pub const fn from_u64(modulus: &U256, value: u64) -> U256 {
    from_u256(modulus, &[value, 0, 0, 0])
}

/// Decodes a reduced Montgomery residue into an ordinary unsigned integer.
///
/// Returns `value * R^-1 mod modulus`, strictly below `modulus`. The input
/// must be strictly below `modulus` and is interpreted as Montgomery form.
///
/// # Panics
///
/// Panics if [`assert_modulus`] rejects `modulus`, or `value >= modulus`.
pub const fn to_u256(modulus: &U256, value: &U256) -> U256 {
    assert_modulus(modulus);
    assert!(!ge(value, modulus), "value must be reduced");
    reduce_wide(
        modulus,
        &[value[0], value[1], value[2], value[3], 0, 0, 0, 0],
    )
}

/// Raises a Montgomery residue to an unsigned integer power.
///
/// `base` must be in Montgomery form and strictly below `modulus`.
/// `exponent` is an ordinary 256-bit integer. The result is
/// reduced and in Montgomery form. A zero exponent returns the
/// [Montgomery representation of one](one), including when `base`
/// is zero.
///
/// # Panics
///
/// Panics if [`assert_modulus`] rejects `modulus`, or `base >= modulus`,
/// including when the exponent is zero.
pub const fn pow(modulus: &U256, base: &U256, exponent: &U256) -> U256 {
    let mut accumulator = one(modulus);
    assert!(!ge(base, modulus), "base must be reduced");
    let mut bit = 256;
    while bit > 0 {
        bit -= 1;
        accumulator = mul(modulus, &accumulator, &accumulator);
        if exponent[bit / 64] >> (bit % 64) & 1 == 1 {
            accumulator = mul(modulus, &accumulator, base);
        }
    }
    accumulator
}

/// Inverts a nonzero field element in Montgomery form.
///
/// The caller must supply a prime `modulus` and a nonzero Montgomery residue
/// `value` strictly below it. The result is the reduced Montgomery
/// representation of the inverse, computed by raising the represented field
/// element to `modulus - 2` using Fermat's little theorem.
///
/// Primality is not checked. For a composite modulus, the result is not
/// guaranteed to be an inverse even when `value` is invertible.
///
/// # Panics
///
/// Panics if [`assert_modulus`] rejects `modulus`, or `value` is zero or
/// greater than or equal to `modulus`.
///
/// # Examples
///
/// ```
/// # use zakura_bento_core as bento;
/// use bento::const_arithmetic::{U256, m255};
///
/// const MODULUS: U256 = [97, 0, 0, 0];
/// const VALUE: U256 = m255::from_u64(&MODULUS, 7);
/// const INVERSE: U256 = m255::invert_prime(&MODULUS, &VALUE);
/// assert_eq!(INVERSE, m255::from_u64(&MODULUS, 14)); // 7 * 14 mod 97 = 1
/// assert_eq!(m255::mul(&MODULUS, &VALUE, &INVERSE), m255::one(&MODULUS));
/// ```
pub const fn invert_prime(modulus: &U256, value: &U256) -> U256 {
    assert_modulus(modulus);
    assert!(
        value[0] | value[1] | value[2] | value[3] != 0,
        "cannot invert zero"
    );
    pow(modulus, value, &sub_u64(modulus, 2))
}
