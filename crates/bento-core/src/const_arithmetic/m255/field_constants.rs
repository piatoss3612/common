//! Field constants derived from a modulus, generator, and two-adicity.
//!
//! Exponents and cofactors use ordinary integer limbs; field-valued results
//! use Montgomery form. Primality and the generator's order are properties of
//! the caller's field parameters, not established by these derivations.

use super::super::U256;
use super::super::u256::{add_with_carry, div_exact_u64, odd_cofactor, shr, sub_u64};
use super::{MontgomeryContext, assert_modulus};

/// Derives a root of unity for the field's subgroup of power-of-two order.
///
/// Returns the reduced Montgomery representation of `generator^t`, where
/// `t = (modulus - 1) / 2^two_adicity` is the [`odd_cofactor`]. `generator` is
/// an ordinary integer, interpreted modulo `modulus`.
///
/// The result has order exactly `2^two_adicity` when `modulus` is prime and
/// `generator` generates its nonzero residues under multiplication. Primality
/// and generator order are not checked. The factorization supplied through
/// `two_adicity` is checked by [`odd_cofactor`].
///
/// # Panics
///
/// Panics if `modulus` is outside the
/// [supported domain](super::assert_modulus), or [`odd_cofactor`]
/// rejects `two_adicity`.
///
/// # Examples
///
/// ```
/// # use zakura_bento_core as bento;
/// use bento::const_arithmetic::{U256, m255};
///
/// // 97 - 1 = 3 * 2^5, and 5 generates the multiplicative group.
/// const MODULUS: U256 = [97, 0, 0, 0];
/// const ROOT: U256 = m255::two_adic_root_of_unity(&MODULUS, 5, 5);
/// assert_eq!(ROOT, m255::from_u64(&MODULUS, 28)); // 5^3 mod 97
/// assert_eq!(
///     m255::pow(&MODULUS, &ROOT, &[32, 0, 0, 0]),
///     m255::one(&MODULUS),
/// );
/// ```
pub const fn two_adic_root_of_unity(modulus: &U256, generator: u64, two_adicity: u32) -> U256 {
    assert_modulus(modulus);
    let exponent = odd_cofactor(modulus, two_adicity);
    let context = MontgomeryContext::new(*modulus);
    let base = context.from_u64(generator);
    context.pow(&base, &exponent)
}

/// Derives a generator for the field's subgroup of odd order.
///
/// Returns the reduced Montgomery representation of `generator^(2^two_adicity)`.
/// `generator` is an ordinary integer, interpreted modulo `modulus`.
///
/// For the subgroup guarantee, `modulus` must be prime, `generator` must
/// generate its nonzero residues under multiplication, and
/// `modulus - 1 = t * 2^two_adicity` must hold with `t` odd. The result then
/// has order `t`. Primality and generator order are not checked. The
/// factorization supplied through `two_adicity` is checked by [`odd_cofactor`].
///
/// # Panics
///
/// Panics if `modulus` is outside the
/// [supported domain](super::assert_modulus), or [`odd_cofactor`]
/// rejects `two_adicity`.
pub const fn odd_order_generator(modulus: &U256, generator: u64, two_adicity: u32) -> U256 {
    assert_modulus(modulus);
    let _ = odd_cofactor(modulus, two_adicity);
    let context = MontgomeryContext::new(*modulus);
    let mut value = context.from_u64(generator);
    let mut squaring = 0;
    while squaring < two_adicity {
        value = context.mul(&value, &value);
        squaring += 1;
    }
    value
}

/// Derives a primitive cube root of unity in Montgomery form.
///
/// Returns the reduced Montgomery representation of
/// `generator^((modulus - 1) / 3)`. `generator` is an ordinary integer,
/// interpreted modulo `modulus`, and `modulus - 1` must be divisible by three.
///
/// The result has order exactly three when `modulus` is prime and `generator`
/// generates its nonzero residues under multiplication. Primality and generator
/// order are not checked. The other primitive cube root is this one's square;
/// callers choose which root their field or protocol uses.
///
/// # Panics
///
/// Panics if `modulus` is outside the
/// [supported domain](super::assert_modulus), or `modulus - 1` is
/// not divisible by three.
pub const fn cube_root_of_unity(modulus: &U256, generator: u64) -> U256 {
    assert_modulus(modulus);
    let exponent = div_exact_u64(&sub_u64(modulus, 1), 3);
    let context = MontgomeryContext::new(*modulus);
    let base = context.from_u64(generator);
    context.pow(&base, &exponent)
}

/// Returns the reduced Montgomery representation of the inverse of two.
///
/// The represented integer is `(modulus + 1) / 2`. Primality is not required;
/// two is invertible modulo every supported odd modulus.
///
/// # Panics
///
/// Panics if `modulus` is outside the
/// [supported domain](super::assert_modulus).
pub const fn two_inverse(modulus: &U256) -> U256 {
    assert_modulus(modulus);
    // For odd p, (p + 1) / 2 = (p >> 1) + 1. The four-limb sum cannot
    // overflow because p >> 1 < 2^254.
    let (half_plus_one, carry) = add_with_carry(&shr(modulus, 1), &[1, 0, 0, 0]);
    assert!(carry == 0);
    MontgomeryContext::new(*modulus).from_u256(&half_plus_one)
}
