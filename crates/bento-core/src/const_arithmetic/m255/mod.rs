//! Montgomery arithmetic for odd moduli below `2^255`, using four `u64` words.
//!
//! A value `x` is represented by the reduced integer `x * R mod p`, where
//! `R = 2^256` and `p` is the modulus. The modulus capacity is 255 bits,
//! including smaller moduli. The unused top bit lets reduced sums fit a
//! [`U256`](super::U256) and bounds carries during [`reduce_wide`].
//!
//! [`from_u256`] and [`from_u64`] enter this representation; [`to_u256`]
//! decodes it. For Montgomery operands, [`mul`] and [`pow`] preserve this
//! representation. Field-valued derivations also return Montgomery residues.
//! [`add`] preserves either ordinary or Montgomery form.
//! The parameter helpers [`pow2_mod`] and [`r2`] explicitly return ordinary
//! residues, and exponents supplied to [`pow`] are ordinary full-width integers.
//!
//! All operations taking a modulus check `2 < p < 2^255` and oddness with
//! [`assert_modulus`], even for a zero exponent or an empty table.
//! [`reduction_coefficient`] takes only an odd low word, without a full modulus.
//! Prime inversion and subgroup-order guarantees have additional mathematical
//! assumptions documented on their functions; primality and generator order
//! are not checked.
//!
//! See the [parent module example](super#example) for conversion and
//! multiplication. These functions have no constant-time or optimized-runtime
//! guarantee.

mod field_constants;
mod modular;
mod montgomery;
mod tables;

pub use field_constants::{
    cube_root_of_unity, odd_order_generator, two_adic_root_of_unity, two_inverse,
};
pub use modular::{add, assert_modulus, pow2_mod};
pub use montgomery::{
    MontgomeryContext, from_u64, from_u256, invert_prime, mul, one, pow, r2, reduce_wide,
    reduction_coefficient, to_u256,
};
pub use tables::{inverse_powers_of_two, powers, safegcd_corrections_62_64, two_adic_root_tables};

#[cfg(test)]
mod tests;
