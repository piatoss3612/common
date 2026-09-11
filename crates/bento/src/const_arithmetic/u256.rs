//! Compile-time arithmetic on full-width unsigned 256-bit integers.
//!
//! Operands use four little-endian `u64` words, with no attached modulus or
//! Montgomery factor. Every macro requires constant expressions; see the
//! [parent module](super) for composition and evaluation rules.

#[doc(inline)]
pub use bento_core::{
    u256_add_with_carry as add_with_carry, u256_div_exact_u64 as div_exact_u64,
    u256_from_hex as from_hex, u256_ge as ge, u256_mul_wide as mul_wide,
    u256_odd_cofactor as odd_cofactor, u256_round_shifted_ratio as round_shifted_ratio,
    u256_shr as shr, u256_sub_u64 as sub_u64, u256_sub_with_borrow as sub_with_borrow,
    u256_tonelli_shanks_exponent as tonelli_shanks_exponent,
};
