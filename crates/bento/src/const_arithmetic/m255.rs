//! Compile-time Montgomery arithmetic for odd moduli `2 < p < 2^255`.
//!
//! A Montgomery residue represents `x` as the reduced integer `x * 2^256 mod p`,
//! stored in four little-endian `u64` words. [`from_u256!`] and [`from_u64!`]
//! enter this representation; [`to_u256!`] decodes it. [`mul!`] and [`pow!`]
//! preserve it for Montgomery operands. Field derivations return Montgomery
//! residues; [`pow2_mod!`] and [`r2!`] return ordinary residues, and exponents
//! are ordinary full-width integers.
//!
//! Every macro requires constant expressions; see the [parent module](super).
//! Moduli, operand bounds, and factorizations are checked during compilation.
//! Primality and generator order remain the caller's responsibility.
//!
//! ```
//! # use zakura_bento::const_arithmetic::{U256, m255};
//! const MODULUS: U256 = [97, 0, 0, 0];
//! const ROOT: U256 = m255::two_adic_root_of_unity!(&MODULUS, 5, 5);
//! const INVERSE: U256 = m255::invert_prime!(&MODULUS, &ROOT);
//! assert_eq!(m255::to_u256!(&MODULUS, &ROOT), [28, 0, 0, 0]);
//! assert_eq!(m255::mul!(&MODULUS, &ROOT, &INVERSE), m255::one!(&MODULUS));
//! ```
//!
//! Table lengths are inferred from the result type or supplied after a
//! semicolon. The [root tables](two_adic_root_tables!) include every order from
//! `2^0` through `2^two_adicity`, so their length must be `two_adicity + 1`:
//!
//! ```
//! # use zakura_bento::const_arithmetic::{U256, m255};
//! const MODULUS: U256 = [97, 0, 0, 0];
//! const TABLES: ([U256; 6], [U256; 6]) = m255::two_adic_root_tables!(&MODULUS, 5, 5);
//! assert_eq!(m255::two_adic_root_tables!(&MODULUS, 5, 5; 6), TABLES);
//! assert_eq!(TABLES.0[0], m255::one!(&MODULUS));
//! assert_eq!(m255::mul!(&MODULUS, &TABLES.0[5], &TABLES.1[5]), TABLES.0[0]);
//! ```

#[doc(inline)]
pub use bento_core::{
    m255_add as add, m255_assert_modulus as assert_modulus,
    m255_cube_root_of_unity as cube_root_of_unity, m255_from_u64 as from_u64,
    m255_from_u256 as from_u256, m255_inverse_powers_of_two as inverse_powers_of_two,
    m255_invert_prime as invert_prime, m255_mul as mul,
    m255_odd_order_generator as odd_order_generator, m255_one as one, m255_pow as pow,
    m255_pow2_mod as pow2_mod, m255_r2 as r2, m255_reduce_wide as reduce_wide,
    m255_reduction_coefficient as reduction_coefficient,
    m255_safegcd_corrections_62_64 as safegcd_corrections_62_64, m255_to_u256 as to_u256,
    m255_two_adic_root_of_unity as two_adic_root_of_unity,
    m255_two_adic_root_tables as two_adic_root_tables, m255_two_inverse as two_inverse,
};
