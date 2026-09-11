//! Reference integer and Montgomery arithmetic for constant derivation.
//!
//! Use these functions with public parameters during constant evaluation, or
//! as references when testing runtime arithmetic. They provide no constant-time
//! or optimized-runtime guarantee. A panic during constant evaluation causes a
//! compilation error.
//!
//! [`u256`] operates on unsigned integers stored as four little-endian `u64`
//! words, including full-width inputs, exponents, and rounded ratios.
//! [`m255`] uses that storage for Montgomery arithmetic with an odd modulus
//! below `2^255`. Its radix is `2^256`; the spare bit accommodates carries.
//!
//! The array aliases describe storage, without enforcing reduction or
//! distinguishing ordinary integers from Montgomery residues. Each operation
//! documents its representations and bounds. Field derivations additionally
//! rely on caller-supplied primality and generator-order assumptions.
//!
//! # Example
//!
//! Convert ordinary integers into Montgomery form before multiplying, then
//! decode the result back to an ordinary integer:
//!
//! ```
//! # use zakura_bento_core as bento;
//! use bento::const_arithmetic::{U256, m255};
//!
//! const MODULUS: U256 = [97, 0, 0, 0];
//! const A: U256 = m255::from_u64(&MODULUS, 7);
//! const B: U256 = m255::from_u64(&MODULUS, 15);
//! const PRODUCT: U256 = m255::mul(&MODULUS, &A, &B);
//! const ORDINARY: U256 = m255::to_u256(&MODULUS, &PRODUCT);
//! assert_eq!(ORDINARY, [8, 0, 0, 0]); // 7 * 15 mod 97
//! ```

/// An unsigned 256-bit integer stored as four little-endian `u64` words.
///
/// Index zero is least significant. All 256 bits are available. This alias also
/// stores exponents and residues; it enforces no modulus or Montgomery factor.
pub type U256 = [u64; 4];

/// An unsigned 320-bit integer stored as five little-endian `u64` words.
///
/// Index zero is least significant. Used for [`u256::round_shifted_ratio`]
/// results, whose width can exceed that of the denominator.
pub type U320 = [u64; 5];

/// An unsigned 512-bit integer stored as eight little-endian `u64` words.
///
/// Index zero is least significant. Holds the exact product of two [`U256`]
/// values, as returned by [`u256::mul_wide`].
pub type U512 = [u64; 8];

pub mod m255;
pub mod u256;

mod macros;
mod word;

#[cfg(test)]
mod test_support;
