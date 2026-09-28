//! Compile-time integer and Montgomery arithmetic for public parameters.
//!
//! Each macro evaluates its arguments and its result inside `const { ... }`.
//! Arguments may be literals, constants, associated constants (including those
//! of generic types), const generic parameters, or other constant expressions.
//! They cannot capture local variables or function parameters, even inside a
//! `const fn` or another const block. Use named constants or nested macro calls
//! to compose calculations. Failed input checks produce compilation errors,
//! including in runtime expressions.
//!
//! [`u256`] uses full-width unsigned integers; [`m255`] uses Montgomery residues
//! for odd moduli below `2^255`. These are reference algorithms for public
//! parameters and provide no constant-time guarantee. Runtime arithmetic belongs
//! in the consuming crate. The facade exposes macros and storage aliases;
//! implementation functions and arithmetic contexts remain in `bento-core`.
//!
//! ```
//! # use zakura_bento as bento;
//! use bento::const_arithmetic::{U256, m255};
//!
//! const MODULUS: U256 = [97, 0, 0, 0];
//! const A: U256 = m255::from_u64!(&MODULUS, 7);
//! const B: U256 = m255::from_u64!(&MODULUS, 15);
//! const PRODUCT: U256 = m255::mul!(&MODULUS, &A, &B);
//! assert_eq!(m255::to_u256!(&MODULUS, &PRODUCT), [8, 0, 0, 0]);
//! ```
//!
//! ```compile_fail
//! # use zakura_bento::const_arithmetic::u256;
//! fn shift(value: [u64; 4]) -> [u64; 4] {
//!     u256::shr!(&value, 1)
//! }
//! ```

/// An unsigned 256-bit integer stored as four little-endian `u64` words.
///
/// Index zero is least significant. All 256 bits are available. This alias also
/// stores exponents and residues; it enforces no modulus or Montgomery factor.
pub type U256 = bento_core::const_arithmetic::U256;

/// An unsigned 320-bit integer stored as five little-endian `u64` words.
///
/// Index zero is least significant. Used for [`u256::round_shifted_ratio!`].
pub type U320 = bento_core::const_arithmetic::U320;

/// An unsigned 512-bit integer stored as eight little-endian `u64` words.
///
/// Index zero is least significant. Holds the exact product of two [`U256`]
/// values, as returned by [`u256::mul_wide!`].
pub type U512 = bento_core::const_arithmetic::U512;

pub mod m255;
pub mod u256;
