//! Montgomery-form arithmetic for the two Pasta prime fields.
//!
//! [`field::Fp`] and [`field::Fq`] provide field arithmetic, canonical encodings,
//! inversion, square roots, and product sums without allocation. Constants and
//! fixed exponentiation schedules use this workspace's `bento` support.
//!
//! Arithmetic is variable-time and provides no constant-time guarantee for
//! secret inputs.
//!
//! ```
//! use zakura_udon::{field::Fp, fp_hex};
//!
//! let value = fp_hex!("0x0000000000000000000000000000000000000000000000000000000000000007");
//! assert_eq!(value.mul(&Fp::from_u64(3)), Fp::from_u64(21));
//! assert_eq!(value.mul(&value.invert().unwrap()), Fp::ONE);
//! assert_eq!(Fp::from_bytes(value.to_bytes()), Some(value));
//! ```

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod field;

// Keep macro support anchored to Udon through dependency aliases and re-exports.
// These expose only Bento's const-enforcing macros, never arithmetic functions.
#[doc(hidden)]
pub use bento::const_arithmetic::{
    m255::from_u256 as __m255_from_u256,
    u256::{from_hex as __u256_from_hex, ge as __u256_ge},
};

#[cfg(test)]
extern crate std;
