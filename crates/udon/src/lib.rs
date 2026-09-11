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

#[cfg(test)]
extern crate std;
