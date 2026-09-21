//! Pasta field and curve arithmetic and allocation-free field FFTs.
//!
//! [`field::Fp`] and [`field::Fq`] provide field arithmetic, canonical encodings,
//! inversion, square roots, and product sums without allocation. Constants and
//! fixed exponentiation schedules use this workspace's `bento` support.
//! [`curve`] provides Pallas and Vesta points, canonical encodings,
//! GLV scalar multiplication, batch normalization, and borrowed compact and
//! expanded fixed-base tables. [`curve::msm`] sums dense or indexed inputs with
//! caller-owned scratch and execution.
//! [`fft`] provides power-of-two transforms, cosets, residue expansion, and fused
//! interpolation with caller-owned tables, buffers, scratch, and execution.
//! [`exec`] provides scoped fork/join, task budgets, and borrowed work helpers
//! shared by arithmetic and downstream workloads. [`exec::run`] supplies bounded
//! task claims and typed admission for application schedulers; [`curve::msm::run`]
//! and [`fft::run`] expose incremental arithmetic with exclusively leased scratch.
//! Field elements, nonidentity [`curve::AffinePoint`] values, and cached
//! [`curve::PreparedAffinePoint`] entries implement [`bento::Pod`] for direct
//! embedded storage. Their type docs distinguish mathematical invariants from
//! memory validity. [`stored_form!`] names the field representation; artifact
//! owners identify the curve and record schema.
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
//!
//! # Features
//!
//! Curves and FFTs are always available without feature flags. All current APIs
//! work without an allocator.
//!
//! By default, [`field::PastaField::sqrt`] uses small tables of roots of unity.
//! Enabling `sqrt-table-large` selects a larger table algorithm that reduces
//! work for many square inputs, at the cost of additional static storage.
//! Performance depends on the input and target. Both configurations use
//! compile-time tables without allocation or runtime initialization, and
//! preserve the same public API and stored field representation.
//!
//! Cargo features are additive: any consumer enabling `sqrt-table-large`
//! selects it for that Udon build.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod curve;
pub mod exec;
pub mod fft;
pub mod field;
mod stored_form;

pub use stored_form::{STORED_FORM, StoredForm};

// Keep macro support anchored to Udon through dependency aliases and re-exports.
// These expose only Bento's const-enforcing macros, never arithmetic functions.
#[doc(hidden)]
pub use bento::const_arithmetic::{
    m255::from_u256 as __m255_from_u256,
    u256::{from_hex as __u256_from_hex, ge as __u256_ge},
};

#[cfg(test)]
extern crate std;

#[cfg(test)]
mod test_support;

#[cfg(test)]
extern crate self as zakura_udon;
#[cfg(test)]
#[path = "../tests/execution/frontier.rs"]
mod frontier_tests;
#[cfg(test)]
#[path = "../tests/execution/mixed.rs"]
mod mixed_tests;
