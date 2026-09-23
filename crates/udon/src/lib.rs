//! Pasta field and curve arithmetic and allocation-free field FFTs.
//!
//! [`field::Fp`] and [`field::Fq`] provide field arithmetic, canonical encodings,
//! inversion, square roots and ratios, and product sums without allocation.
//! [`field::PastaField::sqrt_alt`] also returns a root of a fixed nonsquare
//! multiple when the input is nonsquare. Constants and fixed exponentiation
//! schedules use this workspace's `bento` support.
//! [`curve`] provides Pallas and Vesta points, canonical encodings,
//! GLV scalar multiplication, batch normalization, and borrowed compact and
//! expanded fixed-base tables. [`curve::msm`] sums dense or indexed inputs with
//! caller-owned scratch and execution.
//! [`fft`] provides power-of-two transforms, cosets, residue expansion, and fused
//! interpolation with caller-owned tables, buffers, scratch, and execution.
//! [`polynomial`] combines borrowed coefficient slices with caller-supplied
//! weights and implicit zero extension, evaluates polynomials with Horner's rule
//! or retained powers, divides by monic polynomials with retained remainders,
//! constructs vanishing polynomials, and interpolates small distinct point sets.
//! [`exec`] provides scoped fork/join, task budgets, and borrowed work helpers
//! shared by arithmetic and downstream workloads. [`exec::run`] supplies bounded
//! task claims and typed admission for application schedulers; [`curve::msm::run`]
//! and [`fft::execution`] expose incremental arithmetic with exclusively leased scratch.
//! [`field::Field`], [`field::FftField`], and [`curve::Affine`] describe the
//! fields and curves to generic code, with operator forms forwarding to the
//! inherent arithmetic. [`fft::Domain`] runs the generic reference transforms
//! and evaluates vanishing and Lagrange polynomials, [`poly`] provides
//! polynomial utilities over the field traits, and [`poseidon`] carries
//! the Pasta Poseidon parameters.
//! Field elements, nonidentity [`curve::AffinePoint`] values, and cached
//! [`curve::PreparedAffinePoint`] entries implement [`bento::Pod`] for direct
//! embedded storage. Construction establishes their invariants; embedding
//! preserves their exact representations for immediate use. Field arithmetic
//! returns [`field::Loose`] values, which compare as field elements; explicit
//! reduction produces [`field::Reduced`] values for ordering and square roots.
//! [`stored_form!`] names the limb representation; artifact schemas identify
//! the field, reduction state, and curve.
//!
//! Arithmetic is variable-time and provides no constant-time guarantee for
//! secret inputs.
//!
//! ```
//! use zakura_udon::{field::Fp, fp_hex};
//!
//! let value = fp_hex!("0x0000000000000000000000000000000000000000000000000000000000000007");
//! assert_eq!(value.mul(&<Fp>::from_u64(3)).reduce(), <Fp>::from_u64(21).reduce());
//! assert_eq!(value.mul(&value.invert().unwrap()).reduce(), <Fp>::ONE.reduce());
//! assert_eq!(<Fp>::from_bytes(value.to_bytes()).unwrap().reduce(), value.reduce());
//! ```
//!
//! # Features
//!
//! Curves and FFTs are always available without feature flags. All current APIs
//! work without an allocator.
//!
//! By default, square roots and ratios use small tables of roots of unity.
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
#![warn(unreachable_pub)]

mod checks;
pub mod curve;
pub mod exec;
pub mod fft;
pub mod field;
pub mod poly;
pub mod polynomial;
pub mod poseidon;
mod stored_form;

pub use stored_form::STORED_FORM;

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

// The scheduler fixtures shared with `tests/execution/` name the crate by its
// package name; this alias lets `test_support` mount those same files.
#[cfg(test)]
extern crate self as zakura_udon;
