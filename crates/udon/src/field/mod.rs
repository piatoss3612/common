//! Field contracts, generic operations, and optimized Pasta arithmetic.
//!
//! One generic [`PastaField`] serves both fields through the [`PrimeModulus`]
//! marker types, keeping the base and scalar fields distinct at compile time
//! even though the curves form a cycle.
//!
//! ```compile_fail
//! use zakura_udon::field::{Fp, Fq};
//! let _ = Fp::ONE.add(&Fq::ONE);
//! ```
//!
//! A stored field value represents `x * R mod p`, where `R = 2^256`.
//! [`Loose`] values lie in `[0, 2p)` and [`Reduced`] values in `[0, p)`.
//! Protocol bytes and [`CanonicalUint`] instead represent `x` itself.
//! Arithmetic is variable-time; this module provides no constant-time guarantee
//! for secret inputs.
//!
//! [`fraction_prefixes`] computes running products of ordered fractions with
//! caller-owned scratch. [`ConstantPrefix`] borrows an explicit tail after a
//! repeated value for materialization or structured FFT and MSM operations.
//!
//! Operators forward to the inherent methods. [`Field`] describes arithmetic,
//! [`PrimeField`] adds canonical representations of field-specific widths,
//! and [`FftField`], [`CubeRootField`], and [`DeferredField`] describe optional
//! arithmetic capabilities to generic code. Both Pasta fields
//! implement them through the same kernels.

// Generic contracts and helpers; concrete storage and kernels stay in Pasta.
mod batch;
mod encoding;
pub(crate) mod pasta;
mod products;
mod traits;

#[cfg(test)]
mod tests;

pub(crate) use batch::NonzeroInversionLanes;
pub use batch::{batch_invert, batch_invert_groups, batch_invert_with_scratch};
pub use encoding::{low_u64, random};
#[cfg(test)]
pub(crate) use pasta::count_inversions;
pub use pasta::{
    BatchInversionError, CanonicalUint, Fp, Fq, Loose, PallasBase, PallasScalar, PastaField,
    PrimeModulus, ProductSum, Reduced, ReductionState, try_batch_invert_by,
};
pub(crate) use pasta::{invert_nonzero, word};
pub use products::dot;
pub use traits::{CubeRootField, DeferredField, FftField, Field, PrimeField};

pub(crate) use pasta::fill_powers;
pub use pasta::{
    ConstantPrefix, ConstantPrefixError, FractionPrefixError, batch_invert_groups_scaled,
    batch_invert_scaled, fraction_prefixes, fraction_prefixes_in_place, try_batch_invert_scaled_by,
};
