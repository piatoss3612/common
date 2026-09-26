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
//! Operators forward to the inherent methods. The unstable `traits` feature
//! adds the `Field` interface for arithmetic, canonical encodings, transforms,
//! and product accumulation, together with generic sampling, encoding, and
//! dot-product helpers. The Pasta implementations use the same native kernels.

// Generic contracts and helpers; concrete storage and kernels stay in Pasta.
#[cfg(feature = "traits")]
mod encoding;
pub(crate) mod pasta;
#[cfg(feature = "traits")]
mod products;
#[cfg(feature = "traits")]
mod traits;

#[cfg(all(test, feature = "traits"))]
mod tests;

#[cfg(feature = "traits")]
pub use encoding::{low_u64, random};
pub(crate) use pasta::NonzeroInversionLanes;
#[cfg(test)]
pub(crate) use pasta::count_inversions;
pub use pasta::{
    BatchInversionError, CanonicalUint, Fp, Fq, Loose, PallasBase, PallasScalar, PastaField,
    PrimeModulus, ProductSum, Reduced, ReductionState, batch_invert, batch_invert_groups,
    try_batch_invert_by,
};
pub(crate) use pasta::{invert_nonzero, word};
#[cfg(feature = "traits")]
pub use products::{dot, dot_iter};
#[cfg(feature = "traits")]
pub use traits::Field;

pub(crate) use pasta::fill_powers;
pub use pasta::{
    ConstantPrefix, ConstantPrefixError, FractionPrefixError, batch_invert_groups_scaled,
    batch_invert_scaled, fraction_prefixes, fraction_prefixes_in_place, try_batch_invert_scaled_by,
};
