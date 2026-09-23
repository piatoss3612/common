//! Power-of-two field transforms with caller-owned tables, buffers, and execution.
//!
//! [`Transform`] binds a domain and borrowed tables, with synchronous transform
//! conveniences. [`execution::FftPlan`] fixes reusable transform semantics and geometry;
//! its working buffers are borrowed only for execution. [`Expansion`]
//! evaluates a base polynomial on a larger coset without constructing a full
//! zero-padded transform. [`execution::InterpolationPlan`] combines interpolations from
//! several domains into one coefficient vector.
//! [`CosetDomain::evaluate_lagrange`] evaluates a requested natural-index basis
//! range at a field point, with caller-owned output and inversion scratch.
//! [`VanishingDivision`] finishes transformed numerator evaluations into
//! coefficient pieces after division by a subgroup vanishing polynomial.
//! [`ConstantPrefixExpansion`] reuses Lagrange samples to extend evaluations with
//! a constant prefix; [`CosetDomain::interpolate_constant_prefix`] recovers their
//! coefficients directly.
//!
//! [`Domain`] dispatches field transforms through [`crate::field::FftField`]
//! and evaluates vanishing and Lagrange polynomials. Other
//! [`reference::Butterfly`] values default to the [`mod@reference`] transforms.
//!
//! Setup and execution never allocate. Tables may be prepared into mutable
//! slices or borrowed from downstream Bento POD artifacts. Shared resource limits
//! come from [`ExecutionOptions`](crate::exec::ExecutionOptions). Udon chooses
//! arithmetic schedules within those limits. The scoped [`Executor`] supplies parallel
//! execution without requiring a particular runtime or an allocator in Udon.
//! An executor's own allocations are outside Udon's storage requirements.
//! [`execution::FftPlan`] fixes order, normalization, and transform geometry.
//! [`execution::ExpansionPlan`] additionally fixes coefficient storage and residue
//! ordering. Their drivers execute synchronously or expose incremental tasks.
//!
//! Field arithmetic is variable-time, with no constant-time guarantee for
//! secret inputs. Field buffers and table contents use [`PastaField`]'s loose
//! Montgomery representation. Arithmetic preserves its bound at every step.
//!
//! # Validation and working storage
//!
//! Transform construction validates configuration without binding working buffers.
//! Callers must supply working storage matching the resolved operation or task request.
//! Synchronous drivers assert buffer and scratch requirements before mutation.
//! Incremental tasks assert their own resource requirements before writing; a later
//! task's panic does not undo earlier writes. Publishing a failed or cancelled task
//! poisons the run and permits outstanding receipts to drain. Invalid input prefixes
//! report the supported length range separately from unsupported domain sizes.
//!
//! Transform table binding, such as [`Tables::bind`], checks dimensions and trusts
//! contents constructed by the caller. These binders and transform execution use
//! stored entries directly, without validating or reducing them.
//! Preparation returns immutable handles. Configuration checks table
//! compatibility; the generic [`mod@reference`] transforms have their own contracts.
//!
//! Scratch consists of initialized field elements. Its initial values do not
//! affect the result, and it may be reused after execution. Elements beyond the
//! reported requirement remain untouched. A panic may leave partial results;
//! modified buffers still contain loose field representations after unwinding,
//! without a cleanup pass. Custom executors must uphold [`Executor`]'s
//! completion contract on panic as well as on success.
//!
//! # Examples
//!
//! A transform can execute with no precomputation or scratch:
//!
//! ```
//! use zakura_udon::{
//!     exec::{ExecutionOptions, SerialExecutor},
//!     field::Fp,
//!     fft::{Domain, Transform},
//! };
//!
//! let transform = Transform::new(Domain::new(2)?.subgroup());
//! let original = [Fp::ONE, Fp::from_u64(2), Fp::ZERO, Fp::ZERO];
//! let mut values = original;
//! transform.forward(&mut values, ExecutionOptions::default(), &SerialExecutor, &mut [])?;
//! transform.inverse(&mut values, ExecutionOptions::default(), &SerialExecutor, &mut [])?;
//! assert_eq!(values.map(|value| value.reduce()), original.map(|value| value.reduce()));
//! # Ok::<(), zakura_udon::fft::FftError>(())
//! ```
//!
//! Prepare optional tables once into caller-owned arrays. Preparation returns
//! the same transform handle used for execution:
//!
//! ```
//! use zakura_udon::{
//!     exec::{ExecutionOptions, SerialExecutor},
//!     field::Fq,
//!     fft::{Domain, TableRequirements, TablesMut},
//! };
//!
//! const SIZE: usize = 8;
//! const TABLES: TableRequirements = match TableRequirements::for_size(SIZE) {
//!     Ok(required) => required,
//!     Err(_) => panic!("unsupported table size"),
//! };
//! let domain = Domain::for_size(SIZE)?.coset();
//! let mut forward = [Fq::ZERO; TABLES.twiddles];
//! let mut inverse = [Fq::ZERO; TABLES.twiddles];
//! let transform = TablesMut {
//!     forward: Some(&mut forward),
//!     inverse: Some(&mut inverse),
//!     ..TablesMut::default()
//! }.prepare(domain);
//! let coefficients = [Fq::ONE; SIZE];
//! let mut values = coefficients;
//! transform.forward(&mut values, ExecutionOptions::default(), &SerialExecutor, &mut [])?;
//! transform.inverse(&mut values, ExecutionOptions::default(), &SerialExecutor, &mut [])?;
//! assert_eq!(values.map(|value| value.reduce()), coefficients.map(|value| value.reduce()));
//! # Ok::<(), zakura_udon::fft::FftError>(())
//! ```
//!
//! Keep bit-reversed evaluations through a product and feed them directly into
//! interpolation. These orders describe the mathematical layout; Udon selects
//! the arithmetic schedule:
//!
//! ```
//! use zakura_udon::{
//!     field::Fp,
//!     exec::{ExecutionOptions, SerialExecutor},
//!     fft::{Direction, Domain, Expansion, ExpansionOrder, ExpansionStorage,
//!         ElementOrder, InputSupport, StorageLayout, Transform, TransformRequest,
//!         execution::{ExpansionPlan, FftPlan}},
//! };
//!
//! let options = ExecutionOptions::default();
//! let base = Transform::new(Domain::new(2)?.subgroup());
//! let extended = Domain::new(3)?.coset();
//! let expansion = ExpansionPlan::new(
//!     Expansion::new(base, extended, None)?,
//!     ExpansionStorage::Coefficients, ExpansionOrder::BitReversed,
//!     InputSupport::Prefix(2), ElementOrder::Natural,
//!     StorageLayout::Contiguous, options,
//! )?;
//! let coefficients = [Fp::ONE, Fp::from_u64(2)];
//! let mut factor = [Fp::ZERO; 8];
//! expansion.execute(&coefficients, &mut factor, &mut [], None,
//!     &mut [], &SerialExecutor);
//! let mut product = [Fp::ZERO; 8];
//! expansion.execute(&coefficients, &mut product, &mut [], Some(&factor),
//!     &mut [], &SerialExecutor);
//! let inverse = FftPlan::new(Transform::new(extended),
//!     TransformRequest {
//!         input_order: ElementOrder::BitReversed,
//!         ..TransformRequest::new(Direction::Inverse)
//!     }, StorageLayout::Contiguous, options,
//! )?;
//! inverse.execute(None, &mut product, None, &mut [], &SerialExecutor);
//! let expected = [Fp::ONE, Fp::from_u64(4), Fp::from_u64(4)];
//! for (value, expected) in product[..3].iter().zip(expected) {
//!     assert_eq!(value.reduce(), expected);
//! }
//! assert!(product[3..].iter().all(|value| value.is_zero()));
//! # Ok::<(), zakura_udon::fft::FftError>(())
//! ```

use crate::checks::assert_length;
use crate::exec::Executor;
use crate::field::{PastaField, PrimeModulus};

mod constant_prefix;
mod domain;
pub mod execution;
mod expansion;
mod expansion_operation;
mod expansion_scales;
mod factors;
mod interpolation;
mod lagrange;
mod layout;
mod operation;
mod planning;
mod powers;
pub mod reference;
mod stages;
mod tables;
mod transform;
mod vanishing;

pub use constant_prefix::ConstantPrefixExpansion;
pub use domain::{CosetDomain, Domain};
pub use expansion::Expansion;
pub use expansion_operation::{ExpansionOrder, ExpansionStorage, Residue};
pub use expansion_scales::{ExpansionScaleNormalization, ExpansionScales};
pub use interpolation::ClassState;
use interpolation::{Class, interpolate_classes, interpolate_sum};
pub use lagrange::{LagrangeCompletion, LagrangeError};
pub use layout::{
    CoefficientView, ElementOrder, EvaluationLayout, EvaluationView, InverseScale, ResidueLayout,
};
pub use operation::{Direction, InputStorage, InputSupport, StorageLayout, TransformRequest};
use planning::Strategy;
use planning::check_scratch;
pub use powers::{TwiddleDescription, TwiddleStorage, TwiddleTable};
use stages::Codelet;
pub use tables::{TableRequirements, Tables, TablesMut};
pub use transform::Transform;
pub use vanishing::{VanishingDivision, VanishingFactors};

/// An invalid FFT configuration or workspace limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftError {
    /// A domain size is unsupported.
    InvalidSize,
    /// A declared input-prefix length is outside the supported range.
    InvalidPrefix {
        /// Minimum supported length, in field elements.
        min: usize,
        /// Maximum supported length, in field elements.
        max: usize,
        /// Supplied length, in field elements.
        actual: usize,
    },
    /// An element count, index calculation, or byte size overflowed.
    SizeOverflow,
    /// An execution setting or combination of request options is invalid.
    InvalidExecution,
    /// Required arithmetic workspace exceeds the caller's byte ceiling.
    MemoryLimit {
        /// Required bytes for the selected storage layout.
        required: usize,
        /// Caller-provided byte ceiling.
        limit: usize,
    },
    /// A layout, range, or domain relationship is invalid.
    InvalidLayout,
    /// A class is larger than the output domain.
    InvalidClass,
    /// Interpolation has already begun on a class, consuming its evaluations.
    InvalidClassState,
}

impl core::fmt::Display for FftError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidSize => f.write_str("unsupported FFT domain size"),
            Self::InvalidPrefix { min, max, actual } => write!(
                f,
                "input prefix must contain {min}..={max} elements, received {actual}"
            ),
            Self::SizeOverflow => f.write_str("FFT storage or index size overflow"),
            Self::MemoryLimit { required, limit } => write!(
                f,
                "FFT workspace requires {required} bytes, limit is {limit}"
            ),
            Self::InvalidExecution => f.write_str("invalid FFT execution settings"),
            Self::InvalidLayout => f.write_str("invalid FFT layout or range"),
            Self::InvalidClass => f.write_str("interpolation class exceeds the output domain"),
            Self::InvalidClassState => f.write_str("class no longer contains evaluations"),
        }
    }
}

impl core::error::Error for FftError {}

fn check_prefix(actual: usize, min: usize, max: usize) -> Result<(), FftError> {
    if (min..=max).contains(&actual) {
        Ok(())
    } else {
        Err(FftError::InvalidPrefix { min, max, actual })
    }
}

const fn check_field_count(count: usize) -> Result<usize, FftError> {
    // Both sealed Pasta fields have the same four-limb representation.
    check_element_count::<crate::field::Fp>(count)
}

const fn check_element_count<T>(count: usize) -> Result<usize, FftError> {
    let size = core::mem::size_of::<T>();
    if size == 0 || count <= (isize::MAX as usize) / size {
        Ok(count)
    } else {
        Err(FftError::SizeOverflow)
    }
}

const fn check_domain_size(size: usize) -> Result<(), FftError> {
    if !size.is_power_of_two() || <crate::field::Fp>::root_of_unity(size.ilog2()).is_none() {
        return Err(FftError::InvalidSize);
    }
    match check_field_count(size) {
        Ok(_) => Ok(()),
        Err(error) => Err(error),
    }
}

// Ord::min is not const on the workspace's pinned toolchain.
const fn min(left: usize, right: usize) -> usize {
    if left < right { left } else { right }
}

/// Reverses the low `log_size` bits of `index`, discarding the others.
///
/// This is the permutation between natural and bit-reversed transform orders;
/// see [`ElementOrder`]. `log_size = 0` returns zero, and `log_size` of at
/// least `usize::BITS` reverses every bit.
pub fn bit_reverse(index: usize, log_size: u32) -> usize {
    if log_size == 0 {
        0
    } else if log_size >= usize::BITS {
        index.reverse_bits()
    } else {
        index.reverse_bits() >> (usize::BITS - log_size)
    }
}

#[cfg(test)]
mod tests;
