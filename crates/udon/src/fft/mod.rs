//! Power-of-two field transforms with caller-owned tables, buffers, and execution.
//!
//! [`Plan`] transforms coefficients and evaluations in place. [`Expansion`]
//! evaluates a base polynomial on a larger coset without constructing a full
//! zero-padded transform. [`interpolate_classes`] combines interpolations from
//! several domains into one coefficient vector.
//!
//! Setup and execution never allocate. Tables may be prepared into mutable
//! slices or borrowed from downstream Bento POD artifacts. [`ExecutionOptions`]
//! and [`ExpansionOptions`] determine the scratch requirements; their serial
//! settings need no scratch. The scoped [`Executor`] lets callers supply parallel
//! execution without requiring a particular runtime or an allocator in Udon.
//! An executor's own allocations are outside Udon's storage requirements.
//! [`Plan::configure`] fixes order, normalization, backend, and total resource
//! budgets for repeated transforms. [`Expansion::configure`] additionally fixes
//! coefficient storage and residue ordering. Both have const sizing descriptions.
//!
//! Field arithmetic is variable-time, with no constant-time guarantee for
//! secret inputs. Field buffers and table contents must satisfy
//! [`PastaField`]'s reduced Montgomery representation contract for correct
//! arithmetic; this is not a memory-safety requirement.
//!
//! # Validation and working storage
//!
//! The checked APIs validate sizes, execution settings, and scratch lengths
//! before modifying buffers; a returned [`FftError`] leaves them unchanged.
//! Length mismatches identify the buffer parameter or table field, along with
//! its expected and actual lengths. Invalid input prefixes report the
//! supported length range separately from unsupported domain sizes.
//!
//! Checked table binding, such as [`Tables::bind`], validates dimensions and
//! mathematical contents before returning a reusable handle. Native
//! preparation returns the same immutable handles without rescanning entries.
//! Explicit constructors such as [`Tables::bind_trusted`] rely on the caller
//! for correct contents, as documented by each table family;
//! invalid contents can cause incorrect results or panics. Configuration checks
//! compatibility, and execution does not revalidate entries. The generic
//! [`mod@reference`] transforms have their own contracts.
//!
//! Scratch consists of initialized field elements. Its initial values do not
//! affect the result, and it may be reused after execution. Elements beyond the
//! reported requirement remain untouched. A panic may leave partial results;
//! with valid input fields and tables, modified buffers contain reduced field
//! representations after unwinding. Custom executors must uphold [`Executor`]'s
//! completion contract on panic as well as on success.
//!
//! # Examples
//!
//! ```
//! use zakura_udon::{
//!     field::Fp,
//!     fft::{Domain, Plan},
//! };
//!
//! let domain = Domain::new(2).unwrap().subgroup();
//! let plan = Plan::without_tables(domain);
//! let original = [Fp::ONE, Fp::from_u64(2), Fp::ZERO, Fp::ZERO];
//! let mut values = original;
//! plan.forward_serial(&mut values).unwrap();
//! plan.inverse_serial(&mut values).unwrap();
//! assert_eq!(values, original);
//! ```
//!
//! Tables and scratch can also live in ordinary arrays. Prepare tables once,
//! then reuse the plan and scratch across transforms:
//!
//! ```
//! use zakura_udon::{
//!     field::Fq,
//!     fft::{Domain, ExecutionOptions, Plan, SerialExecutor, TableRequirements, TablesMut},
//! };
//!
//! const SIZE: usize = 8;
//! const TABLES: TableRequirements = match TableRequirements::for_size(SIZE) {
//!     Ok(required) => required,
//!     Err(_) => panic!("unsupported table size"),
//! };
//! const OPTIONS: ExecutionOptions = ExecutionOptions {
//!     tile_len: 2,
//!     columns_per_task: 1,
//!     max_tasks: 2,
//! };
//! const SCRATCH: usize = match OPTIONS.requirements(SIZE) {
//!     Ok(required) => required.field_elements,
//!     Err(_) => panic!("unsupported transform configuration"),
//! };
//! let domain = Domain::for_size(SIZE).unwrap().coset(Fq::from_u64(7)).unwrap();
//! let mut forward = [Fq::ZERO; TABLES.twiddles];
//! let mut inverse = [Fq::ZERO; TABLES.twiddles];
//! let tables = TablesMut {
//!     forward: Some(&mut forward),
//!     inverse: Some(&mut inverse),
//!     ..TablesMut::default()
//! }.prepare(domain).unwrap();
//! let plan = Plan::new(tables);
//! let mut scratch = [Fq::ZERO; SCRATCH];
//! let coefficients = [Fq::ONE; SIZE];
//! let mut values = coefficients;
//! plan.forward(&mut values, OPTIONS, &SerialExecutor, &mut scratch).unwrap();
//! plan.inverse(&mut values, OPTIONS, &SerialExecutor, &mut scratch).unwrap();
//! assert_eq!(values, coefficients);
//! ```
//!
//! Keep bit-reversed evaluations through a product and feed them directly into
//! interpolation. Expansion reverses both residue blocks and their inner rows:
//!
//! ```
//! use zakura_udon::{
//!     field::Fp,
//!     fft::{Direction, Domain, Expansion, ExpansionOrder, ExpansionStorage,
//!         ExpansionStrategy, InputOrder, Plan, SerialExecutor, Strategy, TransformRequest},
//! };
//!
//! let base = Plan::without_tables(Domain::new(2).unwrap().subgroup());
//! let extended = Domain::new(3).unwrap().coset(Fp::from_u64(7)).unwrap();
//! let expansion = Expansion::new(base, extended, None).unwrap().configure(
//!     ExpansionOrder::BitReversed,
//!     ExpansionStorage::Coefficients,
//!     ExpansionStrategy::serial(),
//! ).unwrap();
//! let coefficients = [Fp::ONE, Fp::from_u64(2)];
//! let mut factor = [Fp::ZERO; 8];
//! expansion.execute_into(&coefficients, &mut factor, &SerialExecutor, &mut []).unwrap();
//! let mut product = [Fp::ZERO; 8];
//! expansion.execute_product_into(
//!     &coefficients, expansion.view(&factor).unwrap(), &mut product,
//!     &SerialExecutor, &mut [],
//! ).unwrap();
//! let inverse = Plan::without_tables(extended).configure(
//!     TransformRequest {
//!         input_order: InputOrder::BitReversed,
//!         ..TransformRequest::new(Direction::Inverse)
//!     },
//!     Strategy::serial(),
//! ).unwrap();
//! inverse.execute(&mut product, &SerialExecutor, &mut []).unwrap();
//! assert_eq!(&product[..3], &[Fp::ONE, Fp::from_u64(4), Fp::from_u64(4)]);
//! assert!(product[3..].iter().all(|value| *value == Fp::ZERO));
//! ```

use crate::field::{PastaField, PrimeModulus};

mod domain;
mod executor;
mod expansion;
mod expansion_operation;
mod expansion_scales;
mod interpolation;
mod interpolation_parallel;
mod layout;
mod operation;
mod powers;
pub mod reference;
mod stages;
mod tables;
mod transform;

pub use domain::{CosetDomain, Domain};
pub use executor::{ExecutionOptions, Executor, ScratchRequirements, SerialExecutor};
pub use expansion::{Expansion, ExpansionOptions};
pub use expansion_operation::{
    ExpansionDescription, ExpansionOrder, ExpansionRequirements, ExpansionStorage,
    ExpansionStrategy, PreparedExpansion, Residue,
};
pub use expansion_scales::{ExpansionScaleArtifact, ExpansionScaleNormalization, ExpansionScales};
pub use interpolation::{
    Class, ClassState, InputOrder, interpolate_classes, interpolation_scratch,
};
pub use interpolation_parallel::{
    InterpolationOptions, InterpolationRequirements, interpolate_classes_parallel, interpolate_sum,
};
pub use layout::{
    CoefficientTiles, CoefficientView, EvaluationLayout, EvaluationView, ResidueLayout, ResidueView,
};
pub use operation::{
    Backend, Codelet, Direction, Initialization, InputPolicy, InputSupport, InverseScale,
    OperationDescription, OperationRequirements, PreparedOperation, ResourceBudget, Strategy,
    TransformRequest,
};
pub use powers::{PowerTable, TwiddleArtifact, TwiddleDescription, TwiddleStorage, TwiddleTable};
pub use tables::{BoundTables, TableRequirements, Tables, TablesMut};
pub use transform::Plan;

/// An invalid FFT description or insufficient caller storage.
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
    /// A coset shift is zero.
    ZeroShift,
    /// A coset shift has an unreduced Montgomery representation.
    InvalidShift,
    /// An execution setting or combination of request options is invalid.
    InvalidExecution,
    /// The requested storage exceeds an explicit resource ceiling.
    ResourceLimit,
    /// A buffer has the wrong length.
    LengthMismatch {
        /// Name of the buffer parameter or table field with the wrong length.
        buffer: &'static str,
        /// Required length, in elements.
        expected: usize,
        /// Supplied length, in elements.
        actual: usize,
    },
    /// The scratch buffer is too short.
    ScratchTooSmall {
        /// Required length, in field elements.
        required: usize,
        /// Supplied length, in field elements.
        provided: usize,
    },
    /// A prepared table does not match its domain.
    InvalidTables,
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
            Self::ZeroShift => f.write_str("coset shift must be nonzero"),
            Self::InvalidShift => f.write_str("coset shift must have reduced Montgomery limbs"),
            Self::InvalidExecution => f.write_str("invalid FFT execution settings"),
            Self::ResourceLimit => f.write_str("FFT resource ceiling exceeded"),
            Self::LengthMismatch {
                buffer,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "{buffer}: expected {expected} elements, received {actual}"
                )
            }
            Self::ScratchTooSmall { required, provided } => {
                write!(
                    f,
                    "FFT needs {required} scratch elements, received {provided}"
                )
            }
            Self::InvalidTables => f.write_str("FFT table contents do not match the domain"),
            Self::InvalidLayout => f.write_str("invalid FFT layout or range"),
            Self::InvalidClass => f.write_str("interpolation class exceeds the output domain"),
            Self::InvalidClassState => f.write_str("class no longer contains evaluations"),
        }
    }
}

impl core::error::Error for FftError {}

fn is_reduced<M: PrimeModulus>(value: PastaField<M>) -> bool {
    value
        .montgomery_limbs()
        .iter()
        .rev()
        .cmp(M::MODULUS.iter().rev())
        .is_lt()
}

fn check_len(buffer: &'static str, actual: usize, expected: usize) -> Result<(), FftError> {
    if actual == expected {
        Ok(())
    } else {
        Err(FftError::LengthMismatch {
            buffer,
            expected,
            actual,
        })
    }
}

fn check_prefix(actual: usize, min: usize, max: usize) -> Result<(), FftError> {
    if (min..=max).contains(&actual) {
        Ok(())
    } else {
        Err(FftError::InvalidPrefix { min, max, actual })
    }
}

const fn check_field_count(count: usize) -> Result<usize, FftError> {
    // Both sealed Pasta fields have the same four-limb representation.
    if count <= (isize::MAX as usize) / core::mem::size_of::<crate::field::Fp>() {
        Ok(count)
    } else {
        Err(FftError::SizeOverflow)
    }
}

const fn check_domain_size(size: usize) -> Result<(), FftError> {
    if !size.is_power_of_two() || crate::field::Fp::root_of_unity(size.ilog2()).is_none() {
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

fn reverse(index: usize, log_size: u32) -> usize {
    if log_size == 0 {
        0
    } else {
        index.reverse_bits() >> (usize::BITS - log_size)
    }
}

#[cfg(test)]
mod tests;
