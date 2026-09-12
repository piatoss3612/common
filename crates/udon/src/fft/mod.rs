//! Radix-2 field transforms with caller-owned tables, buffers, and execution.
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
//! its expected and actual lengths. Invalid coefficient prefixes report the
//! supported length range separately from unsupported domain sizes.
//! Table contents are checked only by [`Tables::validate`] and
//! [`Expansion::validate_scales`]. Invalid contents can cause incorrect results
//! or panics. The generic [`mod@reference`] transforms have their own contracts.
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
//!     fft::{Domain, ExecutionOptions, Plan, SerialExecutor},
//! };
//!
//! let domain = Domain::new(2).unwrap().subgroup();
//! let plan = Plan::without_tables(domain);
//! let options = ExecutionOptions::serial();
//! let original = [Fp::ONE, Fp::from_u64(2), Fp::ZERO, Fp::ZERO];
//! let mut values = original;
//! plan.forward(&mut values, options, &SerialExecutor, &mut []).unwrap();
//! plan.inverse(&mut values, options, &SerialExecutor, &mut []).unwrap();
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
//! let plan = Plan::new(domain, tables).unwrap();
//! let mut scratch = [Fq::ZERO; SCRATCH];
//! let coefficients = [Fq::ONE; SIZE];
//! let mut values = coefficients;
//! plan.forward(&mut values, OPTIONS, &SerialExecutor, &mut scratch).unwrap();
//! plan.inverse(&mut values, OPTIONS, &SerialExecutor, &mut scratch).unwrap();
//! assert_eq!(values, coefficients);
//! ```

use crate::field::{PastaField, PrimeModulus};

mod domain;
mod executor;
mod expansion;
mod interpolation;
mod layout;
pub mod reference;
mod tables;
mod transform;

pub use domain::{CosetDomain, Domain};
pub use executor::{ExecutionOptions, Executor, ScratchRequirements, SerialExecutor};
pub use expansion::{Expansion, ExpansionOptions};
pub use interpolation::{Class, InputOrder, interpolate_classes, interpolation_scratch};
pub use layout::{CoefficientTiles, ResidueLayout, ResidueView};
pub use tables::{TableRequirements, Tables, TablesMut};
pub use transform::Plan;

/// An invalid FFT description or insufficient caller storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FftError {
    /// A domain size is unsupported.
    InvalidSize,
    /// A coefficient-prefix length is outside the supported range.
    InvalidPrefix {
        /// Minimum supported length, in coefficients.
        min: usize,
        /// Maximum supported length, in coefficients.
        max: usize,
        /// Supplied length, in coefficients.
        actual: usize,
    },
    /// An element count, index calculation, or byte size overflowed.
    SizeOverflow,
    /// A coset shift is zero.
    ZeroShift,
    /// An execution setting is zero or its tile length is not a power of two.
    InvalidExecution,
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
    /// A class is larger than half of the output domain.
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
                "coefficient prefix must contain {min}..={max} elements, received {actual}"
            ),
            Self::SizeOverflow => f.write_str("FFT storage or index size overflow"),
            Self::ZeroShift => f.write_str("coset shift must be nonzero"),
            Self::InvalidExecution => f.write_str("invalid FFT execution settings"),
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
            Self::InvalidClass => f.write_str("interpolation class exceeds half the output domain"),
            Self::InvalidClassState => f.write_str("class no longer contains evaluations"),
        }
    }
}

impl core::error::Error for FftError {}

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
