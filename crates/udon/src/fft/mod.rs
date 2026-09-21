//! Power-of-two field transforms with caller-owned tables, buffers, and execution.
//!
//! [`Plan`] binds a domain and borrowed tables, with synchronous transform
//! conveniences. [`run::FftPlan`] fixes reusable transform semantics and geometry;
//! its working buffers are borrowed only for execution. [`Expansion`]
//! evaluates a base polynomial on a larger coset without constructing a full
//! zero-padded transform. [`run::InterpolationPlan`] combines interpolations from
//! several domains into one coefficient vector.
//!
//! Setup and execution never allocate. Tables may be prepared into mutable
//! slices or borrowed from downstream Bento POD artifacts. [`ExecutionOptions`]
//! and [`ExpansionOptions`] determine the scratch requirements; their serial
//! settings need no scratch. The scoped [`Executor`] lets callers supply parallel
//! execution without requiring a particular runtime or an allocator in Udon.
//! An executor's own allocations are outside Udon's storage requirements.
//! [`run::FftPlan`] fixes order, normalization, and transform geometry.
//! [`run::ExpansionPlan`] additionally fixes coefficient storage and residue
//! ordering. Their drivers execute synchronously or expose incremental tasks.
//!
//! Field arithmetic is variable-time, with no constant-time guarantee for
//! secret inputs. Field buffers and table contents must satisfy
//! [`PastaField`]'s reduced Montgomery representation contract for correct
//! arithmetic; this is not a memory-safety requirement.
//!
//! # Validation and working storage
//!
//! Plan construction validates configuration without binding working buffers.
//! Synchronous drivers validate their complete buffer and scratch bindings before
//! mutation. Incremental tasks validate their own resource lengths before they
//! write; a later task error does not undo writes from earlier tasks. Publication
//! inspects arithmetic errors even when a task returned normally, poisons the
//! run, and permits outstanding receipts to drain.
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
//!     exec::SerialExecutor,
//!     field::Fp,
//!     fft::{Domain, ExecutionOptions, Plan},
//! };
//!
//! let domain = Domain::new(2).unwrap().subgroup();
//! let plan = Plan::without_tables(domain);
//! let original = [Fp::ONE, Fp::from_u64(2), Fp::ZERO, Fp::ZERO];
//! let mut values = original;
//! plan.forward(&mut values, ExecutionOptions::serial(), &SerialExecutor, &mut []).unwrap();
//! plan.inverse(&mut values, ExecutionOptions::serial(), &SerialExecutor, &mut []).unwrap();
//! assert_eq!(values, original);
//! ```
//!
//! Tables and scratch can also live in ordinary arrays. Prepare tables once,
//! then reuse the plan and scratch across transforms:
//!
//! ```
//! use zakura_udon::{
//!     exec::SerialExecutor,
//!     field::Fq,
//!     fft::{Domain, ExecutionOptions, Plan, TableRequirements, TablesMut},
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
//! use core::num::NonZeroUsize;
//! use zakura_udon::{
//!     field::Fp,
//!     exec::SerialExecutor,
//!     fft::{Codelet, Direction, Domain, Expansion, ExpansionOrder, ExpansionStorage,
//!         ElementOrder, InputSupport, Plan, TransformRequest,
//!         run::{ExpansionPlan, FftPlan}},
//! };
//!
//! let tasks = NonZeroUsize::new(1).unwrap();
//! let tile = NonZeroUsize::new(4).unwrap();
//! let base = Plan::without_tables(Domain::new(2).unwrap().subgroup());
//! let extended = Domain::new(3).unwrap().coset(Fp::from_u64(7)).unwrap();
//! let expansion = ExpansionPlan::new(
//!     Expansion::new(base, extended, None).unwrap(),
//!     ExpansionStorage::Coefficients, ExpansionOrder::BitReversed,
//!     InputSupport::Prefix(2), ElementOrder::Natural, tile, Codelet::Radix2,
//! ).unwrap();
//! let coefficients = [Fp::ONE, Fp::from_u64(2)];
//! let mut factor = [Fp::ZERO; 8];
//! expansion.execute(&coefficients, &mut factor, &mut [], None,
//!     &mut [], tasks, &SerialExecutor).unwrap();
//! let mut product = [Fp::ZERO; 8];
//! expansion.execute(&coefficients, &mut product, &mut [], Some(&factor),
//!     &mut [], tasks, &SerialExecutor).unwrap();
//! let inverse = FftPlan::new(Plan::without_tables(extended),
//!     TransformRequest {
//!         input_order: ElementOrder::BitReversed,
//!         ..TransformRequest::new(Direction::Inverse)
//!     }, tile, Codelet::Radix2,
//! ).unwrap();
//! inverse.execute(None, &mut product, None, &mut [], tasks, &SerialExecutor).unwrap();
//! assert_eq!(&product[..3], &[Fp::ONE, Fp::from_u64(4), Fp::from_u64(4)]);
//! assert!(product[3..].iter().all(|value| *value == Fp::ZERO));
//! ```

use crate::exec::Executor;
#[cfg(test)]
use crate::exec::SerialExecutor;
use crate::field::{PastaField, PrimeModulus};

mod domain;
mod execution;
mod expansion;
mod expansion_operation;
mod expansion_scales;
mod finish;
mod interpolation;
mod interpolation_parallel;
mod layout;
mod operation;
mod powers;
pub mod reference;
pub mod run;
mod stages;
mod tables;
mod transform;

pub use domain::{CosetDomain, Domain};
pub use execution::{ExecutionOptions, ScratchRequirements};
pub use expansion::{Expansion, ExpansionOptions};
pub use expansion_operation::{ExpansionOrder, ExpansionStorage, Residue};
pub use expansion_scales::{ExpansionScaleNormalization, ExpansionScales};
pub use interpolation::ClassState;
use interpolation::{Class, interpolate_classes, interpolation_scratch};
use interpolation_parallel::interpolate_sum;
pub use layout::{
    CoefficientView, ElementOrder, EvaluationLayout, EvaluationView, InverseScale, ResidueLayout,
};
pub use operation::{Codelet, Direction, InputStorage, InputSupport, TransformRequest};
pub use powers::{PowerTable, TwiddleDescription, TwiddleStorage, TwiddleTable};
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

fn check_length(buffer: &'static str, expected: usize, actual: usize) -> Result<(), FftError> {
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
