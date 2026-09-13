//! Variable-time multiscalar multiplication with borrowed inputs and scratch.
//!
//! A multiscalar multiplication (MSM) sums products of scalars and curve points.
//! [`Input`] borrows dense or indexed terms without gathering bases.
//! [`PreparedScalars`] retains scalar preparation across changing bases or
//! indices, with no digit scratch needed during execution.
//! [`execute_batch`] groups differently sized inputs under one scoped task
//! budget. Use [`Executor::join`] to compose these jobs with fixed-base products
//! or other work, dividing the caller's budget between simultaneous operations.
//!
//! All buffers and execution resources belong to the caller. No operation
//! allocates or requires a feature flag. [`Input`] checks lengths and indices;
//! its type docs describe the mathematical invariants required of stored data.
//! All operations are variable-time, with no constant-time guarantee for secret
//! bases, scalars, or indices.
//!
//! Indexed inputs can borrow embedded bases directly. Const sizing also works
//! for dense inputs of the same length, and allows execution without an allocator:
//!
//! ```
//! use zakura_udon::{
//!     curve::{
//!         Pallas, PallasAffine, PallasProjective,
//!         msm::{Bases, ExecutionOptions, Input, PreparedScalars, Requirements, Scratch},
//!     },
//!     exec::{SerialExecutor, TaskBudget},
//!     field::{Fp, Fq},
//! };
//!
//! const OPTIONS: ExecutionOptions = ExecutionOptions::SERIAL;
//! const R: Requirements = match Input::<Pallas>::requirements_for_len(3, OPTIONS) {
//!     Ok(r) => r,
//!     Err(_) => panic!("MSM is too large"),
//! };
//! let g = PallasAffine::GENERATOR;
//! let bases = [g, g.neg()];
//! let indices = [0, 1, 0];
//! let scalars = [Fq::from_u64(2), Fq::from_u64(3), Fq::from_u64(5)];
//! let input = Input::indexed(Bases::Affine(&bases), &indices, &scalars)?;
//! let mut digits = [0; R.digits];
//! let mut affine = [g; R.affine];
//! let mut projective = [PallasProjective::IDENTITY; R.projective];
//! let mut field = [Fp::ZERO; R.field];
//! let mut working_indices = [0; R.indices];
//! let mut scratch = Scratch {
//!     digits: &mut digits,
//!     affine: &mut affine,
//!     projective: &mut projective,
//!     field: &mut field,
//!     indices: &mut working_indices,
//! };
//! let result = input.execute(OPTIONS, &SerialExecutor, scratch.reborrow())?;
//! assert_eq!(result, g.mul_projective(&Fq::from_u64(4)));
//! // Reuse the same storage without clearing it.
//! assert_eq!(input.execute(OPTIONS, &SerialExecutor, scratch.reborrow())?, result);
//! // Retain scalar preparation while changing the selected bases.
//! const PREPARED_BYTES: usize = match PreparedScalars::<Pallas>::storage_len(3) {
//!     Ok(bytes) => bytes,
//!     Err(_) => panic!("unsupported size"),
//! };
//! let mut storage = [0; PREPARED_BYTES];
//! let retained = PreparedScalars::<Pallas>::prepare(
//!     &scalars, &mut storage, TaskBudget::SERIAL, &SerialExecutor,
//! )?;
//! let indices = [0, 0, 0];
//! let reused = Input::indexed_prepared(Bases::Affine(&bases), &indices, retained)?;
//! assert_eq!(reused.requirements(OPTIONS)?.digits, 0);
//! assert_eq!(reused.execute(OPTIONS, &SerialExecutor, scratch)?,
//!     g.mul_projective(&Fq::from_u64(10)));
//! # Ok::<(), zakura_udon::curve::CurveError>(())
//! ```

use core::num::NonZeroUsize;

use super::{
    AffinePoint, CurveError, PastaCurve, Point, PreparedAffinePoint, ProjectivePoint, check_length,
    check_scratch, checked_count,
};
use crate::{
    exec::{Executor, TaskBudget},
    field::PastaField,
};

mod buckets;
mod kernels;
mod prepared;
mod recode;
mod schedule;
pub use prepared::PreparedScalars;
// Measured dispatch policy, shared by sizing, recoding, and arithmetic.
const BOOTH_MIN: usize = 128;
const WINDOW_BITS: usize = 8;
const WINDOWS: usize = 128_usize.div_ceil(WINDOW_BITS) + 1;
const BUCKETS: usize = 1 << (WINDOW_BITS - 1);
#[cfg(test)]
mod tests;

/// Borrowed base storage for an [`Input`].
///
/// Affine and cached affine entries are nonidentity and support Bento POD
/// storage. [`Point`] slices can also contain identities. No representation is
/// copied or gathered when constructing an input.
#[derive(Clone, Copy, Debug)]
pub enum Bases<'a, C: PastaCurve> {
    /// Nonidentity affine bases.
    Affine(&'a [AffinePoint<C>]),
    /// Nonidentity bases with cached endomorphism coordinates.
    Prepared(&'a [PreparedAffinePoint<C>]),
    /// Affine bases that may include identity.
    Points(&'a [Point<C>]),
}

impl<C: PastaCurve> Bases<'_, C> {
    /// Returns the number of available bases.
    pub const fn len(&self) -> usize {
        match self {
            Self::Affine(b) => b.len(),
            Self::Prepared(b) => b.len(),
            Self::Points(b) => b.len(),
        }
    }

    /// Returns whether the base slice is empty.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Borrowed MSM terms with checked lengths and indices.
///
/// Cloning this handle copies references only. Its immutable borrows preserve
/// these checks for subsequent executions. Bases and scalars must already
/// satisfy the mathematical invariants of [`AffinePoint`],
/// [`PreparedAffinePoint`], [`Point`], and [`PastaField`], as applicable. These
/// invariants are not checked here, including when inputs come from POD storage.
/// Violations remain memory-safe but can make execution panic or return an
/// incorrect result.
#[derive(Clone, Copy, Debug)]
pub struct Input<'a, C: PastaCurve> {
    bases: Bases<'a, C>,
    scalars: Scalars<'a, C>,
    indices: Option<&'a [u32]>,
}

#[derive(Clone, Copy)]
enum Scalars<'a, C: PastaCurve> {
    Raw(&'a [PastaField<C::Scalar>]),
    Prepared(PreparedScalars<'a, C>),
}

impl<C: PastaCurve + core::fmt::Debug> core::fmt::Debug for Scalars<'_, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::Raw(s) => f.debug_tuple("Raw").field(s).finish(),
            Self::Prepared(s) => f.debug_tuple("Prepared").field(s).finish(),
        }
    }
}

impl<C: PastaCurve> Scalars<'_, C> {
    const fn len(&self) -> usize {
        match self {
            Self::Raw(s) => s.len(),
            Self::Prepared(s) => s.len(),
        }
    }
}

impl<'a, C: PastaCurve> Input<'a, C> {
    /// Borrows dense terms, requiring one scalar for each base.
    ///
    /// Term `i` is `scalars[i] * bases[i]`. Empty inputs are accepted.
    /// Returns [`CurveError::LengthMismatch`] when the lengths differ.
    pub fn new(
        bases: Bases<'a, C>,
        scalars: &'a [PastaField<C::Scalar>],
    ) -> Result<Self, CurveError> {
        check_length("scalars", bases.len(), scalars.len())?;
        Ok(Self {
            bases,
            scalars: Scalars::Raw(scalars),
            indices: None,
        })
    }

    /// Borrows indexed terms without gathering the selected bases.
    ///
    /// Term `i` is `scalars[i] * bases[indices[i]]`. Empty inputs are accepted;
    /// repeated indices contribute separately. Returns
    /// [`CurveError::LengthMismatch`] unless there is one index per scalar, or
    /// [`CurveError::BaseIndexOutOfBounds`] if an index is at least `bases.len()`.
    pub fn indexed(
        bases: Bases<'a, C>,
        indices: &'a [u32],
        scalars: &'a [PastaField<C::Scalar>],
    ) -> Result<Self, CurveError> {
        Self::with_indices(bases, indices, Scalars::Raw(scalars))
    }

    /// Borrows dense bases paired with reusable scalar preparation.
    ///
    /// Pairs scalars and bases in the order passed to their constructors, with
    /// the length and empty-input contracts of [`Self::new`]. Execution requires
    /// no digit scratch; use [`Self::requirements`] to size its remaining buffers.
    pub fn new_prepared(
        bases: Bases<'a, C>,
        scalars: PreparedScalars<'a, C>,
    ) -> Result<Self, CurveError> {
        check_length("scalars", bases.len(), scalars.len())?;
        Ok(Self {
            bases,
            scalars: Scalars::Prepared(scalars),
            indices: None,
        })
    }

    /// Borrows indexed bases paired with reusable scalar preparation.
    ///
    /// Has the index and length contracts of [`Self::indexed`], including its
    /// errors. Execution requires no digit scratch; use [`Self::requirements`]
    /// to size its remaining buffers. Indices may differ between uses of the
    /// same [`PreparedScalars`] handle.
    pub fn indexed_prepared(
        bases: Bases<'a, C>,
        indices: &'a [u32],
        scalars: PreparedScalars<'a, C>,
    ) -> Result<Self, CurveError> {
        Self::with_indices(bases, indices, Scalars::Prepared(scalars))
    }

    fn with_indices(
        bases: Bases<'a, C>,
        indices: &'a [u32],
        scalars: Scalars<'a, C>,
    ) -> Result<Self, CurveError> {
        check_length("indices", scalars.len(), indices.len())?;
        for (position, &index) in indices.iter().enumerate() {
            if u64::from(index) >= bases.len() as u64 {
                return Err(CurveError::BaseIndexOutOfBounds {
                    position,
                    index,
                    bases: bases.len(),
                });
            }
        }
        Ok(Self {
            bases,
            scalars,
            indices: Some(indices),
        })
    }

    /// Returns the number of scalar/base terms.
    pub const fn len(&self) -> usize {
        self.scalars.len()
    }

    /// Returns whether the sum has no terms.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }

    const fn digit_scratch_len(&self) -> Result<usize, CurveError> {
        match self.scalars {
            Scalars::Raw(s) => recode::storage_len(s.len()),
            Scalars::Prepared(_) => Ok(0),
        }
    }

    /// Returns scratch counts for unprepared scalars and execution options.
    ///
    /// Counts are independent of base representation, indices, and scalar values.
    /// This const query supports static buffers and returns
    /// [`CurveError::SizeOverflow`] when a buffer cannot be represented by a
    /// slice. Counts can change between crate versions; obtain them from this API.
    /// For inputs borrowing [`PreparedScalars`], use [`Self::requirements`] to
    /// omit digit scratch.
    pub const fn requirements_for_len(
        terms: usize,
        options: ExecutionOptions,
    ) -> Result<Requirements, CurveError> {
        schedule::single_requirements::<C>(terms, options)
    }

    /// Returns scratch counts for this input and execution policy.
    ///
    /// Equivalent to [`Self::requirements_for_len`] with `self.len()`, except
    /// inputs borrowing [`PreparedScalars`] require zero digit bytes. Returns
    /// [`CurveError::SizeOverflow`] if a buffer exceeds slice limits.
    pub const fn requirements(
        &self,
        options: ExecutionOptions,
    ) -> Result<Requirements, CurveError> {
        match Self::requirements_for_len(self.len(), options) {
            Ok(mut r) => {
                if let Scalars::Prepared(_) = self.scalars {
                    r.digits = 0;
                }
                Ok(r)
            }
            Err(error) => Err(error),
        }
    }

    /// Computes the sum, returning identity when there are no terms.
    ///
    /// Size scratch with [`Self::requirements`] using the same `options`.
    /// Initial contents do not matter; tails beyond the reported counts are
    /// untouched.
    ///
    /// # Errors
    ///
    /// Returns [`CurveError::ScratchTooSmall`] for insufficient scratch or
    /// [`CurveError::SizeOverflow`] if sizing exceeds slice limits. These checks
    /// precede all writes, so returned errors leave scratch unchanged.
    ///
    /// # Panics
    ///
    /// An executor panic may leave scratch partially written. All scoped jobs
    /// finish or unwind before it propagates, as required by [`Executor`].
    /// The buffers can then be reused without clearing them.
    pub fn execute<X: Executor>(
        &self,
        options: ExecutionOptions,
        executor: &X,
        scratch: Scratch<'_, C>,
    ) -> Result<ProjectivePoint<C>, CurveError> {
        let mut output = [ProjectivePoint::IDENTITY];
        execute_batch(
            core::slice::from_ref(self),
            &mut output,
            options,
            executor,
            scratch,
        )?;
        Ok(output[0])
    }
}

/// Concurrency and working storage limits for MSM execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionOptions {
    /// Total allowance for simultaneous work partitions, including nested work.
    pub task_budget: TaskBudget,
    /// Maximum terms staged together in each concurrent arithmetic partition.
    ///
    /// `None` permits staging the whole partition. A cap trades additional passes
    /// over the terms for less temporary point storage. It does not bound total
    /// scratch: digit storage still scales with all terms, and concurrent
    /// partitions each need working storage. Query [`Input::requirements`] or
    /// [`batch_requirements`] to size buffers for the chosen options.
    pub max_terms_per_pass: Option<NonZeroUsize>,
}

impl ExecutionOptions {
    /// Serial execution without a pass cap.
    pub const SERIAL: Self = Self {
        task_budget: TaskBudget::SERIAL,
        max_terms_per_pass: None,
    };
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self::SERIAL
    }
}

/// Minimum scratch lengths, counted in elements of the corresponding slice.
///
/// Obtain counts from [`Input::requirements_for_len`], [`Input::requirements`],
/// or [`batch_requirements`]. Larger buffers can be reused; execution touches
/// only the reported prefixes.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Requirements {
    /// Number of digit bytes.
    pub digits: usize,
    /// Number of nonidentity affine points.
    pub affine: usize,
    /// Number of projective points.
    pub projective: usize,
    /// Number of base-field elements.
    pub field: usize,
    /// Number of temporary `usize` indices used during execution.
    pub indices: usize,
}

/// Caller-owned working slices for one execution or group of MSMs.
///
/// Initialize affine storage with any valid point, such as the generator.
/// Execution initializes every value it reads, so the buffers can be reused
/// without clearing them. An [`Input`] retains no scratch borrows.
#[derive(Debug)]
pub struct Scratch<'a, C: PastaCurve> {
    /// Recoded digit storage, sized by [`Requirements::digits`].
    pub digits: &'a mut [u8],
    /// Affine working storage, sized by [`Requirements::affine`].
    pub affine: &'a mut [AffinePoint<C>],
    /// Projective working storage, sized by [`Requirements::projective`].
    pub projective: &'a mut [ProjectivePoint<C>],
    /// Base-field working storage, sized by [`Requirements::field`].
    pub field: &'a mut [PastaField<C::Base>],
    /// Working indices, sized by [`Requirements::indices`].
    pub indices: &'a mut [usize],
}

impl<'a, C: PastaCurve> Scratch<'a, C> {
    /// Borrows the buffers for an execution while retaining this scratch handle.
    pub fn reborrow(&mut self) -> Scratch<'_, C> {
        Scratch {
            digits: self.digits,
            affine: self.affine,
            projective: self.projective,
            field: self.field,
            indices: self.indices,
        }
    }

    fn checked(self, r: Requirements) -> Result<Self, CurveError> {
        check_scratch("digits", r.digits, self.digits.len())?;
        check_scratch("affine", r.affine, self.affine.len())?;
        check_scratch("projective", r.projective, self.projective.len())?;
        check_scratch("field", r.field, self.field.len())?;
        check_scratch("indices", r.indices, self.indices.len())?;
        Ok(Self {
            digits: &mut self.digits[..r.digits],
            affine: &mut self.affine[..r.affine],
            projective: &mut self.projective[..r.projective],
            field: &mut self.field[..r.field],
            indices: &mut self.indices[..r.indices],
        })
    }
}

/// Returns scratch counts for grouped inputs sharing one task budget.
///
/// The counts depend on input lengths, whether scalars are prepared, and options,
/// including the pass cap.
/// Returns [`CurveError::SizeOverflow`] if a buffer would exceed slice limits.
pub fn batch_requirements<C: PastaCurve>(
    inputs: &[Input<'_, C>],
    options: ExecutionOptions,
) -> Result<Requirements, CurveError> {
    Ok(schedule::Plan::new(inputs, options)?.requirements)
}

/// Computes one projective result per input, preserving input order.
///
/// Inputs can mix base representations and dense or indexed access. They share
/// the task budget in `options`. Size scratch with [`batch_requirements`] using
/// the same inputs and options. The initial-content, scratch-tail, and panic
/// contracts of [`Input::execute`] apply; a panic may also leave output partially
/// written.
///
/// # Errors
///
/// Returns [`CurveError::LengthMismatch`] unless `output.len() == inputs.len()`,
/// [`CurveError::ScratchTooSmall`] for insufficient scratch, or
/// [`CurveError::SizeOverflow`] if sizing exceeds slice limits. All checks
/// precede writes, so returned errors leave both output and scratch unchanged.
pub fn execute_batch<C: PastaCurve, X: Executor>(
    inputs: &[Input<'_, C>],
    output: &mut [ProjectivePoint<C>],
    options: ExecutionOptions,
    executor: &X,
    scratch: Scratch<'_, C>,
) -> Result<(), CurveError> {
    check_length("output", inputs.len(), output.len())?;
    let plan = schedule::Plan::new(inputs, options)?;
    let scratch = scratch.checked(plan.requirements)?;
    schedule::execute(&plan, inputs, output, executor, scratch);
    Ok(())
}
