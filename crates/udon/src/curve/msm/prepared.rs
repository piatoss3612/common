//! Borrowed scalar preparation for repeated MSMs.

use core::marker::PhantomData;

use super::{CurveError, PastaCurve, check_scratch, checked_count, recode};
use crate::{
    exec::{Executor, TaskBudget},
    field::PastaField,
};

/// Prepared MSM scalars borrowing caller-owned byte storage.
///
/// Reuse this handle with [`super::Input::new_prepared`] or
/// [`super::Input::indexed_prepared`] when the same scalar vector multiplies
/// different bases or indices. Preparation includes scalar-dependent dispatch;
/// execution does not read the original scalars or recode them again. Handles
/// may be shared between concurrent executions, each with its own scratch.
///
/// The curve type prevents cross-curve reuse. This is an ephemeral representation,
/// not a POD or serialization format; its size and encoding can change between
/// crate versions. Preparation and execution are variable-time and provide no
/// constant-time guarantee for secret scalars.
#[derive(Clone, Copy, Debug)]
pub struct PreparedScalars<'a, C: PastaCurve> {
    pub(super) digits: &'a [u8],
    terms: usize,
    marker: PhantomData<C>,
}

impl<'a, C: PastaCurve> PreparedScalars<'a, C> {
    /// Returns the required byte count for preparing `terms` scalars.
    ///
    /// Empty vectors require no storage. Returns [`CurveError::SizeOverflow`]
    /// when the scalar or byte buffer would exceed slice limits. Obtain counts
    /// here rather than relying on an internal digit layout.
    pub const fn storage_len(terms: usize) -> Result<usize, CurveError> {
        if let Err(error) = checked_count::<PastaField<C::Scalar>>(terms, 1) {
            return Err(error);
        }
        recode::storage_len(terms)
    }

    /// Prepares scalars into the required prefix of `storage` and borrows it.
    ///
    /// Scalar values must satisfy [`PastaField`]'s reduced-residue invariant;
    /// this is not checked. Violations remain memory-safe but may cause panics
    /// or incorrect results. Empty inputs are accepted. The original scalar
    /// slice is not retained. Initial storage contents do not matter, and the
    /// unused tail is untouched. Preparation uses at most `budget` concurrent
    /// work partitions through `executor`; the handle can be executed with a
    /// different budget or pass cap.
    ///
    /// Returns [`CurveError::SizeOverflow`] if sizing exceeds slice limits, or
    /// [`CurveError::ScratchTooSmall`] if `storage` has fewer bytes than
    /// [`Self::storage_len`] requires. Both checks precede all writes.
    ///
    /// An executor panic may leave storage partially written. All scoped jobs
    /// finish or unwind before propagation, as required by [`Executor`]. The
    /// buffer may then be reused without clearing it.
    pub fn prepare<X: Executor>(
        scalars: &[PastaField<C::Scalar>],
        storage: &'a mut [u8],
        budget: TaskBudget,
        executor: &X,
    ) -> Result<Self, CurveError> {
        let len = Self::storage_len(scalars.len())?;
        check_scratch("digits", len, storage.len())?;
        let digits = &mut storage[..len];
        recode::prepare::<C, X>(scalars, digits, budget, executor);
        Ok(Self {
            digits,
            terms: scalars.len(),
            marker: PhantomData,
        })
    }

    /// Returns the number of prepared scalars.
    pub const fn len(&self) -> usize {
        self.terms
    }

    /// Returns whether the vector contains no scalars.
    pub const fn is_empty(&self) -> bool {
        self.terms == 0
    }
}
