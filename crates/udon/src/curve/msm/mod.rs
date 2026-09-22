//! Variable-time multiscalar multiplication with caller-owned storage.
//!
//! [`run::BatchPlan`] plans and executes contiguous inputs, retaining scheduling
//! metadata for repeated execution of the same immutable scalar rows and bases.
//! [`Selection`] retains validated base mappings across scalar rows. For
//! prepared MSM, cache each base
//! with [`PreparedAffinePoint::from_affine`] and select [`Bases::Prepared`].
//! [`PreparedScalars`] retains classification and GLV data for one scalar row
//! across base sets and execution policies.
//! [`Bases::Compact`] consumes ordinary embedded or prepared compact tables.
//! For incremental task scheduling, [`run::MsmPlan`] retains geometry without
//! borrowing inputs. Reuse it across [`run::MsmRun`] invocations, each of which
//! borrows its inputs and frontier storage. [`run::BatchPlan`] instead borrows
//! both its immutable inputs and mutable planning metadata for its lifetime.
//!
//! [`Input::execute`] selects an implementation from the input's scalar and base
//! facts, [`ExecutionOptions`], and supplied scratch capacities. Resolved plans
//! retain that choice for repeated or incremental execution. Window widths,
//! recoding, and accumulation are implementation details. Digit caches are
//! prepared against a resolved plan so preparation and execution agree.
//!
//! All runtime operations are allocation-free and variable-time, with no
//! constant-time guarantee for secret bases, scalars, or indices. Callers own
//! storage and execution; [`ExecutionOptions`] defines workspace accounting.
//!
//! Reuse cached bases and validated indices across two signed scalar rows.
//! Rebuild the batch plan when its immutable scalar inputs change:
//!
//! ```
//! use zakura_udon::{
//!     curve::{AffinePoint, Pallas, PreparedAffinePoint, ProjectivePoint, msm::*,
//!         msm::run::{BatchPlan, JobStorage, WorkerStorage}},
//!     exec::{ExecutionOptions, SerialExecutor},
//!     field::PastaField,
//! };
//! let bases = [AffinePoint::<Pallas>::GENERATOR; 2];
//! let prepared = bases.map(|base| PreparedAffinePoint::from_affine(&base));
//! let indices = [0, 1, 0];
//! let selection = Selection::indexed(Bases::Prepared(&prepared), &indices)?;
//! let mut jobs = [JobStorage::EMPTY; 1];
//! let mut workers = [WorkerStorage::EMPTY; 1];
//! for row in [[1_i128, -1, 3], [0, 2, 1]] {
//!     let inputs = [selection.with_signed(&row)];
//!     let plan = BatchPlan::new(&inputs, ExecutionOptions::default(),
//!         &mut jobs, &mut workers)?;
//!     let r = plan.requirements();
//!     let mut records = vec![ScalarStorage::ZERO; r.scalars()];
//!     let mut digits = vec![0; r.digits()];
//!     let mut affine = vec![AffinePoint::GENERATOR; r.affine()];
//!     let mut projective = vec![ProjectivePoint::IDENTITY; r.projective()];
//!     let mut field = vec![PastaField::ZERO; r.field()];
//!     let mut indices_scratch = vec![0; r.indices()];
//!     let scratch = Scratch::new(&mut records, &mut digits, &mut affine,
//!         &mut projective, &mut field, &mut indices_scratch);
//!     let mut output = [ProjectivePoint::IDENTITY];
//!     plan.execute(&mut output, &SerialExecutor, scratch);
//!     assert_eq!(output[0], bases[0].mul_projective(&PastaField::<_>::from_u64(3)));
//! }
//! # Ok::<(), zakura_udon::curve::CurveError>(())
//! ```

use super::{
    AffinePoint, CurveError, EisensteinTableBatch, PastaCurve, Point, PreparedAffinePoint,
    ProjectivePoint, assert_length, assert_scratch, checked_count,
};
use crate::exec::{ExecutionOptions, Executor};
use crate::field::{CanonicalUint, PastaField};
#[cfg(test)]
use core::num::NonZeroUsize;

#[cfg(test)]
use crate::exec::TaskBudget;

mod policy;
pub(crate) use policy::{Accumulation, ArithmeticOptions, BatchOptions, Kernel};

mod buckets;
mod kernels;
mod prepared;
mod recode;
mod schedule;

pub mod run;
pub use prepared::{PreparedScalars, ScalarStorage};
#[cfg(test)]
use run::{BatchPlan, JobStorage, WorkerStorage};
const BOOTH_MIN: usize = 128;
#[cfg(test)]
mod tests;

/// Borrowed ordinary, cached, or compact-table base storage.
///
/// Stored affine points must be nonidentity and satisfy their documented curve
/// invariants. Compact tables must satisfy [`EisensteinTableBatch`]'s binding or
/// preparation contract, including the owner's obligations for trusted bindings.
/// MSM construction and execution do not recheck these mathematical invariants;
/// see [`Input`] for the consequences of invalid data.
#[derive(Clone, Copy, Debug)]
pub enum Bases<'a, C: PastaCurve> {
    /// Nonidentity affine points.
    Affine(&'a [AffinePoint<C>]),
    /// Affine points with cached endomorphism coordinates.
    ///
    /// Prepare entries with [`PreparedAffinePoint::from_affine`], or embed
    /// POD entries that satisfy [`PreparedAffinePoint`]'s mathematical invariants.
    /// A [`Selection`] reuses them across scalar rows.
    Prepared(&'a [PreparedAffinePoint<C>]),
    /// Points that may contain identities.
    Points(&'a [Point<C>]),
    /// Borrowed compact tables, reusable across scalar rows.
    Compact(EisensteinTableBatch<'a, C>),
    /// Compact tables with cached endomorphism coordinates.
    CompactPrepared(EisensteinTableBatch<'a, C, PreparedAffinePoint<C>>),
}
impl<C: PastaCurve> Bases<'_, C> {
    /// Number of available bases (not compact table entries).
    pub const fn len(&self) -> usize {
        match self {
            Self::Affine(b) => b.len(),
            Self::Prepared(b) => b.len(),
            Self::Points(b) => b.len(),
            Self::Compact(b) => b.len(),
            Self::CompactPrepared(b) => b.len(),
        }
    }
    /// Whether there are no bases.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Validated base mapping with a lifetime independent of scalar rows.
///
/// Use [`Bases::Prepared`] for a prepared MSM over cached bases. The mapping
/// and base preparation remain reusable after each scalar row is dropped.
///
/// Cloning copies references. Immutable base and index borrows preserve index
/// validation across scalar rows. Every binding requires exactly [`Self::len`] scalars
/// and panics on a mismatch. Only [`Self::with_canonical`] additionally validates
/// scalar values. Base values already carry their point type's invariants.
#[derive(Clone, Copy, Debug)]
pub struct Selection<'a, C: PastaCurve> {
    bases: Bases<'a, C>,
    indices: Option<&'a [u32]>,
}
impl<'a, C: PastaCurve> Selection<'a, C> {
    /// Selects every base in storage order.
    pub const fn new(bases: Bases<'a, C>) -> Self {
        Self {
            bases,
            indices: None,
        }
    }
    /// Selects bases by index without gathering their coordinates.
    ///
    /// Repeated indices contribute separately. Returns
    /// [`CurveError::BaseIndexOutOfBounds`] if an index is at least `bases.len()`.
    pub fn indexed(bases: Bases<'a, C>, indices: &'a [u32]) -> Result<Self, CurveError> {
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
            indices: Some(indices),
        })
    }
    /// Number of selected terms.
    pub const fn len(&self) -> usize {
        match self.indices {
            Some(i) => i.len(),
            None => self.bases.len(),
        }
    }
    /// Whether the selection is empty.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
    fn bind<'s>(&self, scalars: Scalars<'s, C>) -> Input<'s, C>
    where
        'a: 's,
    {
        assert_length("scalars", self.len(), scalars.len());
        Input {
            bases: self.bases,
            indices: self.indices,
            scalars,
        }
    }
    /// Binds a field scalar row with an O(1) length check.
    ///
    /// Scalars use the loose field representation. See [`Selection`] for the
    /// exact-length requirement.
    pub fn with_scalars<'s>(&self, scalars: &'s [PastaField<C::Scalar>]) -> Input<'s, C>
    where
        'a: 's,
    {
        self.bind(Scalars::Raw(scalars))
    }
    /// Binds prepared scalars with an O(1) length check.
    ///
    /// See [`Selection`] for the exact-length requirement.
    pub fn with_prepared_scalars<'s>(&self, scalars: PreparedScalars<'s, C>) -> Input<'s, C>
    where
        'a: 's,
    {
        self.bind(Scalars::Prepared(scalars))
    }
    /// Binds unsigned coefficients whose type guarantees the 128-bit bound.
    ///
    /// See [`Selection`] for the exact-length requirement.
    pub fn with_unsigned<'s>(&self, scalars: &'s [u128]) -> Input<'s, C>
    where
        'a: 's,
    {
        self.bind(Scalars::Unsigned(scalars))
    }
    /// Binds signed coefficients, including `i128::MIN`.
    ///
    /// A negative coefficient subtracts its magnitude's base multiple. See
    /// [`Selection`] for the exact-length requirement.
    pub fn with_signed<'s>(&self, scalars: &'s [i128]) -> Input<'s, C>
    where
        'a: 's,
    {
        self.bind(Scalars::Signed(scalars))
    }
    /// Checks canonical integers against the scalar modulus and a bit bound.
    ///
    /// Each integer must be below the scalar modulus and fit in `bits` bits.
    /// The bound must be in `0..=256`; zero permits only zero coefficients.
    /// Returns [`CurveError::InvalidScalar`] for a violation, recording the first
    /// invalid position, or position zero for an invalid bound, even on an empty
    /// row. No value is truncated. See [`Selection`] for the length requirement.
    pub fn with_canonical<'s>(
        &self,
        scalars: &'s [CanonicalUint],
        bits: usize,
    ) -> Result<Input<'s, C>, CurveError>
    where
        'a: 's,
    {
        assert_length("scalars", self.len(), scalars.len());
        prepared::validate_canonical::<C>(scalars, bits)?;
        Ok(self.bind(Scalars::Canonical(scalars)))
    }
}

/// Borrowed MSM terms with checked lengths and indices.
///
/// Field scalars use [`PastaField`]'s loose representation; stored bases are
/// trusted values with their point type's invariants. Arithmetic is
/// variable-time and gives no constant-time guarantee for secret inputs.
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
    Unsigned(&'a [u128]),
    Signed(&'a [i128]),
    Canonical(&'a [CanonicalUint]),
}
impl<C: PastaCurve> core::fmt::Debug for Scalars<'_, C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Scalars")
            .field("len", &self.len())
            .finish_non_exhaustive()
    }
}
impl<'a, C: PastaCurve> Scalars<'a, C> {
    const fn len(&self) -> usize {
        match self {
            Self::Raw(s) => s.len(),
            Self::Prepared(s) => s.len(),
            Self::Unsigned(s) => s.len(),
            Self::Signed(s) => s.len(),
            Self::Canonical(s) => s.len(),
        }
    }
    fn slice(self, range: core::ops::Range<usize>) -> Self {
        match self {
            Self::Raw(s) => Self::Raw(&s[range]),
            Self::Unsigned(s) => Self::Unsigned(&s[range]),
            Self::Signed(s) => Self::Signed(&s[range]),
            Self::Canonical(s) => Self::Canonical(&s[range]),
            Self::Prepared(s) => Self::Prepared(PreparedScalars {
                // A whole-row bound remains valid for each chunk; execution
                // already retains its selected geometry in the job metadata.
                records: &s.records[range],
                shape: s.shape,
                cached: None,
            }),
        }
    }
}
impl<'a, C: PastaCurve> Input<'a, C> {
    /// Borrows one field scalar per base in storage order.
    ///
    /// Panics unless the lengths are equal.
    pub fn new(bases: Bases<'a, C>, scalars: &'a [PastaField<C::Scalar>]) -> Self {
        Selection::new(bases).with_scalars(scalars)
    }
    /// Checks indices and scalar length, without gathering bases.
    ///
    /// Panics unless each scalar has an index. Index bounds and repeated indices follow
    /// [`Selection::indexed`].
    pub fn indexed(
        bases: Bases<'a, C>,
        indices: &'a [u32],
        scalars: &'a [PastaField<C::Scalar>],
    ) -> Result<Self, CurveError> {
        assert_length("indices", scalars.len(), indices.len());
        Ok(Selection::indexed(bases, indices)?.with_scalars(scalars))
    }
    /// Binds one prepared scalar per base.
    ///
    /// Panics unless the lengths are equal.
    pub fn new_prepared(bases: Bases<'a, C>, scalars: PreparedScalars<'a, C>) -> Self {
        Selection::new(bases).with_prepared_scalars(scalars)
    }
    /// Checks indices and binds prepared scalars.
    ///
    /// Panics unless each scalar has an index. Index bounds and repeated indices follow
    /// [`Selection::indexed`].
    pub fn indexed_prepared(
        bases: Bases<'a, C>,
        indices: &'a [u32],
        scalars: PreparedScalars<'a, C>,
    ) -> Result<Self, CurveError> {
        assert_length("indices", scalars.len(), indices.len());
        Ok(Selection::indexed(bases, indices)?.with_prepared_scalars(scalars))
    }
    /// Retains this input's validated base mapping for another scalar row.
    pub const fn selection(&self) -> Selection<'a, C> {
        Selection {
            bases: self.bases,
            indices: self.indices,
        }
    }
    /// Number of terms.
    pub const fn len(&self) -> usize {
        self.scalars.len()
    }
    /// Whether the sum has no terms.
    pub const fn is_empty(&self) -> bool {
        self.len() == 0
    }
    /// Resolves scratch requirements from the input and resource constraints.
    ///
    /// Returns [`CurveError::SizeOverflow`] if storage counts or byte sizes
    /// cannot be represented, or [`CurveError::MemoryLimit`] if planning finds
    /// no layout within the workspace ceiling. This query does not write or
    /// reserve buffers. Execution can also use smaller supplied buffers by
    /// selecting a fitting implementation.
    pub fn requirements(&self, options: ExecutionOptions) -> Result<Requirements, CurveError> {
        Ok(schedule::Plan::new(core::slice::from_ref(self), options.into())?.requirements)
    }

    /// Computes this sum using caller-owned scratch and scoped execution.
    ///
    /// Udon adapts to every supplied buffer capacity and the workspace ceiling.
    /// An empty input returns the identity. Errors from [`Self::requirements`]
    /// or [`CurveError::ScratchTooSmall`] for insufficient capacity occur before
    /// writes; surplus tails remain untouched. A panic may change scratch,
    /// which can be reused after scoped work has finished unwinding.
    pub fn execute<X: Executor>(
        &self,
        options: ExecutionOptions,
        executor: &X,
        scratch: Scratch<'_, C>,
    ) -> Result<ProjectivePoint<C>, CurveError> {
        let plan = schedule::Plan::with_capacity(
            core::slice::from_ref(self),
            options.into(),
            scratch.capacity(),
        )?;
        let scratch = scratch.checked(plan.requirements);
        let mut output = [ProjectivePoint::IDENTITY];
        schedule::execute(
            &plan,
            core::slice::from_ref(self),
            &mut output,
            executor,
            scratch,
        );
        Ok(output[0])
    }

    /// Conservative const scratch counts for unprepared scalars and ordinary bases.
    ///
    /// Supports field and integer scalar rows with [`Bases::Affine`],
    /// [`Bases::Prepared`], or [`Bases::Points`]. Use [`Self::requirements`] for
    /// retained scalars or compact tables: their memory planning can select a
    /// different layout, whose individual buffer counts may be larger.
    ///
    /// Returns [`CurveError::SizeOverflow`] if sizing exceeds slice limits, or
    /// [`CurveError::MemoryLimit`] under the
    /// [memory policy](BatchOptions::with_memory_limit).
    #[cfg(test)]
    pub(crate) const fn requirements_for_len(
        terms: usize,
        options: BatchOptions,
    ) -> Result<Requirements, CurveError> {
        schedule::single_requirements::<C>(terms, options)
    }
    /// Scratch counts for this input, accounting for retained preparation.
    ///
    /// Errors and memory accounting match [`batch_requirements`].
    #[cfg(test)]
    pub(crate) fn requirements_with(
        &self,
        options: BatchOptions,
    ) -> Result<Requirements, CurveError> {
        batch_requirements(core::slice::from_ref(self), options)
    }
    /// Computes the sum with caller-owned scratch and execution resources.
    ///
    /// An empty input returns the identity. Size scratch with
    /// [`Self::requirements`] and the same options. Initial scratch contents do
    /// not matter; tails beyond the required prefixes remain untouched.
    ///
    /// Sizing errors match [`Self::requirements`]; insufficient scratch returns
    /// [`CurveError::ScratchTooSmall`]. Returned errors precede all writes.
    /// An executor panic may leave scratch partially written; scoped work must
    /// finish unwinding before reuse, as required by [`Executor`].
    #[cfg(test)]
    pub(crate) fn execute_with<X: Executor>(
        &self,
        options: BatchOptions,
        executor: &X,
        scratch: Scratch<'_, C>,
    ) -> Result<ProjectivePoint<C>, CurveError> {
        let mut result = [ProjectivePoint::IDENTITY];
        execute_batch(
            core::slice::from_ref(self),
            &mut result,
            options,
            executor,
            scratch,
        )?;
        Ok(result[0])
    }
}

/// Element counts for initialized, typed caller-owned temporary buffers.
///
/// Obtain these counts from [`run::MsmPlan::requirements`] or
/// [`run::BatchPlan::requirements`] for the execution being sized.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Requirements {
    scalars: usize,
    digits: usize,
    affine: usize,
    projective: usize,
    field: usize,
    indices: usize,
}
impl Requirements {
    fn fits(self, available: Self) -> bool {
        self.scalars <= available.scalars
            && self.digits <= available.digits
            && self.affine <= available.affine
            && self.projective <= available.projective
            && self.field <= available.field
            && self.indices <= available.indices
    }
    fn capacity_error(self, available: Self) -> CurveError {
        for (buffer, required, provided) in [
            ("scalars", self.scalars, available.scalars),
            ("digits", self.digits, available.digits),
            ("affine", self.affine, available.affine),
            ("projective", self.projective, available.projective),
            ("field", self.field, available.field),
            ("indices", self.indices, available.indices),
        ] {
            if provided < required {
                return CurveError::ScratchTooSmall {
                    buffer,
                    required,
                    provided,
                };
            }
        }
        unreachable!("capacity is insufficient")
    }

    /// Scalar classification and GLV records.
    pub const fn scalars(&self) -> usize {
        self.scalars
    }
    /// Recoding bytes.
    pub const fn digits(&self) -> usize {
        self.digits
    }
    /// Affine points.
    pub const fn affine(&self) -> usize {
        self.affine
    }
    /// Projective points, including intermediate results.
    pub const fn projective(&self) -> usize {
        self.projective
    }
    /// Base-field elements.
    pub const fn field(&self) -> usize {
        self.field
    }
    /// Working indices.
    pub const fn indices(&self) -> usize {
        self.indices
    }
    /// Checked total bytes in these buffers for curve `C`.
    ///
    /// Counts only the required prefixes of the six [`Scratch`] buffers, using
    /// `C`'s element sizes. [`run::BatchPlan::temporary_bytes`] also counts metadata.
    /// Returns [`CurveError::SizeOverflow`] if the byte count overflows `usize`.
    pub const fn bytes<C: PastaCurve>(&self) -> Result<usize, CurveError> {
        let counts = [
            self.scalars,
            self.digits,
            self.affine,
            self.projective,
            self.field,
            self.indices,
        ];
        let sizes = [
            core::mem::size_of::<ScalarStorage<C>>(),
            1,
            core::mem::size_of::<AffinePoint<C>>(),
            core::mem::size_of::<ProjectivePoint<C>>(),
            core::mem::size_of::<PastaField<C::Base>>(),
            core::mem::size_of::<usize>(),
        ];
        let mut bytes = 0usize;
        let mut i = 0;
        while i < counts.len() {
            let Some(n) = counts[i].checked_mul(sizes[i]) else {
                return Err(CurveError::SizeOverflow);
            };
            let Some(total) = bytes.checked_add(n) else {
                return Err(CurveError::SizeOverflow);
            };
            bytes = total;
            i += 1;
        }
        Ok(bytes)
    }
}
/// Initialized caller-owned buffers whose contents may be reused between executions.
///
/// Execution overwrites each value before using it. See
/// [`run::BatchPlan::execute`] for sizing, untouched tails, and reuse after an
/// executor panic.
#[derive(Debug)]
pub struct Scratch<'a, C: PastaCurve> {
    scalars: &'a mut [ScalarStorage<C>],
    digits: &'a mut [u8],
    affine: &'a mut [AffinePoint<C>],
    projective: &'a mut [ProjectivePoint<C>],
    field: &'a mut [PastaField<C::Base>],
    indices: &'a mut [usize],
}
impl<'a, C: PastaCurve> Scratch<'a, C> {
    /// Borrows initialized buffers without checking their lengths.
    ///
    /// Execution checks lengths against its [`Requirements`] before writing.
    /// Suitable initializers are [`ScalarStorage::ZERO`],
    /// [`AffinePoint::GENERATOR`], [`ProjectivePoint::IDENTITY`],
    /// [`PastaField::ZERO`], and zero for byte and index storage.
    pub fn new(
        scalars: &'a mut [ScalarStorage<C>],
        digits: &'a mut [u8],
        affine: &'a mut [AffinePoint<C>],
        projective: &'a mut [ProjectivePoint<C>],
        field: &'a mut [PastaField<C::Base>],
        indices: &'a mut [usize],
    ) -> Self {
        Self {
            scalars,
            digits,
            affine,
            projective,
            field,
            indices,
        }
    }
    /// Reborrows storage for another execution.
    pub fn reborrow(&mut self) -> Scratch<'_, C> {
        Scratch {
            scalars: self.scalars,
            digits: self.digits,
            affine: self.affine,
            projective: self.projective,
            field: self.field,
            indices: self.indices,
        }
    }
    fn capacity(&self) -> Requirements {
        Requirements {
            scalars: self.scalars.len(),
            digits: self.digits.len(),
            affine: self.affine.len(),
            projective: self.projective.len(),
            field: self.field.len(),
            indices: self.indices.len(),
        }
    }
    fn checked(self, r: Requirements) -> Self {
        assert_scratch("scalars", r.scalars, self.scalars.len());
        assert_scratch("digits", r.digits, self.digits.len());
        assert_scratch("affine", r.affine, self.affine.len());
        assert_scratch("projective", r.projective, self.projective.len());
        assert_scratch("field", r.field, self.field.len());
        assert_scratch("indices", r.indices, self.indices.len());
        Self {
            scalars: &mut self.scalars[..r.scalars],
            digits: &mut self.digits[..r.digits],
            affine: &mut self.affine[..r.affine],
            projective: &mut self.projective[..r.projective],
            field: &mut self.field[..r.field],
            indices: &mut self.indices[..r.indices],
        }
    }
}
/// Returns scratch counts for inputs sharing one task budget and memory ceiling.
///
/// Counts account for workspace reuse across jobs and concurrent workers; they
/// are not the sum of independent input requirements. Returns
/// [`CurveError::SizeOverflow`] if sizing exceeds slice limits, or
/// [`CurveError::MemoryLimit`] if the
/// [memory policy](BatchOptions::with_memory_limit) finds no fitting layout.
/// Reusable plan metadata is excluded; size a retained plan with
/// [`run::BatchPlan::requirements`] instead.
#[cfg(test)]
pub(crate) fn batch_requirements<C: PastaCurve>(
    inputs: &[Input<'_, C>],
    options: BatchOptions,
) -> Result<Requirements, CurveError> {
    Ok(schedule::Plan::new(inputs, options)?.requirements)
}
/// Computes one result per input, in input order.
///
/// Size scratch with [`batch_requirements`] for the same inputs and options. Panics
/// before writes unless `output.len() == inputs.len()` and scratch meets the
/// requirement. Planning errors match [`batch_requirements`] and precede writes. An
/// executor panic may partially write output and scratch; reuse after unwinding follows
/// [`Input::execute`].
#[cfg(test)]
pub(crate) fn execute_batch<C: PastaCurve, X: Executor>(
    inputs: &[Input<'_, C>],
    output: &mut [ProjectivePoint<C>],
    options: BatchOptions,
    executor: &X,
    scratch: Scratch<'_, C>,
) -> Result<(), CurveError> {
    assert_length("output", inputs.len(), output.len());
    let plan = schedule::Plan::new(inputs, options)?;
    let scratch = scratch.checked(plan.requirements);
    schedule::execute(&plan, inputs, output, executor, scratch);
    Ok(())
}

#[cfg(test)]
mod experiments;
