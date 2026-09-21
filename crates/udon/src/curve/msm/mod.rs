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
//! For incremental task scheduling, use [`run::MsmPlan`] and [`run::MsmRun`].
//!
//! All runtime operations are allocation-free and variable-time, with no
//! constant-time guarantee for secret bases, scalars, or indices. Resource
//! limits are explicit; see [`ExecutionOptions::with_memory_limit`].
//!
//! Reuse cached bases and validated indices across two signed scalar rows.
//! Rebuild the batch plan when its immutable scalar inputs change:
//!
//! ```
//! use zakura_udon::{
//!     curve::{AffinePoint, Pallas, PreparedAffinePoint, ProjectivePoint, msm::*,
//!         msm::run::{BatchPlan, JobStorage, WorkerStorage}},
//!     exec::SerialExecutor,
//!     field::PastaField,
//! };
//! let bases = [AffinePoint::<Pallas>::GENERATOR; 2];
//! let prepared = bases.map(|base| PreparedAffinePoint::from_affine(&base));
//! let indices = [0, 1, 0];
//! let selection = Selection::indexed(Bases::Prepared(&prepared), &indices)?;
//! let mut jobs = [JobStorage::EMPTY; 1];
//! let mut workers = [WorkerStorage::EMPTY; 1];
//! for row in [[1_i128, -1, 3], [0, 2, 1]] {
//!     let inputs = [selection.with_signed(&row)?];
//!     let plan = BatchPlan::new(&inputs, ExecutionOptions::SERIAL,
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
//!     plan.execute(&mut output, &SerialExecutor, scratch)?;
//!     assert_eq!(output[0], bases[0].mul_projective(&PastaField::from_u64(3)));
//! }
//! # Ok::<(), zakura_udon::curve::CurveError>(())
//! ```

use super::{
    AffinePoint, CurveError, EisensteinTableBatch, PastaCurve, Point, PreparedAffinePoint,
    ProjectivePoint, check_length, check_scratch, checked_count,
};
#[cfg(test)]
use crate::exec::Executor;
use crate::{
    exec::TaskBudget,
    field::{CanonicalUint, PastaField},
};
use core::num::NonZeroUsize;

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
/// validation across scalar rows. Every binding requires exactly [`Self::len`]
/// scalars, returning [`CurveError::LengthMismatch`] otherwise. Only
/// [`Self::with_canonical`] additionally validates scalar values.
/// The mathematical invariants of [`Bases`] remain the producer's responsibility.
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
    fn bind<'s>(&self, scalars: Scalars<'s, C>) -> Result<Input<'s, C>, CurveError>
    where
        'a: 's,
    {
        check_length("scalars", self.len(), scalars.len())?;
        Ok(Input {
            bases: self.bases,
            indices: self.indices,
            scalars,
        })
    }
    /// Binds a field scalar row with an O(1) length check.
    ///
    /// Scalars must satisfy [`Input`]'s reduced-residue invariant. See
    /// [`Selection`] for the exact-length requirement.
    pub fn with_scalars<'s>(
        &self,
        scalars: &'s [PastaField<C::Scalar>],
    ) -> Result<Input<'s, C>, CurveError>
    where
        'a: 's,
    {
        self.bind(Scalars::Raw(scalars))
    }
    /// Binds prepared scalars with an O(1) length check.
    ///
    /// See [`Selection`] for the exact-length requirement.
    pub fn with_prepared_scalars<'s>(
        &self,
        scalars: PreparedScalars<'s, C>,
    ) -> Result<Input<'s, C>, CurveError>
    where
        'a: 's,
    {
        self.bind(Scalars::Prepared(scalars))
    }
    /// Binds unsigned coefficients whose type guarantees the 128-bit bound.
    ///
    /// See [`Selection`] for the exact-length requirement.
    pub fn with_unsigned<'s>(&self, scalars: &'s [u128]) -> Result<Input<'s, C>, CurveError>
    where
        'a: 's,
    {
        self.bind(Scalars::Unsigned(scalars))
    }
    /// Binds signed coefficients, including `i128::MIN`.
    ///
    /// A negative coefficient subtracts its magnitude's base multiple. See
    /// [`Selection`] for the exact-length requirement.
    pub fn with_signed<'s>(&self, scalars: &'s [i128]) -> Result<Input<'s, C>, CurveError>
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
        check_length("scalars", self.len(), scalars.len())?;
        prepared::validate_canonical::<C>(scalars, bits)?;
        self.bind(Scalars::Canonical(scalars))
    }
}

/// Borrowed MSM terms with checked lengths and indices.
///
/// Field scalars must satisfy [`PastaField`]'s reduced-residue invariant; stored
/// bases must satisfy [`Bases`]' mathematical invariants. Violations remain
/// memory-safe but may panic or produce incorrect results. Arithmetic is
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
    /// Returns [`CurveError::LengthMismatch`] unless the lengths are equal.
    pub fn new(
        bases: Bases<'a, C>,
        scalars: &'a [PastaField<C::Scalar>],
    ) -> Result<Self, CurveError> {
        Selection::new(bases).with_scalars(scalars)
    }
    /// Checks indices and scalar length, without gathering bases.
    ///
    /// Returns [`CurveError::LengthMismatch`] unless each scalar has an index.
    /// Index bounds and repeated indices follow [`Selection::indexed`].
    pub fn indexed(
        bases: Bases<'a, C>,
        indices: &'a [u32],
        scalars: &'a [PastaField<C::Scalar>],
    ) -> Result<Self, CurveError> {
        check_length("indices", scalars.len(), indices.len())?;
        Selection::indexed(bases, indices)?.with_scalars(scalars)
    }
    /// Binds one prepared scalar per base.
    ///
    /// Returns [`CurveError::LengthMismatch`] unless the lengths are equal.
    pub fn new_prepared(
        bases: Bases<'a, C>,
        scalars: PreparedScalars<'a, C>,
    ) -> Result<Self, CurveError> {
        Selection::new(bases).with_prepared_scalars(scalars)
    }
    /// Checks indices and binds prepared scalars.
    ///
    /// Returns [`CurveError::LengthMismatch`] unless each scalar has an index.
    /// Index bounds and repeated indices follow [`Selection::indexed`].
    pub fn indexed_prepared(
        bases: Bases<'a, C>,
        indices: &'a [u32],
        scalars: PreparedScalars<'a, C>,
    ) -> Result<Self, CurveError> {
        check_length("indices", scalars.len(), indices.len())?;
        Selection::indexed(bases, indices)?.with_prepared_scalars(scalars)
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
    /// Conservative const scratch counts for unprepared scalars and ordinary bases.
    ///
    /// Supports field and integer scalar rows with [`Bases::Affine`],
    /// [`Bases::Prepared`], or [`Bases::Points`]. Use [`Self::requirements`] for
    /// retained scalars or compact tables: their memory planning can select a
    /// different layout, whose individual buffer counts may be larger.
    ///
    /// Returns [`CurveError::SizeOverflow`] if sizing exceeds slice limits, or
    /// [`CurveError::MemoryLimit`] under the
    /// [memory policy](ExecutionOptions::with_memory_limit).
    #[cfg(test)]
    pub(crate) const fn requirements_for_len(
        terms: usize,
        options: ExecutionOptions,
    ) -> Result<Requirements, CurveError> {
        schedule::single_requirements::<C>(terms, options)
    }
    /// Scratch counts for this input, accounting for retained preparation.
    ///
    /// Errors and memory accounting match [`batch_requirements`].
    #[cfg(test)]
    pub(crate) fn requirements(
        &self,
        options: ExecutionOptions,
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
    pub(crate) fn execute<X: Executor>(
        &self,
        options: ExecutionOptions,
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

/// Booth bucket accumulation preference for tuning and kernel comparisons.
///
/// Short-scalar and joint-table kernels do not use this preference.
/// [Streaming buckets](ExecutionOptions::with_streaming_buckets) always use
/// projective accumulation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Accumulation {
    /// Select a policy using the effective pass size.
    Auto,
    /// Affine pair trees with batch inversion.
    Affine,
    /// Projective buckets, with no affine pair scratch.
    Projective,
    /// Affine early passes and a projective final pass.
    Hybrid,
}
/// Caller-selected concurrency, memory, and arithmetic policy.
///
/// Task, pass, and chunk limits are upper bounds; the planner may use less.
/// Explicit kernel selections override earlier selections as documented on the
/// builders. Automatic kernel and scheduling choices may change between releases.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionOptions {
    task_budget: TaskBudget,
    max_terms_per_pass: Option<NonZeroUsize>,
    memory_limit: Option<usize>,
    chunk_size: Option<NonZeroUsize>,
    window_bits: Option<u8>,
    joint_tables: bool,
    streaming: bool,
    accumulation: Accumulation,
}
impl ExecutionOptions {
    /// Serial execution with no caller-imposed memory ceiling.
    pub const SERIAL: Self = Self {
        task_budget: TaskBudget::SERIAL,
        max_terms_per_pass: None,
        memory_limit: None,
        chunk_size: None,
        window_bits: None,
        joint_tables: false,
        streaming: false,
        accumulation: Accumulation::Auto,
    };
    /// Sets the total scoped concurrency allowance.
    pub const fn with_task_budget(mut self, budget: TaskBudget) -> Self {
        self.task_budget = budget;
        self
    }
    /// Caps terms staged for affine buckets or temporary joint tables.
    ///
    /// `None` removes the cap. This does not bound scalar records or recoding
    /// bytes; use [`Self::with_chunk_size`] or [`Self::with_memory_limit`] for
    /// those. Kernels without this staging, including projective buckets and
    /// short-scalar ladders, do not split their arithmetic at the cap.
    pub const fn with_max_terms_per_pass(mut self, cap: Option<NonZeroUsize>) -> Self {
        self.max_terms_per_pass = cap;
        self
    }
    /// Bounds the typed temporary buffer capacity required by a plan, in bytes.
    ///
    /// Counts the required [`Scratch`] prefixes, intermediate results, and any
    /// [`run::BatchPlan`] metadata prefixes reserved by its sizing query. Excludes
    /// surplus buffer tails, inputs, outputs, retained preparation, fixed stack
    /// frames, and executor resources. This is not a process memory limit.
    ///
    /// The planner may reduce staging, concurrency, or chunk size and change
    /// automatic kernel choices, preserving explicit width and accumulator
    /// selections. It returns [`CurveError::MemoryLimit`] if its search finds no
    /// fitting layout. The search is not exhaustive and does not prove that no
    /// possible layout fits.
    pub const fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.memory_limit = Some(bytes);
        self
    }
    /// Caps the terms prepared and recoded together.
    ///
    /// By default, execution completes each chunk's MSM and reuses its workspace.
    /// [Streaming execution](Self::with_streaming_buckets) instead retains window
    /// buckets across chunks. Both bound scratch independently of the total term
    /// count; a [memory ceiling](Self::with_memory_limit) may reduce the chunk size.
    pub const fn with_chunk_size(mut self, terms: NonZeroUsize) -> Self {
        self.chunk_size = Some(terms);
        self
    }
    /// Forces a Booth width for comparisons or application-specific tuning.
    ///
    /// Returns [`CurveError::InvalidMsmWindow`] outside `4..=12`. Overrides a
    /// previous joint-table selection and preserves streaming execution, if set.
    pub const fn with_booth_width(mut self, bits: u32) -> Result<Self, CurveError> {
        if bits < 4 || bits > 12 {
            return Err(CurveError::InvalidMsmWindow { bits });
        }
        self.window_bits = Some(bits as u8);
        self.joint_tables = false;
        Ok(self)
    }
    /// Forces the compact joint ladder for comparisons or retained table reuse.
    ///
    /// Ordinary bases prepare temporary tables within each arithmetic pass.
    /// Compact bases reuse their tables. Overrides a previously selected Booth
    /// width or streaming mode; [`Accumulation`] does not affect this kernel.
    pub const fn with_joint_tables(mut self) -> Self {
        self.window_bits = None;
        self.joint_tables = true;
        self.streaming = false;
        self
    }
    /// Retains projective window buckets while recoding successive chunks.
    ///
    /// This trades a larger fixed bucket workspace for one collapse per window.
    /// It supplies a Booth width and chunk size unless already configured, and
    /// overrides a joint-table selection. Bucket accumulation is serial within
    /// each job; scalar preparation may use the assigned task budget.
    /// The byte ceiling includes all retained window buckets. A later call to
    /// [`Self::with_joint_tables`] restores complete-chunk execution.
    pub const fn with_streaming_buckets(mut self) -> Self {
        self.streaming = true;
        self.joint_tables = false;
        if self.window_bits.is_none() {
            self.window_bits = Some(8);
        }
        if self.chunk_size.is_none() {
            self.chunk_size = NonZeroUsize::new(256);
        }
        self
    }
    /// Selects an [`Accumulation`] preference for nonstreaming Booth kernels.
    pub const fn with_accumulation(mut self, accumulation: Accumulation) -> Self {
        self.accumulation = accumulation;
        self
    }
    /// Requested task budget; a memory-constrained plan may use fewer workers.
    pub const fn task_budget(&self) -> TaskBudget {
        self.task_budget
    }
    /// Requested byte ceiling, if any.
    pub const fn memory_limit(&self) -> Option<usize> {
        self.memory_limit
    }
    /// Requested per-pass term cap.
    pub const fn max_terms_per_pass(&self) -> Option<NonZeroUsize> {
        self.max_terms_per_pass
    }
}
impl Default for ExecutionOptions {
    fn default() -> Self {
        Self::SERIAL
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
    fn checked(self, r: Requirements) -> Result<Self, CurveError> {
        check_scratch("scalars", r.scalars, self.scalars.len())?;
        check_scratch("digits", r.digits, self.digits.len())?;
        check_scratch("affine", r.affine, self.affine.len())?;
        check_scratch("projective", r.projective, self.projective.len())?;
        check_scratch("field", r.field, self.field.len())?;
        check_scratch("indices", r.indices, self.indices.len())?;
        Ok(Self {
            scalars: &mut self.scalars[..r.scalars],
            digits: &mut self.digits[..r.digits],
            affine: &mut self.affine[..r.affine],
            projective: &mut self.projective[..r.projective],
            field: &mut self.field[..r.field],
            indices: &mut self.indices[..r.indices],
        })
    }
}
/// Returns scratch counts for inputs sharing one task budget and memory ceiling.
///
/// Counts account for workspace reuse across jobs and concurrent workers; they
/// are not the sum of independent input requirements. Returns
/// [`CurveError::SizeOverflow`] if sizing exceeds slice limits, or
/// [`CurveError::MemoryLimit`] if the
/// [memory policy](ExecutionOptions::with_memory_limit) finds no fitting layout.
/// Reusable plan metadata is excluded; size a retained plan with
/// [`run::BatchPlan::requirements`] instead.
#[cfg(test)]
pub(crate) fn batch_requirements<C: PastaCurve>(
    inputs: &[Input<'_, C>],
    options: ExecutionOptions,
) -> Result<Requirements, CurveError> {
    Ok(schedule::Plan::new(inputs, options)?.requirements)
}
/// Computes one result per input, in input order.
///
/// Size scratch with [`batch_requirements`] for the same inputs and options.
/// Returns [`CurveError::LengthMismatch`] unless `output.len() == inputs.len()`;
/// sizing and scratch errors match [`Input::execute`]. All returned errors
/// precede writes. An executor panic may partially write output and scratch;
/// reuse after unwinding follows [`Input::execute`].
#[cfg(test)]
pub(crate) fn execute_batch<C: PastaCurve, X: Executor>(
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

#[cfg(test)]
mod experiments;
