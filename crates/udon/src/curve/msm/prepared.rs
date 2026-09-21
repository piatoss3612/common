//! Scalar classification and GLV storage independent of an execution geometry.

use core::marker::PhantomData;

use super::{CurveError, PastaCurve, Scalars, check_scratch, checked_count};
use crate::{
    curve::{glv::decompose_canonical, parameters::GlvParameters},
    exec::{Executor, TaskBudget, for_each_chunk_mut},
    field::{CanonicalUint, PastaField, PrimeModulus, word::subtract_limbs},
};

/// Initialized storage for one prepared scalar.
///
/// Its private representation retains signed GLV components and small-integer
/// classification, independently of window width and arithmetic backend. It is
/// runtime storage for variable-time arithmetic, not a POD or serialization
/// format. Initialize caller-owned arrays with [`Self::ZERO`].
#[derive(Clone, Copy, Debug)]
pub struct ScalarStorage<C: PastaCurve> {
    pub(super) halves: [i128; 2],
    // Magnitude and sign represent the scalar only when bits <= 128. The 255
    // sentinel selects full-width arithmetic, which uses the GLV halves instead.
    pub(super) magnitude: u128,
    pub(super) bits: u8,
    pub(super) negative: bool,
    marker: PhantomData<C>,
}

impl<C: PastaCurve> PartialEq for ScalarStorage<C> {
    fn eq(&self, other: &Self) -> bool {
        self.halves == other.halves
            && self.magnitude == other.magnitude
            && self.bits == other.bits
            && self.negative == other.negative
    }
}
impl<C: PastaCurve> Eq for ScalarStorage<C> {}

impl<C: PastaCurve> ScalarStorage<C> {
    /// Valid initializer representing zero.
    pub const ZERO: Self = Self {
        halves: [0; 2],
        magnitude: 0,
        bits: 0,
        negative: false,
        marker: PhantomData,
    };

    pub(super) fn field(value: &PastaField<C::Scalar>) -> Self {
        Self::canonical(value.to_canonical_uint())
    }

    /// Requires an integer strictly below `C`'s scalar modulus.
    fn canonical(integer: CanonicalUint) -> Self {
        let limbs = integer.limbs();
        let opposite = subtract_limbs(&C::Scalar::MODULUS, &limbs).0;
        let (small, negative) = if limbs[2] | limbs[3] == 0 {
            (limbs, false)
        } else if opposite[2] | opposite[3] == 0 {
            (opposite, true)
        } else {
            (limbs, false)
        };
        let magnitude = u128::from(small[0]) | (u128::from(small[1]) << 64);
        let bits = if small[2] | small[3] == 0 {
            (128 - magnitude.leading_zeros()) as u8
        } else {
            255
        };
        let (a, b) = if bits <= 64 {
            // Small coefficients already satisfy both GLV bounds. Retaining
            // this decomposition avoids lattice rounding for Boolean and
            // bounded-integer inputs, including small negative residues.
            let a = magnitude as i128;
            (if negative { -a } else { a }, 0)
        } else {
            decompose_canonical::<C>(integer)
        };
        debug_assert!(a.unsigned_abs() <= GlvParameters::<C>::BOUNDS[0]);
        debug_assert!(b.unsigned_abs() <= GlvParameters::<C>::BOUNDS[1]);
        Self {
            halves: [a, b],
            magnitude,
            bits,
            negative,
            marker: PhantomData,
        }
    }
}

/// Scalar preparation that can be reused with different bases and policies.
///
/// The original scalar borrow is released after preparation. Classification and
/// GLV decomposition are retained; execution may recode them for its selected
/// kernel. Use [`Self::cache`] to additionally retain a particular recoding.
/// Handles can be shared across concurrent executions with separate scratch.
/// Preparation, caching, and execution are variable-time, with no constant-time
/// guarantee for secret coefficients.
///
/// This example reuses one preparation with two base sets after overwriting the
/// original scalar row:
///
/// ```
/// use zakura_udon::{
///     curve::{AffinePoint, Pallas, ProjectivePoint, msm::*},
///     exec::{SerialExecutor, TaskBudget},
///     field::{Fq, PastaField},
/// };
/// let mut scalars = [Fq::from_u64(1), Fq::from_u64(2)];
/// let mut records = [ScalarStorage::<Pallas>::ZERO; 2];
/// let prepared = PreparedScalars::prepare(
///     &scalars, &mut records, TaskBudget::SERIAL, &SerialExecutor,
/// )?;
/// scalars.fill(Fq::ZERO);
/// let bases = [AffinePoint::<Pallas>::GENERATOR; 2];
/// let opposite = bases.map(|p| p.neg());
/// let inputs = [
///     Input::new_prepared(Bases::Affine(&bases), prepared)?,
///     Input::new_prepared(Bases::Affine(&opposite), prepared)?,
/// ];
/// let options = BatchOptions::default();
/// let mut jobs = [run::JobStorage::EMPTY; 2];
/// let mut workers = [run::WorkerStorage::EMPTY; 1];
/// let plan = run::BatchPlan::new(&inputs, options, &mut jobs, &mut workers)?;
/// let r = plan.requirements();
/// # let mut records = vec![ScalarStorage::ZERO; r.scalars()];
/// # let mut digits = vec![0; r.digits()];
/// # let mut affine = vec![AffinePoint::GENERATOR; r.affine()];
/// # let mut projective = vec![ProjectivePoint::IDENTITY; r.projective()];
/// # let mut field = vec![PastaField::ZERO; r.field()];
/// # let mut indices = vec![0; r.indices()];
/// # let scratch = Scratch::new(&mut records, &mut digits, &mut affine,
/// #     &mut projective, &mut field, &mut indices);
/// // Allocate and initialize scratch from `r`, as in the `msm` module example.
/// let mut output = [ProjectivePoint::IDENTITY; 2];
/// plan.execute(&mut output, &SerialExecutor, scratch)?;
/// let expected = bases[0].mul_projective(&Fq::from_u64(3));
/// assert_eq!(output, [expected, expected.neg()]);
/// # Ok::<(), zakura_udon::curve::CurveError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct PreparedScalars<'a, C: PastaCurve> {
    pub(super) records: &'a [ScalarStorage<C>],
    pub(super) shape: super::recode::Shape,
    pub(super) cached: Option<super::recode::Cache<'a>>,
}

impl<'a, C: PastaCurve> PreparedScalars<'a, C> {
    /// Returns the number of [`ScalarStorage`] entries required for `terms`.
    ///
    /// Returns [`CurveError::SizeOverflow`] if the slice would be too large.
    pub const fn storage_len(terms: usize) -> Result<usize, CurveError> {
        checked_count::<ScalarStorage<C>>(terms, 1)
    }

    /// Classifies and decomposes scalars into the required prefix of `storage`.
    ///
    /// Scalars must satisfy [`PastaField`]'s reduced-residue invariant; see
    /// [`super::Input`] for the consequences of invalid data. Provide initialized
    /// storage with at least [`Self::storage_len`] entries for `scalars.len()`;
    /// entries beyond that prefix are untouched.
    ///
    /// Returns [`CurveError::SizeOverflow`] or [`CurveError::ScratchTooSmall`]
    /// before any writes. An executor panic may partially write storage, which
    /// can be reused without clearing after all scoped jobs finish unwinding,
    /// as required by [`Executor`].
    pub fn prepare<X: Executor>(
        scalars: &[PastaField<C::Scalar>],
        storage: &'a mut [ScalarStorage<C>],
        budget: TaskBudget,
        executor: &X,
    ) -> Result<Self, CurveError> {
        Self::from_source(Scalars::Raw(scalars), storage, budget, executor)
    }

    /// Prepares unsigned 128-bit coefficients without Montgomery conversion.
    ///
    /// The type guarantees the bound; errors and scratch contracts match
    /// [`Self::prepare`].
    pub fn unsigned<X: Executor>(
        scalars: &[u128],
        storage: &'a mut [ScalarStorage<C>],
        budget: TaskBudget,
        executor: &X,
    ) -> Result<Self, CurveError> {
        Self::from_source(Scalars::Unsigned(scalars), storage, budget, executor)
    }

    /// Prepares signed 128-bit coefficients, including `i128::MIN`.
    ///
    /// Negative coefficients subtract their magnitude's base multiple. Errors,
    /// scratch requirements, and executor panic behavior match [`Self::prepare`].
    pub fn signed<X: Executor>(
        scalars: &[i128],
        storage: &'a mut [ScalarStorage<C>],
        budget: TaskBudget,
        executor: &X,
    ) -> Result<Self, CurveError> {
        Self::from_source(Scalars::Signed(scalars), storage, budget, executor)
    }

    /// Prepares canonical integers after checking the modulus and `bits` bound.
    ///
    /// Accepted bounds and scalar errors match [`super::Selection::with_canonical`].
    /// Storage requirements, sizing errors, and executor panic behavior match
    /// [`Self::prepare`]. All returned errors precede storage writes.
    pub fn canonical<X: Executor>(
        scalars: &[CanonicalUint],
        bits: usize,
        storage: &'a mut [ScalarStorage<C>],
        budget: TaskBudget,
        executor: &X,
    ) -> Result<Self, CurveError> {
        validate_canonical::<C>(scalars, bits)?;
        Self::from_source(Scalars::Canonical(scalars), storage, budget, executor)
    }

    fn from_source<X: Executor>(
        source: Scalars<'_, C>,
        storage: &'a mut [ScalarStorage<C>],
        budget: TaskBudget,
        executor: &X,
    ) -> Result<Self, CurveError> {
        let n = Self::storage_len(source.len())?;
        check_scratch("scalars", n, storage.len())?;
        let records = &mut storage[..n];
        prepare(source, records, budget, executor);
        Ok(Self {
            shape: super::recode::Shape::of(records),
            records,
            cached: None,
        })
    }

    /// Returns additional bytes needed to cache recoding for `options`.
    ///
    /// The scalar shape and requested kernel select a recoding for the complete
    /// scalar vector. Chunk, pass, and accumulator settings do not
    /// constrain this retained allocation; it is separate from execution scratch
    /// and scalar record storage. Returns [`CurveError::SizeOverflow`] if the
    /// byte slice would be too large. See [`Self::cache`] for reuse conditions.
    pub fn cache_len(&self, options: super::ArithmeticOptions) -> Result<usize, CurveError> {
        super::recode::Geometry::for_shape(self.len(), self.shape, options).storage_len(self.len())
    }

    /// Caches recoding in caller-owned bytes and returns a new borrowed handle.
    ///
    /// Size storage with [`Self::cache_len`]. Execution can reuse the cache when
    /// its recoding geometry matches, the complete input fits in one chunk, and
    /// streaming is disabled. Otherwise it recodes retained GLV data into scratch;
    /// it never interprets a cache using another geometry.
    ///
    /// Returns [`CurveError::SizeOverflow`] or [`CurveError::ScratchTooSmall`]
    /// before writes. Bytes beyond the required prefix remain untouched.
    pub fn cache(
        &self,
        options: super::ArithmeticOptions,
        storage: &'a mut [u8],
    ) -> Result<Self, CurveError> {
        let geometry = super::recode::Geometry::for_shape(self.len(), self.shape, options);
        let len = geometry.storage_len(self.len())?;
        check_scratch("digits", len, storage.len())?;
        let digits = &mut storage[..len];
        super::recode::write(self.records, geometry, digits);
        Ok(Self {
            records: self.records,
            shape: self.shape,
            cached: Some(super::recode::Cache { geometry, digits }),
        })
    }

    /// Number of prepared coefficients.
    pub const fn len(&self) -> usize {
        self.records.len()
    }
    /// Whether the vector is empty.
    pub const fn is_empty(&self) -> bool {
        self.records.is_empty()
    }
    /// Bytes borrowed for scalar records and optional cached recoding.
    ///
    /// Excludes this handle and any unused tails of the supplied storage.
    pub const fn retained_bytes(&self) -> usize {
        core::mem::size_of_val(self.records)
            + match self.cached {
                Some(c) => c.digits.len(),
                None => 0,
            }
    }
}

pub(super) fn validate_canonical<C: PastaCurve>(
    scalars: &[CanonicalUint],
    bits: usize,
) -> Result<(), CurveError> {
    if bits > 256 {
        return Err(CurveError::InvalidScalar { position: 0 });
    }
    for (position, scalar) in scalars.iter().enumerate() {
        if !scalar.fits_in_bits(bits) || subtract_limbs(&scalar.limbs(), &C::Scalar::MODULUS).1 == 0
        {
            return Err(CurveError::InvalidScalar { position });
        }
    }
    Ok(())
}

pub(super) fn prepare<C: PastaCurve, X: Executor>(
    source: Scalars<'_, C>,
    records: &mut [ScalarStorage<C>],
    budget: TaskBudget,
    executor: &X,
) {
    for_each_chunk_mut(records, 256, budget, executor, |chunk, records, _| {
        let start = chunk * 256;
        prepare_chunk(source.slice(start..start + records.len()), records);
    });
}

// Both drivers call the same bounded conversion kernel. It has no executor and
// cannot retain a resource while requesting another one.
pub(super) fn prepare_chunk<C: PastaCurve>(
    source: Scalars<'_, C>,
    records: &mut [ScalarStorage<C>],
) {
    for (i, record) in records.iter_mut().enumerate() {
        *record = match source {
            Scalars::Prepared(s) => s.records[i],
            Scalars::Raw(s) => ScalarStorage::canonical(s[i].to_canonical_uint()),
            Scalars::Canonical(s) => ScalarStorage::canonical(s[i]),
            Scalars::Unsigned(s) => ScalarStorage::canonical(CanonicalUint::from_limbs([
                s[i] as u64,
                (s[i] >> 64) as u64,
                0,
                0,
            ])),
            Scalars::Signed(s) => {
                let magnitude = s[i].unsigned_abs();
                let mut limbs = [magnitude as u64, (magnitude >> 64) as u64, 0, 0];
                if s[i] < 0 {
                    limbs = subtract_limbs(&C::Scalar::MODULUS, &limbs).0;
                }
                ScalarStorage::canonical(CanonicalUint::from_limbs(limbs))
            }
        };
    }
}
