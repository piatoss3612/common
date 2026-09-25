use super::{CosetDomain, FftError, check_scratch};
use crate::field::{
    ConstantPrefix, PastaField, PrimeModulus, ReductionState, batch_invert, fill_powers,
};
use crate::polynomial::evaluate;

impl<M: PrimeModulus> CosetDomain<M> {
    /// Interpolates a constant prefix and explicit tail into ascending coefficients.
    ///
    /// Input represents natural-order evaluations on this domain. Its length
    /// must equal the domain size `n`, or this returns [`FftError::InvalidLayout`].
    /// With tail length `t`, output needs at least `n` fields and scratch at
    /// least `t`. Unused buffer tails and the input are preserved. All dimension
    /// checks precede writes; insufficient buffers panic before mutation. Rust
    /// borrows keep writable buffers disjoint from each other and input.
    ///
    /// Scratch holds the tail's differences from the constant. Evaluating this
    /// short polynomial recovers the normalized coefficients directly, including
    /// removal of a ZETA shift. Work is `O(n * (t + 1))`, with no allocation,
    /// retained table, inversion, or executor. An empty tail writes just the
    /// constant coefficient and zeros. Both reduction states are accepted.
    /// Arithmetic is variable-time.
    ///
    /// For longer tails, materialize with [`ConstantPrefix::write_values`] and use
    /// [`super::Transform::inverse`]. The crossover depends on the domain size
    /// and available tables; no automatic selection occurs.
    pub fn interpolate_constant_prefix<S: ReductionState>(
        self,
        input: ConstantPrefix<'_, M, S>,
        output: &mut [PastaField<M>],
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        if input.len() != self.size() {
            return Err(FftError::InvalidLayout);
        }
        assert!(output.len() >= self.size(), "coefficient output too short");
        check_scratch(input.tail().len(), scratch.len());
        let output = &mut output[..self.size()];
        let delta = &mut scratch[..input.tail().len()];
        if delta.is_empty() {
            output.fill(PastaField::ZERO);
            output[0] = input.constant();
            return Ok(());
        }
        for (delta, value) in delta.iter_mut().zip(input.tail()) {
            *delta = value.sub(&input.constant());
        }
        // Tail indices are n-t+r. The inverse DFT is therefore
        // c_k = [k=0]*a + (shift^-1 * root^t)^k * delta(root^-k) / n.
        let step = self
            .domain()
            .root()
            .pow_u64(delta.len() as u64)
            .mul(&self.inverse_shift());
        let mut scale = self.domain().size_inverse();
        let mut point = PastaField::ONE;
        for coefficient in output.iter_mut() {
            *coefficient = evaluate(delta, &point).mul(&scale);
            point = point.mul(&self.domain().inverse_root());
            scale = scale.mul(&step);
        }
        output[0] = output[0].add(&input.constant());
        Ok(())
    }
}

/// Retained Lagrange samples for extending constant-prefix evaluations.
///
/// This borrows samples of the base domain's zeroth Lagrange polynomial at
/// every target node, in natural order. Rotating the samples gives every other
/// basis polynomial, so one table serves all tail lengths and values. The base
/// and extended domains can independently use subgroup or ZETA shifts; the
/// target size must be at least the base size. Output is the unique polynomial
/// of degree below the base size evaluated on the target domain.
///
/// [`Self::prepare`] fills caller storage; [`Self::bind`] borrows imported
/// samples whose mathematical contents are the caller's responsibility.
/// Preparation, binding and evaluation are serial, allocation-free, and
/// variable-time. The table retains one field per target node. Execution
/// needs no scratch; preparation accepts bounded inversion scratch.
///
/// For longer tails, use [`ConstantPrefix::write_values`] followed by
/// [`super::Expansion::evaluations`] for a subgroup base, or inverse and forward
/// transforms for a coset base. Callers select the method and retain its tables.
///
/// ```
/// use zakura_udon::{
///     fft::{ConstantPrefixExpansion, Domain},
///     field::{ConstantPrefix, Fp},
/// };
/// let base = Domain::new(2)?.subgroup();
/// let target = Domain::new(3)?.subgroup();
/// let tail = [<Fp>::from_u64(9)];
/// let input = ConstantPrefix::new(4, &<Fp>::from_u64(3), &tail)?;
/// let mut samples = [Fp::ZERO; 8];
/// let plan = ConstantPrefixExpansion::prepare(base, target, &mut samples,
///     &mut [Fp::ZERO; 8])?;
/// let mut output = [Fp::ZERO; 8];
/// plan.evaluate(input, &mut output)?;
/// assert_eq!(output[0].reduce(), <Fp>::from_u64(3).reduce());
/// assert_eq!(output[6].reduce(), tail[0].reduce());
/// # Ok::<(), Box<dyn std::error::Error>>(())
/// ```
#[derive(Clone, Copy)]
pub struct ConstantPrefixExpansion<'a, M: PrimeModulus> {
    base: CosetDomain<M>,
    extended: CosetDomain<M>,
    samples: &'a [PastaField<M>],
}

impl<M: PrimeModulus> core::fmt::Debug for ConstantPrefixExpansion<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ConstantPrefixExpansion")
            .field("base", &self.base)
            .field("extended", &self.extended)
            .field("samples", &self.samples)
            .finish()
    }
}

impl<'a, M: PrimeModulus> ConstantPrefixExpansion<'a, M> {
    /// Prepares `extended.size()` samples, preserving any storage tail.
    ///
    /// Returns [`FftError::InvalidLayout`] if the base is larger than the target.
    /// Insufficient sample storage panics. Both checks precede writes to either
    /// buffer. Initial contents are ignored. Scratch uses the bounded batching
    /// contract of [`batch_invert`]; empty scratch is valid, and entries beyond
    /// the target size are untouched. With full scratch at most one inversion is
    /// needed. Work outside inversion is linear in the target size.
    pub fn prepare(
        base: CosetDomain<M>,
        extended: CosetDomain<M>,
        storage: &'a mut [PastaField<M>],
        scratch: &mut [PastaField<M>],
    ) -> Result<Self, FftError> {
        check_domains(base, extended)?;
        assert!(
            storage.len() >= extended.size(),
            "Lagrange samples too short"
        );
        let samples = &mut storage[..extended.size()];
        if base.size() == 1 {
            samples.fill(PastaField::ONE);
        } else if base == extended {
            samples.fill(PastaField::ZERO);
            samples[0] = PastaField::ONE;
        } else {
            let ratio = extended.shift().mul(&base.inverse_shift());
            fill_powers(ratio, extended.domain().root(), samples);
            for value in samples.iter_mut() {
                *value = value.sub(&PastaField::<M>::ONE);
            }
            batch_invert(samples, scratch);
            let (mut numerator, step, inverse_size) = sample_progression(base, extended);
            for value in samples.iter_mut() {
                *value = value.mul(&numerator.sub(&inverse_size));
                numerator = numerator.mul(&step);
            }
            if base.shift().reduce() == extended.shift().reduce() {
                // The quotient has a removable singularity at the first node.
                samples[0] = PastaField::ONE;
            }
        }
        Ok(Self {
            base,
            extended,
            samples,
        })
    }

    /// Borrows imported natural-order samples with constant work.
    ///
    /// Returns [`FftError::InvalidLayout`] if the base is larger than the target
    /// or the sample count differs from `extended.size()`. The caller must
    /// supply samples of the base domain's zeroth Lagrange polynomial at each
    /// target node in natural order. Contents are not checked; incorrect samples
    /// give incorrect results. Binding performs no arithmetic and needs no
    /// scratch.
    pub fn bind(
        base: CosetDomain<M>,
        extended: CosetDomain<M>,
        samples: &'a [PastaField<M>],
    ) -> Result<Self, FftError> {
        check_domains(base, extended)?;
        if samples.len() != extended.size() {
            return Err(FftError::InvalidLayout);
        }
        Ok(Self {
            base,
            extended,
            samples,
        })
    }

    /// Domain of the input evaluations.
    pub const fn base(self) -> CosetDomain<M> {
        self.base
    }

    /// Domain and natural ordering of the output evaluations.
    pub const fn extended(self) -> CosetDomain<M> {
        self.extended
    }

    /// Samples of the base domain's zeroth basis polynomial on the target.
    pub const fn samples(self) -> &'a [PastaField<M>] {
        self.samples
    }

    /// Writes extended evaluations from natural-order constant-prefix input.
    ///
    /// Returns [`FftError::InvalidLayout`] if input length differs from the base
    /// size. Output needs at least `extended.size()` fields, or this panics.
    /// Validation precedes all writes. Unused output tails and input are
    /// preserved. Rust borrows enforce disjoint writable storage.
    ///
    /// For a tail of length `t` and target size `N`, work is `O(N * (t + 1))`,
    /// with no scratch or inversions. An empty tail fills the constant. Full
    /// tails are valid but may favor ordinary FFT extension. Both input
    /// reduction states are accepted.
    pub fn evaluate<S: ReductionState>(
        self,
        input: ConstantPrefix<'_, M, S>,
        output: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        if input.len() != self.base.size() {
            return Err(FftError::InvalidLayout);
        }
        let size = self.extended.size();
        assert!(output.len() >= size, "extended output too short");
        let output = &mut output[..size];
        output.fill(input.constant());
        let stride = size / self.base.size();
        for (i, value) in input.tail().iter().enumerate() {
            let delta = value.sub(&input.constant());
            if delta.is_zero() {
                continue;
            }
            // L_i(x) = L_0(x * root_base^-i). Tail indices run from n-t
            // to n-1. Split the rotated table at its wrap instead of taking
            // a modular index for every output. offset <= size cannot overflow.
            let offset = stride * (input.tail().len() - i);
            let (first, second) = output.split_at_mut(size - offset);
            for (out, sample) in first.iter_mut().zip(&self.samples[offset..]) {
                *out = delta.mul_add(sample, out);
            }
            for (out, sample) in second.iter_mut().zip(&self.samples[..offset]) {
                *out = delta.mul_add(sample, out);
            }
        }
        Ok(())
    }
}

fn check_domains<M: PrimeModulus>(
    base: CosetDomain<M>,
    extended: CosetDomain<M>,
) -> Result<(), FftError> {
    if base.size() > extended.size() {
        Err(FftError::InvalidLayout)
    } else {
        Ok(())
    }
}

fn sample_progression<M: PrimeModulus>(
    base: CosetDomain<M>,
    extended: CosetDomain<M>,
) -> (PastaField<M>, PastaField<M>, PastaField<M>) {
    let inverse_size = base.domain().size_inverse();
    let first = extended
        .shift()
        .mul(&base.inverse_shift())
        .pow_u64(base.size() as u64)
        .mul(&inverse_size);
    let step = extended.domain().root().pow_u64(base.size() as u64);
    (first, step, inverse_size)
}
