//! Optional generic operations over the shared FFT domain descriptor.

use super::lagrange::Finish;
use super::{CosetDomain, Domain, FftError, LagrangeError};
use crate::field::{Field, PastaField, PrimeModulus, ReductionState};

impl<F: Field> Domain<F> {
    /// Constructs a domain using the field's canonical roots and normalization.
    ///
    /// This constructs the descriptor for implementations of [`Field::domain`]
    /// without calling that method. Generic callers use `F::domain(log_size)`
    /// to select the field's constructor; Pasta adapters wrap the native one.
    ///
    /// Size one is supported. Returns [`FftError::InvalidSize`] above the
    /// field's two-adicity, or [`FftError::SizeOverflow`] if the element count
    /// does not fit `usize` or its slice would exceed `isize::MAX` bytes.
    pub fn from_field(log_size: u32) -> Result<Self, FftError> {
        let root = F::root_of_unity(log_size).ok_or(FftError::InvalidSize)?;
        let inverse_root = F::root_of_unity_inverse(log_size).ok_or(FftError::InvalidSize)?;
        Self::from_roots(log_size, root, inverse_root, || {
            F::power_of_two_inverse(log_size)
        })
    }

    /// Returns the elements `1, root, root^2, ...` in natural order.
    ///
    /// Each element is one multiplication from its predecessor, so the
    /// iterator only runs front to back.
    pub fn elements(self) -> impl ExactSizeIterator<Item = F> {
        let mut current = F::ONE;
        (0..self.size()).map(move |_| {
            let element = current;
            current *= self.root();
            element
        })
    }

    /// Evaluates the vanishing polynomial `X^size - 1` of the domain at `x`.
    pub fn vanishing(self, x: F) -> F {
        self.power_of_size(x, F::square) - F::ONE
    }

    /// Returns whether `x` is an element of the domain.
    pub fn contains(self, x: F) -> bool {
        self.vanishing(x).is_zero()
    }

    /// Replaces coefficients with evaluations at the elements, in natural order.
    ///
    /// Dispatches directly to the field's required [`Field::fft`] implementation.
    ///
    /// # Panics
    ///
    /// Panics before mutation if `values.len()` is not the domain size.
    pub fn transform(self, values: &mut [F]) {
        assert_eq!(values.len(), self.size(), "transform input length");
        F::fft(self, values);
    }

    /// Replaces natural-order evaluations with normalized coefficients.
    ///
    /// Dispatches directly to the field's required [`Field::ifft`] implementation.
    ///
    /// # Panics
    ///
    /// Panics before mutation if `values.len()` is not the domain size.
    pub fn inverse_transform(self, values: &mut [F]) {
        assert_eq!(values.len(), self.size(), "transform input length");
        F::ifft(self, values);
    }

    /// Evaluates the first `evaluations.len()` Lagrange basis polynomials at `x`.
    ///
    /// Writes `l_i(x)` to `evaluations[i]`, where `l_i` is one at `root^i` and
    /// zero at every other element. With `v(X) = X^size - 1`,
    /// `l_i(x) = v(x) * root^i / (size * (x - root^i))`, and the divisions
    /// share one inversion when `scratch` has at least one element per
    /// evaluation. Smaller scratch bounds each batch; empty scratch inverts
    /// each denominator separately. Initial contents do not matter and entries
    /// beyond the evaluation count remain untouched.
    ///
    /// Dispatches through [`Field::lagrange_evaluations`]; Pasta uses
    /// [`super::CosetDomain::evaluate_lagrange`].
    ///
    /// If `x` is the element `root^i`, the evaluations are one at `i` and zero
    /// elsewhere. They are written directly and `Some(i)` is returned so a
    /// caller can recognize the case; `i` may lie beyond the written prefix.
    /// Otherwise the result is `None`.
    ///
    /// # Panics
    ///
    /// Panics before mutation if `evaluations` is longer than the domain.
    pub fn lagrange_evaluations(
        self,
        x: F,
        evaluations: &mut [F],
        scratch: &mut [F],
    ) -> Option<usize> {
        assert!(
            evaluations.len() <= self.size(),
            "Lagrange evaluations exceed the domain size"
        );
        F::lagrange_evaluations(self, x, evaluations, scratch)
    }
}

impl<M: PrimeModulus> Domain<PastaField<M>> {
    /// Returns the `i` with `root^i = x` for an `x` known to be in the subgroup.
    ///
    /// The discrete logarithm is taken bit by bit: `root^(size/2)` is the
    /// unique element of order two, so `x^(size / 2^(j+1))` is `-1` exactly
    /// when bit `j` of the logarithm is set, once the lower bits have been
    /// cleared. This costs `O(log_size^2)` squarings instead of a scan.
    fn index_of(self, mut x: PastaField<M>) -> usize {
        let mut index = 0;
        // `root^-(2^j)` at iteration `j`.
        let mut inverse_power = self.inverse_root();
        for bit in 0..self.log_size() {
            let mut test = x;
            for _ in 0..(self.log_size() - 1 - bit) {
                test = test.square();
            }
            if !test.is_one() {
                index |= 1 << bit;
                x = x.mul(&inverse_power);
            }
            inverse_power = inverse_power.square();
        }
        index
    }
}

impl<M: PrimeModulus> CosetDomain<M> {
    // The consumer prefix API also returns a node's full-domain index, even
    // outside the prefix. Reuse preparation's classification and any local
    // hit before doing the additional lookup that this contract requires.
    pub(crate) fn evaluate_lagrange_with_index(
        self,
        point: &PastaField<M, impl ReductionState>,
        output: &mut [PastaField<M>],
        scratch: &mut [PastaField<M>],
    ) -> Result<Option<usize>, LagrangeError> {
        #[cfg(test)]
        NATIVE_LAGRANGE_COUNT.with(|count| count.set(count.get() + 1));
        let completion = self.prepare_lagrange(point, 0..output.len(), output)?;
        let domain = self.domain();
        let index = if output.is_empty() || self.size() == 1 {
            // These native preparations return directly without testing
            // membership. A singleton basis is one even at an off-domain point.
            let relative = self.relative_point(point);
            let power = domain.power_of_size(relative, PastaField::square);
            power.is_one().then(|| domain.index_of(relative))
        } else {
            match completion.finish {
                Finish::Scale(_) => None,
                Finish::Delta(Some(index)) => Some(index),
                Finish::Delta(None) => Some(domain.index_of(self.relative_point(point))),
            }
        };
        completion.evaluate(output, scratch)?;
        Ok(index)
    }
}

#[cfg(test)]
std::thread_local! {
    static NATIVE_LAGRANGE_COUNT: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn count_native_lagrange_evaluations(f: impl FnOnce()) -> usize {
    NATIVE_LAGRANGE_COUNT.with(|count| {
        let before = count.get();
        f();
        count.get() - before
    })
}
