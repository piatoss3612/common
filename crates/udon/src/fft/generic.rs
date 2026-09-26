//! Optional generic operations over the shared FFT domain descriptor.

use super::lagrange::Finish;
use super::{CosetDomain, Domain, LagrangeError};
use crate::field::{FftField, PastaField, PrimeModulus, ReductionState};

impl<F: FftField> Domain<F> {
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
    /// Dispatches directly to the field's required [`FftField::fft`] implementation.
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
    /// Dispatches directly to the field's required [`FftField::ifft`] implementation.
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
    /// Dispatches through [`FftField::lagrange_evaluations`]; Pasta uses
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

impl<F: Copy + Eq + core::ops::MulAssign> Domain<F> {
    /// Returns the `i` with `root^i = x` for an `x` known to be in the subgroup.
    ///
    /// The discrete logarithm is taken bit by bit: `root^(size/2)` is the
    /// unique element of order two, so `x^(size / 2^(j+1))` is `-1` exactly
    /// when bit `j` of the logarithm is set, once the lower bits have been
    /// cleared. This costs `O(log_size^2)` squarings instead of a scan.
    fn index_of(self, mut x: F, one: F, square: impl Fn(&F) -> F) -> usize {
        let mut index = 0;
        // `root^-(2^j)` at iteration `j`.
        let mut inverse_power = self.inverse_root();
        for bit in 0..self.log_size() {
            let mut test = x;
            for _ in 0..(self.log_size() - 1 - bit) {
                test = square(&test);
            }
            if test != one {
                index |= 1 << bit;
                x *= inverse_power;
            }
            inverse_power = square(&inverse_power);
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
        let completion = self.prepare_lagrange(point, 0..output.len(), output)?;
        let domain = self.domain();
        let index = if output.is_empty() || self.size() == 1 {
            // These native preparations return directly without testing
            // membership. A singleton basis is one even at an off-domain point.
            let relative = self.relative_point(point);
            let power = domain.power_of_size(relative, PastaField::square);
            power
                .is_one()
                .then(|| domain.index_of(relative, PastaField::ONE, PastaField::square))
        } else {
            match completion.finish {
                Finish::Scale(_) => None,
                Finish::Delta(Some(index)) => Some(index),
                Finish::Delta(None) => Some(domain.index_of(
                    self.relative_point(point),
                    PastaField::ONE,
                    PastaField::square,
                )),
            }
        };
        completion.evaluate(output, scratch)?;
        Ok(index)
    }
}

// Generic fields may use the field-operation formula; Pasta overrides the hook
// with its native range evaluator and scaled batch inversion.
pub(crate) fn lagrange_evaluations<F: FftField>(
    domain: Domain<F>,
    x: F,
    evaluations: &mut [F],
    scratch: &mut [F],
) -> Option<usize> {
    assert!(
        evaluations.len() <= domain.size(),
        "Lagrange evaluations exceed the domain size"
    );
    #[cfg(test)]
    LAGRANGE_COUNT.with(|count| count.set(count.get() + 1));
    let vanishing = domain.vanishing(x);
    if vanishing.is_zero() {
        let index = domain.index_of(x, F::ONE, F::square);
        for (position, evaluation) in evaluations.iter_mut().enumerate() {
            *evaluation = if position == index { F::ONE } else { F::ZERO };
        }
        return Some(index);
    }

    // Every difference is nonzero because x is outside the domain.
    let mut power = F::ONE;
    for evaluation in evaluations.iter_mut() {
        *evaluation = x - power;
        power *= domain.root();
    }
    F::batch_invert(evaluations, scratch);

    let mut numerator = vanishing * domain.size_inverse();
    for evaluation in evaluations.iter_mut() {
        *evaluation *= numerator;
        numerator *= domain.root();
    }
    None
}

#[cfg(test)]
std::thread_local! {
    static LAGRANGE_COUNT: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(test)]
pub(super) fn count_lagrange_evaluations(f: impl FnOnce()) -> usize {
    LAGRANGE_COUNT.with(|count| {
        let before = count.get();
        f();
        count.get() - before
    })
}
