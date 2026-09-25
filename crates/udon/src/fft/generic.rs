//! Optional generic operations over the shared FFT domain descriptor.

use super::{Domain, FftError};
use crate::field::FftField;

impl<F: FftField> Domain<F> {
    /// Constructs a domain of `2^log_size` field elements.
    ///
    /// `log_size = 0` gives the singleton subgroup containing one. Returns
    /// [`FftError::InvalidSize`] above the field's two-adicity (32 for both
    /// Pasta fields), or [`FftError::SizeOverflow`] if the element count does
    /// not fit `usize` or a slice of that length would exceed `isize::MAX`
    /// bytes.
    pub fn new(log_size: u32) -> Result<Self, FftError> {
        let root = F::root_of_unity(log_size).ok_or(FftError::InvalidSize)?;
        let inverse_root = F::root_of_unity_inverse(log_size).ok_or(FftError::InvalidSize)?;
        Self::from_roots(log_size, root, inverse_root, || {
            F::power_of_two_inverse(log_size)
        })
    }

    /// Constructs a domain from its nonzero power-of-two element count.
    ///
    /// Returns [`FftError::InvalidSize`] for zero or a non-power-of-two length;
    /// other size limits and errors are those of [`Self::new`].
    pub fn for_size(size: usize) -> Result<Self, FftError> {
        if !size.is_power_of_two() {
            return Err(FftError::InvalidSize);
        }
        Self::new(size.ilog2())
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

    /// Returns `x^size` by `log_size` squarings.
    fn power_of_size(self, x: F) -> F {
        (0..self.log_size()).fold(x, |power, _| power.square())
    }

    /// Evaluates the vanishing polynomial `X^size - 1` of the domain at `x`.
    pub fn vanishing(self, x: F) -> F {
        self.power_of_size(x) - F::ONE
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
    /// share one inversion through `scratch`, which needs one element per
    /// evaluation. Its initial contents do not matter.
    ///
    /// If `x` is the element `root^i`, the evaluations are one at `i` and zero
    /// elsewhere. They are written directly and `Some(i)` is returned so a
    /// caller can recognize the case; `i` may lie beyond the written prefix.
    /// Otherwise the result is `None`.
    ///
    /// # Panics
    ///
    /// Panics before mutation if `evaluations` is longer than the domain or
    /// `scratch` is shorter than `evaluations`.
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
        assert!(
            scratch.len() >= evaluations.len(),
            "Lagrange evaluation scratch must cover every evaluation"
        );

        let vanishing = self.vanishing(x);
        if vanishing.is_zero() {
            let index = self.index_of(x);
            for (position, evaluation) in evaluations.iter_mut().enumerate() {
                *evaluation = if position == index { F::ONE } else { F::ZERO };
            }
            return Some(index);
        }

        // Every difference is nonzero because x is outside the domain.
        let mut power = F::ONE;
        for evaluation in evaluations.iter_mut() {
            *evaluation = x - power;
            power *= self.root();
        }
        F::batch_invert(evaluations, scratch);

        let mut numerator = vanishing * self.size_inverse();
        for evaluation in evaluations.iter_mut() {
            *evaluation *= numerator;
            numerator *= self.root();
        }
        None
    }

    /// Returns the `i` with `root^i = x` for an `x` known to be in the domain.
    ///
    /// The discrete logarithm is taken bit by bit: `root^(size/2)` is the
    /// unique element of order two, so `x^(size / 2^(j+1))` is `-1` exactly
    /// when bit `j` of the logarithm is set, once the lower bits have been
    /// cleared. This costs `O(log_size^2)` squarings instead of a scan.
    fn index_of(self, mut x: F) -> usize {
        let mut index = 0;
        // `root^-(2^j)` at iteration `j`.
        let mut inverse_power = self.inverse_root();
        for bit in 0..self.log_size() {
            let mut test = x;
            for _ in 0..(self.log_size() - 1 - bit) {
                test = test.square();
            }
            if test != F::ONE {
                index |= 1 << bit;
                x *= inverse_power;
            }
            inverse_power = inverse_power.square();
        }
        index
    }
}
