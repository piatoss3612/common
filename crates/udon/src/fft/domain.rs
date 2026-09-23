//! Radix-2 evaluation domains and their coset shifts.

use super::{
    FftError, check_element_count, factors::Shift,
    reference::{Butterfly, inverse_transform, transform},
};
use crate::field::{FftField, PastaField, PrimeModulus, batch_invert_with_scratch};

/// A radix-2 subgroup with its canonical root of unity and the scalars
/// transforms over it need.
///
/// For size `n`, the subgroup consists of `root^j` for `0 <= j < n`. Roots
/// come from [`FftField::root_of_unity`]; if `small` and `large` are domains
/// in the same field, `large.root()^(large.size()/small.size())` equals
/// `small.root()` whenever `small.size() <= large.size()`.
/// Domains in the same field compare equal exactly when their sizes match.
///
/// `Domain<PastaField<M>>` configures the Pasta transforms through
/// [`Self::subgroup`] and [`Self::coset`]. The generic methods run the
/// [`reference`](super::reference) transforms over any [`Butterfly`] value and
/// evaluate the domain's vanishing and Lagrange polynomials.
#[derive(Clone, Copy, Debug)]
pub struct Domain<F> {
    log_size: u32,
    size: usize,
    root: F,
    inverse_root: F,
    size_inverse: F,
}

impl<F: FftField> PartialEq for Domain<F> {
    fn eq(&self, other: &Self) -> bool {
        // Construction fixes every field parameter from the validated size.
        self.size == other.size
    }
}

impl<F: FftField> Eq for Domain<F> {}

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
        let size = 1usize.checked_shl(log_size).ok_or(FftError::SizeOverflow)?;
        check_element_count::<F>(size)?;
        Ok(Self {
            log_size,
            size,
            root,
            inverse_root,
            size_inverse: F::power_of_two_inverse(log_size),
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

    /// The base-two logarithm of the size.
    pub const fn log_size(self) -> u32 {
        self.log_size
    }
    /// The number of coefficients or evaluations.
    pub const fn size(self) -> usize {
        self.size
    }
    /// The root defining natural evaluation order.
    pub const fn root(self) -> F {
        self.root
    }
    /// The inverse of [`Self::root`].
    pub const fn inverse_root(self) -> F {
        self.inverse_root
    }
    /// The multiplicative inverse of the domain size in the field.
    pub const fn size_inverse(self) -> F {
        self.size_inverse
    }

    /// Returns the elements `1, root, root^2, ...` in natural order.
    ///
    /// Each element is one multiplication from its predecessor, so the
    /// iterator only runs front to back.
    pub fn elements(self) -> impl ExactSizeIterator<Item = F> {
        let mut current = F::ONE;
        (0..self.size).map(move |_| {
            let element = current;
            current *= self.root;
            element
        })
    }

    /// Returns `x^size` by `log_size` squarings.
    fn power_of_size(self, x: F) -> F {
        (0..self.log_size).fold(x, |power, _| power.square())
    }

    /// Evaluates the vanishing polynomial `X^size - 1` of the domain at `x`.
    pub fn vanishing(self, x: F) -> F {
        self.power_of_size(x) - F::ONE
    }

    /// Returns whether `x` is an element of the domain.
    pub fn contains(self, x: F) -> bool {
        self.vanishing(x).is_zero()
    }

    /// Replaces coefficients with evaluations at the elements, in natural
    /// order, using the reference [`transform`].
    ///
    /// # Panics
    ///
    /// Panics before mutation if `values.len()` is not the domain size.
    pub fn transform<V: Butterfly<F>>(self, values: &mut [V]) {
        assert_eq!(values.len(), self.size, "transform input length");
        transform(values, &self.root);
    }

    /// Replaces natural-order evaluations with coefficients, using the
    /// reference [`inverse_transform`].
    ///
    /// # Panics
    ///
    /// Panics before mutation if `values.len()` is not the domain size.
    pub fn inverse_transform<V: Butterfly<F>>(self, values: &mut [V]) {
        assert_eq!(values.len(), self.size, "transform input length");
        inverse_transform(values, &self.inverse_root, &self.size_inverse);
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
            evaluations.len() <= self.size,
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
            power *= self.root;
        }
        batch_invert_with_scratch(evaluations, scratch);

        let mut numerator = vanishing * self.size_inverse;
        for evaluation in evaluations.iter_mut() {
            *evaluation *= numerator;
            numerator *= self.root;
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
        let mut inverse_power = self.inverse_root;
        for bit in 0..self.log_size {
            let mut test = x;
            for _ in 0..(self.log_size - 1 - bit) {
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

impl<M: PrimeModulus> Domain<PastaField<M>> {
    /// Treats the subgroup as a transform domain with shift one.
    pub fn subgroup(self) -> CosetDomain<M> {
        CosetDomain::new(self, Shift::Subgroup)
    }

    /// Constructs the evaluation points `ZETA * root^j` for `0 <= j < size`.
    pub fn coset(self) -> CosetDomain<M> {
        CosetDomain::new(self, Shift::Zeta)
    }
}

/// A radix-2 subgroup or its coset shifted by [`PastaField::ZETA`].
///
/// Natural evaluation row `j` is `shift * domain.root()^j`, for `0 <= j < size`.
/// Construct these domains with [`Domain::subgroup`] or [`Domain::coset`].
/// Equality identifies the same ordered evaluation points, including the shift.
///
/// The [`Transform::inverse`](super::Transform::inverse) transform returns
/// coefficients of the original polynomial, removing the shift and normalizing
/// by the size.
/// [`Self::evaluate_lagrange`] evaluates selected basis polynomials at a point;
/// [`Self::prepare_lagrange`] exposes denominators for shared inversion batches.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct CosetDomain<M: PrimeModulus> {
    domain: Domain<PastaField<M>>,
    pub(super) shift: Shift,
}

impl<M: PrimeModulus> core::fmt::Debug for CosetDomain<M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CosetDomain")
            .field("domain", &self.domain)
            .field("shift", &self.shift)
            .finish()
    }
}

impl<M: PrimeModulus> CosetDomain<M> {
    /// Whether both descriptors identify the same ordered evaluation points.
    pub fn same_domain(self, other: Self) -> bool {
        self == other
    }

    fn new(domain: Domain<PastaField<M>>, shift: Shift) -> Self {
        Self { domain, shift }
    }

    pub(super) const fn is_subgroup(self) -> bool {
        matches!(self.shift, Shift::Subgroup)
    }

    /// The underlying subgroup.
    pub const fn domain(self) -> Domain<PastaField<M>> {
        self.domain
    }
    /// The number of evaluation points.
    pub const fn size(self) -> usize {
        self.domain.size()
    }
    /// The first evaluation point: one or [`PastaField::ZETA`].
    pub const fn shift(self) -> PastaField<M> {
        self.shift.value()
    }
    /// One or [`PastaField::ZETA_INVERSE`], respectively.
    pub const fn inverse_shift(self) -> PastaField<M> {
        self.shift.inverse()
    }
}
