//! Radix-2 evaluation domains and their coset shifts.

use super::{FftError, check_element_count, factors::Shift};
use crate::field::{PastaField, PrimeModulus};

/// A radix-2 subgroup with its canonical root of unity and the scalars
/// transforms over it need.
///
/// For size `n`, the subgroup consists of `root^j` for `0 <= j < n`.
/// Pasta roots are compatible across sizes: the larger domain's root raised
/// to the size ratio is the smaller domain's root.
/// Domains in the same field compare equal exactly when their sizes match.
///
/// [`Self::subgroup`] and [`Self::coset`] configure native Pasta transforms with
/// caller-owned tables, scratch, and execution. The unstable `traits` feature
/// also supports consumer field implementations and adds generic evaluation
/// and transform methods to this same descriptor.
#[derive(Clone, Copy, Debug)]
pub struct Domain<F> {
    log_size: u32,
    size: usize,
    root: F,
    inverse_root: F,
    size_inverse: F,
}

impl<F> PartialEq for Domain<F> {
    fn eq(&self, other: &Self) -> bool {
        // Construction fixes every field parameter from the validated size.
        self.size == other.size
    }
}

impl<F> Eq for Domain<F> {}

#[cfg(not(feature = "traits"))]
impl<M: PrimeModulus> Domain<PastaField<M>> {
    /// Constructs a Pasta domain of `2^log_size` elements.
    ///
    /// Returns [`FftError::InvalidSize`] above the field's two-adicity, or
    /// [`FftError::SizeOverflow`] if its slice would exceed addressable memory.
    pub fn new(log_size: u32) -> Result<Self, FftError> {
        Self::pasta(log_size)
    }

    /// Constructs a domain from its nonzero power-of-two element count.
    pub fn for_size(size: usize) -> Result<Self, FftError> {
        Self::pasta_for_size(size)
    }
}

impl<F: Copy> Domain<F> {
    pub(super) fn from_roots(
        log_size: u32,
        root: F,
        inverse_root: F,
        size_inverse: impl FnOnce() -> F,
    ) -> Result<Self, FftError> {
        let size = 1usize.checked_shl(log_size).ok_or(FftError::SizeOverflow)?;
        check_element_count::<F>(size)?;
        Ok(Self {
            log_size,
            size,
            root,
            inverse_root,
            size_inverse: size_inverse(),
        })
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
}

impl<M: PrimeModulus> Domain<PastaField<M>> {
    // Native setup uses inherent field operations in every feature configuration.
    pub(super) fn pasta(log_size: u32) -> Result<Self, FftError> {
        let root = PastaField::root_of_unity(log_size).ok_or(FftError::InvalidSize)?;
        let inverse_root =
            PastaField::root_of_unity_inverse(log_size).ok_or(FftError::InvalidSize)?;
        Self::from_roots(log_size, root, inverse_root, || {
            PastaField::power_of_two_inverse(log_size)
        })
    }

    pub(super) fn pasta_for_size(size: usize) -> Result<Self, FftError> {
        if !size.is_power_of_two() {
            return Err(FftError::InvalidSize);
        }
        Self::pasta(size.ilog2())
    }

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
