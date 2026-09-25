use super::{FftError, PastaField, PrimeModulus, check_field_count};

/// A radix-2 subgroup and its canonical Pasta root of unity.
///
/// For size `n`, the subgroup consists of `root^j` for `0 <= j < n`.
/// Roots come from [`PastaField::root_of_unity`]; if `small` and `large` are
/// domains in the same field, `large.root()^(large.size()/small.size())` equals
/// `small.root()` whenever `small.size() <= large.size()`.
/// Domains in the same field compare equal exactly when their sizes match.
#[derive(Clone, Copy)]
pub struct Domain<M: PrimeModulus> {
    log_size: u32,
    size: usize,
    root: PastaField<M>,
    inverse_root: PastaField<M>,
    size_inverse: PastaField<M>,
}

impl<M: PrimeModulus> PartialEq for Domain<M> {
    fn eq(&self, other: &Self) -> bool {
        // Construction fixes every field parameter from the validated size.
        self.size == other.size
    }
}

impl<M: PrimeModulus> Eq for Domain<M> {}

impl<M: PrimeModulus> core::fmt::Debug for Domain<M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Domain")
            .field("log_size", &self.log_size)
            .field("size", &self.size)
            .field("root", &self.root)
            .field("inverse_root", &self.inverse_root)
            .field("size_inverse", &self.size_inverse)
            .finish()
    }
}

impl<M: PrimeModulus> Domain<M> {
    /// Constructs a domain of `2^log_size` field elements.
    ///
    /// `log_size = 0` gives the singleton subgroup containing one. Returns
    /// [`FftError::InvalidSize`] above the Pasta two-adicity (32), or
    /// [`FftError::SizeOverflow`] if the element count does not fit `usize` or
    /// a field slice of that length would exceed `isize::MAX` bytes.
    pub fn new(log_size: u32) -> Result<Self, FftError> {
        let root = PastaField::root_of_unity(log_size).ok_or(FftError::InvalidSize)?;
        let size = 1usize.checked_shl(log_size).ok_or(FftError::SizeOverflow)?;
        check_field_count(size)?;
        Ok(Self {
            log_size,
            size,
            root,
            inverse_root: PastaField::root_of_unity_inverse(log_size).unwrap(),
            size_inverse: PastaField::power_of_two_inverse(log_size),
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
    pub const fn root(self) -> PastaField<M> {
        self.root
    }
    /// The inverse of [`Self::root`].
    pub const fn inverse_root(self) -> PastaField<M> {
        self.inverse_root
    }
    /// The multiplicative inverse of the domain size in the field.
    pub const fn size_inverse(self) -> PastaField<M> {
        self.size_inverse
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

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Shift {
    Subgroup,
    Zeta,
}

impl Shift {
    pub const fn value<M: PrimeModulus>(self) -> PastaField<M> {
        match self {
            Self::Subgroup => PastaField::ONE,
            Self::Zeta => PastaField::ZETA,
        }
    }

    pub const fn inverse<M: PrimeModulus>(self) -> PastaField<M> {
        match self {
            Self::Subgroup => PastaField::ONE,
            Self::Zeta => PastaField::ZETA_INVERSE,
        }
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
    domain: Domain<M>,
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

    fn new(domain: Domain<M>, shift: Shift) -> Self {
        Self { domain, shift }
    }

    pub(super) const fn is_subgroup(self) -> bool {
        matches!(self.shift, Shift::Subgroup)
    }

    /// The underlying subgroup.
    pub const fn domain(self) -> Domain<M> {
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
