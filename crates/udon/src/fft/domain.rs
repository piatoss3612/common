use super::{FftError, PastaField, PrimeModulus, check_field_count};

/// A radix-2 subgroup and its canonical Pasta root of unity.
///
/// For size `n`, the subgroup consists of `root^j` for `0 <= j < n`.
/// Roots come from [`PastaField::root_of_unity`]; if `small` and `large` are
/// domains in the same field, `large.root()^(large.size()/small.size())` equals
/// `small.root()` whenever `small.size() <= large.size()`.
#[derive(Clone, Copy)]
pub struct Domain<M: PrimeModulus> {
    log_size: u32,
    size: usize,
    root: PastaField<M>,
    inverse_root: PastaField<M>,
    size_inverse: PastaField<M>,
}

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

    /// Treats the subgroup as a coset with shift one.
    pub fn subgroup(self) -> CosetDomain<M> {
        CosetDomain::with_inverse(self, PastaField::ONE, PastaField::ONE)
    }

    /// Constructs the evaluation points `shift * root^j` for `0 <= j < size`.
    ///
    /// Returns [`FftError::ZeroShift`] when `shift` is zero.
    pub fn coset(self, shift: PastaField<M>) -> Result<CosetDomain<M>, FftError> {
        let inverse = shift.invert().ok_or(FftError::ZeroShift)?;
        Ok(CosetDomain::with_inverse(self, shift, inverse))
    }
}

/// A radix-2 domain multiplied by a nonzero field element.
///
/// Natural evaluation row `j` is the point `shift * domain.root()^j`, for
/// `0 <= j < size`. Shifts need not be outside the subgroup. The
/// [`Transform::inverse`](super::Transform::inverse) transform returns coefficients of
/// the original polynomial, removing the shift and normalizing by the size.
#[derive(Clone, Copy)]
pub struct CosetDomain<M: PrimeModulus> {
    domain: Domain<M>,
    shift: PastaField<M>,
    inverse_shift: PastaField<M>,
    // Normalized scales repeat every three coefficients for order-three shifts.
    pub(super) inverse_scale_cycle: Option<[PastaField<M>; 3]>,
}

impl<M: PrimeModulus> core::fmt::Debug for CosetDomain<M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CosetDomain")
            .field("domain", &self.domain)
            .field("shift", &self.shift)
            .field("inverse_shift", &self.inverse_shift)
            .field("inverse_scale_cycle", &self.inverse_scale_cycle)
            .finish()
    }
}

impl<M: PrimeModulus> CosetDomain<M> {
    /// Whether both descriptors identify the same ordered evaluation points.
    pub fn same_domain(self, other: Self) -> bool {
        self.size() == other.size()
            && self.domain.root().reduce() == other.domain.root().reduce()
            && self.shift.reduce() == other.shift.reduce()
    }

    pub(super) fn with_inverse(
        domain: Domain<M>,
        shift: PastaField<M>,
        inverse_shift: PastaField<M>,
    ) -> Self {
        let square = inverse_shift.square();
        let inverse_scale_cycle =
            if square.mul(&inverse_shift).reduce() == PastaField::<M>::ONE.reduce() {
                let scale = domain.size_inverse();
                Some([scale, scale.mul(&inverse_shift), scale.mul(&square)])
            } else {
                None
            };
        Self {
            domain,
            shift,
            inverse_shift,
            inverse_scale_cycle,
        }
    }

    /// The underlying subgroup.
    pub const fn domain(self) -> Domain<M> {
        self.domain
    }
    /// The number of evaluation points.
    pub const fn size(self) -> usize {
        self.domain.size()
    }
    /// The first evaluation point.
    pub const fn shift(self) -> PastaField<M> {
        self.shift
    }
    /// The multiplicative inverse of the shift.
    pub const fn inverse_shift(self) -> PastaField<M> {
        self.inverse_shift
    }

    pub(super) fn inverse_scale(self, index: usize) -> PastaField<M> {
        match self.inverse_scale_cycle {
            Some(cycle) => cycle[index % 3],
            None => self
                .domain
                .size_inverse()
                .mul(&self.inverse_shift.pow_u64(index as u64)),
        }
    }
}
