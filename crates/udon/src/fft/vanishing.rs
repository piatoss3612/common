use super::{Domain, ElementOrder, FftError, PastaField, PrimeModulus, assert_length, reverse};
use crate::field::{ReductionState, batch_invert};

/// Division by `X^n - 1` on a shifted domain of size `N`, written as pieces.
///
/// This computes the unique polynomial `h` of degree less than `N` satisfying
/// `(X^n - 1) * h = a` modulo `X^N - g^N`, for numerator evaluations at
/// `g * root^j`. The shift `g` may be any nonzero field element with `g^N != 1`.
/// It need not be a shift supported by [`super::CosetDomain`].
///
/// This is an exact polynomial quotient when the numerator has degree less
/// than `N` and is divisible by `X^n - 1`. Otherwise it is modular division;
/// evaluations alone cannot establish divisibility or a degree bound. Requesting
/// fewer than `N / n` pieces discards high coefficients without checking them.
/// Such truncation preserves an exact quotient only if its degree is less than
/// the number of requested coefficients.
///
/// [`Self::write_pieces`] finishes a **forward subgroup transform** of the
/// numerator evaluations. It combines inverse indexing, size normalization,
/// shift removal, division, and writes into caller-owned pieces. The transform
/// may use ordinary or incremental execution; it must complete before finishing.
/// [`Self::prepare_factors`] supplies a compact pointwise alternative when
/// divided evaluations are needed before interpolation.
///
/// ```
/// use zakura_udon::{
///     exec::{ExecutionOptions, SerialExecutor},
///     field::Fp,
///     fft::{Domain, ElementOrder, Transform, VanishingDivision},
/// };
/// let domain = Domain::new(2)?;
/// let shift = <Fp>::ZETA;
/// let division = VanishingDivision::new(domain, &shift, 2)?;
/// // a(X) = (X^2 - 1) * (3 + X).
/// let mut values = core::array::from_fn::<_, 4, _>(|j| {
///     let x = shift.mul(&domain.root().pow_u64(j as u64));
///     x.square().sub(&<Fp>::ONE).mul(&<Fp>::from_u64(3).add(&x))
/// });
/// Transform::new(domain.subgroup()).forward(
///     &mut values, ExecutionOptions::default(), &SerialExecutor, &mut [],
/// )?;
/// let mut low = [Fp::ZERO; 2];
/// division.write_pieces(
///     &values, ElementOrder::Natural, &mut [&mut low], &mut [Fp::ZERO; 2],
/// );
/// assert_eq!(low.map(Fp::reduce), [Fp::from_u64(3), Fp::ONE]);
/// # Ok::<(), zakura_udon::fft::FftError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct VanishingDivision<M: PrimeModulus> {
    domain: Domain<M>,
    shift: PastaField<M>,
    piece_size: usize,
    seed: PastaField<M>,
    untwist: Untwist<M>,
}

#[derive(Clone, Copy, Debug)]
enum Untwist<M: PrimeModulus> {
    Periodic([PastaField<M>; 3]),
    Powers {
        inverse_shift: PastaField<M>,
        column_step: PastaField<M>,
    },
}

impl<M: PrimeModulus> VanishingDivision<M> {
    /// Checks the domain, shift, and number of coefficients in each piece.
    ///
    /// `piece_size` must be a nonzero power of two no larger than `domain.size()`.
    /// Returns [`FftError::InvalidLayout`] for an invalid piece size, zero shift,
    /// or `shift^domain.size() = 1`. Preparation uses constant storage and field
    /// inversions; repeated finishes reuse the returned plan.
    pub fn new<S: ReductionState>(
        domain: Domain<M>,
        shift: &PastaField<M, S>,
        piece_size: usize,
    ) -> Result<Self, FftError> {
        if !piece_size.is_power_of_two() || piece_size > domain.size() || shift.is_zero() {
            return Err(FftError::InvalidLayout);
        }
        let shift = shift.into_loose();
        let c_minus_one = shift
            .pow_u64(domain.size() as u64)
            .sub(&PastaField::<M>::ONE);
        if c_minus_one.is_zero() {
            return Err(FftError::InvalidLayout);
        }
        let inverse_shift = shift.invert().unwrap();
        let untwist = if shift.reduce() == PastaField::<M>::ZETA.reduce()
            || shift.reduce() == PastaField::<M>::ZETA_INVERSE.reduce()
        {
            let scale = domain.size_inverse();
            Untwist::Periodic([
                scale,
                scale.mul(&inverse_shift),
                scale.mul(&inverse_shift.square()),
            ])
        } else {
            Untwist::Powers {
                inverse_shift,
                column_step: inverse_shift.pow_u64(piece_size as u64),
            }
        };
        Ok(Self {
            domain,
            shift,
            piece_size,
            seed: c_minus_one.invert().unwrap(),
            untwist,
        })
    }

    /// The subgroup supplying the size and root, without the evaluation shift.
    pub const fn domain(self) -> Domain<M> {
        self.domain
    }

    /// The numerator's evaluation shift `g`.
    pub const fn shift(self) -> PastaField<M> {
        self.shift
    }

    /// Number of coefficients in each output piece, also the divisor degree.
    pub const fn piece_size(self) -> usize {
        self.piece_size
    }

    /// Maximum number of pieces, and required column scratch for a nonempty finish.
    ///
    /// This is also the number of retained pointwise divisor factors.
    pub const fn piece_count(self) -> usize {
        self.domain.size() / self.piece_size
    }

    /// Finishes a forward subgroup transform into ascending coefficient pieces.
    ///
    /// `transformed` must contain the completed **unscaled forward transform on
    /// `self.domain().subgroup()`** of the numerator's evaluations. Its physical
    /// output order is `order`; do not supply an inverse or coset transform.
    /// These mathematical input requirements are not checked. For example,
    /// [`super::run::FftPlan`] can select either output order and any input order.
    ///
    /// Piece `j` receives coefficients of degrees `j*n .. (j+1)*n`, in ascending
    /// order, where `n = self.piece_size()`. Supply a prefix of at most
    /// [`Self::piece_count`] pieces; an empty prefix does no work and needs no
    /// scratch. Nonempty output needs `piece_count()` scratch fields, reused for
    /// each column. Input is preserved. Piece and scratch tails are untouched;
    /// Rust borrows require writable buffers to be disjoint from each other and
    /// from input. Execution is serial, takes linear work, and allocates nothing.
    ///
    /// # Panics
    ///
    /// Panics before any writes if input length differs from the domain size,
    /// there are too many pieces, any piece is shorter than `piece_size()`, or
    /// scratch is insufficient. Omitted high pieces are never checked for zero;
    /// see the plan's modular division and truncation contract.
    pub fn write_pieces(
        self,
        transformed: &[PastaField<M>],
        order: ElementOrder,
        pieces: &mut [&mut [PastaField<M>]],
        scratch: &mut [PastaField<M>],
    ) {
        let size = self.domain.size();
        let count = self.piece_count();
        assert_length("transformed", size, transformed.len());
        assert!(pieces.len() <= count, "too many coefficient pieces");
        assert!(
            pieces.iter().all(|piece| piece.len() >= self.piece_size),
            "coefficient piece too short"
        );
        if pieces.is_empty() {
            return;
        }
        super::check_scratch(count, scratch.len());
        let column = &mut scratch[..count];
        let mut first = self.domain.size_inverse();
        for offset in 0..self.piece_size {
            let mut factor = first;
            let mut sum = PastaField::ZERO;
            for (j, value) in column.iter_mut().enumerate() {
                let degree = j * self.piece_size + offset;
                // A forward DFT at index -degree is the unscaled inverse at
                // degree. Negation wraps modulo the power-of-two domain size.
                let index = size.wrapping_sub(degree) & (size - 1);
                let index = match order {
                    ElementOrder::Natural => index,
                    ElementOrder::BitReversed => reverse(index, self.domain.log_size()),
                };
                let scale = match self.untwist {
                    Untwist::Periodic(cycle) => cycle[degree % 3],
                    Untwist::Powers { column_step, .. } => {
                        let current = factor;
                        if j + 1 < count {
                            factor = factor.mul(&column_step);
                        }
                        current
                    }
                };
                *value = transformed[index].mul(&scale);
                sum = sum.add(value);
            }
            // In each residue column, a[0] = c*h[m-1] - h[0] and
            // a[j] = h[j-1] - h[j]. Summing gives (c-1)*h[m-1].
            let mut h = sum.mul(&self.seed);
            if pieces.len() == count {
                pieces[count - 1][offset] = h;
            }
            for j in (1..count).rev() {
                h = h.add(&column[j]);
                if j - 1 < pieces.len() {
                    pieces[j - 1][offset] = h;
                }
            }
            if let Untwist::Powers { inverse_shift, .. } = self.untwist
                && offset + 1 < self.piece_size
            {
                first = first.mul(&inverse_shift);
            }
        }
    }

    /// Prepares the `N/n` repeating inverses of `x^n - 1` in caller storage.
    ///
    /// Only the first [`Self::piece_count`] storage elements are written. The
    /// returned handle borrows them and binds their order to this plan. Inversion
    /// scratch may have any length, including zero; larger buffers can reduce
    /// the number of inversions. Its use follows [`batch_invert`].
    ///
    /// Panics before writes if storage is too short. No allocation is required.
    pub fn prepare_factors<'a>(
        self,
        storage: &'a mut [PastaField<M>],
        scratch: &mut [PastaField<M>],
    ) -> VanishingFactors<'a, M> {
        assert!(
            storage.len() >= self.piece_count(),
            "divisor storage too short"
        );
        let factors = &mut storage[..self.piece_count()];
        let mut power = self.shift.pow_u64(self.piece_size as u64);
        let step = self.domain.root().pow_u64(self.piece_size as u64);
        for (index, factor) in factors.iter_mut().enumerate() {
            *factor = power.sub(&PastaField::<M>::ONE);
            if index + 1 < self.piece_count() {
                power = power.mul(&step);
            }
        }
        batch_invert(factors, scratch);
        VanishingFactors {
            division: self,
            factors,
        }
    }
}

/// Borrowed repeating divisor inverses bound to a [`VanishingDivision`] plan.
///
/// Construct with [`VanishingDivision::prepare_factors`]. There is one factor
/// per piece, rather than per evaluation, since `(g*root^j)^n - 1` repeats every
/// `N/n` natural rows. This handle is useful for evaluation-space composition;
/// a subsequent inverse transform and untwist recover the same modular quotient
/// as [`VanishingDivision::write_pieces`].
#[derive(Clone, Copy, Debug)]
pub struct VanishingFactors<'a, M: PrimeModulus> {
    division: VanishingDivision<M>,
    factors: &'a [PastaField<M>],
}

impl<'a, M: PrimeModulus> VanishingFactors<'a, M> {
    /// The domain, shift, and piece size to which these factors belong.
    pub const fn division(self) -> VanishingDivision<M> {
        self.division
    }

    /// The inverse divisors in natural row order for one period.
    pub const fn as_slice(self) -> &'a [PastaField<M>] {
        self.factors
    }

    /// Divides a full domain of numerator evaluations in place.
    ///
    /// `order` is the physical evaluation order. The values must correspond to
    /// this handle's shifted domain; this mathematical requirement is not checked.
    /// No scratch, inversions, or allocation are required during execution.
    /// Panics before writes unless the input length equals the domain size.
    pub fn divide_in_place(self, values: &mut [PastaField<M>], order: ElementOrder) {
        let domain = self.division.domain;
        assert_length("evaluations", domain.size(), values.len());
        for (index, value) in values.iter_mut().enumerate() {
            let row = match order {
                ElementOrder::Natural => index,
                ElementOrder::BitReversed => reverse(index, domain.log_size()),
            };
            *value = value.mul(&self.factors[row % self.factors.len()]);
        }
    }
}
