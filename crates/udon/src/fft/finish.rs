//! Normalization and coset-shift removal shared by the FFT backends.
//!
//! For domain size `n`, shift `s`, and coefficient index `i`, a normalized
//! inverse needs the factor `n^-1 * s^-i`. An unscaled inverse needs only `s^-i`.

use super::{CosetDomain, PastaField, PrimeModulus, domain::Shift};

/// Where a normalized inverse applies its size and shift factors.
#[derive(Clone, Copy)]
pub(super) enum InverseFinish<M: PrimeModulus> {
    /// Divide outputs by the size, then multiply by repeating `s^-i` factors.
    Outputs(Factors<M>),
    /// Use `n^-1 * s^-i`, combining it with butterfly input factors when possible.
    ScaledInputs(Factors<M>),
}

impl<M: PrimeModulus> InverseFinish<M> {
    pub fn select(domain: CosetDomain<M>, combined_table: bool) -> Self {
        if !domain.is_subgroup() && combined_table {
            Self::ScaledInputs(Factors::normalized(domain))
        } else {
            Self::Outputs(Factors::untwist(domain))
        }
    }
}

/// Coefficient factors indexed by natural degree.
///
/// The supported domain shifts have order one or three. Multiplying by a common
/// scale preserves the three-entry cycle.
#[derive(Clone, Copy, Debug)]
pub(super) enum Factors<M: PrimeModulus> {
    Identity,
    Periodic([PastaField<M>; 3]),
}

impl<M: PrimeModulus> Factors<M> {
    pub fn forward(shift: Shift) -> Self {
        match shift {
            Shift::Subgroup => Self::Identity,
            Shift::Zeta => {
                Self::Periodic([PastaField::ONE, PastaField::ZETA, PastaField::ZETA_INVERSE])
            }
        }
    }

    pub fn untwist(domain: CosetDomain<M>) -> Self {
        if domain.is_subgroup() {
            Self::Identity
        } else {
            Self::Periodic([PastaField::ONE, PastaField::ZETA_INVERSE, PastaField::ZETA])
        }
    }

    pub fn normalized(domain: CosetDomain<M>) -> Self {
        let scale = domain.domain().size_inverse();
        match Self::untwist(domain) {
            Self::Identity => Self::Periodic([scale; 3]),
            Self::Periodic(cycle) => {
                Self::Periodic([scale, scale.mul(&cycle[1]), scale.mul(&cycle[2])])
            }
        }
    }

    pub fn scaled(self, scale: PastaField<M>) -> Self {
        match self {
            Self::Identity => Self::Periodic([scale; 3]),
            Self::Periodic(cycle) => Self::Periodic(cycle.map(|power| power.mul(&scale))),
        }
    }

    pub fn at(self, index: usize) -> PastaField<M> {
        match self {
            Self::Identity => PastaField::ONE,
            Self::Periodic(cycle) => cycle[index % 3],
        }
    }

    /// Applies normalized coset factors while retaining cheap size division.
    ///
    /// `self` must come from [`Self::normalized`] for a domain with base-two
    /// logarithm `log_size`, without any additional scaling.
    #[inline]
    pub fn apply_normalized(
        self,
        value: PastaField<M>,
        degree: usize,
        log_size: u32,
    ) -> PastaField<M> {
        // Phase zero has no shift correction. The other phases combine size
        // division and untwisting into one multiplication of a loose value.
        if degree.is_multiple_of(3) {
            crate::field::fft::divide_by_power_of_two(value, log_size)
        } else {
            value.mul(&self.at(degree))
        }
    }
}
