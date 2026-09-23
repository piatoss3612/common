//! Coset shifts and the coefficient factors that apply or remove them.
//!
//! For domain size `n`, shift `s`, and coefficient index `i`, a normalized
//! inverse needs the factor `n^-1 * s^-i`. An unscaled inverse needs only `s^-i`.
//! Expansion residues use arbitrary forward shifts with their own power
//! progressions.

use super::{CosetDomain, PastaField, PrimeModulus, reverse};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Shift {
    Subgroup,
    Zeta,
}

impl Shift {
    pub(super) const fn value<M: PrimeModulus>(self) -> PastaField<M> {
        match self {
            Self::Subgroup => PastaField::ONE,
            Self::Zeta => PastaField::ZETA,
        }
    }

    pub(super) const fn inverse<M: PrimeModulus>(self) -> PastaField<M> {
        match self {
            Self::Subgroup => PastaField::ONE,
            Self::Zeta => PastaField::ZETA_INVERSE,
        }
    }
}

/// Where a normalized inverse applies its size and shift factors.
#[derive(Clone, Copy)]
pub(super) enum InverseFinish<M: PrimeModulus> {
    /// Divide outputs by the size, then multiply by repeating `s^-i` factors.
    Outputs(Factors<M>),
    /// Use `n^-1 * s^-i`, combining it with butterfly input factors when possible.
    ScaledInputs(Factors<M>),
}

impl<M: PrimeModulus> InverseFinish<M> {
    pub(super) fn select(domain: CosetDomain<M>, combined_table: bool) -> Self {
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
    pub(super) fn forward(shift: Shift) -> Self {
        match shift {
            Shift::Subgroup => Self::Identity,
            Shift::Zeta => {
                Self::Periodic([PastaField::ONE, PastaField::ZETA, PastaField::ZETA_INVERSE])
            }
        }
    }

    pub(super) fn untwist(domain: CosetDomain<M>) -> Self {
        if domain.is_subgroup() {
            Self::Identity
        } else {
            Self::Periodic([PastaField::ONE, PastaField::ZETA_INVERSE, PastaField::ZETA])
        }
    }

    pub(super) fn normalized(domain: CosetDomain<M>) -> Self {
        let scale = domain.domain().size_inverse();
        match Self::untwist(domain) {
            Self::Identity => Self::Periodic([scale; 3]),
            Self::Periodic(cycle) => {
                Self::Periodic([scale, scale.mul(&cycle[1]), scale.mul(&cycle[2])])
            }
        }
    }

    pub(super) fn scaled(self, scale: PastaField<M>) -> Self {
        match self {
            Self::Identity => Self::Periodic([scale; 3]),
            Self::Periodic(cycle) => Self::Periodic(cycle.map(|power| power.mul(&scale))),
        }
    }

    pub(super) fn at(self, index: usize) -> PastaField<M> {
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
    pub(super) fn apply_normalized(
        self,
        value: PastaField<M>,
        degree: usize,
        log_size: u32,
    ) -> PastaField<M> {
        // Phase zero has no shift correction. The other phases combine size
        // division and untwisting into one multiplication of a loose value.
        if degree.is_multiple_of(3) {
            crate::field::butterfly::divide_by_power_of_two(value, log_size)
        } else {
            value.mul(&self.at(degree))
        }
    }
}

/// Forward coefficient factors, including arbitrary shifts for expansion residues.
#[derive(Clone, Copy, Debug)]
pub(super) enum ForwardShift<M: PrimeModulus> {
    Domain(Shift),
    Residue {
        shift: PastaField<M>,
        inverse: PastaField<M>,
    },
}

impl<M: PrimeModulus> ForwardShift<M> {
    pub(super) fn for_domain(domain: CosetDomain<M>) -> Self {
        Self::Domain(domain.shift)
    }

    pub(super) fn is_identity(self) -> bool {
        matches!(self, Self::Domain(Shift::Subgroup))
    }

    pub(super) fn shift(self) -> PastaField<M> {
        match self {
            Self::Domain(shift) => shift.value(),
            Self::Residue { shift, .. } => shift,
        }
    }

    pub(super) fn cycle(self) -> Option<Factors<M>> {
        match self {
            Self::Domain(shift) => Some(Factors::forward(shift)),
            Self::Residue { .. } => None,
        }
    }

    pub(super) fn at(self, degree: usize) -> PastaField<M> {
        match self.cycle() {
            Some(cycle) => cycle.at(degree),
            None => self.shift().pow_u64(degree as u64),
        }
    }
}

// Computed starting powers make scaling independent across regions. Incrementing
// a storage index clears its trailing one bits and sets the next zero bit. In
// bit-reversed order, this changes the coefficient degree by an amount determined
// by that trailing-one count. The ratios array stores the corresponding powers
// of the coset shift, so each region can advance without repeated exponentiation.
pub(super) struct BitReversedPowers<M: PrimeModulus> {
    shift: PastaField<M>,
    ratios: [PastaField<M>; 32],
    log_size: u32,
}

impl<M: PrimeModulus> BitReversedPowers<M> {
    pub(super) fn new(shift: PastaField<M>, mut inverse: PastaField<M>, log_size: u32) -> Self {
        let mut ratios = [PastaField::ONE; 32];
        let mut reciprocal = [PastaField::ONE; 32];
        let mut power = shift;
        for index in 0..log_size as usize {
            ratios[index] = power;
            reciprocal[index] = inverse;
            power = power.square();
            inverse = inverse.square();
        }
        ratios[..log_size as usize].reverse();
        let mut prefix = PastaField::ONE;
        for index in 0..log_size as usize {
            ratios[index] = ratios[index].mul(&prefix);
            prefix = prefix.mul(&reciprocal[log_size as usize - 1 - index]);
        }
        Self {
            shift,
            ratios,
            log_size,
        }
    }
    pub(super) fn at(&self, index: usize) -> PastaField<M> {
        self.shift.pow_u64(bit_reverse(index, self.log_size) as u64)
    }
    pub(super) fn next(&self, index: usize, power: PastaField<M>) -> PastaField<M> {
        power.mul(&self.ratios[index.trailing_ones() as usize])
    }
}
