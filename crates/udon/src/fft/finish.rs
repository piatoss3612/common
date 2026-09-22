//! Normalization and coset-shift removal shared by the FFT backends.
//!
//! For domain size `n`, shift `s`, and coefficient index `i`, a normalized
//! inverse needs the factor `n^-1 * s^-i`. An unscaled inverse needs only `s^-i`.

use super::{CosetDomain, PastaField, PrimeModulus, Transform};

/// Where a normalized inverse applies its size and shift factors.
#[derive(Clone, Copy)]
pub(super) enum InverseFinish<M: PrimeModulus> {
    /// Divide outputs by the power-of-two domain size; the shift is one.
    Subgroup,
    /// Divide outputs by the size, then multiply by repeating `s^-i` factors.
    Periodic([PastaField<M>; 3]),
    /// Use `n^-1 * s^-i`, combining it with butterfly input factors when possible.
    ScaledInputs,
}

impl<M: PrimeModulus> InverseFinish<M> {
    pub fn select(domain: CosetDomain<M>, combined_table: bool) -> Self {
        if domain.shift() == PastaField::ONE {
            Self::Subgroup
        } else if domain.inverse_scale_cycle.is_some() && !combined_table {
            // Order-three shifts need only three untwisting factors. A combined
            // finish table instead supplies normalized input factors directly.
            Self::Periodic([
                PastaField::ONE,
                domain.inverse_shift(),
                domain.inverse_shift().square(),
            ])
        } else {
            Self::ScaledInputs
        }
    }
}

/// Coefficient factors indexed by natural degree.
///
/// [`Self::untwist`] supplies shift removal alone; [`Self::normalized`] also
/// includes the inverse size. A task seeds its range with [`Self::at`] and uses
/// [`Self::next`] for consecutive coefficients to avoid repeated exponentiation.
#[derive(Clone, Copy)]
pub(super) enum Factors<'a, M: PrimeModulus> {
    Identity,
    Periodic([PastaField<M>; 3]),
    Table {
        /// Factors for the lower half of the domain.
        values: &'a [PastaField<M>],
        /// `s^(-n/2)`, converting a lower-half factor to its upper-half factor.
        upper: PastaField<M>,
    },
    Progression {
        first: PastaField<M>,
        step: PastaField<M>,
    },
}

impl<'a, M: PrimeModulus> Factors<'a, M> {
    pub fn untwist(domain: CosetDomain<M>) -> Self {
        match InverseFinish::select(domain, false) {
            InverseFinish::Subgroup => Self::Identity,
            InverseFinish::Periodic(cycle) => Self::Periodic(cycle),
            InverseFinish::ScaledInputs => Self::Progression {
                first: PastaField::ONE,
                step: domain.inverse_shift(),
            },
        }
    }

    pub fn normalized(plan: Transform<'a, M>) -> Self {
        if let Some(values) = plan.tables.inverse_scales.filter(|v| !v.is_empty()) {
            Self::Table {
                values,
                upper: Self::untwist(plan.domain).at(plan.domain.size() / 2),
            }
        } else if let Some(cycle) = plan.domain.inverse_scale_cycle {
            Self::Periodic(cycle)
        } else {
            Self::Progression {
                first: plan.domain.domain().size_inverse(),
                step: plan.domain.inverse_shift(),
            }
        }
    }

    /// Returns the factor at `index`, which must be below the domain size.
    pub fn at(self, index: usize) -> PastaField<M> {
        match self {
            Self::Identity => PastaField::ONE,
            Self::Periodic(cycle) => cycle[index % 3],
            Self::Table { values, upper } => {
                if index < values.len() {
                    values[index]
                } else {
                    values[index - values.len()].mul(&upper)
                }
            }
            Self::Progression { first, step } => {
                if index == 0 {
                    first
                } else {
                    first.mul(&step.pow_u64(index as u64))
                }
            }
        }
    }

    /// Advances from `previous`, the factor at `index - 1`.
    ///
    /// Requires `0 < index < domain size` and the same provider for both factors.
    pub fn next(self, index: usize, previous: PastaField<M>) -> PastaField<M> {
        match self {
            Self::Progression { step, .. } => previous.mul(&step),
            _ => self.at(index),
        }
    }
}
