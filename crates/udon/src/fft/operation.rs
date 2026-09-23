use super::{ElementOrder, InverseScale, PastaField, PrimeModulus, reverse};
use super::{domain::Shift, finish::Factors};

/// Mathematical transform direction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    /// Evaluate coefficients on the coset.
    Forward,
    /// Interpolate evaluations to coefficients.
    Inverse,
}

/// Declared nonzero support in natural logical input order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputSupport {
    /// Every domain position may be nonzero.
    Full,
    /// Only positions `0..length` may be nonzero.
    ///
    /// A separate input contains exactly this prefix; in-place execution ignores
    /// and overwrites its tail.
    /// For an inverse these positions are evaluations, not coefficients.
    Prefix(usize),
}

/// Location and lifetime of transform input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputStorage {
    /// Read an immutable input view and write a separate output bank.
    Preserve,
    /// Read and overwrite the writable values bank.
    InPlace,
}

/// Mathematical and storage semantics fixed before executing an operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformRequest {
    /// Forward evaluation or inverse interpolation.
    pub direction: Direction,
    /// Declared input support; a prefix requires natural input order.
    pub support: InputSupport,
    /// Order of coefficients or evaluations in the input.
    pub input_order: ElementOrder,
    /// Desired order of coefficients or evaluations in the output.
    pub output_order: ElementOrder,
    /// Inverse-size factor; forward requests must use [`InverseScale::Normalized`].
    pub inverse_scale: InverseScale,
    /// In-place input or a preserved separate input view.
    pub input_storage: InputStorage,
}

impl TransformRequest {
    /// A full natural-order transform using in-place input.
    pub const fn new(direction: Direction) -> Self {
        Self {
            direction,
            support: InputSupport::Full,
            input_order: ElementOrder::Natural,
            output_order: ElementOrder::Natural,
            inverse_scale: InverseScale::Normalized,
            input_storage: InputStorage::InPlace,
        }
    }
    pub(super) const fn input_len(self, size: usize) -> usize {
        match self.support {
            InputSupport::Full => size,
            InputSupport::Prefix(len) => len,
        }
    }
}

/// Writable storage available to an incremental transform.
///
/// Fragment lengths describe the provider's physical storage, not an arithmetic
/// radix. A whole-bank lease permits order conversion without a retained copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageLayout {
    /// One bank whose complete mutable slice can be leased.
    Contiguous,
    /// Independently leased, equal power-of-two fragments.
    Fragments {
        /// Elements per physical fragment; clamped to the domain size.
        length: core::num::NonZeroUsize,
        /// Whether the provider can also lease the complete bank exclusively.
        whole_bank: bool,
    },
}

/// Small straight-line radix schedules, including differential-test candidates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum Codelet {
    /// Individual radix-2 rounds.
    Radix2,
    /// Four-value local schedules.
    #[cfg(test)]
    Radix4,
    /// Eight-value local schedules.
    #[cfg(test)]
    Radix8,
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
    pub fn for_domain(domain: super::CosetDomain<M>) -> Self {
        Self::Domain(domain.shift)
    }

    pub fn is_identity(self) -> bool {
        matches!(self, Self::Domain(Shift::Subgroup))
    }

    pub fn shift(self) -> PastaField<M> {
        match self {
            Self::Domain(shift) => shift.value(),
            Self::Residue { shift, .. } => shift,
        }
    }

    pub fn cycle(self) -> Option<Factors<M>> {
        match self {
            Self::Domain(shift) => Some(Factors::forward(shift)),
            Self::Residue { .. } => None,
        }
    }

    pub fn at(self, degree: usize) -> PastaField<M> {
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
        self.shift.pow_u64(reverse(index, self.log_size) as u64)
    }
    pub(super) fn next(&self, index: usize, power: PastaField<M>) -> PastaField<M> {
        power.mul(&self.ratios[index.trailing_ones() as usize])
    }
}
