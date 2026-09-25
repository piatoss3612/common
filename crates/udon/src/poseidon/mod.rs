//! Poseidon parameters over the Pasta fields.
//!
//! One instance per field: width 5, rate 4, the `x^5` S-box, 8 full and 56
//! partial rounds, with round constants and MDS matrix produced by the
//! Hades/Poseidon reference generator
//! (<https://extgit.isec.tugraz.at/krypto/hadeshash>, through the
//! `daira/pasta-hadeshash` fork). This module carries the parameters only.
//! The permutation and the sponge built over them belong to the protocol
//! that hashes with them. The unstable `traits` feature also exposes
//! `PoseidonPermutation` and the `PoseidonFp` / `PoseidonFq` views.

use crate::field::{Fp, Fq};

mod pallas_base;
mod pallas_scalar;

/// A Poseidon instance over a field: the shape of the permutation and the
/// tables it runs with, for a state of width `T`.
#[derive(Clone, Copy, Debug)]
pub struct PoseidonParameters<F: 'static, const T: usize> {
    /// The number of full rounds, half before and half after the partial
    /// rounds.
    pub full_rounds: usize,
    /// The number of partial rounds.
    pub partial_rounds: usize,
    /// The S-box exponent.
    pub alpha: u32,
    /// The round constants in application order, one row of `T` per round:
    /// half the full rounds, then the partial rounds, then the remaining full
    /// rounds.
    pub round_constants: &'static [[F; T]],
    /// The MDS matrix.
    pub mds: &'static [[F; T]; T],
}

impl<F: 'static, const T: usize> PoseidonParameters<F, T> {
    /// The state width `T`.
    pub const fn width(&self) -> usize {
        T
    }

    /// The sponge rate: the width less one capacity element.
    pub const fn rate(&self) -> usize {
        T - 1
    }

    /// The total number of rounds.
    pub const fn rounds(&self) -> usize {
        self.full_rounds + self.partial_rounds
    }
}

/// The instance over the Pallas base field [`Fp`], which is also the Vesta
/// scalar field.
pub const PALLAS_BASE: PoseidonParameters<Fp, 5> = PoseidonParameters {
    full_rounds: 8,
    partial_rounds: 56,
    alpha: 5,
    round_constants: &pallas_base::ROUND_CONSTANTS,
    mds: &pallas_base::MDS,
};

/// The instance over the Pallas scalar field [`Fq`], which is also the Vesta
/// base field.
pub const PALLAS_SCALAR: PoseidonParameters<Fq, 5> = PoseidonParameters {
    full_rounds: 8,
    partial_rounds: 56,
    alpha: 5,
    round_constants: &pallas_scalar::ROUND_CONSTANTS,
    mds: &pallas_scalar::MDS,
};

#[cfg(feature = "traits")]
mod traits;
#[cfg(feature = "traits")]
pub use traits::{PoseidonFp, PoseidonFq, PoseidonPermutation};

#[cfg(test)]
mod tests;
