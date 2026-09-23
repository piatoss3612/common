//! Consumer views of the fixed Poseidon parameter sets.

use super::{PALLAS_BASE, PALLAS_SCALAR};
use crate::field::{Field, Fp, Fq};

/// A Poseidon permutation over `F`, described to code generic over the
/// instance.
///
/// The shape constants size states and sponges at compile time; the tables
/// are borrowed from the instance. [`PoseidonFp`] and [`PoseidonFq`] present
/// [`PALLAS_BASE`] and [`PALLAS_SCALAR`] this way.
pub trait PoseidonPermutation<F: Field>: Send + Sync + 'static {
    /// The state width.
    const T: usize;

    /// The sponge rate: how many elements are absorbed or squeezed between
    /// permutations. Smaller than [`T`](Self::T).
    const RATE: usize;

    /// The number of full rounds, half before and half after the partial
    /// rounds.
    const FULL_ROUNDS: usize;

    /// The number of partial rounds, which apply the S-box to the first state
    /// element only.
    const PARTIAL_ROUNDS: usize;

    /// The S-box exponent: the map `x -> x^ALPHA`, a permutation of `F`.
    const ALPHA: isize;

    /// Returns the round constants, one row of [`T`](Self::T) per round in
    /// application order.
    fn round_constants(&self) -> impl Iterator<Item = &[F]>;

    /// Returns the rows of the MDS matrix.
    fn mds_matrix(&self) -> impl ExactSizeIterator<Item = &[F]>;
}

/// The [`PALLAS_BASE`] instance as a [`PoseidonPermutation`] over [`Fp`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PoseidonFp;

/// The [`PALLAS_SCALAR`] instance as a [`PoseidonPermutation`] over [`Fq`].
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PoseidonFq;

macro_rules! poseidon_permutation {
    ($name:ident, $field:ty, $parameters:expr) => {
        impl PoseidonPermutation<$field> for $name {
            const T: usize = $parameters.width();
            const RATE: usize = $parameters.rate();
            const FULL_ROUNDS: usize = $parameters.full_rounds;
            const PARTIAL_ROUNDS: usize = $parameters.partial_rounds;
            const ALPHA: isize = $parameters.alpha as isize;

            fn round_constants(&self) -> impl Iterator<Item = &[$field]> {
                $parameters.round_constants.iter().map(|row| &row[..])
            }

            fn mds_matrix(&self) -> impl ExactSizeIterator<Item = &[$field]> {
                $parameters.mds.iter().map(|row| &row[..])
            }
        }
    };
}

poseidon_permutation!(PoseidonFp, Fp, PALLAS_BASE);
poseidon_permutation!(PoseidonFq, Fq, PALLAS_SCALAR);
