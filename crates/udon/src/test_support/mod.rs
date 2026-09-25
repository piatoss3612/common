//! Deterministic sampling and integer conversions shared by arithmetic tests.

use crate::field::{CanonicalUint, PastaField, PrimeModulus};
use num_bigint::BigUint;
use std::vec::Vec;

pub(crate) const CORPUS_SEED: u64 = 0x243f_6a88_85a3_08d3;

pub(crate) fn xorshift64(state: &mut u64) -> u64 {
    *state ^= *state << 13;
    *state ^= *state >> 7;
    *state ^= *state << 17;
    *state
}

/// An infinite reproducible sequence below `2^254`, canonical for either field.
///
/// Callers add the boundary cases needed by their operations separately.
pub(crate) fn field_samples<M: PrimeModulus>() -> impl Iterator<Item = PastaField<M>> {
    let mut state = CORPUS_SEED;
    core::iter::from_fn(move || {
        let mut limbs = core::array::from_fn(|_| xorshift64(&mut state));
        limbs[3] &= (1 << 62) - 1;
        Some(PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs)).unwrap())
    })
}

pub(crate) fn integer(limbs: &[u64]) -> BigUint {
    BigUint::from_bytes_le(
        &limbs
            .iter()
            .flat_map(|limb| limb.to_le_bytes())
            .collect::<Vec<_>>(),
    )
}

pub(crate) fn modulus<M: PrimeModulus>() -> BigUint {
    integer(&M::MODULUS)
}

/// Twice the modulus: the exclusive bound of the loose representation.
pub(crate) fn twice_modulus<M: PrimeModulus>() -> BigUint {
    modulus::<M>() * 2u8
}

/// The largest loose Montgomery integer, `2p - 1`, as limbs.
pub(crate) fn max_loose_limbs<M: PrimeModulus>() -> [u64; 4] {
    let digits = (twice_modulus::<M>() - 1u8).to_u64_digits();
    let mut limbs = [0; 4];
    limbs[..digits.len()].copy_from_slice(&digits);
    limbs
}

pub(crate) mod admission;
#[path = "../../tests/support/fft_pipeline.rs"]
pub(crate) mod fft_pipeline;
pub(crate) mod fft_run;
#[path = "../../tests/support/msm_run.rs"]
pub(crate) mod msm_run;
#[path = "../../tests/support/run_pool.rs"]
pub(crate) mod run_pool;
