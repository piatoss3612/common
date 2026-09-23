//! Deterministic Pasta samples and independent integer arithmetic references.

pub(super) use super::{
    CanonicalUint, PallasBase, PallasScalar, PastaField, PrimeModulus, Reduced, ReductionState,
};
pub(super) use num_bigint::{BigInt, BigUint};
pub(super) use std::{vec, vec::Vec};

pub(crate) fn limbs<const N: usize>(value: &BigUint) -> [u64; N] {
    let digits = value.to_u64_digits();
    assert!(digits.len() <= N);
    let mut result = [0; N];
    result[..digits.len()].copy_from_slice(&digits);
    result
}

pub(super) fn field<M: PrimeModulus>(value: &BigUint) -> PastaField<M> {
    PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs(value))).unwrap()
}

pub(in crate::field) fn check_value<M: PrimeModulus, S: ReductionState>(
    actual: PastaField<M, S>,
    expected: &BigUint,
) -> Result<(), &'static str> {
    let p = modulus::<M>();
    let expected = expected % &p;
    let stored = integer(&actual.montgomery_limbs());
    if stored >= integer(&PastaField::<M, S>::BOUND) {
        return Err("stored value exceeds its reduction-state bound");
    }
    if &stored % &p != (&expected << 256usize) % &p {
        return Err("Montgomery residue differs from the integer reference");
    }
    if integer(&actual.reduce().montgomery_limbs()) != &stored % &p {
        return Err("reduction differs from the integer reference");
    }
    if BigUint::from_bytes_le(&actual.to_bytes()) != expected {
        return Err("encoding differs from the integer reference");
    }
    Ok(())
}

pub(in crate::field) fn assert_value<M: PrimeModulus, S: ReductionState>(
    actual: PastaField<M, S>,
    expected: &BigUint,
) {
    assert_eq!(
        check_value(actual, expected),
        Ok(()),
        "{actual:?}, expected {expected}"
    );
}

/// Loose Montgomery representatives, with extra weight at representation bounds.
pub(crate) fn arbitrary_field<M: PrimeModulus>()
-> impl proptest::strategy::Strategy<Value = PastaField<M>> {
    use proptest::prelude::*;

    let p = modulus::<M>();
    let boundaries = [
        BigUint::from(0u8),
        BigUint::from(1u8),
        &p - 1u8,
        p.clone(),
        &p + 1u8,
        &p * 2u8 - 1u8,
    ]
    .map(|stored| PastaField::from_montgomery_limbs(limbs(&stored)));
    prop_oneof![
        1 => proptest::sample::select(boundaries.to_vec()),
        4 => any::<[u64; 4]>().prop_map(|mut stored| {
            // Both Pasta moduli exceed 2^254, so these limbs are below 2p.
            stored[3] &= (1 << 63) - 1;
            PastaField::from_montgomery_limbs(stored)
        }),
    ]
}

pub(super) fn deterministic_bytes<const N: usize>(state: &mut u64) -> [u8; N] {
    core::array::from_fn(|_| xorshift64(state) as u8)
}

pub(in crate::field) fn samples<M: PrimeModulus>(count: usize) -> Vec<(PastaField<M>, BigUint)> {
    let p = modulus::<M>();
    let mut integers = vec![
        BigUint::from(0u8),
        BigUint::from(1u8),
        BigUint::from(2u8),
        BigUint::from(u64::MAX),
        BigUint::from(1u8) << 64usize,
        BigUint::from(u128::MAX),
        &p - 2u8,
        &p - 1u8,
    ];
    let mut state = 0x6a09_e667_f3bc_c909;
    integers.extend(
        (0..count).map(|_| BigUint::from_bytes_le(&deterministic_bytes::<64>(&mut state)) % &p),
    );
    let mut samples = integers
        .into_iter()
        .map(|value| (field::<M>(&value), value))
        .collect::<Vec<_>>();

    // Canonical integer boundaries do not map to Montgomery limb boundaries.
    let inverse_r = (BigUint::from(1u8) << 256usize).modpow(&(&p - 2u8), &p);
    let twice = &p * 2u8;
    let raw_values = [
        BigUint::from(1u8),
        &p - 2u8,
        &p - 1u8,
        p.clone(),
        &p + 1u8,
        &twice - 2u8,
        &twice - 1u8,
    ]
    .into_iter()
    .chain(
        (0..count).map(|_| BigUint::from_bytes_le(&deterministic_bytes::<32>(&mut state)) % &twice),
    );
    for raw in raw_values {
        samples.push((
            PastaField::from_montgomery_limbs(limbs(&raw)),
            raw * &inverse_r % &p,
        ));
    }
    samples
}

pub(super) fn signed_mod(value: BigInt, modulus: &BigUint) -> BigUint {
    let p = BigInt::from(modulus.clone());
    ((value % &p + &p) % &p).to_biguint().unwrap()
}

pub(super) fn signed62(limbs: &[i64; 5]) -> BigInt {
    limbs
        .iter()
        .rev()
        .fold(BigInt::from(0), |value, limb| (value << 62usize) + limb)
}

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

pub(crate) fn twice_modulus<M: PrimeModulus>() -> BigUint {
    modulus::<M>() * 2u8
}

pub(crate) fn max_loose_limbs<M: PrimeModulus>() -> [u64; 4] {
    let digits = (twice_modulus::<M>() - 1u8).to_u64_digits();
    let mut limbs = [0; 4];
    limbs[..digits.len()].copy_from_slice(&digits);
    limbs
}
