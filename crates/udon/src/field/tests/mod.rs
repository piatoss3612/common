//! Independent integer references for runtime field arithmetic.

pub(super) use super::{CanonicalUint, PallasBase, PallasScalar, PastaField, PrimeModulus};
use crate::test_support::xorshift64;
pub(super) use crate::test_support::{integer, modulus};
pub(super) use num_bigint::{BigInt, BigUint};
pub(super) use std::{vec, vec::Vec};

mod arithmetic;
mod constants;
mod encoding;
mod kernels;
mod parameters;
mod uint;

pub(super) fn limbs<const N: usize>(value: &BigUint) -> [u64; N] {
    let digits = value.to_u64_digits();
    assert!(digits.len() <= N);
    let mut result = [0; N];
    result[..digits.len()].copy_from_slice(&digits);
    result
}

pub(super) fn field<M: PrimeModulus>(value: &BigUint) -> PastaField<M> {
    PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs(value))).unwrap()
}

pub(super) fn assert_value<M: PrimeModulus>(actual: PastaField<M>, expected: &BigUint) {
    let p = modulus::<M>();
    let expected = expected % &p;
    let stored = integer(&actual.montgomery_limbs());
    assert!(stored < p, "field operations must preserve reduced storage");
    assert_eq!(stored, (&expected << 256usize) % &p);
    assert_eq!(BigUint::from_bytes_le(&actual.to_bytes()), expected);
}

pub(super) fn deterministic_bytes<const N: usize>(state: &mut u64) -> [u8; N] {
    core::array::from_fn(|_| xorshift64(state) as u8)
}

pub(super) fn samples<M: PrimeModulus>(count: usize) -> Vec<(PastaField<M>, BigUint)> {
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
    for raw in [BigUint::from(1u8), &p - 2u8, &p - 1u8] {
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
