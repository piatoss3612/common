use crate::field::pasta::test_support::{assert_value, samples};
use crate::field::pasta::test_support::{field_samples, modulus};
use crate::field::{
    Field, PallasBase, PallasScalar, PastaField, PrimeModulus, batch_invert, batch_invert_groups,
    batch_invert_with_scratch, count_inversions,
};
use num_bigint::BigUint;
use std::{vec, vec::Vec};

fn exercise<M: PrimeModulus>() {
    let samples = samples::<M>(37);
    let original: Vec<_> = samples.iter().map(|(value, _)| *value).collect();
    let mut values = original.clone();
    let sentinel = PastaField::from_u64(19);
    let mut scratch = vec![sentinel; values.len() + 3];
    batch_invert(&mut values, &mut scratch);
    let p = modulus::<M>();
    for ((actual, (value, integer)), expected) in values.iter().zip(&samples).zip(&original) {
        assert_eq!(
            (*actual).reduce(),
            (value.invert().unwrap_or(PastaField::<_>::ZERO)).reduce()
        );
        assert_value(*actual, &integer.modpow(&(&p - 2u8), &p));
        assert_eq!((*value).reduce(), (*expected).reduce());
    }
    assert_eq!(
        scratch[values.len()..]
            .iter()
            .map(|value| value.reduce())
            .collect::<Vec<_>>(),
        [sentinel; 3]
            .iter()
            .map(|value| value.reduce())
            .collect::<Vec<_>>()
    );
    // Reuse dirty prefixes, crossing both odd and empty group boundaries.
    let (a, rest) = values.split_at_mut(3);
    let (b, c) = rest.split_at_mut(4);
    batch_invert_groups(&mut [&mut [][..], a, &mut [], b, c, &mut []], &mut scratch);
    assert_eq!(
        (values)
            .iter()
            .map(|value| value.reduce())
            .collect::<Vec<_>>(),
        (original)
            .iter()
            .map(|value| value.reduce())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        scratch[values.len()..]
            .iter()
            .map(|value| value.reduce())
            .collect::<Vec<_>>(),
        [sentinel; 3]
            .iter()
            .map(|value| value.reduce())
            .collect::<Vec<_>>()
    );

    for capacity in [0, 1, 2, 7, values.len() - 1] {
        let mut short = vec![sentinel; capacity];
        let mut values = original.clone();
        let (a, b) = values.split_at_mut(1);
        batch_invert_groups(&mut [a, b], &mut short);
        for (value, original) in values.iter().zip(&original) {
            assert_eq!(
                (*value).reduce(),
                (original.invert().unwrap_or(PastaField::<_>::ZERO)).reduce()
            );
        }
        batch_invert(&mut values, &mut short);
        assert_eq!(
            (values)
                .iter()
                .map(|value| value.reduce())
                .collect::<Vec<_>>(),
            (original)
                .iter()
                .map(|value| value.reduce())
                .collect::<Vec<_>>()
        );
    }

    for len in [0, 1, 2, 7] {
        let mut zeros = vec![PastaField::<M>::ZERO; len];
        batch_invert(&mut zeros, &mut scratch);
        assert!(zeros.iter().all(PastaField::is_zero));
    }
    batch_invert_groups::<PastaField<M>>(&mut [] as &mut [&mut [PastaField<M>]], &mut []);
    let mut singleton = [sentinel];
    batch_invert(&mut singleton, &mut scratch);
    assert_eq!(
        (singleton[0].mul(&sentinel)).reduce(),
        (PastaField::<_>::ONE).reduce()
    );
    let mut a = [PastaField::ZERO, sentinel, PastaField::ZERO];
    let mut b = [PastaField::ZERO, sentinel];
    batch_invert_groups(&mut [&mut a[..], &mut b[..]], &mut scratch);
    assert_eq!(
        (a).iter().map(|value| value.reduce()).collect::<Vec<_>>(),
        ([PastaField::<_>::ZERO, singleton[0], PastaField::<_>::ZERO])
            .iter()
            .map(|value| value.reduce())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        (b).iter().map(|value| value.reduce()).collect::<Vec<_>>(),
        ([PastaField::<_>::ZERO, singleton[0]])
            .iter()
            .map(|value| value.reduce())
            .collect::<Vec<_>>()
    );
}

#[test]
fn zero_preserving_batches() {
    exercise::<PallasBase>();
    exercise::<PallasScalar>();
}

fn single_slice_batches<M: PrimeModulus>() {
    type Invert<F> = fn(&mut [F], &mut [F]);
    // Exercise the trait method and both public names against the independent
    // integer inverse.
    let methods: [(&str, Invert<PastaField<M>>); 3] = [
        ("batch_invert", batch_invert::<PastaField<M>>),
        ("with_scratch", batch_invert_with_scratch::<PastaField<M>>),
        ("trait", <PastaField<M> as Field>::batch_invert),
    ];
    let p = modulus::<M>();
    let samples = samples::<M>(9);
    let sentinel = PastaField::<M>::from_u64(19);
    for all_zero in [false, true] {
        let original: Vec<_> = samples
            .iter()
            .map(|(value, _)| if all_zero { PastaField::ZERO } else { *value })
            .collect();
        let integers: Vec<_> = samples
            .iter()
            .map(|(_, integer)| {
                if all_zero {
                    BigUint::from(0u8)
                } else {
                    integer.clone()
                }
            })
            .collect();
        let expected: Vec<_> = integers
            .iter()
            .map(|integer| integer.modpow(&(&p - 2u8), &p))
            .collect();
        for len in [0, 1, 2, 3, 7, original.len()] {
            for capacity in [0, 1, 2, 3, 7, len.saturating_sub(1), len, len + 3] {
                let expected_inversions = original[..len]
                    .chunks(capacity.max(1))
                    .filter(|batch| batch.iter().any(|value| !value.is_zero()))
                    .count();
                for (name, invert) in methods {
                    let mut values = original[..len].to_vec();
                    let mut scratch = vec![sentinel; capacity + 2];
                    // Reusing dirty scratch must also invert the result back.
                    for expected in [&expected[..len], &integers[..len]] {
                        let inversions = count_inversions(|| {
                            invert(&mut values, &mut scratch[..capacity]);
                        });
                        assert_eq!(
                            inversions, expected_inversions,
                            "{name}, length {len}, capacity {capacity}, all zero {all_zero}"
                        );
                        for (actual, expected) in values.iter().zip(expected) {
                            assert_value(*actual, expected);
                        }
                        assert!(scratch[len.min(capacity)..].iter().all(|value| {
                            value.montgomery_limbs() == sentinel.montgomery_limbs()
                        }));
                    }
                }
            }
        }
    }
}

#[test]
fn batch_entry_points_share_the_scratch_contract() {
    single_slice_batches::<PallasBase>();
    single_slice_batches::<PallasScalar>();
}

fn group_boundaries<M: PrimeModulus>() {
    let sentinel = PastaField::<M>::from_u64(19);
    let mut scratch = [sentinel; 10];
    for len in 0..=7 {
        for zeros in 0..1 << len {
            let original: Vec<_> = (0..len)
                .map(|i| {
                    if zeros & (1 << i) != 0 {
                        PastaField::ZERO
                    } else {
                        PastaField::from_u64(i as u64 + 2)
                    }
                })
                .collect();
            let expected: Vec<_> = original
                .iter()
                .map(|value| value.invert().unwrap_or(PastaField::ZERO))
                .collect();
            for split in 0..=len {
                for capacity in 0..=scratch.len() {
                    let mut values = original.clone();
                    let unused = len.min(capacity);
                    let tail = scratch[unused..].to_vec();
                    let (left, right) = values.split_at_mut(split);
                    let inversions = count_inversions(|| {
                        batch_invert_groups(
                            &mut [&mut [][..], left, &mut [], right, &mut []],
                            &mut scratch[..capacity],
                        );
                    });
                    let expected_inversions = original
                        .chunks(capacity.max(1))
                        .filter(|batch| batch.iter().any(|value| !value.is_zero()))
                        .count();
                    assert_eq!(
                        inversions, expected_inversions,
                        "length {len}, zeros {zeros}, split {split}, capacity {capacity}"
                    );
                    assert_eq!(
                        (values)
                            .iter()
                            .map(|value| value.reduce())
                            .collect::<Vec<_>>(),
                        (expected)
                            .iter()
                            .map(|value| value.reduce())
                            .collect::<Vec<_>>(),
                        "length {len}, zeros {zeros}, split {split}, capacity {capacity}"
                    );
                    assert_eq!(
                        scratch[unused..]
                            .iter()
                            .map(PastaField::montgomery_limbs)
                            .collect::<Vec<_>>(),
                        tail.iter()
                            .map(PastaField::montgomery_limbs)
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}

#[test]
fn group_boundaries_preserve_zero_positions() {
    group_boundaries::<PallasBase>();
    group_boundaries::<PallasScalar>();
}

fn singleton_groups<M: PrimeModulus>() {
    let original: Vec<_> = field_samples::<M>()
        .filter(|value| !value.is_zero())
        .take(1000)
        .map(|value| [value])
        .collect();
    let mut groups = original.clone();
    let mut scratch = [PastaField::from_u64(19); 64];
    let inversions = count_inversions(|| batch_invert_groups(&mut groups, &mut scratch));
    assert_eq!(inversions, 16);
    for (value, inverse) in original.iter().zip(&groups) {
        assert_eq!(value[0].mul(&inverse[0]).reduce(), PastaField::ONE);
    }

    // The final partial batch must leave the previous batch's scratch tail.
    let mut full_batches = original[..960].to_vec();
    let mut full_scratch = [PastaField::from_u64(19); 64];
    batch_invert_groups(&mut full_batches, &mut full_scratch);
    for (actual, expected) in scratch[40..].iter().zip(&full_scratch[40..]) {
        assert_eq!(actual.montgomery_limbs(), expected.montgomery_limbs());
    }
}

#[test]
fn singleton_groups_share_bounded_inversions() {
    singleton_groups::<PallasBase>();
    singleton_groups::<PallasScalar>();
}
