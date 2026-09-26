use super::*;
use crate::field::pasta::test_support::field_samples;
use crate::field::{
    BatchInversionError, batch_invert, batch_invert_groups, batch_invert_groups_scaled,
    batch_invert_scaled, count_inversions, try_batch_invert_by, try_batch_invert_scaled_by,
};

fn inversion_endpoints<M: PrimeModulus>() {
    let values: Vec<_> = field_samples::<M>()
        .filter(|x| !x.is_zero())
        .take(257)
        .collect();
    for n in [0, 1, 2, 3, 4, 7, 8, 31, 32, 127, 128, 257] {
        let mut input = values[..n].to_vec();
        let mut prefix = vec![PastaField::from_u64(91); n + 1];
        crate::field::invert_nonzero(&mut input, &mut prefix);
        assert_eq!(
            (prefix[n]).reduce(),
            (PastaField::<_>::from_u64(91)).reduce()
        );
        for (value, inverse) in values.iter().zip(&input) {
            assert_eq!(
                (value.mul(inverse)).reduce(),
                (PastaField::<_>::ONE).reduce()
            );
        }
        // Reuse dirty prefixes with unit denominators and both lane parities.
        input.fill(PastaField::ONE);
        crate::field::invert_nonzero(&mut input, &mut prefix);
        assert!(input.iter().all(|x| x.reduce() == PastaField::ONE));
    }
}

#[test]
fn batch_inversion_handles_lane_endpoints() {
    inversion_endpoints::<crate::field::PallasBase>();
    inversion_endpoints::<crate::field::PallasScalar>();
}

#[derive(Debug, PartialEq)]
enum VisitError {
    Inversion(BatchInversionError),
    Stop,
}
impl From<BatchInversionError> for VisitError {
    fn from(error: BatchInversionError) -> Self {
        Self::Inversion(error)
    }
}

fn record_inversion<M: PrimeModulus>() {
    // The records deliberately have data unrelated to their denominators.
    let records: Vec<_> = (0..37)
        .map(|i| (i * 17, PastaField::<M>::from_u64(i as u64 + 1)))
        .collect();
    let sentinel = PastaField::from_u64(99);
    for len in [0, 1, 2, 7, 36, 37] {
        for capacity in [0, 1, 2, 7, 37, 40] {
            let mut scratch = vec![sentinel; capacity];
            let mut seen = vec![false; len];
            try_batch_invert_by(
                &records[..len],
                |record| record.1,
                &mut scratch,
                |index, record, inverse| {
                    assert_eq!(record.0, records[index].0);
                    assert_eq!(
                        record.1.montgomery_limbs(),
                        records[index].1.montgomery_limbs()
                    );
                    assert_eq!(
                        (record.1.mul(&inverse)).reduce(),
                        (PastaField::<_>::ONE).reduce()
                    );
                    assert!(!seen[index]);
                    seen[index] = true;
                    Ok::<_, VisitError>(())
                },
            )
            .unwrap();
            assert!(seen.iter().all(|x| *x));
            assert!(
                scratch[len.min(capacity)..]
                    .iter()
                    .all(|x| x.montgomery_limbs() == sentinel.montgomery_limbs())
            );
        }
    }
    for zero in [0, 18, 36] {
        let mut invalid = records.clone();
        invalid[zero].1 = PastaField::ZERO;
        for capacity in [0, 2, 40] {
            let mut scratch = vec![sentinel; capacity];
            assert_eq!(
                try_batch_invert_by(
                    &invalid,
                    |record| record.1,
                    &mut scratch,
                    |_, _, _| -> Result<(), VisitError> { panic!("visited invalid batch") },
                ),
                Err(VisitError::Inversion(
                    BatchInversionError::ZeroDenominator { index: zero }
                ))
            );
            assert!(
                scratch
                    .iter()
                    .all(|x| x.montgomery_limbs() == sentinel.montgomery_limbs())
            );
        }
    }
    for capacity in [0, 2, 40] {
        let mut seen = Vec::new();
        let mut scratch = vec![sentinel; capacity];
        assert_eq!(
            try_batch_invert_by(
                &records,
                |record| record.1,
                &mut scratch,
                |index, record, inverse| {
                    assert_eq!(
                        (record.1.mul(&inverse)).reduce(),
                        (PastaField::<_>::ONE).reduce()
                    );
                    if seen.len() == 3 {
                        return Err(VisitError::Stop);
                    }
                    seen.push(index);
                    Ok(())
                },
            ),
            Err(VisitError::Stop)
        );
        assert_eq!(seen.len(), 3);
    }
}

#[test]
fn records_preserve_indices_and_validate_before_visiting() {
    record_inversion::<PallasBase>();
    record_inversion::<PallasScalar>();
}

#[test]
fn scaled_records_validate_before_visiting_and_stop_on_error() {
    scaled_records::<PallasBase>();
    scaled_records::<PallasScalar>();
}

fn scaled_records<M: PrimeModulus>() {
    let p = modulus::<M>();
    let corpus: Vec<_> = samples::<M>(17)
        .into_iter()
        .filter(|(_, integer)| *integer != BigUint::from(0u8))
        .collect();
    let records: Vec<_> = corpus
        .iter()
        .enumerate()
        .map(|(index, (value, _))| (index * 17, *value))
        .collect();
    let sentinel = PastaField::from_u64(91);
    for (scale, scale_integer) in samples::<M>(2) {
        let expected: Vec<_> = corpus
            .iter()
            .map(|(_, integer)| integer.modpow(&(&p - 2u8), &p) * &scale_integer % &p)
            .collect();
        for len in [0, 1, 2, 3, 7, 8, 31, 32, 33, records.len()] {
            for capacity in [0, 1, 2, 7, 32, records.len() + 3] {
                let mut scratch = vec![sentinel; capacity];
                let mut seen = vec![false; len];
                let inversions = count_inversions(|| {
                    try_batch_invert_scaled_by(
                        &records[..len],
                        |record| record.1,
                        &scale,
                        &mut scratch,
                        |index, record, inverse| {
                            assert_eq!(record.0, records[index].0);
                            assert_eq!(
                                record.1.montgomery_limbs(),
                                records[index].1.montgomery_limbs()
                            );
                            assert_value(inverse, &expected[index]);
                            assert!(!seen[index]);
                            seen[index] = true;
                            Ok::<_, VisitError>(())
                        },
                    )
                    .unwrap();
                });
                assert_eq!(inversions, len.div_ceil(capacity.max(1)));
                assert!(seen.iter().all(|visited| *visited));
                assert!(
                    scratch[len.min(capacity)..]
                        .iter()
                        .all(|value| { value.montgomery_limbs() == sentinel.montgomery_limbs() })
                );
            }
        }

        // A zero scale cannot suppress validation, including loose zeros in
        // later batches. Rejection leaves all scratch and visitor state intact.
        for zero in [0, 7, records.len() - 1] {
            let mut invalid = records.clone();
            invalid[zero].1 = PastaField::from_montgomery_limbs(M::MODULUS);
            for capacity in [0, 2, 7, records.len() + 3] {
                let mut scratch = vec![sentinel; capacity];
                assert_eq!(
                    try_batch_invert_scaled_by(
                        &invalid,
                        |record| record.1,
                        &scale,
                        &mut scratch,
                        |_, _, _| -> Result<(), VisitError> { panic!("visited invalid batch") },
                    ),
                    Err(VisitError::Inversion(
                        BatchInversionError::ZeroDenominator { index: zero }
                    ))
                );
                assert!(
                    scratch
                        .iter()
                        .all(|value| { value.montgomery_limbs() == sentinel.montgomery_limbs() })
                );
            }
        }

        for capacity in [0, 1, 2, 7, records.len() + 3] {
            for stop_after in [0, 1, 3, 8, records.len() - 1] {
                let mut seen = vec![false; records.len()];
                let mut calls = 0;
                let mut scratch = vec![sentinel; capacity];
                assert_eq!(
                    try_batch_invert_scaled_by(
                        &records,
                        |record| record.1,
                        &scale,
                        &mut scratch,
                        |index, _, inverse| {
                            assert_value(inverse, &expected[index]);
                            assert!(!seen[index]);
                            calls += 1;
                            if calls == stop_after + 1 {
                                return Err(VisitError::Stop);
                            }
                            seen[index] = true;
                            Ok(())
                        },
                    ),
                    Err(VisitError::Stop)
                );
                assert_eq!(calls, stop_after + 1);
                assert_eq!(seen.iter().filter(|seen| **seen).count(), stop_after);
                assert!(
                    scratch[records.len().min(capacity)..]
                        .iter()
                        .all(|value| { value.montgomery_limbs() == sentinel.montgomery_limbs() })
                );
            }
        }
    }
}

#[test]
fn scaled_batches_match_integer_inverses() {
    scaled_batches::<PallasBase>();
    scaled_batches::<PallasScalar>();
}

fn scaled_batches<M: PrimeModulus>() {
    let p = modulus::<M>();
    let corpus = samples::<M>(17);
    let sentinel = PastaField::from_u64(91);
    let loose_zero = PastaField::from_montgomery_limbs(M::MODULUS);
    // Both scales and denominators include Montgomery limb boundaries, loose
    // zero, ordinary integer boundaries, and deterministic full-width values.
    for (scale, scale_integer) in samples::<M>(2) {
        for zero_parity in [None, Some(0), Some(1), Some(2)] {
            let samples: Vec<_> = corpus
                .iter()
                .enumerate()
                .map(|(i, (value, integer))| {
                    if zero_parity.is_some_and(|parity| parity == 2 || i % 2 == parity) {
                        (loose_zero, BigUint::from(0u8))
                    } else {
                        (*value, integer.clone())
                    }
                })
                .collect();
            let expected: Vec<_> = samples
                .iter()
                .map(|(_, integer)| integer.modpow(&(&p - 2u8), &p) * &scale_integer % &p)
                .collect();
            for len in [0, 1, 2, 3, 7, 8, 31, 32, 33, samples.len()] {
                for capacity in [0, 1, 2, 7, 31, 32, samples.len(), samples.len() + 3] {
                    let original: Vec<_> = samples[..len].iter().map(|(value, _)| *value).collect();
                    let mut scratch = vec![sentinel; capacity];
                    let expected_inversions = original
                        .chunks(capacity.max(1))
                        .filter(|chunk| chunk.iter().any(|value| !value.is_zero()))
                        .count();
                    // Compare the slice and grouped facades, reusing dirty
                    // scratch and crossing empty, odd, and singleton groups.
                    for grouped in [false, true] {
                        let mut values = original.clone();
                        let inversions = count_inversions(|| {
                            if grouped {
                                let (first, rest) = values.split_at_mut(len.min(1));
                                let (middle, last) = rest.split_at_mut(rest.len().min(3));
                                batch_invert_groups_scaled(
                                    &mut [&mut [][..], first, &mut [], middle, last, &mut []],
                                    &scale,
                                    &mut scratch,
                                );
                            } else {
                                batch_invert_scaled(&mut values, &scale, &mut scratch);
                            }
                        });
                        assert_eq!(inversions, expected_inversions);
                        for (actual, expected) in values.iter().zip(&expected) {
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
    batch_invert_groups_scaled::<M>(&mut [] as &mut [&mut [PastaField<M>]], &loose_zero, &mut []);
}

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
    batch_invert_groups::<M>(&mut [] as &mut [&mut [PastaField<M>]], &mut []);
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
    // Compare native and consumer entry points against the integer inverse.
    let methods: &[(&str, Invert<PastaField<M>>)] = &[
        ("native", batch_invert::<M>),
        #[cfg(feature = "traits")]
        ("trait", |values, scratch| {
            use crate::field::{Field, FieldAdapter};
            FieldAdapter::batch_invert(
                FieldAdapter::from_slice_mut(values),
                FieldAdapter::from_slice_mut(scratch),
            );
        }),
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
                for &(name, invert) in methods {
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
