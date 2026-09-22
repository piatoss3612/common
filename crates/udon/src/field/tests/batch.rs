use super::*;
use crate::field::inversion::count_inversions;
use crate::field::{BatchInversionError, batch_invert, batch_invert_groups, try_batch_invert_by};
use crate::test_support::field_samples;

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
