use super::*;
use crate::field::{BatchInversionError, batch_invert, batch_invert_groups};
use crate::test_support::field_samples;

fn exercise<M: PrimeModulus>() {
    let samples = samples::<M>(37);
    let original: Vec<_> = samples.iter().map(|(value, _)| *value).collect();
    let mut values = original.clone();
    let sentinel = PastaField::from_u64(19);
    let mut scratch = vec![sentinel; values.len() + 3];
    batch_invert(&mut values, &mut scratch).unwrap();
    let p = modulus::<M>();
    for ((actual, (value, integer)), expected) in values.iter().zip(&samples).zip(&original) {
        assert_eq!(*actual, value.invert().unwrap_or(PastaField::ZERO));
        assert_value(*actual, &integer.modpow(&(&p - 2u8), &p));
        assert_eq!(*value, *expected);
    }
    assert_eq!(&scratch[values.len()..], &[sentinel; 3]);
    // Reuse dirty prefixes, crossing both odd and empty group boundaries.
    let (a, rest) = values.split_at_mut(3);
    let (b, c) = rest.split_at_mut(4);
    batch_invert_groups(&mut [&mut [], a, &mut [], b, c, &mut []], &mut scratch).unwrap();
    assert_eq!(values, original);
    assert_eq!(&scratch[values.len()..], &[sentinel; 3]);

    let before = values.clone();
    let mut short = vec![sentinel; values.len() - 1];
    let (a, b) = values.split_at_mut(1);
    assert_eq!(
        batch_invert_groups(&mut [a, b], &mut short),
        Err(BatchInversionError::ScratchTooSmall {
            required: before.len(),
            provided: short.len()
        })
    );
    assert_eq!(values, before);
    assert!(short.iter().all(|value| *value == sentinel));
    assert!(batch_invert(&mut values, &mut short).is_err());
    assert_eq!(values, before);
    assert!(short.iter().all(|value| *value == sentinel));

    for len in [0, 1, 2, 7] {
        let mut zeros = vec![PastaField::<M>::ZERO; len];
        batch_invert(&mut zeros, &mut scratch).unwrap();
        assert!(zeros.iter().all(PastaField::is_zero));
    }
    batch_invert_groups::<M>(&mut [], &mut []).unwrap();
    let mut singleton = [sentinel];
    batch_invert(&mut singleton, &mut scratch).unwrap();
    assert_eq!(singleton[0].mul(&sentinel), PastaField::ONE);
    let mut a = [PastaField::ZERO, sentinel, PastaField::ZERO];
    let mut b = [PastaField::ZERO, sentinel];
    batch_invert_groups(&mut [&mut a, &mut b], &mut scratch).unwrap();
    assert_eq!(a, [PastaField::ZERO, singleton[0], PastaField::ZERO]);
    assert_eq!(b, [PastaField::ZERO, singleton[0]]);
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
                let mut values = original.clone();
                let tail = scratch[len..].to_vec();
                let (left, right) = values.split_at_mut(split);
                batch_invert_groups(&mut [&mut [], left, &mut [], right, &mut []], &mut scratch)
                    .unwrap();
                assert_eq!(
                    values, expected,
                    "length {len}, zeros {zeros}, split {split}"
                );
                assert_eq!(&scratch[len..], tail);
            }
        }
    }
}

#[test]
fn group_boundaries_preserve_zero_positions() {
    group_boundaries::<PallasBase>();
    group_boundaries::<PallasScalar>();
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
        assert_eq!(prefix[n], PastaField::from_u64(91));
        for (value, inverse) in values.iter().zip(&input) {
            assert_eq!(value.mul(inverse), PastaField::ONE);
        }
        // Reuse dirty prefixes with unit denominators and both lane parities.
        input.fill(PastaField::ONE);
        crate::field::invert_nonzero(&mut input, &mut prefix);
        assert!(input.iter().all(|x| *x == PastaField::ONE));
    }
}

#[test]
fn batch_inversion_handles_lane_endpoints() {
    inversion_endpoints::<crate::field::PallasBase>();
    inversion_endpoints::<crate::field::PallasScalar>();
}
