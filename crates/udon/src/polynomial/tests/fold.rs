use super::*;
use crate::field::{PallasBase, PallasScalar};
use crate::test_support::{field_samples, integer, modulus};
use num_bigint::BigUint;
use std::{vec, vec::Vec};

fn from_raw<M: PrimeModulus>(raw: &BigUint) -> PastaField<M> {
    let digits = raw.to_u64_digits();
    let mut limbs = [0; 4];
    limbs[..digits.len()].copy_from_slice(&digits);
    PastaField::from_montgomery_limbs(limbs)
}

fn oracle<M: PrimeModulus, S: ReductionState, W: ReductionState>(
    inputs: &[&[PastaField<M, S>]],
    weights: &[PastaField<M, W>],
) -> Vec<BigUint> {
    let p = modulus::<M>();
    let inverse_r_squared = (BigUint::from(1u8) << 512usize).modinv(&p).unwrap();
    let extent = inputs.iter().map(|input| input.len()).max().unwrap_or(0);
    (0..extent)
        .map(|j| {
            let mut sum = BigUint::from(0u8);
            for (input, weight) in inputs.iter().zip(weights) {
                if let Some(value) = input.get(j) {
                    sum += integer(&value.montgomery_limbs()) * integer(&weight.montgomery_limbs());
                }
            }
            sum * &inverse_r_squared % &p
        })
        .collect()
}

fn check<M: PrimeModulus, S: ReductionState, W: ReductionState>(
    inputs: &[&[PastaField<M, S>]],
    weights: &[PastaField<M, W>],
    expected: &[BigUint],
) {
    let p = modulus::<M>();
    let sentinel = from_raw::<M>(&(&p * 2u8 - 1u8));
    let mut output = vec![sentinel; expected.len() + 3];
    assert_eq!(
        fold_weighted(inputs, weights, &mut output),
        Ok(expected.len())
    );
    for (actual, expected) in output.iter().zip(expected) {
        let raw = integer(&actual.montgomery_limbs());
        assert!(raw < &p * 2u8);
        assert_eq!(raw % &p, (expected << 256usize) % &p);
    }
    assert!(
        output[expected.len()..]
            .iter()
            .all(|value| { value.montgomery_limbs() == sentinel.montgomery_limbs() })
    );
    // Reuse a dirty output and supply exactly the requested extent.
    assert_eq!(
        fold_weighted(inputs, weights, &mut output[..expected.len()]),
        Ok(expected.len())
    );
}

fn check_fields<M: PrimeModulus>() {
    let p = modulus::<M>();
    let mut values: Vec<_> = field_samples::<M>().take(25).collect();
    for raw in [
        BigUint::from(0u8),
        BigUint::from(1u8),
        &p - 1u8,
        p.clone(),
        &p + 1u8,
        &p * 2u8 - 1u8,
        integer(&PastaField::<M>::ONE.montgomery_limbs()) + &p,
    ] {
        values.push(from_raw(&raw));
    }
    let reduced: Vec<_> = values.iter().map(|value| value.reduce()).collect();
    let loose_one = *values.last().unwrap();
    let loose_zero = from_raw::<M>(&p);
    for count in [0, 1, 2, 3, 4, 8, 33, 129] {
        for length in [0, 1, 2, 3, 4, 5, 7, 8, 17] {
            for layout in 0..3 {
                let ranges: Vec<_> = (0..count)
                    .map(|i| {
                        let len = match layout {
                            0 => length,
                            1 => (i * 7) % (length + 1),
                            _ => (length + i * 7) % (length + 1),
                        };
                        let start = i % (values.len() - len + 1);
                        start..start + len
                    })
                    .collect();
                let inputs: Vec<_> = ranges.iter().map(|range| &values[range.clone()]).collect();
                let reduced_inputs: Vec<_> =
                    ranges.iter().map(|range| &reduced[range.clone()]).collect();
                for kind in 0..4 {
                    let weights: Vec<_> = (0..count)
                        .map(|i| match kind {
                            0 => values[(i * 13 + 7) % values.len()],
                            1 => loose_one,
                            2 => loose_zero,
                            _ => match i % 3 {
                                0 => loose_one,
                                1 => loose_zero,
                                _ => values[(i * 11 + 3) % values.len()],
                            },
                        })
                        .collect();
                    let reduced_weights: Vec<_> = weights.iter().map(|w| w.reduce()).collect();
                    let expected = oracle(&inputs, &weights);
                    check(&inputs, &weights, &expected);
                    check(&inputs, &reduced_weights, &expected);
                    check(&reduced_inputs, &weights, &expected);
                    check(&reduced_inputs, &reduced_weights, &expected);
                }
            }
        }
    }
    // Repeated maximal loose products require high-limb carry accounting.
    // Input and weight slices deliberately share the same storage.
    let maximum = from_raw::<M>(&(&p * 2u8 - 1u8));
    let edges = [PastaField::ZERO, loose_zero, loose_one, maximum];
    for a in &edges {
        for b in &edges {
            let inputs = [&edges[..], &edges[..2]];
            for weights in [[*a, *b], [*b, *a]] {
                check(&inputs, &weights, &oracle(&inputs, &weights));
                let inputs = [inputs[1], inputs[0]];
                check(&inputs, &weights, &oracle(&inputs, &weights));
            }
            check(&[&[*a]], &[*b], &oracle(&[&[*a]], &[*b]));
        }
    }
    let weights = vec![maximum; 4096];
    let inputs = vec![&weights[..5]; weights.len()];
    check(&inputs, &weights, &oracle(&inputs, &weights));
    // Cancellation must not trim the requested extent.
    let weights = [PastaField::ONE, PastaField::<M>::ONE.neg()];
    let inputs = [&values[..], &values[..]];
    check(&inputs, &weights, &vec![BigUint::from(0u8); values.len()]);
}

#[test]
fn weighted_folds_match_integer_sums() {
    check_fields::<PallasBase>();
    check_fields::<PallasScalar>();
}

fn check_errors<M: PrimeModulus>() {
    let values = [PastaField::<M>::ONE; 5];
    let original = [from_raw::<M>(&modulus::<M>()); 4];
    let mut output = original;
    let inputs = [&values[..]];
    for weights in [&[][..], &values[..2]] {
        assert_eq!(
            fold_weighted(&inputs, weights, &mut output),
            Err(FoldError::WeightCount)
        );
        assert_eq!(
            output.map(|x| x.montgomery_limbs()),
            original.map(|x| x.montgomery_limbs())
        );
    }
    for weight in [PastaField::<M>::ZERO, PastaField::ONE] {
        assert_eq!(
            fold_weighted(&inputs, &[weight], &mut output),
            Err(FoldError::OutputTooShort {
                required: 5,
                actual: 4
            })
        );
        assert_eq!(
            output.map(|x| x.montgomery_limbs()),
            original.map(|x| x.montgomery_limbs())
        );
    }
    assert_eq!(
        fold_weighted(&[&values[..0]], &values[..0], &mut []),
        Err(FoldError::WeightCount)
    );
    assert_eq!(fold_weighted(&[&values[..0]], &values[..1], &mut []), Ok(0));
    assert_eq!(
        fold_weighted(&inputs[..0], &values[..0], &mut output),
        Ok(0)
    );
    assert_eq!(
        output.map(|x| x.montgomery_limbs()),
        original.map(|x| x.montgomery_limbs())
    );
}

#[test]
fn invalid_shapes_leave_output_unchanged() {
    check_errors::<PallasBase>();
    check_errors::<PallasScalar>();
}
