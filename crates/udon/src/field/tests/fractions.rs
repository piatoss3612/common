use super::*;
use crate::field::{
    FractionPrefixError, count_inversions, fraction_prefixes, fraction_prefixes_in_place,
};
use crate::test_support::field_samples;

fn from_raw<M: PrimeModulus>(raw: &BigUint) -> PastaField<M> {
    let digits = raw.to_u64_digits();
    let mut limbs = [0; 4];
    limbs[..digits.len()].copy_from_slice(&digits);
    PastaField::from_montgomery_limbs(limbs)
}

fn oracle<M: PrimeModulus>(
    numerators: &[PastaField<M>],
    denominators: &[PastaField<M>],
    initial: &PastaField<M>,
) -> Vec<BigUint> {
    let p = modulus::<M>();
    let inverse_r = (BigUint::from(1u8) << 256usize).modinv(&p).unwrap();
    let decode = |value: &PastaField<M>| integer(&value.montgomery_limbs()) * &inverse_r % &p;
    let mut product = decode(initial);
    let mut output = vec![product.clone()];
    for (num, den) in numerators.iter().zip(denominators) {
        let inverse = decode(den).modinv(&p).unwrap_or_default();
        product = product * decode(num) * inverse % &p;
        output.push(product.clone());
    }
    output
}

fn check<M: PrimeModulus, N: ReductionState, D: ReductionState, I: ReductionState>(
    numerators: &[PastaField<M, N>],
    denominators: &[PastaField<M, D>],
    initial: &PastaField<M, I>,
    expected: &[BigUint],
) {
    let p = modulus::<M>();
    let n = numerators.len();
    let sentinel = from_raw::<M>(&(&p * 2u8 - 1u8));
    let mut scratch = vec![sentinel; n + 3];
    let mut output = vec![sentinel; n + 4];
    let assert_output = |output: &[PastaField<M>], scratch: &[PastaField<M>]| {
        assert_eq!(output[0].montgomery_limbs(), initial.montgomery_limbs());
        for (actual, expected) in output.iter().zip(expected) {
            let raw = integer(&actual.montgomery_limbs());
            assert!(raw < &p * 2u8);
            assert_eq!(raw % &p, (expected << 256usize) % &p);
        }
        assert!(
            output[n + 1..]
                .iter()
                .chain(&scratch[n..])
                .all(|value| value.montgomery_limbs() == sentinel.montgomery_limbs())
        );
    };
    let inversions = count_inversions(|| {
        assert_eq!(
            fraction_prefixes(numerators, denominators, initial, &mut output, &mut scratch),
            Ok(n + 1)
        );
    });
    // The counter measures actual nonzero inversions. A failed inversion of
    // the zero aggregate returns before the inverter and does not count.
    let expected_inversions =
        usize::from(!initial.is_zero() && denominators.first().is_some_and(|den| !den.is_zero()));
    assert_eq!(inversions, expected_inversions);
    assert_output(&output, &scratch);

    for (out, num) in output.iter_mut().zip(numerators) {
        *out = num.into_loose();
    }
    // Reuse the previous call's dirty scratch; no initialization is required.
    let inversions = count_inversions(|| {
        assert_eq!(
            fraction_prefixes_in_place(&mut output, denominators, initial, &mut scratch),
            Ok(n + 1)
        );
    });
    assert_eq!(inversions, expected_inversions);
    assert_output(&output, &scratch);
    assert_eq!(
        fraction_prefixes(
            numerators,
            denominators,
            initial,
            &mut output[..n + 1],
            &mut scratch[..n]
        ),
        Ok(n + 1)
    );
    assert_output(&output, &scratch);
}

fn check_states<M: PrimeModulus>(
    numerators: &[PastaField<M>],
    denominators: &[PastaField<M>],
    initial: &PastaField<M>,
) {
    let expected = oracle(numerators, denominators, initial);
    let reduced_n: Vec<_> = numerators.iter().map(|x| x.reduce()).collect();
    let reduced_d: Vec<_> = denominators.iter().map(|x| x.reduce()).collect();
    check(numerators, denominators, initial, &expected);
    check(numerators, &reduced_d, initial, &expected);
    check(&reduced_n, denominators, initial, &expected);
    check(&reduced_n, &reduced_d, initial, &expected);
    check(numerators, denominators, &initial.reduce(), &expected);
    check(numerators, &reduced_d, &initial.reduce(), &expected);
    check(&reduced_n, denominators, &initial.reduce(), &expected);
    check(&reduced_n, &reduced_d, &initial.reduce(), &expected);
}

fn check_field<M: PrimeModulus>() {
    let p = modulus::<M>();
    let zero = from_raw::<M>(&p);
    let one = from_raw::<M>(&(integer(&PastaField::<M>::ONE.montgomery_limbs()) + &p));
    let maximum = from_raw::<M>(&(&p * 2u8 - 1u8));
    let values: Vec<_> = field_samples::<M>().take(260).collect();
    for n in [0, 1, 2, 3, 4, 5, 8, 17, 32, 65, 129] {
        let numerators = &values[..n];
        let denominators = &values[n..2 * n];
        for initial in [PastaField::ZERO, zero, one, maximum] {
            check_states(numerators, denominators, &initial);
        }
        for index in 0..n {
            let mut den = denominators.to_vec();
            den[index] = zero;
            check_states(numerators, &den, &maximum);
            // Multiple zeros must retain the first original zero's position,
            // including when it was the high member of a denominator pair.
            den[n - 1] = PastaField::ZERO;
            check_states(numerators, &den, &one);
            let mut num = numerators.to_vec();
            num[index] = zero;
            check_states(&num, denominators, &maximum);
            check_states(&num, &den, &maximum);
            if index != 0 {
                num[index - 1] = zero;
                check_states(&num, &den, &maximum);
            }
        }
        check_states(&vec![zero; n], &vec![zero; n], &maximum);
        // Shared slices with equal products still follow the literal zero
        // recurrence, rather than cancelling a zero numerator/denominator.
        let mut shared = numerators.to_vec();
        if n > 0 {
            shared[n / 2] = zero;
        }
        check_states(&shared, &shared, &one);
    }
    let edges: Vec<_> = [
        BigUint::from(0u8),
        BigUint::from(1u8),
        &p - 1u8,
        p.clone(),
        &p + 1u8,
        &p * 2u8 - 1u8,
        integer(&one.montgomery_limbs()),
    ]
    .iter()
    .map(from_raw::<M>)
    .collect();
    for num in &edges {
        for den in &edges {
            check_states(&[*num; 7], &[*den; 7], &maximum);
        }
    }
}

#[test]
fn fraction_prefixes_match_integer_recurrence() {
    check_field::<PallasBase>();
    check_field::<PallasScalar>();
}

fn check_errors<M: PrimeModulus>() {
    let sentinel = from_raw::<M>(&modulus::<M>());
    let numerators = [PastaField::<M>::ONE; 5];
    let original = [sentinel; 8];
    for initial in [PastaField::<M>::ZERO, PastaField::ONE] {
        for n in 0..=5 {
            for out_len in 0..n + 1 {
                let mut output = original;
                let mut scratch = original;
                let error = Err(FractionPrefixError::OutputTooShort {
                    required: n + 1,
                    actual: out_len,
                });
                assert_eq!(
                    fraction_prefixes(
                        &numerators[..n],
                        &numerators[..n],
                        &initial,
                        &mut output[..out_len],
                        &mut scratch
                    ),
                    error
                );
                assert_eq!(
                    fraction_prefixes_in_place(
                        &mut output[..out_len],
                        &numerators[..n],
                        &initial,
                        &mut scratch
                    ),
                    error
                );
                assert_eq!(
                    output.map(|x| x.montgomery_limbs()),
                    original.map(|x| x.montgomery_limbs())
                );
                assert_eq!(
                    scratch.map(|x| x.montgomery_limbs()),
                    original.map(|x| x.montgomery_limbs())
                );
            }
            for scratch_len in 0..n {
                let mut output = original;
                let mut scratch = original;
                let error = Err(FractionPrefixError::ScratchTooShort {
                    required: n,
                    actual: scratch_len,
                });
                assert_eq!(
                    fraction_prefixes(
                        &numerators[..n],
                        &numerators[..n],
                        &initial,
                        &mut output,
                        &mut scratch[..scratch_len]
                    ),
                    error
                );
                assert_eq!(
                    fraction_prefixes_in_place(
                        &mut output,
                        &numerators[..n],
                        &initial,
                        &mut scratch[..scratch_len]
                    ),
                    error
                );
                assert_eq!(
                    output.map(|x| x.montgomery_limbs()),
                    original.map(|x| x.montgomery_limbs())
                );
                assert_eq!(
                    scratch.map(|x| x.montgomery_limbs()),
                    original.map(|x| x.montgomery_limbs())
                );
            }
            for other in 0..=5 {
                if n == other {
                    continue;
                }
                let mut output = original;
                let mut scratch = original;
                assert_eq!(
                    fraction_prefixes(
                        &numerators[..n],
                        &numerators[..other],
                        &initial,
                        &mut output,
                        &mut scratch
                    ),
                    Err(FractionPrefixError::LengthMismatch)
                );
                assert_eq!(
                    output.map(|x| x.montgomery_limbs()),
                    original.map(|x| x.montgomery_limbs())
                );
                assert_eq!(
                    scratch.map(|x| x.montgomery_limbs()),
                    original.map(|x| x.montgomery_limbs())
                );
            }
        }
    }
}

#[test]
fn invalid_lengths_leave_both_buffers_unchanged() {
    check_errors::<PallasBase>();
    check_errors::<PallasScalar>();
}
