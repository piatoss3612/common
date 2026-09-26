use super::*;
use crate::field::pasta::test_support::{field_samples, integer, modulus};
use crate::field::{PallasBase, PallasScalar};
use crate::polynomial::evaluate;
use num_bigint::BigUint;
use std::{vec, vec::Vec};

fn from_raw<M: PrimeModulus>(raw: &BigUint) -> PastaField<M> {
    let digits = raw.to_u64_digits();
    let mut limbs = [0; 4];
    limbs[..digits.len()].copy_from_slice(&digits);
    PastaField::from_montgomery_limbs(limbs)
}

fn check_linear<M: PrimeModulus, S: ReductionState>(
    coefficients: &[PastaField<M>],
    point: &PastaField<M, S>,
) {
    let p = modulus::<M>();
    let x = canonical(&[*point]).remove(0);
    let original = canonical(coefficients);
    let sentinel = from_raw::<M>(&(&p * 2u8 - 1u8));
    let mut storage = coefficients.to_vec();
    storage.extend([sentinel; 3]);
    let n = coefficients.len();
    let split = divide_linear_in_place(&mut storage[..n], point);
    assert_eq!(split, n.min(1));
    let remainder = storage[..split]
        .first()
        .copied()
        .unwrap_or(PastaField::ZERO);
    if n == 1 {
        assert_eq!(
            storage[0].montgomery_limbs(),
            coefficients[0].montgomery_limbs()
        );
    }
    // The degree-one monic API has exactly the same split and field values.
    let mut monic = coefficients.to_vec();
    let divisor = [point.neg(), PastaField::ONE];
    assert_eq!(divide_monic_in_place(&mut monic, &divisor), Ok(split));
    let actual = canonical(&storage[..n]);
    assert_eq!(canonical(&monic), actual);
    assert!(
        storage[n..]
            .iter()
            .all(|a| { a.montgomery_limbs() == sentinel.montgomery_limbs() })
    );

    // Multiply the returned quotient by the divisor and add the remainder,
    // using only integer coefficients rather than another division recurrence.
    let remainder_integer = actual.first().cloned().unwrap_or_default();
    let mut reconstructed = vec![BigUint::from(0u8); n.max(1)];
    reconstructed[0] = remainder_integer.clone();
    for (index, q) in actual[split..].iter().enumerate() {
        reconstructed[index] = (&reconstructed[index] + &p - &x * q % &p) % &p;
        reconstructed[index + 1] = (&reconstructed[index + 1] + q) % &p;
    }
    assert_eq!(&reconstructed[..n], original);
    if n == 0 {
        assert_eq!(reconstructed[0], BigUint::from(0u8));
    }

    // A direct integer power sum independently checks the retained evaluation.
    let mut power = BigUint::from(1u8);
    let mut evaluated = BigUint::from(0u8);
    for coefficient in &original {
        evaluated = (evaluated + coefficient * &power) % &p;
        power = power * &x % &p;
    }
    assert_eq!(remainder_integer, evaluated);
    assert_eq!(remainder.reduce(), evaluate(coefficients, point).reduce());

    // A caller can reuse a known evaluation away from the divisor's root.
    let y = point.add(&PastaField::<M>::from_u64(7));
    let denominator_inverse = y.sub(point).invert().unwrap();
    let quotient_at_y = evaluate(coefficients, &y)
        .sub(&remainder)
        .mul(&denominator_inverse);
    assert_eq!(
        quotient_at_y.reduce(),
        evaluate(&storage[split..n], &y).reduce()
    );
}

fn linear_field<M: PrimeModulus>() {
    let p = modulus::<M>();
    let mut values: Vec<_> = field_samples::<M>().take(1025).collect();
    let mut points = vec![
        PastaField::<M>::ZERO,
        PastaField::ONE,
        PastaField::<M>::ONE.neg(),
        PastaField::ZETA,
        values[0],
        values[1],
    ];
    for raw in [
        BigUint::from(1u8),
        &p - 1u8,
        p.clone(),
        &p + 1u8,
        &p * 2u8 - 1u8,
        integer(&PastaField::<M>::ONE.montgomery_limbs()) + &p,
    ] {
        points.push(from_raw(&raw));
    }
    values[..points.len()].copy_from_slice(&points);
    for point in points {
        for length in [0, 1, 2, 3, 4, 7, 8, 9, 31, 32, 33, 64, 65, 129, 1025] {
            check_linear(&values[..length], &point);
            check_linear(&values[..length], &point.reduce());
        }
        for coefficients in [
            vec![PastaField::ZERO; 65],
            vec![from_raw::<M>(&p); 65],
            vec![from_raw::<M>(&(&p * 2u8 - 1u8)); 65],
            vec![values[6], values[7], PastaField::ZERO, from_raw(&p)],
            vec![point.neg(), PastaField::ONE],
        ] {
            check_linear(&coefficients, &point);
        }
    }

    // Each division advances through the quotient suffix, preserving remainders.
    let point = PastaField::<M>::from_u64(7);
    let mut coefficients = [point.square(), point.double().neg(), PastaField::ONE];
    let mut offset = 0;
    for _ in 0..2 {
        let split = divide_linear_in_place(&mut coefficients[offset..], &point);
        assert_eq!(split, 1);
        offset += split;
        assert!(coefficients[..offset].iter().all(PastaField::is_zero));
    }
    let split = divide_linear_in_place(&mut coefficients[offset..], &point);
    assert_eq!(split, 1);
    assert!(coefficients[offset].is_one());
    offset += split;
    assert_eq!(
        divide_linear_in_place(&mut coefficients[offset..], &point),
        0
    );
}

#[test]
fn linear_division_reconstructs_integer_coefficients() {
    linear_field::<PallasBase>();
    linear_field::<PallasScalar>();
}

fn canonical<M: PrimeModulus, S: ReductionState>(values: &[PastaField<M, S>]) -> Vec<BigUint> {
    let p = modulus::<M>();
    let inverse_r = (BigUint::from(1u8) << 256usize).modinv(&p).unwrap();
    values
        .iter()
        .map(|value| {
            let raw = integer(&value.montgomery_limbs());
            assert!(raw < &p * 2u8);
            raw * &inverse_r % &p
        })
        .collect()
}

fn multiply(a: &[BigUint], b: &[BigUint], p: &BigUint) -> Vec<BigUint> {
    let mut product = vec![BigUint::from(0u8); a.len() + b.len() - 1];
    for (i, a) in a.iter().enumerate() {
        for (j, b) in b.iter().enumerate() {
            product[i + j] = (&product[i + j] + a * b) % p;
        }
    }
    product
}

fn check_monic<M: PrimeModulus, S: ReductionState>(
    input: &[PastaField<M>],
    divisor: &[PastaField<M, S>],
) {
    let p = modulus::<M>();
    let sentinel = from_raw::<M>(&(&p * 2u8 - 1u8));
    let mut storage = input.to_vec();
    storage.extend([sentinel; 3]);
    let original_limbs: Vec<_> = input.iter().map(PastaField::montgomery_limbs).collect();
    let split = divide_monic_in_place(&mut storage[..input.len()], divisor).unwrap();
    let degree = divisor.len() - 1;
    assert_eq!(split, input.len().min(degree));
    assert!(
        storage[input.len()..]
            .iter()
            .all(|v| v.montgomery_limbs() == sentinel.montgomery_limbs())
    );
    if degree == 0 || input.len() <= degree {
        assert_eq!(
            storage[..input.len()]
                .iter()
                .map(PastaField::montgomery_limbs)
                .collect::<Vec<_>>(),
            original_limbs
        );
    }
    let (remainder, quotient) = storage[..input.len()].split_at(split);
    let q = canonical(quotient);
    let d = canonical(divisor);
    // Convolution and addition uniquely establish division with deg(r) < deg(d),
    // without sharing the descending division recurrence.
    let mut reconstructed = vec![BigUint::from(0u8); input.len()];
    if !q.is_empty() {
        reconstructed = multiply(&q, &d, &p);
    }
    for (coefficient, remainder) in reconstructed.iter_mut().zip(canonical(remainder)) {
        *coefficient = (&*coefficient + remainder) % &p;
    }
    assert_eq!(reconstructed, canonical(input));
}

fn monic_field<M: PrimeModulus>() {
    let p = modulus::<M>();
    let boundaries = [
        BigUint::from(0u8),
        BigUint::from(1u8),
        &p - 1u8,
        p.clone(),
        &p + 1u8,
        &p * 2u8 - 1u8,
        integer(&PastaField::<M>::ONE.montgomery_limbs()) + &p,
    ]
    .map(|raw| from_raw::<M>(&raw));
    let mut values: Vec<_> = field_samples::<M>().take(257).collect();
    values[..boundaries.len()].copy_from_slice(&boundaries);
    for degree in [0, 1, 2, 3, 4, 5, 8, 17, 32] {
        let mut dense: Vec<_> = values.iter().rev().take(degree).copied().collect();
        dense.push(PastaField::ONE);
        let mut boundary: Vec<_> = boundaries.iter().cycle().take(degree).copied().collect();
        boundary.push(boundaries[6]); // A loose representative of one is monic.
        let mut power = vec![from_raw::<M>(&p); degree];
        power.push(PastaField::ONE);
        for divisor in [dense, boundary, power] {
            let reduced: Vec<_> = divisor.iter().map(|v| v.reduce()).collect();
            for length in [
                0,
                1,
                degree.saturating_sub(1),
                degree,
                degree + 1,
                degree + 2,
                65,
                257,
            ] {
                check_monic(&values[..length], &divisor);
                check_monic(&values[..length], &reduced);
            }
            for repeated in [PastaField::ZERO, from_raw(&p), from_raw(&(&p * 2u8 - 1u8))] {
                check_monic(&vec![repeated; 65], &divisor);
            }
            let mut trailing = values[..65].to_vec();
            trailing[60..].fill(from_raw(&p));
            check_monic(&trailing, &divisor);
            check_monic(&divisor, &divisor);
        }
    }
}

#[test]
fn monic_division_reconstructs_integer_coefficients() {
    monic_field::<PallasBase>();
    monic_field::<PallasScalar>();
}

fn invalid_divisors<M: PrimeModulus>() {
    let p = modulus::<M>();
    let original = [from_raw::<M>(&(&p * 2u8 - 1u8)); 8];
    let mut output = original;
    for n in [0, 1, 8] {
        for divisor in [
            vec![],
            vec![PastaField::ZERO],
            vec![from_raw(&p)],
            vec![PastaField::ONE, PastaField::ZERO],
            vec![PastaField::from_u64(2)],
        ] {
            assert_eq!(
                divide_monic_in_place(&mut output[..n], &divisor),
                Err(if divisor.is_empty() {
                    MonicDivisionError::EmptyDivisor
                } else {
                    MonicDivisionError::NonMonicDivisor
                })
            );
            assert_eq!(
                output.map(|v| v.montgomery_limbs()),
                original.map(|v| v.montgomery_limbs())
            );
        }
    }
}

#[test]
fn invalid_divisors_preserve_storage() {
    invalid_divisors::<PallasBase>();
    invalid_divisors::<PallasScalar>();
}

#[cfg(feature = "traits")]
mod consumer {
    use super::*;
    use crate::field::pasta::test_support::{count_mul_adds, field_samples};
    use crate::field::{FieldAdapter, PallasBase, PallasScalar};
    type Fp = FieldAdapter<PallasBase>;
    use crate::polynomial::{divide_linear_rev, evaluate};

    #[test]
    fn descending_quotient_matches_in_place_division_and_identity() {
        fn check<M: PrimeModulus>() {
            for count in [0, 1, 2, 3, 7, 16] {
                let original: Vec<_> = field_samples::<M>().take(count).collect();
                for point in field_samples::<M>().take(5) {
                    let mut in_place = original.clone();
                    let split = divide_linear_in_place(&mut in_place, &point);
                    let mut descending = Vec::new();
                    assert_eq!(
                        count_mul_adds(|| {
                            descending = divide_linear_rev(
                                original.iter().copied().map(FieldAdapter::new),
                                FieldAdapter::new(point),
                            )
                            .map(FieldAdapter::into_inner)
                            .collect();
                        }),
                        count.saturating_sub(2)
                    );
                    assert_eq!(descending.len(), count.saturating_sub(1));
                    descending.reverse();
                    assert!(
                        descending
                            .iter()
                            .map(|value| value.reduce())
                            .eq(in_place[split..].iter().map(|value| value.reduce()))
                    );
                    let remainder = evaluate(&original, &point);
                    for x in field_samples::<M>().take(4) {
                        assert_eq!(
                            evaluate(&original, &x).reduce(),
                            evaluate(&descending, &x)
                                .mul(&x.sub(&point))
                                .add(&remainder)
                                .reduce()
                        );
                    }
                }
            }
        }
        check::<PallasBase>();
        check::<PallasScalar>();
    }

    #[test]
    fn descending_quotient_reads_coefficients_incrementally() {
        let reads = core::cell::Cell::new(0);
        let values = [1, 2, 3, 4].map(Fp::from);
        let coefficients = values.into_iter().inspect(|_| reads.set(reads.get() + 1));
        let mut quotient = divide_linear_rev(coefficients, Fp::from(2));
        assert_eq!(reads.get(), 1);
        assert_eq!(quotient.next(), Some(Fp::from(4)));
        assert_eq!(reads.get(), 3);
        assert_eq!(quotient.next(), Some(Fp::from(11)));
        assert_eq!(quotient.next(), Some(Fp::from(24)));
        assert_eq!(quotient.next(), None);
        assert_eq!(reads.get(), 4);
    }
}
