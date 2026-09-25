use super::*;
use crate::{
    curve::{Pallas, PastaCurve, ProjectivePoint, Vesta},
    test_support::{integer, modulus},
};
use num_bigint::BigUint;

// Bypass production scalar decomposition, recoding, and multiplication tables.
fn generator_multiple<C: PastaCurve>(scalar: &BigUint) -> ProjectivePoint<C> {
    let mut result = ProjectivePoint::IDENTITY;
    for bit in (0..scalar.bits()).rev() {
        result = result.double();
        if scalar.bit(bit) {
            result = result.add(&ProjectivePoint::GENERATOR);
        }
    }
    result
}

// The scalar DFT uses integer sums and powers independently of the FFT schedule.
// Encoding roots and scales and converting results to generator multiples still
// use production field and curve arithmetic.
fn direct<C: PastaCurve>(
    coefficients: &[BigUint],
    root: &PastaField<C::Scalar>,
    scale: &PastaField<C::Scalar>,
) -> Vec<ProjectivePoint<C>> {
    let q = modulus::<C::Scalar>();
    let root = BigUint::from_bytes_le(&root.to_bytes());
    let scale = BigUint::from_bytes_le(&scale.to_bytes());
    (0..coefficients.len())
        .map(|row| {
            let sum: BigUint = coefficients
                .iter()
                .enumerate()
                .map(|(column, coefficient)| {
                    coefficient * root.modpow(&BigUint::from(row * column), &q)
                })
                .sum();
            generator_multiple::<C>(&(sum * &scale % &q))
        })
        .collect()
}

fn loose<M: PrimeModulus>(value: PastaField<M>) -> PastaField<M> {
    let mut limbs = (integer(&value.reduce().montgomery_limbs()) + modulus::<M>()).to_u64_digits();
    limbs.resize(4, 0);
    PastaField::from_montgomery_limbs(limbs.try_into().unwrap())
}

fn exercise<C: PastaCurve>() {
    let q = modulus::<C::Scalar>();
    for log_size in 0..=6 {
        let domain = Domain::<C::Scalar>::new(log_size).unwrap();
        let size = domain.size();
        let dense: Vec<_> = field_samples::<C::Scalar>()
            .take(size)
            .map(|s| BigUint::from_bytes_le(&s.to_bytes()))
            .collect();
        for shape in 0..5 {
            let coefficients: Vec<_> = (0..size)
                .map(|i| match shape {
                    0 => BigUint::from(0_u32),
                    1 => BigUint::from(7_u32),
                    2 => {
                        if i % 2 == 0 {
                            BigUint::from(1_u32)
                        } else {
                            &q - 1_u32
                        }
                    }
                    3 => BigUint::from(u32::from(i == size - 1)),
                    _ => dense[i].clone(),
                })
                .collect();
            let original: Vec<_> = coefficients.iter().map(generator_multiple::<C>).collect();
            let forward = direct::<C>(&coefficients, &domain.root(), &PastaField::ONE);
            let inverse = direct::<C>(
                &coefficients,
                &domain.inverse_root(),
                &domain.size_inverse(),
            );
            for affine in [false, true] {
                let input: Vec<_> = original
                    .iter()
                    .map(|point| {
                        if affine {
                            point.to_point().to_projective()
                        } else {
                            // Change Jacobian scales through complete arithmetic,
                            // including identity and equal/opposite intermediates.
                            point
                                .add(&ProjectivePoint::GENERATOR)
                                .sub(&ProjectivePoint::GENERATOR)
                        }
                    })
                    .collect();
                assert_eq!(input, original);
                let mut actual = input.clone();
                reference::transform(&mut actual, &loose(domain.root()));
                assert_eq!(actual, forward, "forward log={log_size} shape={shape}");
                reference::inverse_transform(
                    &mut actual,
                    &loose(domain.inverse_root()),
                    &loose(domain.size_inverse()),
                );
                assert_eq!(actual, original, "round trip log={log_size} shape={shape}");

                // An inverse must also agree on arbitrary input, independently
                // of any matching error in the preceding forward transform.
                actual.copy_from_slice(&input);
                reference::inverse_transform(
                    &mut actual,
                    &domain.inverse_root(),
                    &domain.size_inverse(),
                );
                assert_eq!(actual, inverse, "inverse log={log_size} shape={shape}");
            }
            if log_size <= 4 && shape == 4 {
                for scale in [PastaField::ZERO, PastaField::ONE, PastaField::from_u64(7)] {
                    let mut actual = original.clone();
                    reference::inverse_transform(
                        &mut actual,
                        &domain.inverse_root(),
                        &loose(scale),
                    );
                    assert_eq!(
                        actual,
                        direct::<C>(&coefficients, &domain.inverse_root(), &scale)
                    );
                }
            }
        }
    }
}

#[test]
fn projective_transforms_match_integer_dfts_and_independent_inverses() {
    exercise::<Pallas>();
    exercise::<Vesta>();
}

fn lengths<C: PastaCurve>() {
    for size in [0, 3, 5, 6, 17] {
        let original: Vec<_> = (0..size + 2)
            .map(|i| generator_multiple::<C>(&BigUint::from(i + 1)))
            .collect();
        for inverse in [false, true] {
            let mut actual = original.clone();
            let result = catch_unwind(AssertUnwindSafe(|| {
                if inverse {
                    reference::inverse_transform(
                        &mut actual[..size],
                        &PastaField::ONE,
                        &PastaField::ONE,
                    );
                } else {
                    reference::transform(&mut actual[..size], &PastaField::ONE);
                }
            }));
            assert!(result.is_err());
            assert_eq!(
                actual, original,
                "invalid lengths must reject before mutation"
            );
        }
    }
    for size in [1, 2, 4, 8, 16] {
        let domain = Domain::<C::Scalar>::for_size(size).unwrap();
        let mut values = vec![ProjectivePoint::<C>::GENERATOR; size + 2];
        let tail = [ProjectivePoint::IDENTITY, ProjectivePoint::GENERATOR.neg()];
        values[size..].copy_from_slice(&tail);
        reference::transform(&mut values[..size], &domain.root());
        reference::inverse_transform(
            &mut values[..size],
            &domain.inverse_root(),
            &domain.size_inverse(),
        );
        assert_eq!(values[size..], tail);
        assert!(
            values[..size]
                .iter()
                .all(|p| *p == ProjectivePoint::GENERATOR)
        );
    }
}

#[test]
fn projective_transform_length_checks_and_slice_boundaries() {
    lengths::<Pallas>();
    lengths::<Vesta>();
}
