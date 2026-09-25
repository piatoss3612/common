use super::{Butterfly, Twiddle, inverse_transform, transform};
use crate::curve::{Pallas, PastaCurve, ProjectivePoint, Vesta};
use crate::fft::Domain;
use crate::field::pasta::test_support::{field_samples, integer, modulus};
use crate::field::{PastaField, PrimeModulus};
use num_bigint::BigUint;
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    vec,
    vec::Vec,
};

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
        let domain = Domain::<PastaField<C::Scalar>>::new(log_size).unwrap();
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
                assert_eq!(
                    reference::count_transforms(|| {
                        reference::transform(&mut actual, &loose(domain.root()))
                    }),
                    1
                );
                assert_eq!(actual, forward, "forward log={log_size} shape={shape}");
                inverse_transform(
                    &mut actual,
                    &loose(domain.inverse_root()),
                    &loose(domain.size_inverse()),
                );
                assert_eq!(actual, original, "round trip log={log_size} shape={shape}");

                // An inverse must also agree on arbitrary input, independently
                // of any matching error in the preceding forward transform.
                actual.copy_from_slice(&input);
                inverse_transform(&mut actual, &domain.inverse_root(), &domain.size_inverse());
                assert_eq!(actual, inverse, "inverse log={log_size} shape={shape}");
            }
            if log_size <= 4 && shape == 4 {
                for scale in [PastaField::ZERO, PastaField::ONE, PastaField::from_u64(7)] {
                    let mut actual = original.clone();
                    inverse_transform(&mut actual, &domain.inverse_root(), &loose(scale));
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
                    inverse_transform(&mut actual[..size], &PastaField::ONE, &PastaField::ONE);
                } else {
                    transform(&mut actual[..size], &PastaField::ONE);
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
        let domain = Domain::<PastaField<C::Scalar>>::for_size(size).unwrap();
        let mut values = vec![ProjectivePoint::<C>::GENERATOR; size + 2];
        let tail = [ProjectivePoint::IDENTITY, ProjectivePoint::GENERATOR.neg()];
        values[size..].copy_from_slice(&tail);
        transform(&mut values[..size], &domain.root());
        inverse_transform(
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

#[test]
fn generic_reference_supports_a_foreign_field() {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct F17(u32);
    impl Twiddle for F17 {
        const ONE: Self = Self(1);
        fn multiply(&self, rhs: &Self) -> Self {
            Self(self.0 * rhs.0 % 17)
        }
        fn square(&self) -> Self {
            self.multiply(self)
        }
    }
    impl Butterfly<Self> for F17 {
        fn scaled(&self, rhs: &Self) -> Self {
            Self(self.0 * rhs.0 % 17)
        }
        fn add(&self, rhs: &Self) -> Self {
            Self((self.0 + rhs.0) % 17)
        }
        fn negated(&self) -> Self {
            Self((17 - self.0) % 17)
        }
    }
    for size in [1usize, 2, 4, 8, 16] {
        let root = F17(3u32.pow(16 / size as u32) % 17);
        let input: Vec<_> = (0..size).map(|i| F17((i as u32 + 7) % 17)).collect();
        let mut output = input.clone();
        transform(&mut output, &root);
        for (row, output) in output.iter().enumerate() {
            let mut expected = 0;
            for (column, value) in input.iter().enumerate() {
                let power = (0..row * column).fold(1, |p, _| p * root.0 % 17);
                expected = (expected + value.0 * power) % 17;
            }
            assert_eq!(*output, F17(expected));
        }
        let inverse_root = F17((1..17).find(|x| x * root.0 % 17 == 1).unwrap());
        let inverse_size = F17((1..17).find(|x| x * size as u32 % 17 == 1).unwrap());
        inverse_transform(&mut output, &inverse_root, &inverse_size);
        assert_eq!(output, input);
    }
    for size in [0, 3] {
        let mut values = vec![F17(1); size];
        assert!(catch_unwind(AssertUnwindSafe(|| transform(&mut values, &F17(1)))).is_err());
    }
}

#[test]
fn generic_reference_forward_supports_noninvertible_lengths() {
    // (Z/4Z)[i]/(i^2 + 1) has a valid fourth root even though four is zero.
    // Requiring an invertible length for the forward transform would exclude it.
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct Z4i(u32, u32);
    impl Twiddle for Z4i {
        const ONE: Self = Self(1, 0);
        fn multiply(&self, rhs: &Self) -> Self {
            Self(
                (self.0 * rhs.0 + 16 - self.1 * rhs.1) % 4,
                (self.0 * rhs.1 + self.1 * rhs.0) % 4,
            )
        }
        fn square(&self) -> Self {
            self.multiply(self)
        }
    }
    impl Butterfly<Self> for Z4i {
        fn scaled(&self, rhs: &Self) -> Self {
            self.multiply(rhs)
        }
        fn add(&self, rhs: &Self) -> Self {
            Self((self.0 + rhs.0) % 4, (self.1 + rhs.1) % 4)
        }
        fn negated(&self) -> Self {
            Self((4 - self.0) % 4, (4 - self.1) % 4)
        }
    }
    fn power(value: Z4i, exponent: usize) -> Z4i {
        (0..exponent).fold(Z4i::ONE, |acc, _| acc.multiply(&value))
    }

    for size in [1, 2, 4] {
        let root = power(Z4i(0, 1), 4 / size);
        assert_eq!(power(root, size), Z4i::ONE);
        for exponent in 1..size {
            assert_ne!(power(root, exponent), Z4i::ONE);
            let character_sum =
                (0..size).fold(Z4i(0, 0), |sum, i| sum.add(&power(root, i * exponent)));
            assert_eq!(character_sum, Z4i(0, 0));
        }
        if size > 1 {
            assert_eq!(power(root, size / 2), Z4i::ONE.negated());
            // A nonzero annihilator proves the length is not a unit.
            assert_eq!(Z4i(size as u32 % 4, 0).multiply(&Z4i(2, 0)), Z4i(0, 0));
        }
        for fixture in 0..=size {
            let input: Vec<_> = (0..size)
                .map(|i| {
                    if fixture == size {
                        Z4i((i as u32 + 1) % 4, (i as u32 + 3) % 4)
                    } else {
                        Z4i(u32::from(i == fixture), 0)
                    }
                })
                .collect();
            let expected: Vec<_> = (0..size)
                .map(|row| {
                    input
                        .iter()
                        .enumerate()
                        .fold(Z4i(0, 0), |sum, (column, value)| {
                            sum.add(&value.multiply(&power(root, row * column)))
                        })
                })
                .collect();
            let mut output = input;
            transform(&mut output, &root);
            assert_eq!(output, expected, "size={size} fixture={fixture}");
        }
    }
}
