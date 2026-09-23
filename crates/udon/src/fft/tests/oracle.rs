//! Direct polynomial evaluation and reference transforms.

use super::*;

pub(super) fn evaluate<M: PrimeModulus>(
    coefficients: &[PastaField<M>],
    point: PastaField<M>,
) -> PastaField<M> {
    coefficients
        .iter()
        .rev()
        .fold(PastaField::ZERO, |acc, coefficient| {
            acc.mul(&point).add(coefficient)
        })
}

pub(super) fn direct<M: PrimeModulus>(
    coefficients: &[PastaField<M>],
    domain: CosetDomain<M>,
) -> Vec<PastaField<M>> {
    let mut point = domain.shift();
    (0..domain.size())
        .map(|_| {
            let value = evaluate(coefficients, point);
            point = point.mul(&domain.domain().root());
            value
        })
        .collect()
}

pub(super) fn check_forward<M: PrimeModulus>(
    coefficients: &[PastaField<M>],
    domain: CosetDomain<M>,
    actual: &[PastaField<M>],
) -> Result<(), &'static str> {
    if actual == direct(coefficients, domain) {
        Ok(())
    } else {
        Err("FFT differs from direct polynomial evaluation")
    }
}

pub(super) fn reference_coset<M: PrimeModulus>(
    coefficients: &[PastaField<M>],
    domain: CosetDomain<M>,
) -> Vec<PastaField<M>> {
    let mut values = vec![PastaField::ZERO; domain.size()];
    let mut scale = PastaField::ONE;
    for (out, coefficient) in values.iter_mut().zip(coefficients) {
        *out = coefficient.mul(&scale);
        scale = scale.mul(&domain.shift());
    }
    reference::transform(&mut values, &domain.domain().root());
    values
}

#[test]
fn generic_reference_supports_a_foreign_field() {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct F17(u32);
    impl reference::Twiddle for F17 {
        const ONE: Self = Self(1);
        fn multiply(&self, rhs: &Self) -> Self {
            Self(self.0 * rhs.0 % 17)
        }
        fn square(&self) -> Self {
            self.multiply(self)
        }
    }
    impl reference::Butterfly<Self> for F17 {
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
        reference::transform(&mut output, &root);
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
        reference::inverse_transform(&mut output, &inverse_root, &inverse_size);
        assert_eq!(output, input);
    }
    for size in [0, 3] {
        let mut values = vec![F17(1); size];
        assert!(
            catch_unwind(AssertUnwindSafe(|| reference::transform(
                &mut values,
                &F17(1)
            )))
            .is_err()
        );
    }
}

#[test]
fn generic_reference_forward_supports_noninvertible_lengths() {
    use reference::{Butterfly, Twiddle};

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
            reference::transform(&mut output, &root);
            assert_eq!(output, expected, "size={size} fixture={fixture}");
        }
    }
}

pub(super) fn ordered<M: PrimeModulus>(
    values: &[PastaField<M>],
    order: ElementOrder,
) -> Vec<PastaField<M>> {
    (0..values.len())
        .map(|i| {
            values[if order == ElementOrder::Natural {
                i
            } else {
                bit_reverse(i, values.len().ilog2())
            }]
        })
        .collect()
}

pub(super) fn inverse_direct<M: PrimeModulus>(
    evaluations: &[PastaField<M>],
    domain: CosetDomain<M>,
    normalized: bool,
) -> Vec<PastaField<M>> {
    (0..domain.size())
        .map(|degree| {
            let step = domain.domain().inverse_root().pow_u64(degree as u64);
            let mut power = PastaField::ONE;
            let mut sum = PastaField::ZERO;
            for value in evaluations {
                sum = sum.add(&value.mul(&power));
                power = power.mul(&step);
            }
            sum = sum.mul(&domain.inverse_shift().pow_u64(degree as u64));
            if normalized {
                sum.mul(&domain.domain().size_inverse())
            } else {
                sum
            }
        })
        .collect()
}
