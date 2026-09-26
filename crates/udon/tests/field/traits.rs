use super::field_model;

#[test]
fn one_field_bound_supports_roots_transforms_and_accumulation() {
    use num_bigint::BigUint;
    use zakura_udon::{
        fft::{Domain, FftError},
        field::Field,
    };

    fn check<F: Field>() {
        let integer = |value: F| BigUint::from_bytes_le(value.to_bytes().as_ref());
        let modulus = integer(-F::ONE) + 1u8;
        let order = &modulus - 1u8;
        let two_power = BigUint::from(1u8) << F::TWO_ADICITY as usize;
        let odd_order = &order / &two_power;
        assert!(odd_order.bit(0));
        assert_eq!(&order % &two_power, BigUint::from(0u8));
        assert_eq!(
            integer(F::ROOT_OF_UNITY),
            integer(F::MULTIPLICATIVE_GENERATOR).modpow(&odd_order, &modulus)
        );
        assert_eq!(F::ROOT_OF_UNITY * F::ROOT_OF_UNITY_INVERSE, F::ONE);
        assert_eq!(F::TWO_INVERSE.double(), F::ONE);
        assert_eq!(
            integer(F::DELTA),
            integer(F::MULTIPLICATIVE_GENERATOR).modpow(&two_power, &modulus)
        );
        assert_eq!(F::ZETA.pow_u64(3), F::ONE);
        assert_eq!(F::ZETA == F::ONE, &order % 3u8 != BigUint::from(0u8));
        assert_eq!(F::root_of_unity(F::TWO_ADICITY + 1), None);
        assert_eq!(F::root_of_unity_inverse(F::TWO_ADICITY + 1), None);
        assert_eq!(F::domain(F::TWO_ADICITY + 1), Err(FftError::InvalidSize));
        assert_eq!(
            Domain::<F>::from_field(F::TWO_ADICITY + 1),
            Err(FftError::InvalidSize)
        );
        for value in [0, 1, 1u128 << 64, u128::MAX] {
            assert_eq!(
                integer(F::from_u128(value)),
                BigUint::from(value) % &modulus
            );
        }

        assert_eq!(F::reduce(F::Accumulator::default()), F::ZERO);
        let mut accumulator = F::Accumulator::default();
        let mut expected = F::ZERO;
        for i in 0..65 {
            let lhs = -F::from(i * i + 1);
            let rhs = F::from(i * 7 + 3);
            assert_eq!(
                integer(lhs.mul_add(&rhs, &F::ONE)),
                (integer(lhs) * integer(rhs) + 1u8) % &modulus
            );
            F::mul_accumulate(&mut accumulator, &lhs, &rhs);
            expected += lhs * rhs;
        }
        assert_eq!(F::reduce(accumulator), expected);

        for log_size in 0..=F::TWO_ADICITY.min(3) {
            let domain = F::domain(log_size).unwrap();
            let from_field = Domain::<F>::from_field(log_size).unwrap();
            assert_eq!(from_field, domain);
            assert_eq!(from_field.root(), domain.root());
            assert_eq!(from_field.inverse_root(), domain.inverse_root());
            assert_eq!(from_field.size_inverse(), domain.size_inverse());
            assert_eq!(domain.root() * domain.inverse_root(), F::ONE);
            assert_eq!(
                F::from(domain.size() as u64) * domain.size_inverse(),
                F::ONE
            );
            let coefficients: Vec<_> = (0..domain.size())
                .map(|i| F::from((i * i + 7) as u64))
                .collect();
            let mut evaluations = coefficients.clone();
            domain.transform(&mut evaluations);
            for (point, actual) in domain.elements().zip(&evaluations) {
                let expected = coefficients
                    .iter()
                    .enumerate()
                    .map(|(i, coefficient)| *coefficient * point.pow_u64(i as u64))
                    .sum::<F>();
                assert_eq!(*actual, expected);
            }
            domain.inverse_transform(&mut evaluations);
            assert_eq!(evaluations, coefficients);

            let nodes: Vec<_> = domain.elements().collect();
            for point in nodes.iter().copied().chain([F::ZERO]) {
                // Interpolate each basis directly, independently of the
                // evaluator's vanishing-polynomial formula.
                let basis: Vec<_> = nodes
                    .iter()
                    .enumerate()
                    .map(|(i, node)| {
                        nodes
                            .iter()
                            .enumerate()
                            .filter(|(j, _)| *j != i)
                            .map(|(_, other)| (point - other) * (*node - other).invert().unwrap())
                            .product::<F>()
                    })
                    .collect();
                for count in [0, 1, domain.size()] {
                    for capacity in [0, 1, count + 1] {
                        let mut prefix = vec![F::ONE; count];
                        let mut scratch = vec![F::ONE; capacity];
                        assert_eq!(
                            domain.lagrange_evaluations(point, &mut prefix, &mut scratch),
                            nodes.iter().position(|node| *node == point)
                        );
                        assert_eq!(prefix, basis[..count]);
                        assert!(scratch.iter().skip(count).all(|value| *value == F::ONE));
                    }
                }
            }
        }
    }

    check::<zakura_udon::field::FieldAdapter<zakura_udon::field::PallasBase>>();
    check::<zakura_udon::field::FieldAdapter<zakura_udon::field::PallasScalar>>();
    check::<field_model::BlsBase>();
    check::<field_model::BlsScalar>();
    check::<field_model::JubjubScalar>();
    check::<field_model::Small>();
    check::<field_model::SmallScalar>();
}

#[test]
#[should_panic(expected = "lengths must agree")]
fn reference_product_sums_reject_unequal_lengths() {
    use field_model::Small;
    use zakura_udon::field::Field;

    let _ = Small::sum_of_products_slice(&[Small::ONE, Small::ONE], &[Small::ONE]);
}

#[test]
fn reference_product_sums_match_integer_arithmetic() {
    use num_bigint::BigUint;
    use zakura_udon::field::Field;

    fn check<F: Field>() {
        let lhs: Vec<_> = (0..65).map(|i| -F::from(i * i + 1)).collect();
        let rhs: Vec<_> = (0..65).map(|i| F::from(i * 7 + 3)).collect();
        let integer = |value: F| BigUint::from_bytes_le(value.to_bytes().as_ref());
        let modulus = integer(-F::ONE) + 1u8;
        for length in [0, 1, 2, 3, 31, 32, 33, 63, 64, 65] {
            let (lhs, rhs) = (&lhs[..length], &rhs[..length]);
            let expected: BigUint = lhs
                .iter()
                .zip(rhs)
                .map(|(lhs, rhs)| integer(*lhs) * integer(*rhs))
                .sum();
            assert_eq!(
                integer(F::sum_of_products_slice(lhs, rhs)),
                &expected % &modulus
            );
            assert_eq!(
                integer(F::sum_of_product_pairs(
                    lhs.iter().rev().zip(rhs.iter().rev())
                )),
                expected % &modulus
            );
        }
    }

    check::<field_model::BlsBase>();
    check::<field_model::BlsScalar>();
    check::<field_model::JubjubScalar>();
    check::<field_model::Small>();
}

#[test]
fn generic_representations_match_the_field_modulus() {
    use num_bigint::BigUint;
    use zakura_udon::field::Field;

    fn check<F: Field>()
    where
        F::Limbs: AsMut<[u64]>,
    {
        let modulus = BigUint::from_bytes_le(
            &F::MODULUS
                .as_ref()
                .iter()
                .flat_map(|limb| limb.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        assert_eq!(u64::from(F::NUM_BITS), modulus.bits());
        assert_eq!(F::CAPACITY + 1, F::NUM_BITS);
        let integers = [
            BigUint::from(0u8),
            BigUint::from(1u8),
            BigUint::from(u64::MAX),
            (BigUint::from(1u8) << 256usize) + 7u8,
            (BigUint::from(1u8) << 320usize) + 11u8,
            &modulus - 1u8,
        ];
        for integer in integers {
            let integer = integer % &modulus;
            let mut limbs = F::MODULUS;
            limbs.as_mut().fill(0);
            let digits = integer.to_u64_digits();
            limbs.as_mut()[..digits.len()].copy_from_slice(&digits);
            let value = F::from_limbs(limbs).unwrap();
            let bytes = value.to_bytes();
            assert_eq!(BigUint::from_bytes_le(bytes.as_ref()), integer);
            assert_eq!(F::from_bytes(bytes), Some(value));
            let bits = value.to_le_bits();
            assert_eq!(bits.as_ref().len(), bytes.as_ref().len() * 8);
            for (index, bit) in bits.as_ref().iter().enumerate() {
                assert_eq!(*bit, integer.bit(index as u64));
            }
            assert_eq!(value.square().sqrt().unwrap().square(), value.square());
            if !value.is_zero() {
                assert_eq!(value * value.invert().unwrap(), F::ONE);
            }
        }
        assert!(F::from_limbs(F::MODULUS).is_none());
        for input in [[0u8; 64], [0xa5; 64], [0xff; 64]] {
            let mut draws = 0;
            let sample = F::random(|bytes| {
                draws += 1;
                bytes.copy_from_slice(&input);
            });
            assert_eq!(draws, 1);
            assert_eq!(sample, F::from_uniform_bytes(&input));
            assert_eq!(
                BigUint::from_bytes_le(sample.to_bytes().as_ref()),
                BigUint::from_bytes_le(&input) % &modulus
            );
        }
        let mut values = [F::ZERO, F::from(2), F::from(3)];
        F::batch_invert(&mut values, &mut [F::ZERO; 1]);
        assert_eq!(values[0], F::ZERO);
        assert_eq!(values[1] * F::from(2), F::ONE);
        assert_eq!(values[2] * F::from(3), F::ONE);
        let originals = [F::ZERO, F::from(2), F::ZERO, F::from(3), -F::ONE];
        for capacity in [0, 1, 2, 5, 7] {
            let mut values = originals;
            let (first, second) = values.split_at_mut(2);
            let mut scratch = vec![F::ONE; capacity];
            F::batch_invert_groups(&mut [first, &mut [], second], &mut scratch);
            for (original, inverse) in originals.iter().zip(values) {
                assert_eq!(inverse, original.invert().unwrap_or(F::ZERO));
            }
            assert!(
                scratch
                    .iter()
                    .skip(originals.len())
                    .all(|value| *value == F::ONE)
            );
        }
        assert_eq!(
            zakura_udon::polynomial::evaluate_iter(&[F::from(2), F::from(3)], F::from(4)),
            F::from(14)
        );
    }

    check::<zakura_udon::field::FieldAdapter<zakura_udon::field::PallasBase>>();
    check::<zakura_udon::field::FieldAdapter<zakura_udon::field::PallasScalar>>();
    check::<field_model::BlsBase>();
    check::<field_model::BlsScalar>();
    check::<field_model::JubjubScalar>();
    check::<field_model::Small>();
    // A high limb outside a short encoding must be rejected, not discarded.
    assert!(field_model::Small::from_limbs([1 << 32]).is_none());
}
