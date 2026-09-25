use super::field_model;

#[test]
fn generic_representations_support_bls_and_jubjub_fields() {
    use num_bigint::BigUint;
    use zakura_udon::field::{PrimeField, low_u64, random};

    fn check<F: PrimeField>()
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
            assert_eq!(low_u64(&value), digits.first().copied().unwrap_or(0));
            assert_eq!(value.square().sqrt().unwrap().square(), value.square());
            if !value.is_zero() {
                assert_eq!(value * value.invert().unwrap(), F::ONE);
            }
        }
        assert!(F::from_limbs(F::MODULUS).is_none());
        let mut draws = 0;
        let sample = random::<F>(|bytes| {
            draws += 1;
            bytes.fill(0xa5);
        });
        assert_eq!(draws, 1);
        assert_eq!(
            BigUint::from_bytes_le(sample.to_bytes().as_ref()),
            BigUint::from_bytes_le(&[0xa5; 64]) % &modulus
        );
        let mut values = [F::ZERO, F::from(2), F::from(3)];
        F::batch_invert(&mut values, &mut [F::ZERO; 1]);
        assert_eq!(values[0], F::ZERO);
        assert_eq!(values[1] * F::from(2), F::ONE);
        assert_eq!(values[2] * F::from(3), F::ONE);
        assert_eq!(
            zakura_udon::poly::evaluate(&[F::from(2), F::from(3)], F::from(4)),
            F::from(14)
        );
    }

    check::<field_model::BlsBase>();
    check::<field_model::BlsScalar>();
    check::<field_model::JubjubScalar>();
    check::<field_model::Small>();
    // A high limb outside a short encoding must be rejected, not discarded.
    assert!(field_model::Small::from_limbs([1 << 32]).is_none());
}
