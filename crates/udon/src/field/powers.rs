use super::{PastaField, PrimeModulus};

pub(crate) fn fill_powers<M: PrimeModulus>(
    first: PastaField<M>,
    step: PastaField<M>,
    values: &mut [PastaField<M>],
) {
    if values.len() < 8 {
        if let Some((head, tail)) = values.split_first_mut() {
            let mut power = first;
            *head = power;
            for value in tail {
                power = power.mul(&step);
                *value = power;
            }
        }
        return;
    }

    // Even and odd powers advance independently by step^2, so one product
    // need not wait for the preceding entry's multiplication to finish.
    let stride = step.square();
    let mut even = first;
    let mut odd = first.mul(&step);
    let (head, tail) = values.split_at_mut(2);
    head.copy_from_slice(&[even, odd]);
    let mut pairs = tail.chunks_exact_mut(2);
    for pair in pairs.by_ref() {
        even = even.mul(&stride);
        odd = odd.mul(&stride);
        pair.copy_from_slice(&[even, odd]);
    }
    if let [last] = pairs.into_remainder() {
        *last = even.mul(&stride);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{PallasBase, PallasScalar, word::compare_limbs};
    use crate::test_support::field_samples;
    use std::vec;

    fn power_oracle<M: PrimeModulus>() {
        let mut largest = M::TWICE_MODULUS;
        largest[0] -= 1;
        let samples = [
            PastaField::ZERO,
            PastaField::from_montgomery_limbs(M::MODULUS),
            PastaField::from_montgomery_limbs(largest),
            PastaField::ONE,
            PastaField::<M>::ONE.neg(),
            PastaField::from_u64(7),
            field_samples::<M>().next().unwrap(),
        ];
        for first in samples {
            for step in samples {
                for len in (0..=35).chain([63, 64, 65, 255, 256, 257]) {
                    let sentinel = PastaField::from_u64(19);
                    let mut values = vec![sentinel; len + 2];
                    fill_powers(first, step, &mut values[1..len + 1]);
                    for (i, value) in values[1..len + 1].iter().enumerate() {
                        assert_eq!(value.reduce(), first.mul(&step.pow_u64(i as u64)).reduce());
                        // Every entry keeps the loose bound and encodes canonically.
                        assert!(
                            compare_limbs(&value.montgomery_limbs(), &M::TWICE_MODULUS).is_lt()
                        );
                        assert_eq!(
                            PastaField::<M>::from_bytes(value.to_bytes())
                                .map(|value| value.reduce()),
                            Some(value.reduce())
                        );
                    }
                    assert_eq!(values[0].montgomery_limbs(), sentinel.montgomery_limbs());
                    assert_eq!(
                        values[len + 1].montgomery_limbs(),
                        sentinel.montgomery_limbs()
                    );
                }
            }
        }
    }

    #[test]
    fn power_recurrences_match_independent_exponentiation() {
        power_oracle::<PallasBase>();
        power_oracle::<PallasScalar>();
    }
}
