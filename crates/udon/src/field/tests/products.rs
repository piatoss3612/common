use crate::field::pasta::count_slice_sums;
use crate::field::pasta::test_support::{assert_value, limbs, modulus, samples};
use crate::field::{Field, Fp, PallasBase, PallasScalar, PastaField, PrimeModulus, dot, dot_iter};
use num_bigint::BigUint;
use std::{vec, vec::Vec};

#[test]
#[should_panic(expected = "lengths must agree")]
fn dot_rejects_unequal_lengths() {
    let _ = dot(&[<Fp>::ONE, <Fp>::ONE], &[<Fp>::ONE]);
}

#[test]
#[should_panic(expected = "equal length")]
fn dot_iter_rejects_unequal_lengths() {
    let _ = dot_iter([<Fp>::ONE, <Fp>::ONE].iter().rev(), [<Fp>::ONE].iter());
}

fn check_dot<M: PrimeModulus>() {
    let p = modulus::<M>();
    let mut values = samples::<M>(64);
    let inverse_r = (BigUint::from(1u8) << 256usize).modinv(&p).unwrap();
    for raw in [BigUint::from(1u8), &p - 1u8, &p - 2u8] {
        values.push((
            PastaField::from_montgomery_limbs(limbs(&raw)),
            &raw * &inverse_r % &p,
        ));
    }
    for length in [
        0, 1, 2, 3, 4, 7, 11, 12, 23, 31, 32, 33, 63, 64, 65, 66, 67, 68, 69, 257, 4096,
    ] {
        let mut expected = BigUint::from(0u8);
        let mut strided_expected = BigUint::from(0u8);
        let mut lhs = Vec::new();
        let mut rhs = Vec::new();
        for index in 0..length {
            let (a, x) = &values[index % values.len()];
            let (b, y) = &values[(index * 13 + 5) % values.len()];
            lhs.push(*a);
            rhs.push(*b);
            expected += x * y;
            if index % 2 == 0 {
                strided_expected += x * y;
            }
        }
        for product in [
            dot::<PastaField<M>>,
            <PastaField<M> as Field>::sum_of_products_slice,
        ] {
            assert_eq!(
                count_slice_sums(|| assert_value(product(&lhs, &rhs), &expected)),
                1
            );
        }
        assert_eq!(
            count_slice_sums(|| {
                assert_value(dot_iter(lhs.iter().rev(), rhs.iter().rev()), &expected);
                assert_value(
                    dot_iter(lhs.iter().step_by(2), rhs.iter().step_by(2)),
                    &strided_expected,
                );
            }),
            // The native iterator path uses slice kernels for two or three
            // pairs, including the shortened sequence from step_by(2).
            usize::from((2..=3).contains(&length))
                + usize::from((2..=3).contains(&length.div_ceil(2)))
        );
        let maximal = PastaField::<M>::from_montgomery_limbs(limbs(&(&p * 2u8 - 1u8)));
        let maximal_integer = BigUint::from_bytes_le(&maximal.to_bytes());
        let repeated = vec![maximal; length];
        assert_value(
            dot(&repeated, &repeated),
            &(&maximal_integer * &maximal_integer * length),
        );
    }
}

#[test]
fn dot_matches_integer_products() {
    check_dot::<PallasBase>();
    check_dot::<PallasScalar>();
}
