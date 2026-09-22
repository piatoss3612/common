use super::*;
use crate::field::tests::*;

fn check_products<M: PrimeModulus>() {
    let p = modulus::<M>();
    let mut values = samples::<M>(64);
    let inverse_r = (BigUint::from(1u8) << 256usize).modinv(&p).unwrap();
    for raw in [BigUint::from(1u8), &p - 1u8, &p - 2u8] {
        values.push((
            PastaField::from_montgomery_limbs(limbs(&raw)),
            &raw * &inverse_r % &p,
        ));
    }
    for (index, (a, x)) in values.iter().enumerate() {
        let (b, y) = &values[(index * 7 + 3) % values.len()];
        for (other, (c, z)) in values.iter().enumerate() {
            let (d, w) = &values[(other * 13 + 5) % values.len()];
            let product = z * w % &p;
            assert_value(a.mul_sub_product(b, c, d), &(x * y + &p - &product));
            assert_value(
                a.mul_sub_double_product(b, c, d),
                &(x * y + &p * 2u8 - &product * 2u8),
            );
        }
        assert_value(a.mul_sub_product(a, a, a), &BigUint::from(0u8));
        assert_value(a.mul_sub_double_product(a, a, a), &(&p - x * x % &p));
        assert_value(a.mul_sub_product(a, &PastaField::ZERO, a), &(x * x));
    }
    for length in [
        0, 1, 2, 3, 4, 7, 11, 12, 23, 31, 32, 33, 63, 64, 65, 66, 67, 68, 69, 257, 4096,
    ] {
        let mut expected = BigUint::from(0u8);
        let mut mixed_expected = BigUint::from(0u8);
        let mut mixed = ProductSum::<M>::new();
        let mut lhs = Vec::new();
        let mut rhs = Vec::new();
        for index in 0..length {
            let (a, x) = &values[index % values.len()];
            let (b, y) = &values[(index * 13 + 5) % values.len()];
            lhs.push(*a);
            rhs.push(*b);
            expected += x * y;
            if index % 3 == 0 {
                mixed.add_term(a);
                mixed_expected += x;
            } else {
                mixed.add_product(a, b);
                mixed_expected += x * y;
            }
        }
        assert_value(
            PastaField::<M>::sum_of_products_slice(&lhs, &rhs),
            &expected,
        );
        assert_value(
            PastaField::<M>::sum_of_product_pairs(lhs.iter().zip(&rhs)),
            &expected,
        );
        assert_value(mixed.finish(), &mixed_expected);
        let maximal = PastaField::<M>::from_montgomery_limbs(limbs(&(&p - 1u8)));
        let maximal_integer = BigUint::from_bytes_le(&maximal.to_bytes());
        let repeated = vec![maximal; length];
        assert_value(
            PastaField::<M>::sum_of_products_slice(&repeated, &repeated),
            &(&maximal_integer * &maximal_integer * length),
        );
    }
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = PastaField::<M>::sum_of_products_slice(&[PastaField::ONE], &[]);
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = PastaField::<M>::sum_of_products_slice(&[], &[PastaField::ONE]);
        }))
        .is_err()
    );

    // Literal lengths instantiate each array API specialization.
    macro_rules! check_arrays {
        ($($length:literal),* $(,)?) => {$(
            let lhs = core::array::from_fn::<_, $length, _>(|i| values[i % values.len()].0);
            let rhs = core::array::from_fn::<_, $length, _>(|i| values[(i * 7 + 1) % values.len()].0);
            let expected = (0..$length).fold(BigUint::from(0u8), |sum, i| sum + &values[i % values.len()].1 * &values[(i * 7 + 1) % values.len()].1);
            assert_value(PastaField::<M>::sum_of_products(&lhs, &rhs), &expected);
        )*};
    }
    check_arrays!(0, 1, 2, 3, 4, 11, 12, 23, 31, 32, 33, 64, 65);
}

#[test]
fn product_differences_and_sums_match_integer_arithmetic() {
    check_products::<PallasBase>();
    check_products::<PallasScalar>();
}

fn check_overflow<M: PrimeModulus>() {
    let p = modulus::<M>();
    let radix = BigUint::from(1u8) << 256usize;
    let inverse_r = radix.modpow(&(&p - 2u8), &p);
    // The accumulator stores products scaled by R²; finish returns a field
    // element scaled by R, which assert_value compares to an ordinary integer.
    let inverse_r_squared = &inverse_r * &inverse_r % &p;
    let maximum = (BigUint::from(1u8) << 576usize) - 1u8;
    let full = || ProductSum::<M> {
        wide: [u64::MAX; 8],
        carry: u64::MAX,
        marker: PhantomData,
    };
    let term = PastaField::<M>::from_montgomery_limbs(limbs(&(&p - 1u8)));
    let stored = integer(&term.montgomery_limbs());
    let mut sum = full();
    sum.add_product(&term, &term);
    assert_value(
        sum.finish(),
        &((&maximum + &stored * &stored) * &inverse_r_squared),
    );
    let mut sum = full();
    sum.add_term(&term);
    assert_value(
        sum.finish(),
        &((&maximum + &stored * &radix) * &inverse_r_squared),
    );
    let mut sum = full();
    sum.merge(&full());
    sum.add_product(&term, &term);
    sum.add_term(&term);
    assert_value(
        sum.finish(),
        &((&maximum * 2u8 + &stored * &stored + &stored * &radix) * &inverse_r_squared),
    );
    assert_value(full().finish(), &(&maximum * &inverse_r_squared));
    assert_value(ProductSum::<M>::default().finish(), &BigUint::from(0u8));
}

#[test]
fn all_accumulator_mutations_preserve_overflow_bits() {
    check_overflow::<PallasBase>();
    check_overflow::<PallasScalar>();
}
