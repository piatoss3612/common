use super::*;
use crate::field::pasta::test_support::*;

fn check_inverses<M: PrimeModulus>() {
    let p = modulus::<M>();
    let exponent = &p - 2u8;
    let mut values = samples::<M>(4096);
    for small in 1u64..64 {
        values.push((PastaField::from_u64(small), BigUint::from(small)));
        values.push((PastaField::<_>::from_u64(small).neg(), &p - small));
    }
    for (value, x) in values {
        if x == BigUint::from(0u8) {
            assert_eq!((value.invert()).map(|value| value.reduce()), None);
        } else {
            let inverse = value.invert().unwrap();
            assert_value(inverse, &x.modpow(&exponent, &p));
            assert_value(value.mul(&inverse), &BigUint::from(1u8));
        }
    }
}

#[test]
fn safegcd_inversion_matches_independent_fermat_exponentiation() {
    check_inverses::<PallasBase>();
    check_inverses::<PallasScalar>();
}
