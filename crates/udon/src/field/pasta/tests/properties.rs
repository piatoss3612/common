//! Randomized loose-representation checks against ordinary integer arithmetic.

use super::*;
use crate::field::Field;
use proptest::{
    prelude::*,
    test_runner::{FileFailurePersistence, TestCaseResult},
};

fn arithmetic<M: PrimeModulus>(a: PastaField<M>, b: PastaField<M>) -> TestCaseResult {
    let p = modulus::<M>();
    let inverse_r = (BigUint::from(1u8) << 256usize).modpow(&(&p - 2u8), &p);
    let x = integer(&a.montgomery_limbs()) * &inverse_r % &p;
    let y = integer(&b.montgomery_limbs()) * &inverse_r % &p;
    for (actual, expected) in [
        (a.add(&b), &x + &y),
        (a.sub(&b), &x + &p - &y),
        (a.mul(&b), &x * &y),
        (a.mul_add(&b, &a), &x * &y + &x),
        (a.square(), &x * &x),
        (a.reduce().mul(&b.reduce()), &x * &y),
    ] {
        prop_assert_eq!(check_value(actual, &expected), Ok(()));
    }
    if x == BigUint::from(0u8) {
        prop_assert!(a.invert().is_none());
    } else {
        prop_assert_eq!(
            check_value(a.invert().unwrap(), &x.modpow(&(&p - 2u8), &p)),
            Ok(())
        );
    }
    // The square root may have either sign; check its integer square.
    let root = a.square().sqrt().unwrap();
    let root_integer = integer(&root.montgomery_limbs()) * &inverse_r % &p;
    prop_assert_eq!(&root_integer * &root_integer % &p, &x * &x % &p);
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        failure_persistence: Some(std::boxed::Box::new(FileFailurePersistence::WithSource("regressions"))),
        .. ProptestConfig::default()
    })]

    #[test]
    fn fp_arithmetic(a in arbitrary_field::<PallasBase>(), b in arbitrary_field::<PallasBase>()) {
        arithmetic(a, b)?;
    }

    #[test]
    fn fq_arithmetic(a in arbitrary_field::<PallasScalar>(), b in arbitrary_field::<PallasScalar>()) {
        arithmetic(a, b)?;
    }
}

#[test]
fn integer_oracle_rejects_a_corrupted_result_at_the_modulus() {
    fn check<M: PrimeModulus>() {
        let p = modulus::<M>();
        let inverse_r = (BigUint::from(1u8) << 256usize).modpow(&(&p - 2u8), &p);
        for stored in [&p - 1u8, p.clone(), &p + 1u8] {
            let a = PastaField::<M>::from_montgomery_limbs(limbs(&stored));
            let x = &stored * &inverse_r % &p;
            let expected = &x * &x;
            let actual = a.square();
            assert_eq!(check_value(actual, &expected), Ok(()));
            let sentinel = if stored == p {
                actual.add(&PastaField::<M>::ONE)
            } else {
                actual
            };
            assert_eq!(check_value(sentinel, &expected).is_err(), stored == p);
        }
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}
