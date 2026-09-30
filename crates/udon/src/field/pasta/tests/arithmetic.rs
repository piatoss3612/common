use super::*;

fn check_pair<M: PrimeModulus, S: ReductionState, T: ReductionState>(
    a: &PastaField<M, S>,
    b: &PastaField<M, T>,
    x: &BigUint,
    y: &BigUint,
    p: &BigUint,
) {
    assert_value(a.add(b), &(x + y));
    assert_value(a.sub(b), &(x + p - y));
    assert_value(a.mul(b), &(x * y));
    assert_value(a.mul_add(b, a), &(x * y + x));
    assert_value(a.mul_sub(b, a), &(x * y + p - x));
}

fn check_arithmetic<M: PrimeModulus>() {
    let p = modulus::<M>();
    let values = samples::<M>(64);
    assert_value(PastaField::<M>::default(), &BigUint::from(0u8));
    assert_eq!(
        (PastaField::<M>::ZERO).reduce(),
        (PastaField::<_>::ZERO).reduce()
    );
    assert_eq!(
        (PastaField::<M>::ONE).reduce(),
        (PastaField::<_>::ONE).reduce()
    );
    for (a, x) in &values {
        assert_eq!(a.is_zero(), x == &BigUint::from(0u8));
        assert_eq!(a.is_one(), x == &BigUint::from(1u8));
        assert_eq!(a.reduce().is_one(), x == &BigUint::from(1u8));
        assert_value(a.neg(), &(&p - x));
        assert_value(a.square(), &(x * x));
        assert_value(a.double(), &(x * 2u8));
        assert_value(a.half(), &(x * ((&p + 1u8) >> 1usize)));
        assert_value(a.triple(), &(x * 3u8));
        assert_value(a.mul_by_4(), &(x * 4u8));
        assert_value(a.mul_by_8(), &(x * 8u8));
        assert_value(a.reduce().neg(), &(&p - x));
        if !a.is_zero() {
            let negated = a.reduce().negate_nonzero();
            assert_value(negated, &(&p - x));
            assert!(integer(&negated.montgomery_limbs()) < p);
        }
        assert_value(a.reduce().square(), &(x * x));
        assert_value(a.reduce().double(), &(x * 2u8));
        assert_value(a.reduce().half(), &(x * ((&p + 1u8) >> 1usize)));
        assert_eq!(std::format!("{a:?}"), std::format!("0x{x:064x}"));
        for (b, y) in &values {
            check_pair(a, b, x, y, &p);
            check_pair(&a.reduce(), b, x, y, &p);
            check_pair(a, &b.reduce(), x, y, &p);
            check_pair(&a.reduce(), &b.reduce(), x, y, &p);
            let difference = a.reduce().sub_reduced(&b.reduce());
            assert_value(difference, &(x + &p - y));
            assert!(integer(&difference.montgomery_limbs()) < p);
            assert_eq!(a.reduce().cmp(&b.reduce()), x.cmp(y));
            assert_eq!(a.reduce().partial_cmp(&b.reduce()), Some(x.cmp(y)));
            assert_eq!(a.reduce() == b.reduce(), x == y);
        }
        for exponent in [0, 1, 2, 63, 64, 65, 1 << 63, u64::MAX] {
            assert_value(a.pow_u64(exponent), &x.modpow(&BigUint::from(exponent), &p));
        }
    }
}

#[test]
fn one_recognizes_both_montgomery_representatives() {
    fn check<M: PrimeModulus>() {
        const { assert!(PastaField::<M>::ONE.is_one()) };
        const { assert!(PastaField::<M, Reduced>::ONE.is_one()) };
        let p = modulus::<M>();
        let one = (BigUint::from(1u8) << 256usize) % &p;
        for stored in [one.clone(), one + p] {
            for candidate in [&stored - 1u8, stored.clone(), &stored + 1u8] {
                let value = PastaField::<M>::from_montgomery_limbs(limbs(&candidate));
                assert_eq!(value.is_one(), candidate == stored);
                assert_eq!(value.reduce().is_one(), candidate == stored);
            }
        }
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}

fn halving_stored_endpoints<M: PrimeModulus>() {
    let p = modulus::<M>();
    for stored in [
        BigUint::from(0u8),
        BigUint::from(1u8),
        BigUint::from(2u8),
        &p - 2u8,
        &p - 1u8,
        p.clone(),
        &p + 1u8,
        &p * 2u8 - 1u8,
    ] {
        let value = PastaField::<M>::from_montgomery_limbs(limbs(&stored));
        assert_eq!((value.half().double()).reduce(), (value).reduce());
        let expected = if stored.bit(0) {
            (&stored + &p) >> 1usize
        } else {
            &stored >> 1usize
        };
        assert_eq!(integer(&value.half().montgomery_limbs()), expected);
    }
}

#[test]
fn halving_handles_stored_residue_endpoints() {
    halving_stored_endpoints::<PallasBase>();
    halving_stored_endpoints::<PallasScalar>();
}

#[test]
fn arithmetic_and_ordering_match_integer_operations() {
    check_arithmetic::<PallasBase>();
    check_arithmetic::<PallasScalar>();
}

#[test]
fn iterator_products_match_integer_products() {
    fn check<M: PrimeModulus>() {
        let p = modulus::<M>();
        let samples = samples::<M>(16);
        // Rotations cover zero in each position, including the loose zero
        // stored as the modulus, without making every product vanish.
        for offset in 0..samples.len() {
            for length in [0, 1, 2, 3, 8, 17, 65] {
                let factors: Vec<_> = samples.iter().cycle().skip(offset).take(length).collect();
                let expected = factors
                    .iter()
                    .fold(BigUint::from(1u8), |product, (_, integer)| {
                        product * integer % &p
                    });
                let loose: Vec<_> = factors.iter().map(|(value, _)| *value).collect();
                let reduced: Vec<_> = loose.iter().map(|value| value.reduce()).collect();
                let borrowed = loose.iter().fold(PastaField::ONE, |a, b| a.mul(b));
                let owned = loose
                    .iter()
                    .copied()
                    .fold(PastaField::ONE, |a, b| a.mul(&b));
                #[cfg(feature = "traits")]
                {
                    use crate::field::FieldAdapter;
                    let wrapped = FieldAdapter::from_slice(&loose);
                    assert_value(
                        wrapped.iter().product::<FieldAdapter<M>>().into_inner(),
                        &expected,
                    );
                    assert_value(
                        wrapped
                            .iter()
                            .copied()
                            .product::<FieldAdapter<M>>()
                            .into_inner(),
                        &expected,
                    );
                }
                assert_value(borrowed, &expected);
                assert_value(owned, &expected);
                assert_value(
                    reduced.iter().fold(PastaField::ONE, |a, b| a.mul(b)),
                    &expected,
                );
                assert_value(
                    reduced.into_iter().fold(PastaField::ONE, |a, b| a.mul(&b)),
                    &expected,
                );
            }
        }
    }

    check::<PallasBase>();
    check::<PallasScalar>();
}

fn check_square_roots<M: PrimeModulus>() {
    let p = modulus::<M>();
    let euler_exponent = (&p - 1u8) >> 1usize;
    let chain_exponent = (((&p - 1u8) >> 32usize) - 1u8) >> 1usize;
    let mut nonsquares = 0;
    for (value, x) in samples::<M>(256) {
        assert_value(M::pow_sqrt_exponent(&value), &x.modpow(&chain_exponent, &p));
        // Euler's criterion establishes existence independently of Tonelli-Shanks.
        let has_root =
            x == BigUint::from(0u8) || x.modpow(&euler_exponent, &p) == BigUint::from(1u8);
        let root = value.reduce().sqrt();
        assert_eq!(root.is_some(), has_root);
        if let Some(root) = root {
            let y = BigUint::from_bytes_le(&root.to_bytes());
            assert_value(root, &y);
            assert_eq!(&y * &y % &p, x);
        } else {
            nonsquares += 1;
        }
        let square = &x * &x % &p;
        let root = field::<M>(&square).reduce().sqrt().unwrap();
        let y = BigUint::from_bytes_le(&root.to_bytes());
        assert_value(root, &y);
        assert_eq!(&y * &y % &p, square);
    }
    assert!(nonsquares > 0);
}

#[test]
fn square_roots_match_eulers_criterion() {
    check_square_roots::<PallasBase>();
    check_square_roots::<PallasScalar>();
}
