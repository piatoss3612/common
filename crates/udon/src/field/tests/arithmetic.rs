use super::*;

fn check_arithmetic<M: PrimeModulus>() {
    let p = modulus::<M>();
    let values = samples::<M>(64);
    assert_value(PastaField::<M>::default(), &BigUint::from(0u8));
    assert_eq!(PastaField::<M>::ZERO, PastaField::ZERO);
    assert_eq!(PastaField::<M>::ONE, PastaField::ONE);
    for (a, x) in &values {
        assert_eq!(a.is_zero(), x == &BigUint::from(0u8));
        assert_value(a.neg(), &(&p - x));
        assert_value(a.square(), &(x * x));
        assert_value(a.double(), &(x * 2u8));
        assert_value(a.half(), &(x * ((&p + 1u8) >> 1usize)));
        assert_value(a.triple(), &(x * 3u8));
        assert_value(a.mul_by_4(), &(x * 4u8));
        assert_value(a.mul_by_8(), &(x * 8u8));
        assert_eq!(std::format!("{a:?}"), std::format!("0x{x:064x}"));
        for (b, y) in &values {
            assert_value(a.add(b), &(x + y));
            assert_value(a.sub(b), &(x + &p - y));
            assert_value(a.mul(b), &(x * y));
            assert_value(a.mul_add(b, a), &(x * y + x));
            assert_value(a.mul_sub(b, a), &(x * y + &p - x));
            assert_eq!(a.cmp(b), x.cmp(y));
            assert_eq!(a.partial_cmp(b), Some(x.cmp(y)));
            assert_eq!(a == b, x == y);
        }
        for exponent in [0, 1, 2, 63, 64, 65, 1 << 63, u64::MAX] {
            assert_value(a.pow_u64(exponent), &x.modpow(&BigUint::from(exponent), &p));
        }
    }
}

fn halving_stored_endpoints<M: PrimeModulus>() {
    let p = modulus::<M>();
    for stored in [
        BigUint::from(0u8),
        BigUint::from(1u8),
        BigUint::from(2u8),
        &p - 2u8,
        &p - 1u8,
    ] {
        let value = PastaField::<M>::from_montgomery_limbs(limbs(&stored));
        assert_eq!(value.half().double(), value);
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
        let root = value.sqrt();
        assert_eq!(root.is_some(), has_root);
        if let Some(root) = root {
            let y = BigUint::from_bytes_le(&root.to_bytes());
            assert_value(root, &y);
            assert_eq!(&y * &y % &p, x);
        } else {
            nonsquares += 1;
        }
        let square = &x * &x % &p;
        let root = field::<M>(&square).sqrt().unwrap();
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
