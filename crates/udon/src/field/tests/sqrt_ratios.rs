use super::*;

fn check_result<M: PrimeModulus>(
    result: (bool, PastaField<M, Reduced>),
    ratio: &BigUint,
    nonsquare: &BigUint,
    p: &BigUint,
) {
    let (is_square, root) = result;
    // Euler's criterion and the root equation use ordinary integer arithmetic,
    // independently of the field's exponent preparation and root correction.
    let expected_square = *ratio == BigUint::from(0u8)
        || ratio.modpow(&((p - 1u8) >> 1usize), p) == BigUint::from(1u8);
    assert_eq!(is_square, expected_square);
    let expected = if is_square {
        ratio.clone()
    } else {
        ratio * nonsquare % p
    };
    let root_integer = BigUint::from_bytes_le(&root.to_bytes());
    assert_value(root, &root_integer);
    assert_eq!(&root_integer * &root_integer % p, expected);
}

fn check_pair<M: PrimeModulus>(
    numerator: PastaField<M, Reduced>,
    denominator: PastaField<M, Reduced>,
    n: &BigUint,
    d: &BigUint,
    nonsquare: &BigUint,
    p: &BigUint,
) {
    let result = numerator.sqrt_ratio(&denominator);
    if *d == BigUint::from(0u8) {
        assert_eq!(result, (*n == BigUint::from(0u8), PastaField::ZERO));
    } else {
        let ratio = n * d.modpow(&(p - 2u8), p) % p;
        check_result(result, &ratio, nonsquare, p);
        let root = BigUint::from_bytes_le(&result.1.to_bytes());
        let expected = if result.0 {
            n.clone()
        } else {
            n * nonsquare % p
        };
        assert_eq!(&root * &root * d % p, expected);
    }
}

fn check<M: PrimeModulus>() {
    let p = modulus::<M>();
    let nonsquare = BigUint::from(5u8).modpow(&((&p - 1u8) >> 32usize), &p);
    assert_value(PastaField::<M, Reduced>::SQRT_NONSQUARE, &nonsquare);
    assert_eq!(nonsquare.modpow(&((&p - 1u8) >> 1usize), &p), &p - 1u8);
    let zero = BigUint::from(0u8);
    let values = samples::<M>(64);
    for (index, (value, n)) in values.iter().enumerate() {
        let value = value.reduce();
        check_result(value.sqrt_alt(), n, &nonsquare, &p);
        assert_eq!(value.sqrt_alt().0, value.sqrt().is_some());
        for (other, d) in [
            &values[(index + 17) % values.len()],
            &values[(index * 31 + 3) % values.len()],
            &values[index],
            &values[0],
            &values[1],
        ] {
            check_pair(value, other.reduce(), n, d, &nonsquare, &p);
        }
        check_pair(PastaField::ZERO, value, &zero, n, &nonsquare, &p);
    }
    // Cross canonical and raw Montgomery boundaries, including a loose zero
    // represented by p before reducing at the public root boundary.
    for (a, n) in &values[..8] {
        for (b, d) in &values[72..79] {
            check_pair(a.reduce(), b.reduce(), n, d, &nonsquare, &p);
        }
    }
}

#[test]
fn alternate_roots_and_ratios_match_integer_arithmetic_without_inversions() {
    assert_eq!(
        crate::field::count_inversions(|| {
            check::<PallasBase>();
            check::<PallasScalar>();
        }),
        0
    );
}

fn subgroup_boundaries<M: PrimeModulus>() {
    let p = modulus::<M>();
    let r = BigUint::from(5u8).modpow(&((&p - 1u8) >> 32usize), &p);
    // Vary every byte and carry boundary of the inverse exponent. The ratio
    // uses a full-width denominator so its preparation also affects correction.
    let d = &p - 17u8;
    let denominator = field::<M>(&d).reduce();
    let exponents = (0..4).flat_map(|byte| (0..256u64).map(move |digit| digit << (8 * byte)));
    for exponent in exponents.chain([
        255, 256, 257, 65535, 65536, 65537, 0xffffff, 0x1000000, 0xfffffffe, 0xffffffff,
    ]) {
        let a = r.modpow(&BigUint::from(exponent), &p);
        check_result(field::<M>(&a).reduce().sqrt_alt(), &a, &r, &p);
        let numerator = field::<M>(&(&a * &d % &p)).reduce();
        check_result(numerator.sqrt_ratio(&denominator), &a, &r, &p);
    }
}

#[test]
fn alternate_roots_and_ratios_cover_subgroup_carries() {
    subgroup_boundaries::<PallasBase>();
    subgroup_boundaries::<PallasScalar>();
}
