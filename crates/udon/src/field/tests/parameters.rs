use super::*;
use crate::field::parameters::TWO_ADICITY;
use bento::const_arithmetic::u256;

// Runtime square roots use a generated multiplication schedule instead of the
// ordinary exponent `(t - 1) / 2`, where `p - 1 = t * 2^32`. Derive that
// exponent here so the schedule and the fixed vectors can be checked against it.
pub(super) const PALLAS_BASE_SQRT_EXPONENT: [u64; 4] =
    u256::tonelli_shanks_exponent!(&PallasBase::MODULUS, TWO_ADICITY);
pub(super) const PALLAS_SCALAR_SQRT_EXPONENT: [u64; 4] =
    u256::tonelli_shanks_exponent!(&PallasScalar::MODULUS, TWO_ADICITY);

fn check_parameters<M: PrimeModulus>(zeta_power: u32, sqrt_exponent: [u64; 4]) {
    let p = modulus::<M>();
    let one = BigUint::from(1u8);
    let five = BigUint::from(5u8);
    assert_eq!(integer(&M::R), (&one << 256usize) % &p);
    assert_eq!(integer(&M::R2), (&one << 512usize) % &p);
    assert_eq!(integer(&M::R3), (&one << 768usize) % &p);
    assert!(integer(&M::R2) + integer(&M::R3) < p);
    let maximal_product = (&p - 1u8).pow(2);
    assert!(&maximal_product * 3u8 < &p << 256usize);
    assert!(&maximal_product * 4u8 >= &p << 256usize);
    assert_eq!(integer(&M::B448), (&one << 448usize) % &p);
    assert_eq!(M::MODULUS[0].wrapping_mul(M::MONTGOMERY_INV), u64::MAX);
    assert_eq!(signed62(&M::MODULUS_SIGNED62), BigInt::from(p.clone()));
    let odd_cofactor = (&p - 1u8) >> 32usize;
    assert!(odd_cofactor.bit(0));
    assert_eq!(integer(&sqrt_exponent), (&odd_cofactor - 1u8) >> 1usize);
    let inverse_two = (&p + 1u8) >> 1usize;
    assert_value(PastaField::<M>::TWO_INVERSE, &inverse_two);
    for exponent in (0..=33).chain([40, 256, u32::MAX]) {
        assert_value(
            PastaField::<M>::power_of_two_inverse(exponent),
            &inverse_two.modpow(&BigUint::from(exponent), &p),
        );
    }
    for (exponent, entry) in M::POWER_OF_TWO_INVERSES.iter().enumerate() {
        assert_value(
            PastaField::<M>::from_montgomery_limbs(*entry),
            &inverse_two.modpow(&BigUint::from(exponent), &p),
        );
    }
    for (batch, entry) in M::SAFEGCD_CORRECTIONS.iter().enumerate() {
        assert_value(
            PastaField::<M>::from_montgomery_limbs(*entry),
            &(&one << (2 * (batch + 1))),
        );
    }
    for log_size in 0..=32 {
        let expected = five.modpow(&((&p - 1u8) >> log_size), &p);
        let root = PastaField::<M>::root_of_unity(log_size).unwrap();
        assert_value(root, &expected);
        assert_value(
            PastaField::<M>::root_of_unity_inverse(log_size).unwrap(),
            &expected.modpow(&(&p - 2u8), &p),
        );
        assert_eq!(expected.modpow(&(&one << log_size), &p), one);
        if log_size != 0 {
            assert_ne!(expected.modpow(&(&one << (log_size - 1)), &p), one);
        }
    }
    for log_size in [33, 64, u32::MAX] {
        assert_eq!(
            (PastaField::<M>::root_of_unity(log_size)).map(|value| value.reduce()),
            None
        );
        assert_eq!(
            (PastaField::<M>::root_of_unity_inverse(log_size)).map(|value| value.reduce()),
            None
        );
    }
    assert_value(PastaField::<M>::DELTA, &five.modpow(&(&one << 32usize), &p));
    let zeta = five
        .modpow(&((&p - 1u8) / 3u8), &p)
        .modpow(&BigUint::from(zeta_power), &p);
    assert_ne!(zeta, one);
    assert_eq!(zeta.modpow(&BigUint::from(3u8), &p), one);
    assert_value(PastaField::<M>::ZETA, &zeta);
    assert_value(PastaField::<M>::ZETA_INVERSE, &zeta.modpow(&(&p - 2u8), &p));
}

#[test]
fn parameter_derivations_match_big_integers() {
    check_parameters::<PallasBase>(2, PALLAS_BASE_SQRT_EXPONENT);
    check_parameters::<PallasScalar>(1, PALLAS_SCALAR_SQRT_EXPONENT);
}
