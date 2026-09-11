use super::*;
use crate::field::{montgomery, word};

#[test]
fn limb_kernels_match_full_width_integer_arithmetic() {
    for a in [0, 1, 1 << 63, u64::MAX] {
        for b in [0, 1, 1 << 63, u64::MAX] {
            for carry in [0, 1, u64::MAX] {
                let (low, high) = word::adc(a, b, carry);
                assert_eq!(integer(&[low, high]), BigUint::from(a) + b + carry);
                for accumulator in [0, 1, u64::MAX] {
                    let (low, high) = word::mac(accumulator, a, b, carry);
                    assert_eq!(
                        integer(&[low, high]),
                        BigUint::from(a) * b + accumulator + carry
                    );
                }
            }
            for borrow in [0, 1] {
                let difference = BigInt::from(a) - b - borrow;
                let (low, high) = word::sbb(a, b, borrow);
                assert_eq!(high != 0, difference < BigInt::from(0));
                assert_eq!(
                    BigUint::from(low),
                    signed_mod(difference, &(BigUint::from(1u8) << 64usize))
                );
            }
        }
    }
    let mut values = vec![[0; 4], [1, 0, 0, 0], [u64::MAX; 4], [0, 0, 0, 1 << 63]];
    let mut state = 0xa54f_f53a_5f1d_36f1;
    values.extend(
        (0..64).map(|_| CanonicalUint::from_le_bytes(deterministic_bytes(&mut state)).limbs()),
    );
    let radix = BigUint::from(1u8) << 256usize;
    for a in &values {
        let x = integer(a);
        assert_eq!(integer(&word::square_wide(a)), &x * &x);
        for b in &values {
            let y = integer(b);
            assert_eq!(integer(&word::multiply_wide(a, b)), &x * &y);
            assert_eq!(word::compare_limbs(a, b), x.cmp(&y));
            let (difference, borrow) = word::subtract_limbs(a, b);
            assert_eq!(borrow != 0, x < y);
            assert_eq!(integer(&difference), (&x + &radix - &y) % &radix);
        }
    }
}

fn check_montgomery<M: PrimeModulus>() {
    let p = modulus::<M>();
    let radix = BigUint::from(1u8) << 256usize;
    let inverse_r = radix.modpow(&(&p - 2u8), &p);
    let mut values = samples::<M>(64)
        .into_iter()
        .map(|(_, x)| x)
        .collect::<Vec<_>>();
    values.extend([p.clone(), &p + 1u8, &radix - 1u8]);
    for a in &values {
        for b in &values {
            let product = a * b;
            if product >= &p * &radix {
                continue;
            }
            let expected = &product * &inverse_r % &p;
            assert_eq!(
                integer(&montgomery::montgomery_multiply::<M>(&limbs(a), &limbs(b))),
                expected
            );
            assert_eq!(
                integer(&montgomery::montgomery_reduce::<M>(limbs(&product))),
                expected
            );
        }
    }
    for input in [
        BigUint::from(0u8),
        &p * &radix - 1u8,
        &p * &radix,
        &p * (&radix + &p) - 1u8,
    ] {
        let raw = integer(&montgomery::montgomery_reduce_unreduced::<M>(limbs(&input)));
        assert!(raw < &p * 3u8);
        assert_eq!(&raw % &p, &input * &inverse_r % &p);
        let reduced = montgomery::reduce_once::<M>(montgomery::reduce_once::<M>(limbs(&raw)));
        assert_eq!(integer(&reduced), &input * &inverse_r % &p);
    }
}

#[test]
fn montgomery_kernels_cover_their_full_input_bounds() {
    check_montgomery::<PallasBase>();
    check_montgomery::<PallasScalar>();
}

fn check_lazy_squares<M: PrimeModulus>() {
    let p = modulus::<M>();
    let radix = BigUint::from(1u8) << 256usize;
    let inverse_r = radix.modinv(&p).unwrap();
    // Check the actual unreduced intermediates against exact REDC, not just
    // field equality, at every supported run length.
    let negative_inverse = (&radix - p.modinv(&radix).unwrap()) % &radix;
    for (value, _) in samples::<M>(32) {
        let mut raw = value.limbs;
        let mut expected = integer(&raw);
        for count in 0..=256 {
            assert_eq!(integer(&raw), expected);
            assert!(expected < &p * 2u8);
            assert_eq!(
                integer(&montgomery::square_run::<M>(&value.limbs, count, None)),
                &expected % &p,
            );
            let factor = limbs(&(&p - 1u8));
            assert_eq!(
                integer(&montgomery::square_run::<M>(
                    &value.limbs,
                    count,
                    Some(&factor)
                )),
                &expected * (&p - 1u8) * &inverse_r % &p,
            );
            if count != 256 {
                let square = &expected * &expected;
                assert!(square < &p * &radix);
                let q = &square * &negative_inverse % &radix;
                expected = (&square + q * &p) / &radix;
                raw = montgomery::montgomery_reduce_unreduced::<M>(word::square_wide(&raw));
            }
        }
    }
}

#[test]
fn lazy_squares_preserve_exact_redc_bounds_for_every_run_length() {
    check_lazy_squares::<PallasBase>();
    check_lazy_squares::<PallasScalar>();
}
