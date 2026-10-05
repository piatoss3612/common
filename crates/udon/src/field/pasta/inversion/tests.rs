use super::*;
use crate::field::pasta::safegcd::{
    SAFEGCD_BATCHES, SIGNED62_MASK, bezout_offset, divsteps_62, to_signed62, update_fg,
};
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

fn check_divsteps<M: PrimeModulus>() {
    let mut terminal_signs = [false; 2];
    for (_, x) in samples::<M>(256) {
        if x == BigUint::from(0u8) {
            continue;
        }
        let mut f = to_signed62(&M::MODULUS);
        let mut g = to_signed62(&limbs(&x));
        let mut expected_f = BigInt::from(modulus::<M>());
        let mut expected_g = BigInt::from(x);
        let mut delta = 1;
        for _ in 0..SAFEGCD_BATCHES {
            let low_f = f[0] as u64 | ((f[1] as u64) << 62);
            let low_g = g[0] as u64 | ((g[1] as u64) << 62);
            let (next_delta, matrix) = divsteps_62(delta, low_f, low_g);
            for row in matrix.chunks_exact(2) {
                assert!(row[0].unsigned_abs() + row[1].unsigned_abs() <= 1 << 62);
            }
            let [u, v, q, r] = matrix.map(i128::from);
            assert_eq!(u * r - v * q, 1 << 62);
            // Execute individual divsteps with exact signed integers. Unlike
            // the runtime core, this uses neither truncated words nor a matrix.
            for _ in 0..62 {
                let odd = &expected_g % 2u8 != BigInt::from(0);
                if delta > 0 && odd {
                    delta = 1 - delta;
                    let next_g = (&expected_g - &expected_f) >> 1usize;
                    expected_f = expected_g;
                    expected_g = next_g;
                } else {
                    delta += 1;
                    if odd {
                        expected_g += &expected_f;
                    }
                    expected_g >>= 1usize;
                }
            }
            assert_eq!(delta, next_delta);
            (f, g) = update_fg(&f, &g, matrix);
            assert_eq!(signed62(&f), expected_f);
            assert_eq!(signed62(&g), expected_g);
            for row in [f, g] {
                assert!(
                    row[..4]
                        .iter()
                        .all(|limb| (0..=SIGNED62_MASK).contains(limb))
                );
            }
            if g == [0; 5] {
                break;
            }
        }
        assert_eq!(expected_g, BigInt::from(0));
        assert!(expected_f == BigInt::from(1) || expected_f == BigInt::from(-1));
        terminal_signs[usize::from(expected_f < BigInt::from(0))] = true;
    }
    assert_eq!(terminal_signs, [true; 2]);
}

#[test]
fn batched_divsteps_match_signed_integer_steps_and_converge() {
    check_divsteps::<PallasBase>();
    check_divsteps::<PallasScalar>();
}

fn check_bezout_rows<M: PrimeModulus>() {
    let p = modulus::<M>();
    let inverse_radix = (BigUint::from(1u8) << 64usize).modpow(&(&p - 2u8), &p);
    let bound = 1i64 << 62;
    let rows = [
        (bound, 0),
        (-bound, 0),
        (0, bound),
        (0, -bound),
        (bound / 2, -bound / 2),
        (-1, 1),
    ];
    let values = samples::<M>(16);
    for (a, x) in &values {
        for (b, y) in &values {
            for (u, v) in rows {
                let expected = signed_mod(
                    BigInt::from(x.clone()) * u + BigInt::from(y.clone()) * v,
                    &p,
                ) * &inverse_radix
                    % &p;
                assert_value(
                    PastaField::<M, Reduced>::bezout_row_update(u, &a.reduce(), v, &b.reduce()),
                    &expected,
                );
            }
        }
    }
    assert_eq!(integer(&bezout_offset(&M::MODULUS)), &p << 63usize);
}

#[test]
fn signed_bezout_rows_cover_extreme_coefficients() {
    check_bezout_rows::<PallasBase>();
    check_bezout_rows::<PallasScalar>();
}
