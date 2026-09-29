use super::*;
use crate::field::pasta::test_support::*;

fn active_value(value: &Signed62, len: usize) -> BigInt {
    value.0[..len]
        .iter()
        .rev()
        .fold(BigInt::from(0), |acc, limb| (acc << 62usize) + limb)
}

fn from_integer(mut value: BigInt) -> Signed62 {
    let mask = BigInt::from(MASK62);
    let mut out = [0; 5];
    for limb in &mut out[..4] {
        *limb = (&value & &mask).try_into().unwrap();
        value >>= 62usize;
    }
    out[4] = value.try_into().unwrap();
    Signed62(out)
}

fn check_trace<M: PrimeModulus>() {
    let p = modulus::<M>();
    let inverse_radix = (BigUint::from(1u8) << 62usize).modinv(&p).unwrap();
    let mut terminal_signs = [false; 2];
    let mut shrunk = false;
    for (value, _) in samples::<M>(256) {
        let input = value.reduce().limbs;
        if input == [0; 4] {
            assert_eq!(invert::<M>(&input), None);
            continue;
        }
        let mut f = Signed62(M::MODULUS_62);
        let mut g = pack62(&input);
        let mut d = Signed62([0; 5]);
        let mut e = Signed62(M::R2_62);
        let mut eta = -1;
        let mut len = 5;
        let mut terminated = false;
        for batch in 1..=12 {
            let (t, next_eta) = divsteps_62_var(eta, f.0[0] as u64, g.0[0] as u64);
            eta = next_eta;
            let (old_f, old_g) = (active_value(&f, len), active_value(&g, len));
            let (old_d, old_e) = (signed62(&d.0), signed62(&e.0));
            let numerator_f = &old_f * t.u + &old_g * t.v;
            let numerator_g = &old_f * t.q + &old_g * t.r;
            let radix = BigInt::from(1u64 << 62);
            assert_eq!(&numerator_f % &radix, BigInt::from(0));
            assert_eq!(&numerator_g % &radix, BigInt::from(0));
            assert_eq!(
                i128::from(t.u) * i128::from(t.r) - i128::from(t.v) * i128::from(t.q),
                1 << 62
            );

            if batch == 1 {
                update_fg_62_first::<M>(&mut f, &mut g, &t);
            } else {
                update_fg_62_var(len, &mut f, &mut g, &t);
            }
            assert_eq!(active_value(&f, len), numerator_f >> 62usize);
            assert_eq!(active_value(&g, len), numerator_g >> 62usize);
            let expected_d = signed_mod(&old_d * t.u + &old_e * t.v, &p) * &inverse_radix % &p;
            let expected_e = signed_mod(&old_d * t.q + &old_e * t.r, &p) * &inverse_radix % &p;
            let terminal = is_zero(&g, len);
            if batch == 1 {
                assert!(!terminal);
                (d, e) = update_de_62_first::<M>(&e, &t);
            } else if terminal {
                update_d_only_62::<M>(&mut d, &e, t.u, t.v);
            } else {
                update_de_62::<M>(&mut d, &mut e, &t);
            }
            assert_eq!(signed_mod(signed62(&d.0), &p), expected_d);
            if !terminal {
                assert_eq!(signed_mod(signed62(&e.0), &p), expected_e);
            }
            // Check the Montgomery-scaled Bezout invariant independently.
            assert_eq!(
                signed_mod(BigInt::from(integer(&input)) * signed62(&d.0), &p),
                signed_mod(BigInt::from(integer(&M::R2)) * active_value(&f, len), &p)
            );
            if terminal {
                let sign = f.0[len - 1];
                let expected = if sign < 0 {
                    (&p - expected_d) % &p
                } else {
                    expected_d
                };
                terminal_signs[usize::from(sign < 0)] = true;
                normalize_62::<M>(&mut d, sign);
                assert_eq!(integer(&unpack62(&d)), expected);
                assert_eq!(invert_counted::<M>(&input), Some((unpack62(&d), batch)));
                terminated = true;
                break;
            }
            let before = (active_value(&f, len), active_value(&g, len));
            shrink_len(&mut f, &mut g, &mut len);
            shrunk |= len < 5;
            assert_eq!((active_value(&f, len), active_value(&g, len)), before);
        }
        assert!(terminated);
    }
    assert_eq!(terminal_signs, [true; 2]);
    assert!(shrunk);
}

#[test]
fn divsteps_and_coefficient_updates_match_exact_integer_arithmetic() {
    check_trace::<PallasBase>();
    check_trace::<PallasScalar>();
}

fn check_coefficient_boundaries<M: PrimeModulus>() {
    let p = modulus::<M>();
    let m = BigInt::from(p.clone());
    let inverse_radix = (BigUint::from(1u8) << 62usize).modinv(&p).unwrap();
    let values = [
        -&m * 2u8 + 1u8,
        -&m - 1u8,
        -&m,
        -&m + 1u8,
        BigInt::from(-1),
        BigInt::from(0),
        BigInt::from(1),
        &m - 1u8,
    ];
    let bound = 1i64 << 62;
    let rows = [
        (bound, 0),
        (-bound, 0),
        (0, bound),
        (0, -bound),
        (bound / 2, -bound / 2),
        (-1, 1),
        (bound - 1, 1),
    ];
    for x in &values {
        for sign in [-1, 1] {
            let mut normalized = from_integer(x.clone());
            normalize_62::<M>(&mut normalized, sign);
            assert_eq!(integer(&unpack62(&normalized)), signed_mod(x * sign, &p));
        }
        for y in &values {
            for (u, v) in rows {
                let (mut d, mut e) = (from_integer(x.clone()), from_integer(y.clone()));
                let mut top_only = d;
                let t = Trans2x2 { u, v, q: -v, r: u };
                update_d_only_62::<M>(&mut top_only, &e, u, v);
                update_de_62::<M>(&mut d, &mut e, &t);
                assert_eq!(d, top_only);
                assert_eq!(
                    signed_mod(signed62(&d.0), &p),
                    signed_mod(x * u + y * v, &p) * &inverse_radix % &p
                );
                assert_eq!(
                    signed_mod(signed62(&e.0), &p),
                    signed_mod(x * -v + y * u, &p) * &inverse_radix % &p
                );
            }
        }
    }
}

#[test]
fn signed_coefficients_and_normalization_cover_range_boundaries() {
    check_coefficient_boundaries::<PallasBase>();
    check_coefficient_boundaries::<PallasScalar>();
}

#[test]
#[should_panic(expected = "f must be odd")]
fn invariant_checks_remain_active_in_release_tests() {
    divsteps_62_var(-1, 2, 1);
}

/// Fixed regression vectors: internal little-endian representations with
/// their exact expected 62-divstep batch counts, produced by an
/// independently validated simulation of this exact algorithm. If a count
/// drifts, the port has deviated from the validated control flow:
/// investigate the divergence, do not re-pin the count.
const PALLAS_VECTORS: &[([u64; 4], u32)] = &[
    (
        [
            0x34786d38fffffffd,
            0x992c350be41914ad,
            0xffffffffffffffff,
            0x3fffffffffffffff,
        ],
        7,
    ),
    ([0x00000000000005d1, 0, 0, 0], 8),
    (
        [
            0xe096c0a18679d7ae,
            0x2cc4e34bc6b6f06a,
            0x20eddc12b5a7661d,
            0x0c3c5e66537210ad,
        ],
        8,
    ),
    ([1, 0, 0, 0], 9),
    (
        [
            0xa162a7d34ad63d62,
            0xba71beb748b1fa25,
            0x68dc1330fab3847b,
            0x2dc205e082f2c197,
        ],
        10,
    ),
    (
        [
            0x7ceb4d8e9534361d,
            0x8d7879adc97a8e79,
            0x19ed0538ef7b15eb,
            0x35fb58b0ee06c5b4,
        ],
        10,
    ),
];

const VESTA_VECTORS: &[([u64; 4], u32)] = &[
    (
        [
            0x5b2b3e9cfffffffd,
            0x992c350be3420567,
            0xffffffffffffffff,
            0x3fffffffffffffff,
        ],
        7,
    ),
    (
        [
            0x3ea12df55a259593,
            0xd9c58668e724391e,
            0xfcf9d370dd78552a,
            0x1f7fd517b4f7efeb,
        ],
        8,
    ),
    (
        [
            0x4973c2bd762bc27a,
            0x9085d079ecab3a12,
            0x9f66128174a6731a,
            0x3e7853d1fb18fcca,
        ],
        8,
    ),
    ([1, 0, 0, 0], 9),
    ([0x00000ba5f061296c, 0, 0, 0], 10),
    (
        [
            0xc00e79606e554fa8,
            0x13f9947f444f41d3,
            0xd5a780db63e83468,
            0x337e275425a385f3,
        ],
        10,
    ),
];

#[test]
fn batch_counts_match_pasta_curves_regression_vectors() {
    fn check<M: PrimeModulus>(vectors: &[([u64; 4], u32)]) {
        let p = modulus::<M>();
        assert_eq!(signed62(&M::MODULUS_62), BigInt::from(p.clone()));
        assert_eq!(signed62(&M::R2_62), BigInt::from(integer(&M::R2)));
        for &(input, expected_batches) in vectors {
            let (output, batches) = invert_counted::<M>(&input).unwrap();
            assert_eq!(batches, expected_batches);
            assert!(integer(&output) < p);
            assert_eq!(integer(&input) * integer(&output) % &p, integer(&M::R2));
        }
    }
    check::<PallasBase>(PALLAS_VECTORS);
    check::<PallasScalar>(VESTA_VECTORS);
}
