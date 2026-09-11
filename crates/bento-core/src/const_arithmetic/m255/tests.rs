//! Const derivations checked over small and large moduli.

use super::*;
use crate::const_arithmetic::test_support::{integer, limbs, samples};
use crate::const_arithmetic::{U256, u256};
use num_bigint::BigUint;

extern crate std;

#[test]
fn montgomery_arithmetic_matches_independent_full_width_integers() {
    let radix = BigUint::from(1u8) << 256usize;
    for mut modulus in samples() {
        modulus[0] |= 1;
        modulus[3] &= u64::MAX >> 1;
        if modulus == [1, 0, 0, 0] {
            modulus[0] = 3;
        }
        let p = integer(&modulus);
        let radix_inverse = radix.modinv(&p).unwrap();
        assert_eq!(one(&modulus), limbs(&(&radix % &p)));
        assert_eq!(r2(&modulus), limbs(&((&radix * &radix) % &p)));

        // REDC's exclusive upper bound exercises carry propagation even for
        // small moduli and for moduli with zero words between nonzero words.
        let largest_wide = &p * &radix - 1u8;
        assert_eq!(
            reduce_wide(&modulus, &limbs(&largest_wide)),
            limbs(&((&largest_wide * &radix_inverse) % &p)),
        );
        for a in samples() {
            let a_integer = integer(&a);
            let a_reduced = &a_integer % &p;
            let encoded = from_u256(&modulus, &a);
            assert_eq!(encoded, limbs(&((&a_integer * &radix) % &p)));
            assert_eq!(to_u256(&modulus, &encoded), limbs(&a_reduced));

            for b in samples() {
                let b_integer = integer(&b);
                let b_reduced = &b_integer % &p;
                assert_eq!(
                    add(&modulus, &limbs(&a_reduced), &limbs(&b_reduced)),
                    limbs(&((&a_reduced + &b_reduced) % &p)),
                );
                // One reduced operand suffices for the documented product
                // bound; a retains all 256 bits.
                assert_eq!(
                    mul(&modulus, &a, &limbs(&b_reduced)),
                    limbs(&((&a_integer * &b_reduced * &radix_inverse) % &p)),
                );
                let wide = (&a_reduced << 256) + &b_integer;
                assert_eq!(
                    reduce_wide(&modulus, &limbs(&wide)),
                    limbs(&((&wide * &radix_inverse) % &p)),
                );
            }
            assert_eq!(
                pow(&modulus, &encoded, &a),
                limbs(&((a_integer.modpow(&a_integer, &p) * &radix) % &p)),
            );
        }
    }
}

#[test]
fn field_derivations_match_independent_integers() {
    for (modulus, generator) in [
        ([7, 0, 0, 0], 3),
        ([13, 0, 0, 0], 2),
        ([97, 0, 0, 0], 5),
        (
            u256::from_hex("0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001"),
            7,
        ),
    ] {
        let p = integer(&modulus);
        let radix = (BigUint::from(1u8) << 256) % &p;
        let order = &p - 1u8;
        let two_adicity = order.trailing_zeros().unwrap() as u32;
        let cofactor = &order >> two_adicity as usize;
        let g = BigUint::from(generator);

        assert_eq!(u256::odd_cofactor(&modulus, two_adicity), limbs(&cofactor));
        assert_eq!(
            u256::tonelli_shanks_exponent(&modulus, two_adicity),
            limbs(&((&cofactor - 1u8) >> 1)),
        );
        assert_eq!(
            two_adic_root_of_unity(&modulus, generator, two_adicity),
            limbs(&((g.modpow(&cofactor, &p) * &radix) % &p)),
        );
        assert_eq!(
            odd_order_generator(&modulus, generator, two_adicity),
            limbs(&((g.modpow(&(BigUint::from(1u8) << two_adicity as usize), &p) * &radix) % &p)),
        );
        let cube_root = cube_root_of_unity(&modulus, generator);
        assert_eq!(
            cube_root,
            limbs(&((g.modpow(&(&order / 3u8), &p) * &radix) % &p))
        );
        assert_ne!(cube_root, one(&modulus));
        assert_eq!(pow(&modulus, &cube_root, &[3, 0, 0, 0]), one(&modulus));

        for value in samples().into_iter().skip(1) {
            let ordinary = integer(&value) % &p;
            if ordinary == BigUint::from(0u8) {
                continue;
            }
            assert_eq!(
                invert_prime(&modulus, &from_u256(&modulus, &value)),
                limbs(&((ordinary.modinv(&p).unwrap() * &radix) % &p)),
            );
        }
    }
}

#[test]
fn subgroup_derivations_reject_incorrect_two_adicity() {
    for derive in [two_adic_root_of_unity, odd_order_generator] {
        for two_adicity in [0, 1, 3, 6, 256, u32::MAX] {
            assert!(std::panic::catch_unwind(|| derive(&[69, 0, 0, 0], 2, two_adicity)).is_err());
        }
        // The factorization is meaningful even for a composite modulus.
        assert!(std::panic::catch_unwind(|| derive(&[69, 0, 0, 0], 2, 2)).is_ok());
    }
}

#[test]
fn field_derivations_check_their_input_bounds() {
    assert!(std::panic::catch_unwind(|| cube_root_of_unity(&[11, 0, 0, 0], 2)).is_err());
    assert!(std::panic::catch_unwind(|| invert_prime(&[97, 0, 0, 0], &[97, 0, 0, 0])).is_err());
    assert!(std::panic::catch_unwind(|| inverse_powers_of_two::<258>(&[97, 0, 0, 0])).is_err());
    // Both positions in addition have the same reduced-operand contract.
    for operand in [[97, 0, 0, 0], [u64::MAX; 4]] {
        assert!(std::panic::catch_unwind(|| add(&[97, 0, 0, 0], &[0; 4], &operand)).is_err());
    }
    for modulus in [[0; 4], [1, 0, 0, 0], [2, 0, 0, 0], [u64::MAX; 4]] {
        assert!(std::panic::catch_unwind(|| pow(&modulus, &[0; 4], &[0; 4])).is_err());
        assert!(std::panic::catch_unwind(|| mul(&modulus, &[0; 4], &[0; 4])).is_err());
    }
}

#[test]
fn full_width_tables_are_const_evaluable_and_match_independent_integers() {
    const MODULUS: U256 = [u64::MAX - 18, u64::MAX, u64::MAX, (1 << 63) - 1];
    const SINGLE: [U256; 1] = inverse_powers_of_two(&MODULUS);
    const SHORT: [U256; 17] = inverse_powers_of_two(&MODULUS);
    const INVERSES: [U256; 257] = inverse_powers_of_two(&MODULUS);
    const CORRECTIONS: [U256; 12] = safegcd_corrections_62_64(&MODULUS);
    let p = integer(&MODULUS);
    assert_eq!(SINGLE, [INVERSES[0]]);
    assert_eq!(SHORT, INVERSES[..17]);
    for (k, entry) in INVERSES.into_iter().enumerate() {
        assert_eq!(entry, limbs(&((BigUint::from(1u8) << (256 - k)) % &p)));
    }
    for (batch, entry) in CORRECTIONS.into_iter().enumerate() {
        assert_eq!(
            entry,
            limbs(&((BigUint::from(1u8) << (256 + 2 * (batch + 1))) % &p))
        );
    }
}

#[test]
fn safegcd_tables_use_the_requested_batch_limit() {
    const MODULUS: U256 = [97, 0, 0, 0];
    const EMPTY: [U256; 0] = safegcd_corrections_62_64(&MODULUS);
    const SINGLE: [U256; 1] = safegcd_corrections_62_64(&MODULUS);
    const TABLE: [U256; 12] = safegcd_corrections_62_64(&MODULUS);

    assert_eq!(EMPTY, [[0u64; 4]; 0]);
    assert_eq!(SINGLE, [TABLE[0]]);
    // Native integer modular arithmetic is independent of the limb routines.
    let radix = (0..256).fold(1u64, |value, _| value * 2 % 97);
    let mut correction = radix;
    for entry in TABLE {
        correction = correction * 4 % 97;
        assert_eq!(entry, [correction, 0, 0, 0]);
    }
}

#[test]
fn small_prime_derivations_hold() {
    let p: U256 = [97, 0, 0, 0];
    assert_eq!(u256::odd_cofactor(&p, 5), [3, 0, 0, 0]);

    for a in 0..97u64 {
        for b in 0..97 {
            assert_eq!(
                add(&p, &[a, 0, 0, 0], &[b, 0, 0, 0]),
                [(a + b) % 97, 0, 0, 0]
            );
        }
    }

    // Montgomery form round-trips every residue.
    for value in 0..97u64 {
        let form = from_u256(&p, &[value, 0, 0, 0]);
        assert_eq!(to_u256(&p, &form), [value, 0, 0, 0]);
    }

    // The root has order exactly 2^5; 5^32 = 35 generates the odd-order subgroup.
    let one = one(&p);
    let root = two_adic_root_of_unity(&p, 5, 5);
    let mut value = root;
    for _ in 0..5 {
        assert_ne!(value, one);
        value = mul(&p, &value, &value);
    }
    assert_eq!(value, one);
    assert_eq!(odd_order_generator(&p, 5, 5), from_u64(&p, 35));

    assert_eq!(mul(&p, &from_u64(&p, 2), &two_inverse(&p)), one);
}

#[test]
fn large_modulus_derivations_hold() {
    // The BLS12-381 scalar field has two-adicity 32 and generator 7.
    let p = u256::from_hex("0x73eda753299d7d483339d80809a1d80553bda402fffe5bfeffffffff00000001");
    let cofactor = u256::odd_cofactor(&p, 32);
    let reconstructed = u256::mul_wide(&cofactor, &[1 << 32, 0, 0, 0]);
    let minus_one = u256::sub_u64(&p, 1);
    assert_eq!(
        [
            reconstructed[0],
            reconstructed[1],
            reconstructed[2],
            reconstructed[3]
        ],
        minus_one
    );
    assert_eq!(reconstructed[4..], [0; 4]);

    let one = one(&p);
    let root = two_adic_root_of_unity(&p, 7, 32);
    let mut value = root;
    for _ in 0..32 {
        assert_ne!(value, one);
        value = mul(&p, &value, &value);
    }
    assert_eq!(value, one);
}

#[test]
#[should_panic(expected = "operands must be reduced")]
fn add_rejects_operand_equal_to_modulus() {
    let p = [97, 0, 0, 0];
    add(&p, &p, &[0; 4]);
}

#[test]
#[should_panic(expected = "modulus must be odd")]
fn add_rejects_even_modulus() {
    add(&[6, 0, 0, 0], &[1, 0, 0, 0], &[2, 0, 0, 0]);
}

#[test]
#[should_panic(expected = "modulus must be below 2^255")]
fn add_rejects_modulus_at_or_above_2_pow_255() {
    add(&[1, 0, 0, 1 << 63], &[0; 4], &[0; 4]);
}

#[test]
fn conversions_and_reduction_cover_the_spare_bit_boundary() {
    // For p = 2^255 - 19, R = 2p + 38. Thus R mod p = 38 and
    // (R - 1) * R mod p = 37 * 38 = 1406, independently of limb reduction.
    const P: U256 = [u64::MAX - 18, u64::MAX, u64::MAX, (1 << 63) - 1];
    const ENCODED: U256 = from_u256(&P, &[u64::MAX; 4]);
    const DECODED: U256 = to_u256(&P, &ENCODED);
    assert_eq!(one(&P), [38, 0, 0, 0]);
    assert_eq!(r2(&P), [1444, 0, 0, 0]);
    assert_eq!(ENCODED, [1406, 0, 0, 0]);
    assert_eq!(DECODED, [37, 0, 0, 0]);

    let minus_one = u256::sub_u64(&P, 1);
    assert_eq!(add(&P, &minus_one, &minus_one), u256::sub_u64(&P, 2));

    // The largest accepted wide value is p * R - 1. Its result is -38^-1
    // modulo p; this vector was computed with Python's integer modular inverse.
    let upper_boundary = [
        u64::MAX,
        u64::MAX,
        u64::MAX,
        u64::MAX,
        P[0] - 1,
        P[1],
        P[2],
        P[3],
    ];
    assert_eq!(
        reduce_wide(&P, &upper_boundary),
        [
            0xbca1_af28_6bca_1ae3,
            0xa1af_286b_ca1a_f286,
            0xaf28_6bca_1af2_86bc,
            0x686b_ca1a_f286_bca1,
        ]
    );
}

#[test]
fn montgomery_products_work_over_small_primes_and_composites() {
    // Both operands may exceed p when their product remains below p * R.
    // For p = 97, R mod p = 61 and R^-1 mod p = 35: 1 * 2 * 35 = 70.
    assert_eq!(
        mul(&[97, 0, 0, 0], &[98, 0, 0, 0], &[99, 0, 0, 0]),
        [70, 0, 0, 0]
    );

    for p in [3u64, 9, 15, 97] {
        let modulus = [p, 0, 0, 0];
        let radix = (0..256).fold(1u64, |value, _| value * 2 % p);
        for a in 0..p {
            for b in 0..p {
                assert_eq!(
                    mul(&modulus, &from_u64(&modulus, a), &from_u64(&modulus, b)),
                    [a * b % p * radix % p, 0, 0, 0]
                );
            }
        }
        assert_eq!(
            to_u256(&modulus, &from_u64(&modulus, u64::MAX)),
            [u64::MAX % p, 0, 0, 0]
        );
        assert_eq!(
            to_u256(&modulus, &two_inverse(&modulus)),
            [p.div_ceil(2), 0, 0, 0]
        );
    }
}

#[test]
fn powers_use_ordinary_full_width_exponents() {
    const P: U256 = [97, 0, 0, 0];
    const POWER: U256 = pow(&P, &from_u64(&P, 5), &[u64::MAX; 4]);
    // 2^256 - 1 = 63 (mod 96), and 5^63 = 51 (mod 97).
    assert_eq!(to_u256(&P, &POWER), [51, 0, 0, 0]);
    assert_eq!(pow(&P, &[0; 4], &[0; 4]), one(&P));
    assert_eq!(pow2_mod(&P, 0), [1, 0, 0, 0]);
}

#[test]
fn inverse_power_table_includes_both_radix_endpoints() {
    const P: U256 = [97, 0, 0, 0];
    const EMPTY: [U256; 0] = inverse_powers_of_two(&P);
    const TABLE: [U256; 257] = inverse_powers_of_two(&P);
    assert_eq!(EMPTY, [[0; 4]; 0]);
    let mut expected = (0..256).fold(1u64, |value, _| value * 2 % 97);
    for entry in TABLE {
        assert_eq!(entry, [expected, 0, 0, 0]);
        expected = expected * 49 % 97; // 49 is the ordinary inverse of two.
    }
    assert_eq!(TABLE[256], [1, 0, 0, 0]);
}

#[test]
fn reduction_coefficient_has_a_word_sized_domain() {
    const FOR_ONE: u64 = reduction_coefficient(1);
    assert_eq!(FOR_ONE, u64::MAX);
    assert_eq!(reduction_coefficient(u64::MAX), 1);
    for low in [3u64, 97, 0x992d_30ed_0000_0001, (1 << 63) + 1] {
        assert_eq!(low.wrapping_mul(reduction_coefficient(low)), u64::MAX);
    }
}

#[test]
#[should_panic(expected = "low word must be odd")]
fn reduction_coefficient_rejects_even_words() {
    reduction_coefficient(2);
}

#[test]
#[should_panic(expected = "modulus must be odd")]
fn zero_power_still_checks_the_modulus() {
    pow2_mod(&[2, 0, 0, 0], 0);
}

#[test]
#[should_panic(expected = "modulus must exceed 2")]
fn empty_inverse_table_still_checks_the_modulus() {
    inverse_powers_of_two::<0>(&[1, 0, 0, 0]);
}

#[test]
#[should_panic(expected = "modulus must be below 2^255")]
fn empty_safegcd_table_still_checks_the_modulus() {
    safegcd_corrections_62_64::<0>(&[1, 0, 0, 1 << 63]);
}

#[test]
#[should_panic(expected = "wide input must be below modulus * R")]
fn wide_reduction_rejects_the_exclusive_bound() {
    reduce_wide(&[97, 0, 0, 0], &[0, 0, 0, 0, 97, 0, 0, 0]);
}

#[test]
#[should_panic(expected = "wide input must be below modulus * R")]
fn multiplication_checks_the_product_bound() {
    mul(&[97, 0, 0, 0], &[0, 0, 1, 0], &[0, 0, 97, 0]);
}

#[test]
#[should_panic(expected = "base must be reduced")]
fn zero_exponent_still_requires_a_reduced_base() {
    pow(&[97, 0, 0, 0], &[97, 0, 0, 0], &[0; 4]);
}

#[test]
#[should_panic(expected = "value must be reduced")]
fn decoding_requires_a_reduced_residue() {
    to_u256(&[97, 0, 0, 0], &[97, 0, 0, 0]);
}

#[test]
#[should_panic(expected = "cannot invert zero")]
fn prime_inversion_rejects_zero() {
    invert_prime(&[97, 0, 0, 0], &[0; 4]);
}
