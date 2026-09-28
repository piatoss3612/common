//! Full-width integer arithmetic and parameter derivations.

use super::*;
use crate::const_arithmetic::test_support::{integer, limbs, samples};
use num_bigint::BigUint;

extern crate std;

#[test]
fn full_width_operations_match_independent_integers() {
    let radix = BigUint::from(1u8) << 256;
    for a in samples() {
        let a_integer = integer(&a);
        for b in samples() {
            let b_integer = integer(&b);
            assert_eq!(ge(&a, &b), a_integer >= b_integer);

            let sum = &a_integer + &b_integer;
            assert_eq!(
                add_with_carry(&a, &b),
                (limbs(&(&sum % &radix)), u64::from(sum >= radix))
            );
            assert_eq!(
                sub_with_borrow(&a, &b),
                (
                    limbs(&((&a_integer + &radix - &b_integer) % &radix)),
                    u64::from(a_integer < b_integer),
                )
            );
            assert_eq!(mul_wide(&a, &b), limbs(&(&a_integer * &b_integer)));
        }
        for shift in 0..256 {
            assert_eq!(shr(&a, shift), limbs(&(&a_integer >> shift as usize)));
        }
        for divisor in [1, 3, 97, 1 << 63, u64::MAX] {
            let quotient = &a_integer / divisor;
            let dividend = limbs(&(&quotient * divisor));
            assert_eq!(div_exact_u64(&dividend, divisor), limbs(&quotient));
            if a_integer >= BigUint::from(divisor) {
                assert_eq!(sub_u64(&a, divisor), limbs(&(&a_integer - divisor)));
            }
        }
    }
}

#[test]
fn hexadecimal_parsing_preserves_digit_order_and_case() {
    for value in samples() {
        let encoded = std::format!("0x{:064x}", integer(&value));
        assert_eq!(from_hex(&encoded), value);
        let uppercase = std::format!("0x{:064X}", integer(&value));
        assert_eq!(from_hex(&uppercase), value);
    }
    for malformed in [
        "",
        "0x",
        "0x1",
        " 0x",
        "0X0000000000000000000000000000000000000000000000000000000000000000",
    ] {
        assert!(std::panic::catch_unwind(|| from_hex(malformed)).is_err());
    }
    for bad_digit in ["g", "_", " ", "é"] {
        let encoded = std::format!("0x{}{}", "0".repeat(64 - bad_digit.len()), bad_digit);
        assert!(std::panic::catch_unwind(|| from_hex(&encoded)).is_err());
    }
    let too_long = std::format!("0x{}", "0".repeat(65));
    assert!(std::panic::catch_unwind(|| from_hex(&too_long)).is_err());
}

#[test]
fn integer_operations_reject_out_of_domain_inputs() {
    assert!(std::panic::catch_unwind(|| sub_u64(&[0; 4], 1)).is_err());
    for shift in [256, u32::MAX] {
        assert!(std::panic::catch_unwind(|| shr(&[0; 4], shift)).is_err());
    }
    assert!(std::panic::catch_unwind(|| div_exact_u64(&[0; 4], 0)).is_err());
    assert!(std::panic::catch_unwind(|| div_exact_u64(&[5, 0, 0, 0], 2)).is_err());
    for two_adicity in [0, 256, u32::MAX] {
        assert!(std::panic::catch_unwind(|| odd_cofactor(&[97, 0, 0, 0], two_adicity)).is_err());
    }
    for modulus in [[0; 4], [1, 0, 0, 0], [2, 0, 0, 0], [98, 0, 0, 0]] {
        assert!(std::panic::catch_unwind(|| odd_cofactor(&modulus, 1)).is_err());
    }
    for shift in [1, 63, 65, 385, 448, u32::MAX] {
        assert!(std::panic::catch_unwind(|| round_shifted_ratio(&[1, 0, 0, 0], 0, shift)).is_err());
    }
}

#[test]
fn arithmetic_uses_all_256_bits() {
    const MAX: U256 =
        from_hex("0xffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffff");
    const SUM: (U256, u64) = add_with_carry(&MAX, &[1, 0, 0, 0]);
    const DIFFERENCE: (U256, u64) = sub_with_borrow(&[0; 4], &[1, 0, 0, 0]);
    const PRODUCT: U512 = mul_wide(&MAX, &MAX);

    assert_eq!(MAX, [u64::MAX; 4]);
    assert_eq!(SUM, ([0; 4], 1));
    assert_eq!(DIFFERENCE, (MAX, 1));
    // (2^256 - 1)^2 = 2^512 - 2^257 + 1.
    assert_eq!(
        PRODUCT,
        [1, 0, 0, 0, u64::MAX - 1, u64::MAX, u64::MAX, u64::MAX]
    );
    assert!(ge(&MAX, &MAX));
    assert!(ge(&[0, 0, 0, 1 << 63], &[u64::MAX, u64::MAX, u64::MAX, 0]));
    assert!(!ge(&[u64::MAX, 0, 0, 0], &[0, 1, 0, 0]));
    assert_eq!(shr(&MAX, 0), MAX);
    assert_eq!(shr(&MAX, 64), [u64::MAX, u64::MAX, u64::MAX, 0]);
    assert_eq!(shr(&MAX, 255), [1, 0, 0, 0]);
    assert_eq!(div_exact_u64(&MAX, 3), [0x5555_5555_5555_5555; 4]);
}

#[test]
fn field_exponents_accept_moduli_above_the_montgomery_bound() {
    const MODULUS: U256 = [u64::MAX; 4];
    const COFACTOR: U256 = odd_cofactor(&MODULUS, 1);
    const EXPONENT: U256 = tonelli_shanks_exponent(&MODULUS, 1);
    assert_eq!(COFACTOR, [u64::MAX, u64::MAX, u64::MAX, (1 << 63) - 1]);
    assert_eq!(EXPONENT, [u64::MAX, u64::MAX, u64::MAX, (1 << 62) - 1]);

    // p - 1 can also have its only set bit at position 255.
    assert_eq!(odd_cofactor(&[1, 0, 0, 1 << 63], 255), [1, 0, 0, 0]);
    assert_eq!(tonelli_shanks_exponent(&[1, 0, 0, 1 << 63], 255), [0; 4]);
}

#[test]
fn rounded_ratios_match_native_integer_division() {
    for denominator in 1..32u128 {
        for value in [0, 1, 5, 17, u64::MAX as u128] {
            for shift in [0, 64] {
                let numerator = value << shift;
                let quotient = numerator / denominator;
                let remainder = numerator % denominator;
                let rounded = quotient + u128::from(2 * remainder >= denominator);
                assert_eq!(
                    round_shifted_ratio(&[denominator as u64, 0, 0, 0], value, shift),
                    [rounded as u64, (rounded >> 64) as u64, 0, 0, 0]
                );
            }
        }
    }
    // (2^128 - 1) * 2^384 / 2^192 uses the quotient's full 320-bit width.
    const WIDE: crate::const_arithmetic::U320 = round_shifted_ratio(&[0, 0, 0, 1], u128::MAX, 384);
    assert_eq!(WIDE, [0, 0, 0, u64::MAX, u64::MAX]);
}

#[test]
#[should_panic(expected = "denominator must be nonzero")]
fn ratio_rejects_zero_denominator_even_for_zero_numerator() {
    round_shifted_ratio(&[0; 4], 0, 0);
}

#[test]
fn rounded_ratios_match_independent_full_width_division() {
    for denominator in samples().into_iter().skip(1) {
        let d = integer(&denominator);
        for value in [0, 1, 5, u64::MAX as u128, 1 << 64, 1 << 127, u128::MAX] {
            for shift in [0, 64, 128, 192, 256, 320, 384] {
                let numerator = BigUint::from(value) << shift as usize;
                let rounded = (&numerator + (&d >> 1usize)) / &d;
                if rounded.bits() <= 320 {
                    assert_eq!(
                        round_shifted_ratio(&denominator, value, shift),
                        limbs(&rounded),
                        "denominator={denominator:x?}, value={value:x}, shift={shift}",
                    );
                } else {
                    assert!(
                        std::panic::catch_unwind(|| {
                            round_shifted_ratio(&denominator, value, shift)
                        })
                        .is_err()
                    );
                }
            }
        }
    }
    // The remainder needs a 257th bit when dividing this numerator by 2^256-1.
    const FULL_WIDTH: crate::const_arithmetic::U320 =
        round_shifted_ratio(&[u64::MAX; 4], u128::MAX, 384);
    assert_eq!(FULL_WIDTH, [1, 0, u64::MAX, u64::MAX, 0]);
}

#[test]
#[should_panic(expected = "quotient exceeds five limbs")]
fn ratio_rejects_a_quotient_exceeding_320_bits() {
    round_shifted_ratio(&[1, 0, 0, 0], 1, 320);
}

#[test]
#[should_panic(expected = "two_adicity exceeds the trailing zeros")]
fn odd_cofactor_rejects_overstated_two_adicity() {
    // 68 >> 6 is odd, but shifting discards a set bit below the boundary.
    odd_cofactor(&[69, 0, 0, 0], 6);
}

#[test]
fn odd_cofactor_accepts_the_exact_two_adicity() {
    assert_eq!(odd_cofactor(&[69, 0, 0, 0], 2), [17, 0, 0, 0]);
}

#[test]
#[should_panic(expected = "two_adicity is not maximal")]
fn odd_cofactor_rejects_understated_two_adicity() {
    odd_cofactor(&[69, 0, 0, 0], 1);
}
