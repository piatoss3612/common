use num_bigint::BigInt;

use super::*;
use crate::curve::{
    digits::{centered_digit, signed_window_digits},
    parameters::GlvParameters,
};

#[test]
fn packed_midpoint_carries_reconstruct_signed_extremes() {
    for width in [2, 4, 8] {
        for negative in [false, true] {
            for value in [0, 1, 127, 128, 129, 255, 256, u128::MAX, i128::MAX as u128] {
                let mut carry = 0;
                let mut magnitude = value;
                let mut digits = Vec::new();
                for _ in 0..128 / width {
                    let digit = centered_digit(
                        (magnitude & ((1 << width) - 1)) as u16,
                        negative,
                        &mut carry,
                        width,
                    );
                    assert!((-(1 << (width - 1))..1 << (width - 1)).contains(&digit));
                    assert_eq!(digit as i8 as i16, digit);
                    digits.push(digit);
                    magnitude >>= width;
                }
                let mut actual = BigInt::from(if negative { -carry } else { carry });
                for d in digits.into_iter().rev() {
                    actual = (actual << width) + d;
                }
                let expected = BigInt::from(value);
                assert_eq!(actual, if negative { -expected } else { expected });
            }
        }
    }
}

#[test]
fn signed_windows_reconstruct_partial_windows_and_carries() {
    for w in 2..=8 {
        let n = 128_usize.div_ceil(w);
        let mut values = vec![0, 1, u128::MAX, i128::MAX as u128];
        for bit in (w - 1..128).step_by(w) {
            let power = 1_u128 << bit;
            values.extend([power - 1, power, power + 1]);
        }
        for value in values {
            let (digits, carry) = signed_window_digits(value, w);
            let mut integer = BigInt::from(u8::from(carry));
            for &digit in digits[..n].iter().rev() {
                assert!((-(1 << (w - 1))..1 << (w - 1)).contains(&digit));
                integer = (integer << w) + digit;
            }
            assert_eq!(integer, BigInt::from(value));
        }
    }
}

#[test]
fn pasta_lattice_bounds_allow_only_second_half_width_two_carries() {
    for bounds in [
        GlvParameters::<Pallas>::BOUNDS,
        GlvParameters::<Vesta>::BOUNDS,
    ] {
        // Final carry is monotone in the magnitude. These conservative integer
        // upper bounds exclude every other width and the first width-2 half.
        for w in 2..=8 {
            for (half, bound) in bounds.into_iter().enumerate() {
                assert_eq!(signed_window_digits(bound, w).1, w == 2 && half == 1);
            }
        }
    }
}
