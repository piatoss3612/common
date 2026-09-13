//! Binary ladder for short scalars and independent multiplication tests.

use super::{PastaCurve, ProjectivePoint};
use crate::field::PastaField;

/// Centers an unsigned window, applying its magnitude's sign and updating carry.
///
/// Requires `width` in `2..=12`, `window < 2^width`, and `carry` in `0..=1`.
pub(super) fn centered_digit(window: u16, negative: bool, carry: &mut i16, width: u32) -> i16 {
    let value = window as i16 + *carry;
    let half = 1 << (width - 1);
    // Resolve the midpoint toward the negative digit, including when recoding a
    // negative GLV half. Thus width eight always fits in i8, even at a carry tie.
    *carry = i16::from(if negative {
        value > half
    } else {
        value >= half
    });
    let digit = value - (*carry << width);
    if negative { -digit } else { digit }
}

pub(super) fn multiply<C: PastaCurve>(
    scalar: &PastaField<C::Scalar>,
    add_base: impl Fn(&ProjectivePoint<C>) -> ProjectivePoint<C>,
) -> ProjectivePoint<C> {
    let scalar = scalar.to_canonical_uint();
    let mut result = ProjectivePoint::IDENTITY;
    if let Some(high) = scalar.highest_set_bit() {
        for bit in (0..=high).rev() {
            result = result.double();
            if scalar.bit(bit).unwrap() {
                result = add_base(&result);
            }
        }
    }
    result
}
