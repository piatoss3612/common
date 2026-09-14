//! Binary ladder for short scalars and independent multiplication tests.

use super::{PastaCurve, ProjectivePoint};
use crate::field::CanonicalUint;
#[cfg(test)]
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

#[cfg(test)]
pub(super) fn multiply<C: PastaCurve>(
    scalar: &PastaField<C::Scalar>,
    add_base: impl Fn(&ProjectivePoint<C>) -> ProjectivePoint<C>,
) -> ProjectivePoint<C> {
    multiply_canonical(scalar.to_canonical_uint(), add_base)
}

/// Multiplies a fixed base by an unsigned integer.
///
/// Each `add_base` call must add the same base to the supplied point.
pub(super) fn multiply_canonical<C: PastaCurve>(
    scalar: CanonicalUint,
    add_base: impl Fn(&ProjectivePoint<C>) -> ProjectivePoint<C>,
) -> ProjectivePoint<C> {
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
