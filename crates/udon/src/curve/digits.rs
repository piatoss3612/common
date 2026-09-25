//! Signed-digit recoders shared by fixed-base tables and MSM preparation.

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

/// Encodes a magnitude in signed width-`w` digits and a final carry.
///
/// Requires `w` in `2..=8`; digits are in `[-2^(w - 1), 2^(w - 1) - 1]`.
/// Only the first `ceil(128 / w)` digits are used; the remaining digits are zero.
pub(super) fn signed_window_digits(mut magnitude: u128, w: usize) -> ([i16; 64], bool) {
    // Zero-pad the last partial window. Pasta's lattice bounds are
    // |k1| < (a + b)/2 + 1 and |k2| < (b + d)/2 + 1, where d = a + b.
    // Only k2 at width 2 can carry, so only that layout stores a carry entry.
    let mut digits = [0; 64];
    let mut carry = 0;
    for digit in &mut digits[..128_usize.div_ceil(w)] {
        *digit = centered_digit(
            (magnitude & ((1 << w) - 1)) as u16,
            false,
            &mut carry,
            w as u32,
        );
        magnitude >>= w;
    }
    (digits, carry != 0)
}
