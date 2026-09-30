//! Rounded fixed-point ratios used to derive scalar-decomposition constants.
//!
//! This is integer long division, independent of Montgomery representation.
//! The five-limb result accommodates ratios scaled by `2^384` for 256-bit
//! group orders and 128-bit lattice components.

use super::super::word::adc;
use super::super::{U256, U320};
use super::{add_with_carry, ge, sub_with_borrow};

/// Rounds a scaled integer ratio to the nearest unsigned 320-bit integer.
///
/// Returns `round(value * 2^shift / denominator)` as five little-endian limbs,
/// with the least significant word at index zero. Halfway cases round upward.
/// All inputs and the result are ordinary integers, without a Montgomery
/// factor. Such ratios can approximate division by a group order when deriving
/// constants for scalar decomposition.
///
/// `shift` must be a multiple of 64 in `0..=384`, and the rounded quotient
/// must fit 320 bits. Any nonzero 256-bit denominator is accepted, including
/// even and composite denominators.
///
/// # Panics
///
/// Panics if the denominator is zero, `shift` is unsupported, or the rounded
/// quotient does not fit.
///
/// # Examples
///
/// ```
/// # use zakura_bento_core as bento;
/// use bento::const_arithmetic::{U320, u256};
///
/// // A fixed-point approximation to 1/10, scaled by 2^64.
/// const TENTH: U320 = u256::round_shifted_ratio(&[10, 0, 0, 0], 1, 64);
/// assert_eq!(TENTH, [1_844_674_407_370_955_162, 0, 0, 0, 0]);
/// // With no scaling, the halfway value 5/2 rounds upward to 3.
/// assert_eq!(u256::round_shifted_ratio(&[2, 0, 0, 0], 5, 0), [3, 0, 0, 0, 0]);
/// ```
pub const fn round_shifted_ratio(denominator: &U256, value: u128, shift: u32) -> U320 {
    assert!(
        denominator[0] | denominator[1] | denominator[2] | denominator[3] != 0,
        "denominator must be nonzero"
    );
    assert!(
        shift.is_multiple_of(64) && shift <= 384,
        "shift must be a multiple of 64 at most 384"
    );
    let limb_shift = (shift / 64) as usize;
    let mut numerator = [0u64; 8];
    numerator[limb_shift] = value as u64;
    numerator[limb_shift + 1] = (value >> 64) as u64;

    let mut quotient = [0u64; 5];
    let mut remainder: U256 = [0; 4];
    let mut bit = 512;
    while bit > 0 {
        bit -= 1;
        // The old remainder is below the denominator, so appending one bit
        // needs at most one subtraction. Retain the 257th bit for full-width
        // denominators; a wrapped subtraction consumes that bit.
        let top = remainder[3] >> 63;
        let mut index = 3;
        while index > 0 {
            remainder[index] = remainder[index] << 1 | remainder[index - 1] >> 63;
            index -= 1;
        }
        remainder[0] = remainder[0] << 1 | numerator[bit / 64] >> (bit % 64) & 1;
        if top != 0 || ge(&remainder, denominator) {
            (remainder, _) = sub_with_borrow(&remainder, denominator);
            assert!(bit < 320, "quotient exceeds five limbs");
            quotient[bit / 64] |= 1 << (bit % 64);
        }
    }
    // Round half away from zero: bump the quotient when 2r >= denominator.
    let (doubled, carry) = add_with_carry(&remainder, &remainder);
    if carry == 1 || ge(&doubled, denominator) {
        let mut index = 0;
        let mut carry = 1u64;
        while index < 5 && carry == 1 {
            (quotient[index], carry) = adc(quotient[index], 0, carry);
            index += 1;
        }
        assert!(carry == 0, "rounded quotient exceeds five limbs");
    }
    quotient
}
