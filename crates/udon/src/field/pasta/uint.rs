//! Modulus-independent unsigned integers and bit windows.

use super::ENCODED_SIZE;
use super::word::{add_limbs, compare_limbs};

/// An unsigned 256-bit integer used for field bit decomposition.
///
/// There is no Montgomery factor or attached modulus. Field implementations
/// own the range checks when converting this integer to a field element.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CanonicalUint {
    limbs: [u64; 4],
}

impl CanonicalUint {
    /// Constructs an integer from four little-endian words.
    #[inline]
    pub const fn from_limbs(limbs: [u64; 4]) -> Self {
        Self { limbs }
    }

    /// Decodes a 32-byte little-endian integer.
    #[inline]
    pub fn from_le_bytes(bytes: [u8; ENCODED_SIZE]) -> Self {
        let mut limbs = [0; 4];
        let mut index = 0;
        while index < limbs.len() {
            limbs[index] =
                u64::from_le_bytes(bytes[index * 8..(index + 1) * 8].try_into().unwrap());
            index += 1;
        }
        Self { limbs }
    }

    /// Encodes this integer as 32 little-endian bytes.
    #[inline]
    pub fn to_le_bytes(self) -> [u8; ENCODED_SIZE] {
        let mut bytes = [0; ENCODED_SIZE];
        let mut index = 0;
        while index < self.limbs.len() {
            bytes[index * 8..(index + 1) * 8].copy_from_slice(&self.limbs[index].to_le_bytes());
            index += 1;
        }
        bytes
    }

    /// Returns the four little-endian limbs.
    #[inline]
    pub const fn limbs(self) -> [u64; 4] {
        self.limbs
    }

    /// Returns the selected little-endian bit, if it is in range.
    #[inline]
    pub const fn bit(self, index: usize) -> Option<bool> {
        if index < ENCODED_SIZE * 8 {
            Some(self.limbs[index / 64] & (1 << (index % 64)) != 0)
        } else {
            None
        }
    }

    /// Returns the index of the most significant set bit, or `None` for zero.
    #[inline]
    pub const fn highest_set_bit(self) -> Option<usize> {
        let mut index = 4;
        while index > 0 {
            index -= 1;
            if self.limbs[index] != 0 {
                return Some(index * 64 + (63 - self.limbs[index].leading_zeros() as usize));
            }
        }
        None
    }

    /// Extracts up to 64 bits, including windows crossing a limb boundary.
    ///
    /// Returns `None` unless `1 <= width <= 64` and the entire window lies
    /// within bits `0..256`. Bit zero is the least significant bit.
    #[inline]
    pub const fn window(self, offset: usize, width: usize) -> Option<u64> {
        let Some(end) = offset.checked_add(width) else {
            return None;
        };
        if width == 0 || width > 64 || end > ENCODED_SIZE * 8 {
            return None;
        }

        let limb = offset / 64;
        let shift = offset % 64;
        let mut value = self.limbs[limb] >> shift;
        if shift != 0 && limb + 1 < self.limbs.len() {
            value |= self.limbs[limb + 1] << (64 - shift);
        }
        if width < 64 {
            value &= (1 << width) - 1;
        }
        Some(value)
    }

    /// Returns this integer shifted right, or zero when `shift >= 256`.
    #[inline]
    pub const fn shr(self, shift: usize) -> Self {
        if shift >= ENCODED_SIZE * 8 {
            return Self { limbs: [0; 4] };
        }
        let limb_shift = shift / 64;
        let bit_shift = shift % 64;
        let mut limbs = [0; 4];
        let mut index = 0;
        while index + limb_shift < 4 {
            limbs[index] = self.limbs[index + limb_shift] >> bit_shift;
            if bit_shift != 0 && index + limb_shift + 1 < 4 {
                limbs[index] |= self.limbs[index + limb_shift + 1] << (64 - bit_shift);
            }
            index += 1;
        }
        Self { limbs }
    }

    /// Extracts an arbitrary-width bit range as another fixed-width integer.
    ///
    /// Bit `offset` becomes bit zero of the result. Returns `None` for an
    /// empty range or a range extending beyond bit 255.
    #[inline]
    pub fn bit_slice(self, offset: usize, width: usize) -> Option<Self> {
        if width == 0 || offset.checked_add(width)? > ENCODED_SIZE * 8 {
            return None;
        }
        let mut value = self.shr(offset);
        let complete_limbs = width / 64;
        let remaining_bits = width % 64;
        let first_zero_limb = complete_limbs + usize::from(remaining_bits != 0);
        for limb in &mut value.limbs[first_zero_limb..] {
            *limb = 0;
        }
        if remaining_bits != 0 {
            value.limbs[complete_limbs] &= (1 << remaining_bits) - 1;
        }
        Some(value)
    }

    /// Adds a small unsigned integer, returning `None` on 256-bit overflow.
    #[inline]
    pub fn checked_add_u128(self, rhs: u128) -> Option<Self> {
        let (limbs, carry) = add_limbs(&self.limbs, &[rhs as u64, (rhs >> 64) as u64, 0, 0]);
        (carry == 0).then_some(Self { limbs })
    }

    /// Returns whether this integer is strictly smaller than `2^bits`.
    ///
    /// Zero fits in zero bits; every value fits when `bits >= 256`.
    #[inline]
    pub fn fits_in_bits(self, bits: usize) -> bool {
        bits >= ENCODED_SIZE * 8 || self.shr(bits).limbs == [0; 4]
    }

    /// Constructs `2^exponent` when it fits in 256 bits.
    #[inline]
    pub fn power_of_two(exponent: usize) -> Option<Self> {
        (exponent < ENCODED_SIZE * 8).then(|| {
            let mut limbs = [0; 4];
            limbs[exponent / 64] = 1 << (exponent % 64);
            Self { limbs }
        })
    }
}

impl Ord for CanonicalUint {
    #[inline]
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        compare_limbs(&self.limbs, &other.limbs)
    }
}

impl PartialOrd for CanonicalUint {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}
