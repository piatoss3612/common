//! Shared operations on individual 64-bit words.

/// Adds two words and an incoming carry, returning the low word and carry.
#[inline]
pub(super) const fn adc(a: u64, b: u64, carry: u64) -> (u64, u64) {
    let sum = a as u128 + b as u128 + carry as u128;
    (sum as u64, (sum >> 64) as u64)
}

/// Subtracts one word and a borrow bit; the outgoing borrow is zero or one.
#[inline]
pub(super) const fn sbb(a: u64, b: u64, borrow: u64) -> (u64, u64) {
    let difference = (a as u128).wrapping_sub(b as u128 + borrow as u128);
    (difference as u64, (difference >> 127) as u64)
}
