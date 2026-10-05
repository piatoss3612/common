//! Small integer multiples using the Pasta modulus shape.

use super::{
    PastaField, PrimeModulus,
    word::{add_limbs, mac, sbb},
};

// Fold the bits above bit 253 using p = 2^254 + c. For k <= 8 and
// x < 2p, q <= 16 and q*c < p, so one addition of p repairs a borrow.
#[inline]
pub(super) fn multiply<M: PrimeModulus, const K: u64>(limbs: [u64; 4]) -> PastaField<M> {
    const {
        assert!(K > 0 && K <= 8);
    }
    let mut limbs = limbs;
    let mut carry = 0;
    let mut index = 0;
    while index < 4 {
        (limbs[index], carry) = mac(0, limbs[index], K, carry);
        index += 1;
    }
    let q = (carry << 2) | (limbs[3] >> 62);
    limbs[3] &= (1 << 62) - 1;
    let (c0, carry) = mac(0, M::MODULUS[0], q, 0);
    let (c1, c2) = mac(0, M::MODULUS[1], q, carry);
    let (r0, borrow) = sbb(limbs[0], c0, 0);
    let (r1, borrow) = sbb(limbs[1], c1, borrow);
    let (r2, borrow) = sbb(limbs[2], c2, borrow);
    let (r3, borrow) = sbb(limbs[3], 0, borrow);
    let mut limbs = [r0, r1, r2, r3];
    if borrow != 0 {
        limbs = add_limbs(&limbs, &M::MODULUS).0;
    }
    PastaField::from_montgomery(limbs)
}
