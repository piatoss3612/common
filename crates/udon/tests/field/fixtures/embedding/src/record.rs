//! The field record shared by the build script and its consumer.

use udon::field::{Fp, Fq, PastaField, PrimeModulus, Reduced};

#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
pub struct FieldValues {
    pub fp: [Fp; 8],
    pub fq: [Fq; 8],
    pub fp_reduced: [Fp<Reduced>; 8],
    pub fq_reduced: [Fq<Reduced>; 8],
}

pub fn samples<M: PrimeModulus>() -> [PastaField<M>; 8] {
    let mut modulus_plus_one = M::MODULUS;
    modulus_plus_one[0] += 1;
    let mut maximum = [0; 4];
    let mut carry = 0;
    for (output, limb) in maximum.iter_mut().zip(M::MODULUS) {
        *output = (limb << 1) | carry;
        carry = limb >> 63;
    }
    maximum[0] -= 1;
    let squares = [0, 1, 7, u64::MAX].map(|n| PastaField::<M>::from_u64(n).square());
    [
        squares[0],
        squares[1],
        squares[2],
        squares[3],
        PastaField::from_montgomery_limbs(M::MODULUS),
        PastaField::from_montgomery_limbs(modulus_plus_one),
        PastaField::from_montgomery_limbs(maximum),
        PastaField::<M>::ONE.neg(),
    ]
}
