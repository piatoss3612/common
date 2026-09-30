//! Integer exponents derived from the factorization of a field's group order.

use super::{U256, shr, sub_u64};

/// Returns the odd cofactor of `modulus - 1` as an ordinary integer.
///
/// The result `t` satisfies `modulus - 1 = t * 2^two_adicity`, with `t` odd.
/// Thus `two_adicity` must be exactly the number of trailing zero bits of
/// `modulus - 1`. Any odd 256-bit `modulus > 1` is accepted; primality and the
/// usual modular-arithmetic upper bound are not required.
///
/// # Panics
///
/// Panics if `two_adicity` is outside `1..256`, or if the stated factorization
/// does not hold with an odd `t`. This includes even moduli and `modulus <= 1`.
pub const fn odd_cofactor(modulus: &U256, two_adicity: u32) -> U256 {
    assert!(two_adicity >= 1 && two_adicity < 256);
    let minus_one = sub_u64(modulus, 1);
    let mut bit = 0;
    while bit < two_adicity {
        assert!(
            minus_one[(bit / 64) as usize] >> (bit % 64) & 1 == 0,
            "two_adicity exceeds the trailing zeros of modulus - 1"
        );
        bit += 1;
    }
    let cofactor = shr(&minus_one, two_adicity);
    assert!(cofactor[0] & 1 == 1, "two_adicity is not maximal");
    cofactor
}

/// Derives an initial exponent for Tonelli–Shanks square root computation.
///
/// Returns the ordinary integer `(t - 1) / 2`, where `t` is the
/// [`odd_cofactor`] in `modulus - 1 = t * 2^two_adicity`. It can be passed
/// directly as the exponent to [`m255::pow`](super::super::m255::pow) when that
/// operation supports the modulus. Computing this integer does not require
/// `modulus` to be prime or below `2^255`.
///
/// # Panics
///
/// Panics if [`odd_cofactor`] rejects `modulus` or `two_adicity`.
pub const fn tonelli_shanks_exponent(modulus: &U256, two_adicity: u32) -> U256 {
    shr(&sub_u64(&odd_cofactor(modulus, two_adicity), 1), 1)
}
