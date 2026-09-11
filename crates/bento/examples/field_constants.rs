//! Derives roots for a power-of-two subgroup from public field parameters.

use zakura_bento::const_arithmetic::{U256, m255};

// This small field keeps the decoded constants easy to inspect:
// 97 is prime, 5 generates its nonzero residues, and 97 - 1 = 3 * 2^5.
const MODULUS: U256 = [97, 0, 0, 0];
const TWO_ADICITY: u32 = 5;
const ROOT: U256 = m255::two_adic_root_of_unity!(&MODULUS, 5, TWO_ADICITY);
const ROOT_INVERSE: U256 = m255::invert_prime!(&MODULUS, &ROOT);

fn main() {
    // Field constants are Montgomery residues. Decode them for comparison
    // with ordinary integers; exponents are already ordinary integers.
    assert_eq!(m255::to_u256!(&MODULUS, &ROOT), [28, 0, 0, 0]);
    assert_eq!(m255::to_u256!(&MODULUS, &ROOT_INVERSE), [52, 0, 0, 0]);

    let one = m255::one!(&MODULUS);
    const ORDER: u64 = 1 << TWO_ADICITY;
    assert_eq!(m255::pow!(&MODULUS, &ROOT, &[ORDER, 0, 0, 0]), one);
    assert_ne!(m255::pow!(&MODULUS, &ROOT, &[ORDER / 2, 0, 0, 0]), one);
    assert_eq!(m255::mul!(&MODULUS, &ROOT, &ROOT_INVERSE), one);
}
