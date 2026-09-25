use super::bento::const_arithmetic::{U256, U320, U512, m255, u256};

const MODULUS: U256 =
    u256::from_hex!("0x0000000000000000000000000000000000000000000000000000000000000061",);
const A: U256 = m255::from_u64!(&MODULUS, 7);
const B: U256 = m255::from_u256!(&MODULUS, &[15, 0, 0, 0]);

pub trait Parameters {
    const MODULUS: U256;
}

pub struct SmallPrime;

impl Parameters for SmallPrime {
    const MODULUS: U256 = MODULUS;
}

// Generic associated constants and const parameters stay available in the
// inline const block, including when its result is used by a runtime function.
pub fn generic_inverses<M: Parameters, const N: usize>() -> [U256; N] {
    m255::inverse_powers_of_two!(&M::MODULUS; N)
}

pub fn generic_powers<M: Parameters, const N: usize>() -> [U256; N] {
    m255::powers!(&M::MODULUS, &m255::from_u64!(&M::MODULUS, 7); N)
}

pub fn check() {
    assert_eq!(MODULUS, [97, 0, 0, 0]);
    assert!(u256::ge!(&MODULUS, &[96, 0, 0, 0]));
    assert_eq!(
        u256::add_with_carry!(&[u64::MAX; 4], &[1, 0, 0, 0]),
        ([0; 4], 1)
    );
    assert_eq!(
        u256::sub_with_borrow!(&[0; 4], &[1, 0, 0, 0]),
        ([u64::MAX; 4], 1)
    );
    assert_eq!(u256::sub_u64!(&MODULUS, 1), [96, 0, 0, 0]);
    assert_eq!(u256::shr!(&[0, 3, 0, 0], 64), [3, 0, 0, 0]);
    assert_eq!(u256::div_exact_u64!(&[96, 0, 0, 0], 3), [32, 0, 0, 0]);
    let product: U512 = u256::mul_wide!(&[u64::MAX, 0, 0, 0], &[2, 0, 0, 0]);
    assert_eq!(product, [u64::MAX - 1, 1, 0, 0, 0, 0, 0, 0]);
    assert_eq!(u256::odd_cofactor!(&MODULUS, 5), [3, 0, 0, 0]);
    assert_eq!(u256::tonelli_shanks_exponent!(&MODULUS, 5), [1, 0, 0, 0]);
    let ratio: U320 = u256::round_shifted_ratio!(&[2, 0, 0, 0], 3, 0);
    assert_eq!(ratio, [2, 0, 0, 0, 0]);

    m255::assert_modulus!(&MODULUS);
    assert_eq!(m255::one!(&MODULUS), [61, 0, 0, 0]);
    assert_eq!(m255::r2!(&MODULUS), [35, 0, 0, 0]);
    assert_eq!(m255::pow2_mod!(&MODULUS, 8), [62, 0, 0, 0]);
    assert_eq!(
        MODULUS[0].wrapping_mul(m255::reduction_coefficient!(MODULUS[0])),
        u64::MAX
    );
    assert_eq!(A, [39, 0, 0, 0]);
    assert_eq!(B, [42, 0, 0, 0]);
    assert_eq!(m255::add!(&MODULUS, &A, &B), [81, 0, 0, 0]);
    assert_eq!(
        m255::reduce_wide!(&MODULUS, &[61, 0, 0, 0, 0, 0, 0, 0]),
        [1, 0, 0, 0]
    );
    assert_eq!(
        m255::to_u256!(&MODULUS, &m255::mul!(&MODULUS, &A, &B,)),
        [8, 0, 0, 0]
    );
    assert_eq!(m255::pow!(&MODULUS, &A, &[3, 0, 0, 0]), [68, 0, 0, 0]);
    assert_eq!(m255::invert_prime!(&MODULUS, &A), [78, 0, 0, 0]);
    assert_eq!(m255::two_adic_root_of_unity!(&MODULUS, 5, 5), [59, 0, 0, 0]);
    assert_eq!(m255::odd_order_generator!(&MODULUS, 5, 5), [1, 0, 0, 0]);
    assert_eq!(m255::cube_root_of_unity!(&MODULUS, 5), [1, 0, 0, 0]);
    assert_eq!(m255::two_inverse!(&MODULUS), [79, 0, 0, 0]);

    // Table lengths support both type inference and an explicit const argument.
    let powers: [U256; 4] = m255::powers!(&MODULUS, &A,);
    assert_eq!(
        powers,
        [[61, 0, 0, 0], [39, 0, 0, 0], [79, 0, 0, 0], [68, 0, 0, 0]]
    );
    assert_eq!(m255::powers!(&MODULUS, &A; 4,), powers);
    assert_eq!(generic_powers::<SmallPrime, 4>(), powers);
    assert_eq!(generic_powers::<SmallPrime, 0>(), [[0u64; 4]; 0]);
    let inverses: [U256; 3] = m255::inverse_powers_of_two!(&MODULUS,);
    assert_eq!(inverses, [[61, 0, 0, 0], [79, 0, 0, 0], [88, 0, 0, 0]]);
    assert_eq!(generic_inverses::<SmallPrime, 3>(), inverses);
    assert_eq!(generic_inverses::<SmallPrime, 0>(), [[0u64; 4]; 0]);
    let corrections: [U256; 2] = m255::safegcd_corrections_62_64!(&MODULUS);
    assert_eq!(corrections, [[50, 0, 0, 0], [6, 0, 0, 0]]);
    assert_eq!(m255::safegcd_corrections_62_64!(&MODULUS; 2,), corrections);
    let roots: ([U256; 6], [U256; 6]) = m255::two_adic_root_tables!(&MODULUS, 5, 5);
    assert_eq!(m255::two_adic_root_tables!(&MODULUS, 5, 5; 6,), roots);
    assert_eq!(roots.0[0], [61, 0, 0, 0]);
    assert_eq!(roots.1[0], [61, 0, 0, 0]);
    assert_eq!(roots.0[5], [59, 0, 0, 0]);
    assert_eq!(roots.1[5], [68, 0, 0, 0]);
}
