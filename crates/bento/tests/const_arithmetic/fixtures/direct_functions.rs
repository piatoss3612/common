use zakura_bento::const_arithmetic::{m255, u256};

fn main() {
    let _ = u256::from_hex; // rejected
    let _ = u256::ge; // rejected
    let _ = u256::add_with_carry; // rejected
    let _ = u256::sub_with_borrow; // rejected
    let _ = u256::sub_u64; // rejected
    let _ = u256::shr; // rejected
    let _ = u256::div_exact_u64; // rejected
    let _ = u256::mul_wide; // rejected
    let _ = u256::odd_cofactor; // rejected
    let _ = u256::tonelli_shanks_exponent; // rejected
    let _ = u256::round_shifted_ratio; // rejected
    let _ = m255::assert_modulus; // rejected
    let _ = m255::add; // rejected
    let _ = m255::pow2_mod; // rejected
    let _ = m255::one; // rejected
    let _ = m255::r2; // rejected
    let _ = m255::reduction_coefficient; // rejected
    let _ = m255::reduce_wide; // rejected
    let _ = m255::mul; // rejected
    let _ = m255::from_u256; // rejected
    let _ = m255::from_u64; // rejected
    let _ = m255::to_u256; // rejected
    let _ = m255::pow; // rejected
    let _ = m255::invert_prime; // rejected
    let _ = m255::two_adic_root_of_unity; // rejected
    let _ = m255::odd_order_generator; // rejected
    let _ = m255::cube_root_of_unity; // rejected
    let _ = m255::two_inverse; // rejected
    let _ = m255::inverse_powers_of_two::<3>; // rejected
    let _ = m255::safegcd_corrections_62_64::<2>; // rejected
    let _ = m255::powers::<3>; // rejected
    let _ = m255::two_adic_root_tables::<6>; // rejected
}
