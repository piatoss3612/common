use zakura_bento::const_arithmetic::{U256, m255, u256};

fn arithmetic(value: &U256, word: u64, hex: &str) {
    // The harness checks that every marked call has its own diagnostic.
    let _ = u256::from_hex!(hex); // rejected
    let _ = u256::ge!(value, &[0; 4]); // rejected
    let _ = u256::add_with_carry!(value, &[0; 4]); // rejected
    let _ = u256::sub_with_borrow!(value, &[0; 4]); // rejected
    let _ = u256::sub_u64!(value, 1); // rejected
    let _ = u256::shr!(value, 1); // rejected
    let _ = u256::div_exact_u64!(value, 1); // rejected
    let _ = u256::mul_wide!(value, &[0; 4]); // rejected
    let _ = u256::odd_cofactor!(value, 5); // rejected
    let _ = u256::tonelli_shanks_exponent!(value, 5); // rejected
    let _ = u256::round_shifted_ratio!(value, 1, 0); // rejected
    m255::assert_modulus!(value); // rejected
    let _ = m255::add!(value, &[0; 4], &[0; 4]); // rejected
    let _ = m255::pow2_mod!(value, 1); // rejected
    let _ = m255::one!(value); // rejected
    let _ = m255::r2!(value); // rejected
    let _ = m255::reduction_coefficient!(word); // rejected
    let _ = m255::reduce_wide!(value, &[0; 8]); // rejected
    let _ = m255::mul!(value, &[0; 4], &[0; 4]); // rejected
    let _ = m255::from_u256!(value, &[0; 4]); // rejected
    let _ = m255::from_u64!(value, 1); // rejected
    let _ = m255::to_u256!(value, &[0; 4]); // rejected
    let _ = m255::pow!(value, &[0; 4], &[0; 4]); // rejected
    let _ = m255::invert_prime!(value, &[0; 4]); // rejected
    let _ = m255::two_adic_root_of_unity!(value, 5, 5); // rejected
    let _ = m255::odd_order_generator!(value, 5, 5); // rejected
    let _ = m255::cube_root_of_unity!(value, 5); // rejected
    let _ = m255::two_inverse!(value); // rejected
    let _: [U256; 3] = m255::inverse_powers_of_two!(value); // rejected
    let _ = m255::inverse_powers_of_two!(value; 3); // rejected
    let _: [U256; 2] = m255::safegcd_corrections_62_64!(value); // rejected
    let _ = m255::safegcd_corrections_62_64!(value; 2); // rejected
    let _: [U256; 3] = m255::powers!(value, &[0; 4]); // rejected
    let _ = m255::powers!(value, &[0; 4]; 3); // rejected
    let _ = m255::powers!(&[97, 0, 0, 0], value; 3); // rejected
    let _: ([U256; 6], [U256; 6]) = m255::two_adic_root_tables!(value, 5, 5); // rejected
    let _ = m255::two_adic_root_tables!(value, 5, 5; 6); // rejected
}

fn main() {
    arithmetic(
        &[97, 0, 0, 0],
        97,
        "0x0000000000000000000000000000000000000000000000000000000000000061",
    );
}
