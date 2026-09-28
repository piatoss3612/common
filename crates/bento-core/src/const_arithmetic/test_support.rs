//! Independent integer references and deterministic arithmetic inputs.

use num_bigint::BigUint;

use super::U256;

pub(super) fn integer(words: &[u64]) -> BigUint {
    words.iter().rev().fold(BigUint::from(0u8), |value, word| {
        (value << 64) + BigUint::from(*word)
    })
}

pub(super) fn limbs<const N: usize>(value: &BigUint) -> [u64; N] {
    let mut result = [0; N];
    for (index, word) in value.iter_u64_digits().enumerate() {
        result[index] = word;
    }
    result
}

pub(super) fn samples() -> [U256; 32] {
    let mut values = [[0; 4]; 32];
    values[1] = [1, 0, 0, 0];
    values[2] = [u64::MAX; 4];
    values[3] = [0xaaaa_aaaa_aaaa_aaaa; 4];
    values[4] = [0x5555_5555_5555_5555; 4];
    for index in 0..4 {
        values[5 + 2 * index][index] = 1;
        values[6 + 2 * index][index] = u64::MAX;
    }
    // A fixed seed makes mixed-word carry and borrow failures reproducible.
    let mut state = 0x932d_713b_8a9f_046du64;
    for value in &mut values[13..] {
        for word in value {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            *word = state;
        }
    }
    values
}
