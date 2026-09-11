use zakura_bento::const_arithmetic::{U320, u256};

const INVALID: U320 = u256::round_shifted_ratio!(&[1, 0, 0, 0], 1, 320);

fn main() {
    std::hint::black_box(INVALID);
}
