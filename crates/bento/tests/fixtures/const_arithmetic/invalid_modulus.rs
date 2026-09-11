use zakura_bento::const_arithmetic::{U256, m255};

const INVALID: U256 = m255::pow2_mod(&[2, 0, 0, 0], 0);

fn main() {
    std::hint::black_box(INVALID);
}
