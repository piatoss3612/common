use zakura_bento::const_arithmetic::{U256, m255};

const INVALID: U256 = m255::pow!(&[97, 0, 0, 0], &[97, 0, 0, 0], &[0; 4]);

fn main() {
    std::hint::black_box(INVALID);
}
