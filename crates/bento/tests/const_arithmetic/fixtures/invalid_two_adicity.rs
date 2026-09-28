use zakura_bento::const_arithmetic::{U256, m255};

const INVALID: U256 = m255::odd_order_generator!(&[69, 0, 0, 0], 2, 6);

fn main() {
    std::hint::black_box(INVALID);
}
