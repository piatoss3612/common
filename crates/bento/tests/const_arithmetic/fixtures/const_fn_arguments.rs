use zakura_bento::const_arithmetic::u256;

const fn shift(value: [u64; 4]) -> [u64; 4] {
    u256::shr!(&value, 1)
}

fn main() {
    let _ = const { shift([2, 0, 0, 0]) };
}
