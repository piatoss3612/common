use zakura_bento::const_arithmetic::m255;

fn main() {
    let _ = const {
        let modulus = [97, 0, 0, 0];
        m255::one!(&modulus)
    };
}
