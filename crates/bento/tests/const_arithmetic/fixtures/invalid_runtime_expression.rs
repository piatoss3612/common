use zakura_bento::const_arithmetic::m255;

fn main() {
    // Constant validation is mandatory even when the result is discarded.
    let _ = m255::one!(&[6, 0, 0, 0]);
}
