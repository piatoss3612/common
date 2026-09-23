use crate::field::{Fp, dot};

#[test]
#[should_panic(expected = "equal length")]
fn dot_rejects_unequal_lengths() {
    let _ = dot(&[<Fp>::ONE, <Fp>::ONE], &[<Fp>::ONE]);
}
