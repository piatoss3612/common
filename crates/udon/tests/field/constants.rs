use zakura_udon::{
    field::{CanonicalUint, Fp, Fq, Reduced},
    fp_hex, fq_hex,
};

#[test]
fn hex_macros_match_checked_integer_constructors() {
    const FP: Fp = fp_hex!("0x0000000000000000000000000000000100000000000000000123456789abcdef");
    const FQ: Fq = fq_hex!("0x0000000000000000000000000000000100000000000000000123456789ABCDEF",);
    let integer = CanonicalUint::from_limbs([0x0123_4567_89ab_cdef, 0, 1, 0]);
    assert_eq!(
        Fp::<Reduced>::from_canonical_uint(integer),
        Some(FP.reduce())
    );
    assert_eq!(
        Fq::<Reduced>::from_canonical_uint(integer),
        Some(FQ.reduce())
    );
    assert_eq!(FP.to_bytes(), FQ.to_bytes());
}

mod literals {
    pub use zakura_udon::{fp_hex as fp, fq_hex as fq};
}

#[test]
fn reexported_hex_macros_cover_canonical_boundaries() {
    const FP: [Fp; 3] = [
        literals::fp!("0x0000000000000000000000000000000000000000000000000000000000000000"),
        literals::fp!("0x0000000000000000000000000000000000000000000000000000000000000001"),
        literals::fp!("0x40000000000000000000000000000000224698fc094cf91b992d30ed00000000",),
    ];
    const FQ: [Fq; 3] = [
        literals::fq!("0x0000000000000000000000000000000000000000000000000000000000000000"),
        literals::fq!("0x0000000000000000000000000000000000000000000000000000000000000001"),
        literals::fq!("0x40000000000000000000000000000000224698FC0994A8DD8C46EB2100000000",),
    ];
    assert_eq!(
        (FP).iter().map(|value| value.reduce()).collect::<Vec<_>>(),
        ([<Fp>::ZERO, <Fp>::ONE, <Fp>::ONE.neg()])
            .iter()
            .map(|value| value.reduce())
            .collect::<Vec<_>>()
    );
    assert_eq!(
        (FQ).iter().map(|value| value.reduce()).collect::<Vec<_>>(),
        ([<Fq>::ZERO, <Fq>::ONE, <Fq>::ONE.neg()])
            .iter()
            .map(|value| value.reduce())
            .collect::<Vec<_>>()
    );
}

#[test]
fn root_accessors_remain_const() {
    const FP: [Fp; 2] = [
        Fp::root_of_unity(32).unwrap(),
        Fp::root_of_unity_inverse(32).unwrap(),
    ];
    const FQ: [Fq; 2] = [
        Fq::root_of_unity(32).unwrap(),
        Fq::root_of_unity_inverse(32).unwrap(),
    ];
    const FP_NONE: Option<Fp> = Fp::root_of_unity(33);
    const FQ_NONE: Option<Fq> = Fq::root_of_unity_inverse(u32::MAX);
    assert_eq!((FP[0].mul(&FP[1])).reduce(), (<Fp>::ONE).reduce());
    assert_eq!((FQ[0].mul(&FQ[1])).reduce(), (<Fq>::ONE).reduce());
    assert_ne!((FP[0].pow_u64(1 << 31)).reduce(), (<Fp>::ONE).reduce());
    assert_ne!((FQ[0].pow_u64(1 << 31)).reduce(), (<Fq>::ONE).reduce());
    assert_eq!((FP_NONE).map(|value| value.reduce()), None);
    assert_eq!((FQ_NONE).map(|value| value.reduce()), None);
}
