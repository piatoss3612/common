use zakura_udon::{
    field::{
        CanonicalUint, Fp, Fq, PallasBase, PallasScalar, PastaField, PrimeModulus, ProductSum,
    },
    fp_hex, fq_hex,
};

#[test]
fn hex_macros_match_checked_integer_constructors() {
    const FP: Fp = fp_hex!("0x0000000000000000000000000000000100000000000000000123456789abcdef");
    const FQ: Fq = fq_hex!("0x0000000000000000000000000000000100000000000000000123456789ABCDEF",);
    let integer = CanonicalUint::from_limbs([0x0123_4567_89ab_cdef, 0, 1, 0]);
    assert_eq!(Fp::from_canonical_uint(integer), Some(FP));
    assert_eq!(Fq::from_canonical_uint(integer), Some(FQ));
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
    assert_eq!(FP, [Fp::ZERO, Fp::ONE, Fp::ONE.neg()]);
    assert_eq!(FQ, [Fq::ZERO, Fq::ONE, Fq::ONE.neg()]);
}

fn check_encoding_boundaries<M: PrimeModulus>() {
    let modulus = CanonicalUint::from_limbs(M::MODULUS);
    assert!(PastaField::<M>::from_canonical_uint(modulus).is_none());
    assert!(PastaField::<M>::from_bytes(modulus.to_le_bytes()).is_none());
    assert_eq!(
        PastaField::<M>::from_uint_reduced(modulus),
        PastaField::ZERO
    );
    assert!(
        std::panic::catch_unwind(|| { PastaField::<M>::from_montgomery_limbs(M::MODULUS) })
            .is_err()
    );

    for integer in [0, 1, 2, 3, u64::MAX] {
        let value = PastaField::<M>::from_u64(integer);
        assert_eq!(value.is_odd(), integer & 1 == 1);
        assert_eq!(PastaField::<M>::from_bytes(value.to_bytes()), Some(value));
        assert_eq!(
            PastaField::<M>::from_montgomery_limbs(value.montgomery_limbs()),
            value
        );
    }
    assert!(!PastaField::<M>::ONE.neg().is_odd());
    assert!(PastaField::<M>::from_u64(2).neg().is_odd());

    for signed in [i64::MIN, -(1 << 62), -1, 0, 1, 1 << 62, i64::MAX] {
        let value = PastaField::<M>::from_i64(signed);
        let magnitude = PastaField::<M>::from_u64(signed.unsigned_abs());
        if signed < 0 {
            assert_eq!(value.add(&magnitude), PastaField::ZERO);
        } else {
            assert_eq!(value, magnitude);
        }
    }
}

#[test]
fn checked_encodings_enforce_both_moduli() {
    check_encoding_boundaries::<PallasBase>();
    check_encoding_boundaries::<PallasScalar>();
}

#[test]
fn repeatedly_merging_product_sums_preserves_the_field_value() {
    fn check<M: PrimeModulus>() {
        let mut sum = ProductSum::<M>::new();
        sum.add_term(&PastaField::ONE);
        let mut expected = PastaField::<M>::ONE;
        for _ in 0..1024 {
            let mut doubled = ProductSum::new();
            doubled.merge(&sum);
            doubled.merge(&sum);
            sum = doubled;
            expected = expected.double();
        }
        assert_eq!(sum.finish(), expected);
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}

#[test]
#[cfg(target_endian = "little")]
fn stored_limbs_are_validated_before_field_arithmetic() {
    fn check<M: PrimeModulus>() {
        let value = PastaField::<M>::from_u64(7);
        let mut expected = [0; 32];
        for (chunk, limb) in expected.chunks_exact_mut(8).zip(value.montgomery_limbs()) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        assert_eq!(bento::bytes_of(&value.montgomery_limbs()), expected);
        static BYTES: bento::AlignedBytes<32> = bento::AlignedBytes({
            let mut bytes = [0; 32];
            bytes[0] = 1;
            bytes
        });
        assert_eq!(
            PastaField::<M>::from_montgomery_limbs(*BYTES.as_value::<[u64; 4]>()),
            PastaField::<M>::from_montgomery_limbs([1, 0, 0, 0])
        );
        // POD decoding establishes memory validity, so field construction must
        // still reject bytes that represent an unreduced Montgomery residue.
        static INVALID: bento::AlignedBytes<32> = bento::AlignedBytes([0xff; 32]);
        let limbs = *INVALID.as_value::<[u64; 4]>();
        assert!(
            std::panic::catch_unwind(|| PastaField::<M>::from_montgomery_limbs(limbs)).is_err()
        );
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}
