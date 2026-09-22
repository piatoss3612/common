//! Uses embedded field values directly, without conversion or initialization.
#![forbid(unsafe_code)]
#![deny(warnings)]

use udon::field::{Fp, Fq, PallasBase, PallasScalar, PastaField, PrimeModulus, Reduced};

mod record;

bento::embed_struct! {
    static FIELD_VALUES: record::FieldValues =
        concat!(env!("OUT_DIR"), "/field-values-", udon::stored_form!(), ".bin");
}

bento::embed_array! {
    static VALUES: [Fp; 4] =
        concat!(env!("OUT_DIR"), "/fp-values-", udon::stored_form!(), ".bin");
}

// Embedded entries are already fields, even during constant evaluation.
const FP_ZERO: Fp = FIELD_VALUES.fp[0];
const FQ_LOOSE_ZERO: Fq = FIELD_VALUES.fq[4];
const FP_REDUCED: &Fp<Reduced> = &FIELD_VALUES.fp_reduced[2];

fn check<M: PrimeModulus>(loose: &[PastaField<M>; 8], reduced: &[PastaField<M, Reduced>; 8]) {
    let expected = record::samples::<M>();
    assert_eq!(bento::bytes_of(loose), bento::bytes_of(&expected));
    assert_eq!(*reduced, expected.map(|value| value.reduce()));
    assert_eq!(loose[4].montgomery_limbs(), M::MODULUS);
    assert!(loose[4].is_zero());
    assert_eq!(reduced[4].montgomery_limbs(), [0; 4]);
    for i in 0..8 {
        assert_eq!(
            loose[i].mul(&loose[i]).reduce(),
            reduced[i].square().reduce()
        );
        if i < 4 {
            assert_eq!(reduced[i].sqrt().unwrap().square().reduce(), reduced[i]);
        }
    }
}

fn main() {
    assert!(FP_ZERO.is_zero());
    assert_eq!(FQ_LOOSE_ZERO.montgomery_limbs(), PallasScalar::MODULUS);
    assert!(core::ptr::eq(FP_REDUCED, &FIELD_VALUES.fp_reduced[2]));
    assert_eq!(
        bento::bytes_of(VALUES),
        bento::bytes_of(&[<Fp>::ZERO, <Fp>::ONE, <Fp>::from_u64(7), <Fp>::ONE.neg()]),
    );
    assert_eq!(VALUES[2].square().reduce(), Fp::<Reduced>::from_u64(49));
    check(&FIELD_VALUES.fp, &FIELD_VALUES.fp_reduced);
    check(&FIELD_VALUES.fq, &FIELD_VALUES.fq_reduced);
    assert_eq!(FIELD_VALUES.fp[4].montgomery_limbs(), PallasBase::MODULUS);
}
