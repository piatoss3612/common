//! Uses embedded field values directly, without conversion or initialization.
#![forbid(unsafe_code)]
#![deny(warnings)]

use udon::field::{Fp, Fq};

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
const FQ_ONE: Fq = FIELD_VALUES.fq[0];

fn main() {
    assert_eq!(FP_ZERO, Fp::ZERO);
    assert_eq!(FQ_ONE, Fq::ONE);
    assert_eq!(*VALUES, [Fp::ZERO, Fp::ONE, Fp::from_u64(7), Fp::ONE.neg()]);
    assert_eq!(VALUES[2].mul(&VALUES[2]), Fp::from_u64(49));
    for (i, n) in [0, 1, 7, u64::MAX].into_iter().enumerate() {
        let fp = Fp::from_u64(n);
        let fq = Fq::from_u64(n);
        assert_eq!(FIELD_VALUES.fp[i], fp.square());
        assert_eq!(FIELD_VALUES.fq[i], fq.square().add(&Fq::ONE));
        assert_eq!(FIELD_VALUES.fp[i].sqrt().unwrap().square(), fp.square());
        assert_eq!(
            FIELD_VALUES.fq[i].sub(&Fq::ONE).sqrt().unwrap().square(),
            fq.square()
        );
    }
}
