//! Multiplies directly from embedded affine entries with no allocator.
#![no_std]
#![forbid(unsafe_code)]
#![deny(warnings)]

use udon::{
    curve::{Pallas, PastaCurve, Point, Vesta},
    field::{CanonicalUint, PastaField},
};

pub mod record;

bento::embed_struct! {
    static PALLAS: record::Record<Pallas> =
        concat!(env!("OUT_DIR"), "/pallas-fixed-base-", udon::stored_form!(), ".bin");
}
bento::embed_struct! {
    static VESTA: record::Record<Vesta> =
        concat!(env!("OUT_DIR"), "/vesta-fixed-base-", udon::stored_form!(), ".bin");
}

fn exercise_curve<C: PastaCurve>(record: &record::Record<C>) {
    let table = record.table().expect("embedded table must match its base");
    assert_eq!(table.as_slice().as_ptr(), record.entries.as_ptr());
    for scalar in [
        PastaField::ZERO,
        PastaField::ONE,
        PastaField::ONE.neg(),
        PastaField::from_u64(128),
        PastaField::from_canonical_uint(CanonicalUint::from_limbs([
            u64::MAX,
            17,
            u64::MAX,
            1 << 61,
        ]))
        .unwrap(),
    ] {
        let actual = table.mul(&scalar).to_point();
        assert_eq!(actual, record.base.mul_projective(&scalar).to_point());
        assert_eq!(Point::<C>::from_bytes(actual.to_bytes()), Some(actual));
    }
}

pub fn exercise() {
    exercise_curve(PALLAS);
    exercise_curve(VESTA);
}
