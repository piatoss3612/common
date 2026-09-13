//! Multiplies directly from both embedded entry layouts with no allocator.
#![no_std]
#![forbid(unsafe_code)]
#![deny(warnings)]

use udon::{
    curve::{
        AffinePoint, EisensteinScalar, EisensteinTableBatch, Pallas, PastaCurve, Point,
        ProjectivePoint, Vesta,
        msm::{Bases, ExecutionOptions, Input, Requirements, Scratch},
    },
    exec::SerialExecutor,
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
    let (table, cached, compact, compact_cached) =
        record.tables().expect("embedded table must match its base");
    assert_eq!(table.as_slice().as_ptr(), record.entries.as_ptr());
    assert_eq!(cached.as_slice().as_ptr(), record.cached.as_ptr());
    assert_eq!(compact.as_slice().as_ptr(), record.compact.as_ptr());
    assert_eq!(
        compact_cached.as_slice().as_ptr(),
        record.compact_cached.as_ptr()
    );
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
        assert_eq!(cached.mul(&scalar).to_point(), actual);
        assert_eq!(compact.mul(&scalar).to_point(), actual);
        assert_eq!(compact_cached.mul(&scalar).to_point(), actual);
        let prepared = EisensteinScalar::new(&scalar);
        assert_eq!(compact.mul_prepared(&prepared).to_point(), actual);
        assert_eq!(compact_cached.mul_prepared(&prepared).to_point(), actual);
        assert_eq!(Point::<C>::from_bytes(actual.to_bytes()), Some(actual));
    }
    let batch = EisensteinTableBatch::bind(&record.compact).unwrap();
    assert_eq!(batch.as_slice().as_ptr(), record.compact.as_ptr());
    assert_eq!(batch.get(0).unwrap().as_slice(), compact.as_slice());
    exercise_msm(record);
}

fn exercise_msm<C: PastaCurve>(record: &record::Record<C>) {
    const N: usize = 257;
    const OPTIONS: ExecutionOptions = ExecutionOptions {
        max_terms_per_pass: core::num::NonZeroUsize::new(17),
        ..ExecutionOptions::SERIAL
    };
    // Both sealed curve pairings have the same element sizes. Assert that the
    // concrete sizing used for these arrays agrees with each instantiation.
    const R: Requirements = match Input::<Pallas>::requirements_for_len(N, OPTIONS) {
        Ok(r) => r,
        Err(_) => panic!("unsupported MSM size"),
    };
    assert_eq!(Input::<C>::requirements_for_len(N, OPTIONS).unwrap(), R);
    let indices: [u32; N] = core::array::from_fn(|i| (i % 37) as u32);
    let scalars = core::array::from_fn::<_, N, _>(|i| PastaField::from_u64(i as u64 + 1).neg());
    let expected = indices
        .iter()
        .zip(&scalars)
        .fold(ProjectivePoint::IDENTITY, |sum, (&index, scalar)| {
            sum.add(&record.entries[index as usize].mul_projective(scalar))
        });
    let mut digits = [0; R.digits];
    let mut affine = [AffinePoint::GENERATOR; R.affine];
    let mut projective = [ProjectivePoint::IDENTITY; R.projective];
    let mut field = [PastaField::ZERO; R.field];
    let mut working_indices = [0; R.indices];
    let mut scratch = Scratch {
        digits: &mut digits,
        affine: &mut affine,
        projective: &mut projective,
        field: &mut field,
        indices: &mut working_indices,
    };
    for bases in [
        Bases::Affine(&record.entries),
        Bases::Prepared(&record.cached),
    ] {
        let input = Input::indexed(bases, &indices, &scalars).unwrap();
        assert_eq!(
            input
                .execute(OPTIONS, &SerialExecutor, scratch.reborrow())
                .unwrap(),
            expected
        );
    }
}

pub fn exercise() {
    exercise_curve(PALLAS);
    exercise_curve(VESTA);
}
