//! Multiplies directly from both embedded entry layouts with no allocator.
#![no_std]
#![forbid(unsafe_code)]
#![deny(warnings)]

use udon::{
    curve::{
        AffinePoint, CurveTableEntry, EisensteinScalar, EisensteinTableBatch, Pallas, PastaCurve,
        Point, ProjectivePoint, Vesta,
        msm::{
            ArithmeticOptions, Bases, BatchOptions, Input, PreparedScalars, ScalarStorage, Scratch,
            Selection,
            run::{BatchPlan, JobStorage, WorkerStorage},
        },
    },
    exec::{SerialExecutor, TaskBudget},
    fft::reference,
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
bento::embed_struct! {
    static PALLAS_SRS: record::SrsRecord<Pallas> =
        concat!(env!("OUT_DIR"), "/pallas-srs-", udon::stored_form!(), ".bin");
}
bento::embed_struct! {
    static VESTA_SRS: record::SrsRecord<Vesta> =
        concat!(env!("OUT_DIR"), "/vesta-srs-", udon::stored_form!(), ".bin");
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
    const OPTIONS: BatchOptions = BatchOptions::new(
        ArithmeticOptions::DEFAULT.with_max_terms_per_pass(core::num::NonZeroUsize::new(17)),
    )
    .with_memory_limit(8192);
    // Fixed caller-owned capacity; BatchPlan validates its required prefixes.
    let indices: [u32; N] = core::array::from_fn(|i| (i % 37) as u32);
    let scalars = core::array::from_fn::<_, N, _>(|i| PastaField::from_u64(i as u64 + 1).neg());
    let expected = indices
        .iter()
        .zip(&scalars)
        .fold(ProjectivePoint::IDENTITY, |sum, (&index, scalar)| {
            sum.add(&record.entries[index as usize].mul_projective(scalar))
        });
    let mut records = [ScalarStorage::ZERO; 64];
    let mut digits = [0; 4096];
    let mut affine = [AffinePoint::GENERATOR; 128];
    let mut projective = [ProjectivePoint::IDENTITY; 256];
    let mut field = [PastaField::ZERO; 256];
    let mut working_indices = [0; 256];
    let mut scratch = Scratch::new(
        &mut records,
        &mut digits,
        &mut affine,
        &mut projective,
        &mut field,
        &mut working_indices,
    );
    for bases in [
        Bases::Affine(&record.entries),
        Bases::Prepared(&record.cached),
    ] {
        let selection = Selection::indexed(bases, &indices).unwrap();
        let input = selection.with_scalars(&scalars).unwrap();
        let inputs = [input];
        let mut jobs = [JobStorage::EMPTY];
        let mut workers = [WorkerStorage::EMPTY];
        let plan = BatchPlan::new(&inputs, OPTIONS, &mut jobs, &mut workers).unwrap();
        let mut output = [ProjectivePoint::IDENTITY];
        plan.execute(&mut output, &SerialExecutor, scratch.reborrow())
            .unwrap();
        assert_eq!(output[0], expected);
    }
    // Embedded compact layouts must support retained preparation and selection
    // rebinding in a consumer without an allocator.
    let indices = [0; N];
    let mut retained = [ScalarStorage::ZERO; N];
    for bases in [
        Bases::Compact(EisensteinTableBatch::bind(&record.compact).unwrap()),
        Bases::CompactPrepared(EisensteinTableBatch::bind(&record.compact_cached).unwrap()),
    ] {
        let selection = Selection::indexed(bases, &indices).unwrap();
        for row in [scalars, scalars.map(|s| s.neg())] {
            let prepared =
                PreparedScalars::prepare(&row, &mut retained, TaskBudget::SERIAL, &SerialExecutor)
                    .unwrap();
            let inputs = [selection.with_prepared_scalars(prepared).unwrap()];
            let expected = row.iter().fold(PastaField::ZERO, |sum, s| sum.add(s));
            let mut jobs = [JobStorage::EMPTY];
            let mut workers = [WorkerStorage::EMPTY];
            let plan = BatchPlan::new(&inputs, OPTIONS, &mut jobs, &mut workers).unwrap();
            let mut output = [ProjectivePoint::IDENTITY];
            plan.execute(&mut output, &SerialExecutor, scratch.reborrow())
                .unwrap();
            assert_eq!(output[0], record.base.mul_projective(&expected));
        }
    }
}

fn exercise_srs<C: PastaCurve>(record: &record::SrsRecord<C>) {
    let domain = record.domain();
    for entry in record.coefficient.iter().chain(&record.lagrange) {
        let affine = entry.to_affine();
        let (x, y) = affine.coordinates();
        assert!(AffinePoint::<C>::from_xy(*x, *y).is_some() && entry.valid_cache());
    }
    // A direct DFT checks each embedded Lagrange basis and natural row order
    // independently of the generator's in-place butterfly schedule.
    let mut step = PastaField::ONE;
    for lagrange in &record.lagrange {
        let mut power = domain.size_inverse();
        let mut expected = ProjectivePoint::IDENTITY;
        for coefficient in &record.coefficient {
            expected = expected.add(&coefficient.to_affine().mul_projective(&power));
            power = power.mul(&step);
        }
        assert_eq!(
            lagrange.to_affine().to_point(),
            expected.to_point(),
            "embedded SRS must match natural Lagrange order"
        );
        step = step.mul(&domain.inverse_root());
    }
    let mut scalars = [ScalarStorage::ZERO; 64];
    let mut digits = [0; 4096];
    let mut affine = [AffinePoint::GENERATOR; 128];
    let mut projective = [ProjectivePoint::IDENTITY; 256];
    let mut field = [PastaField::ZERO; 256];
    let mut indices = [0; 256];
    let mut scratch = Scratch::new(
        &mut scalars,
        &mut digits,
        &mut affine,
        &mut projective,
        &mut field,
        &mut indices,
    );
    let coefficient = core::array::from_fn::<_, { record::SRS_SIZE }, _>(|i| {
        PastaField::from_u64(i as u64 + 3).invert().unwrap()
    });
    let mut evaluations = coefficient;
    reference::transform(&mut evaluations, &domain.root());
    let coefficient_input = Input::new(Bases::Prepared(&record.coefficient), &coefficient).unwrap();
    let lagrange_input = Input::new(Bases::Prepared(&record.lagrange), &evaluations).unwrap();
    let inputs = [coefficient_input, lagrange_input];
    let mut jobs = [JobStorage::EMPTY; 2];
    let mut workers = [WorkerStorage::EMPTY];
    let plan = BatchPlan::new(&inputs, BatchOptions::default(), &mut jobs, &mut workers).unwrap();
    let mut output = [ProjectivePoint::IDENTITY; 2];
    plan.execute(&mut output, &SerialExecutor, scratch.reborrow())
        .unwrap();
    let [left, right] = output;
    assert_eq!(
        left, right,
        "coefficient and Lagrange commitments must agree"
    );
}

pub fn exercise() {
    exercise_curve(PALLAS);
    exercise_curve(VESTA);
    exercise_srs(PALLAS_SRS);
    exercise_srs(VESTA_SRS);
}
