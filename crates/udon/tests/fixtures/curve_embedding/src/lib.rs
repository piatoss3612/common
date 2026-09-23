//! Multiplies directly from both embedded entry layouts with no allocator.
#![no_std]
#![forbid(unsafe_code)]
#![deny(warnings)]

use udon::{
    curve::{
        AffinePoint, EisensteinScalar, EisensteinTableBatch, FixedBaseTable, Pallas, PastaCurve,
        Point, ProjectivePoint, Vesta,
        msm::{
            Bases, BasisSum, CoalescingKey, CoalescingPlan, IndexedCoalescingPlan, Input,
            PreparedScalars, ScalarStorage, Scratch, Selection, SuffixBasis,
            run::{BatchPlan, JobStorage, WorkerStorage},
        },
    },
    exec::{ExecutionOptions, SerialExecutor, TaskBudget},
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

const PALLAS_TABLES: record::Tables<'static, Pallas> = PALLAS.tables();
const VESTA_TABLES: record::Tables<'static, Vesta> = VESTA.tables();

fn exercise_curve<C: PastaCurve>(record: &record::Record<C>, tables: record::Tables<'_, C>) {
    let (table, cached, compact, compact_cached) = tables;
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
        PastaField::<_>::ONE.neg(),
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
        let scalars = [scalar, scalar.neg(), PastaField::from_u64(3)];
        let mut affine = [AffinePoint::GENERATOR; 33];
        let mut field = [PastaField::ZERO; 33];
        let expected = record.base.mul_projective(&PastaField::from_u64(3));
        assert_eq!(
            FixedBaseTable::sum(&[table; 3], &scalars, &mut affine, &mut field),
            expected
        );
        assert_eq!(
            FixedBaseTable::sum(&[cached; 3], &scalars, &mut affine, &mut field),
            expected
        );
        assert_eq!(FixedBaseTable::sum_scratch_len(&[table; 3]), Ok(192));
    }
    let batch = EisensteinTableBatch::bind(&record.compact);
    assert_eq!(batch.as_slice().as_ptr(), record.compact.as_ptr());
    assert_eq!(batch.get(0).unwrap().as_slice(), compact.as_slice());
    exercise_msm(record);
}

fn exercise_msm<C: PastaCurve>(record: &record::Record<C>) {
    const N: usize = 257;
    const OPTIONS: ExecutionOptions = ExecutionOptions::DEFAULT.with_memory_limit(8192);
    // Fixed caller-owned capacity; BatchPlan validates its required prefixes.
    let indices: [u32; N] = core::array::from_fn(|i| (i % 37) as u32);
    let scalars =
        core::array::from_fn::<_, N, _>(|i| PastaField::<C::Scalar>::from_u64(i as u64 + 1).neg());
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
        let input = selection.with_scalars(&scalars);
        let inputs = [input];
        let mut jobs = [JobStorage::EMPTY];
        let mut workers = [WorkerStorage::EMPTY];
        let plan = BatchPlan::new(&inputs, OPTIONS, &mut jobs, &mut workers).unwrap();
        let mut output = [ProjectivePoint::IDENTITY];
        plan.execute(&mut output, &SerialExecutor, scratch.reborrow());
        assert_eq!(output[0], expected);
    }
    let ordered = indices.map(|index| record.entries[index as usize].to_point());
    let mut keys = [CoalescingKey::EMPTY; N];
    let coalescing = CoalescingPlan::prepare(&ordered, &mut keys);
    let mut points = [Point::IDENTITY; N];
    let mut sums = [PastaField::ZERO; N];
    let input = coalescing.with_scalars(&scalars, &mut points, &mut sums);
    assert_eq!(
        input
            .execute(OPTIONS, &SerialExecutor, scratch.reborrow())
            .unwrap(),
        expected
    );
    let mut order = [0; N];
    let mut merged = [0; N];
    for bases in [
        Bases::Affine(&record.entries),
        Bases::Prepared(&record.cached),
    ] {
        let plan = IndexedCoalescingPlan::prepare(bases, &indices, &mut order).unwrap();
        let input = plan.with_scalars(&scalars, &mut merged, &mut sums);
        assert_eq!(
            input
                .execute(OPTIONS, &SerialExecutor, scratch.reborrow())
                .unwrap(),
            expected
        );
    }
    let region = BasisSum::prepare(&ordered);
    let constant = PastaField::<C::Scalar>::from_u64(7);
    let tail = [PastaField::<C::Scalar>::from_u64(11); 17];
    let row = udon::field::ConstantPrefix::new(N, &constant, &tail).unwrap();
    let mut deltas = [PastaField::ZERO; 17];
    let input = region.tail_corrections(row, &mut deltas);
    let extra = ordered[0].mul_projective(&PastaField::from_u64(13));
    let actual = region
        .sum()
        .mul_projective(&constant)
        .add(
            &input
                .execute(OPTIONS, &SerialExecutor, scratch.reborrow())
                .unwrap(),
        )
        .add(&extra);
    let dense = ordered
        .iter()
        .enumerate()
        .fold(ProjectivePoint::IDENTITY, |sum, (i, base)| {
            sum.add(&base.mul_projective(&if i < N - tail.len() {
                constant
            } else {
                tail[i - (N - tail.len())]
            }))
        })
        .add(&extra);
    assert_eq!(actual, dense);
    let mut suffix = [Point::IDENTITY; N];
    let basis = SuffixBasis::prepare(
        &ordered,
        &mut suffix,
        &mut [ProjectivePoint::IDENTITY; 7],
        &mut [PastaField::ZERO; 3],
    );
    let mut differences = [PastaField::ZERO; N];
    let input = basis.with_scalars(&scalars, &mut differences);
    assert_eq!(
        input
            .execute(OPTIONS, &SerialExecutor, scratch.reborrow())
            .unwrap(),
        expected
    );
    let row = core::array::from_fn::<_, N, _>(|i| (i / 11) as u128);
    let expected = row
        .iter()
        .zip(&ordered)
        .fold(ProjectivePoint::IDENTITY, |sum, (n, base)| {
            sum.add(&base.mul_projective(&PastaField::from_u64(*n as u64)))
        });
    let mut differences = [0; N];
    let input = basis
        .with_monotone_unsigned(&row, &mut differences)
        .unwrap();
    assert_eq!(
        input
            .execute(OPTIONS, &SerialExecutor, scratch.reborrow())
            .unwrap(),
        expected
    );
    // Embedded compact layouts must support retained preparation and selection
    // rebinding in a consumer without an allocator.
    let indices = [0; N];
    let mut retained = [ScalarStorage::ZERO; N];
    for bases in [
        Bases::Compact(EisensteinTableBatch::bind(&record.compact)),
        Bases::CompactPrepared(EisensteinTableBatch::bind(&record.compact_cached)),
    ] {
        let plan = IndexedCoalescingPlan::prepare(bases, &indices, &mut order).unwrap();
        let input = plan.with_scalars(&scalars, &mut merged, &mut sums);
        let sum = scalars.iter().fold(PastaField::ZERO, |sum, s| sum.add(s));
        assert_eq!(
            input
                .execute(OPTIONS, &SerialExecutor, scratch.reborrow())
                .unwrap(),
            record.base.mul_projective(&sum)
        );
        let selection = Selection::indexed(bases, &indices).unwrap();
        let sparse = scalars.map(|s| if s.is_odd() { s } else { PastaField::ZERO });
        let input = selection
            .with_nonzero_scalars(&sparse, &mut merged, &mut sums)
            .unwrap();
        let total = sparse.iter().fold(PastaField::ZERO, |sum, s| sum.add(s));
        assert_eq!(
            input
                .execute(OPTIONS, &SerialExecutor, scratch.reborrow())
                .unwrap(),
            record.base.mul_projective(&total)
        );
        for row in [scalars, scalars.map(|s| s.neg())] {
            let prepared =
                PreparedScalars::prepare(&row, &mut retained, TaskBudget::SERIAL, &SerialExecutor);
            let inputs = [selection.with_prepared_scalars(prepared)];
            let expected = row.iter().fold(PastaField::ZERO, |sum, s| sum.add(s));
            let mut jobs = [JobStorage::EMPTY];
            let mut workers = [WorkerStorage::EMPTY];
            let plan = BatchPlan::new(&inputs, OPTIONS, &mut jobs, &mut workers).unwrap();
            let mut output = [ProjectivePoint::IDENTITY];
            plan.execute(&mut output, &SerialExecutor, scratch.reborrow());
            assert_eq!(output[0], record.base.mul_projective(&expected));
        }
    }
}

fn exercise_srs<C: PastaCurve>(record: &record::SrsRecord<C>) {
    let domain = record.domain();
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
        PastaField::<C::Scalar>::from_u64(i as u64 + 3)
            .invert()
            .unwrap()
    });
    let mut evaluations = coefficient;
    reference::transform(&mut evaluations, &domain.root());
    let coefficient_input = Input::new(Bases::Prepared(&record.coefficient), &coefficient);
    let lagrange_input = Input::new(Bases::Prepared(&record.lagrange), &evaluations);
    let inputs = [coefficient_input, lagrange_input];
    let mut jobs = [JobStorage::EMPTY; 2];
    let mut workers = [WorkerStorage::EMPTY];
    let plan = BatchPlan::new(
        &inputs,
        ExecutionOptions::default(),
        &mut jobs,
        &mut workers,
    )
    .unwrap();
    let mut output = [ProjectivePoint::IDENTITY; 2];
    plan.execute(&mut output, &SerialExecutor, scratch.reborrow());
    let [left, right] = output;
    assert_eq!(
        left, right,
        "coefficient and Lagrange commitments must agree"
    );
}

pub fn exercise() {
    exercise_curve(PALLAS, PALLAS_TABLES);
    exercise_curve(VESTA, VESTA_TABLES);
    exercise_srs(PALLAS_SRS);
    exercise_srs(VESTA_SRS);
}
