//! Transforms directly from embedded tables using only stack-owned work buffers.
#![no_std]
#![forbid(unsafe_code)]
#![deny(warnings)]

use udon::{
    exec::{ExecutionOptions, SerialExecutor, TaskBudget},
    fft::{
        Direction, Domain, EvaluationLayout, EvaluationView, Expansion,
        ExpansionScaleNormalization, ExpansionScales, InputStorage, StorageLayout, Tables,
        TransformRequest, TwiddleTable, run::FftPlan,
    },
    field::{PastaField, PrimeModulus},
};

pub mod record;

bento::embed_struct! {
    static FP_TABLES: record::FpTables =
        concat!(env!("OUT_DIR"), "/fp-fft-", udon::stored_form!(), ".bin");
}
bento::embed_struct! {
    static FQ_TABLES: record::FqTables =
        concat!(env!("OUT_DIR"), "/fq-fft-", udon::stored_form!(), ".bin");
}

pub fn exercise() {
    exercise_field(
        FP_TABLES.header,
        FP_TABLES.tables(),
        &FP_TABLES.residues,
        &FP_TABLES.packed,
    );
    exercise_field(
        FQ_TABLES.header,
        FQ_TABLES.tables(),
        &FQ_TABLES.residues,
        &FQ_TABLES.packed,
    );
}

fn exercise_field<M: PrimeModulus>(
    header: record::Header,
    tables: Tables<'_, M>,
    scales: &[PastaField<M>],
    packed: &[PastaField<M>],
) {
    let domain = Domain::for_size(record::SIZE).unwrap().subgroup();
    let extended = Domain::for_size(record::EXTENDED_SIZE)
        .unwrap()
        .coset(PastaField::ZETA)
        .unwrap();
    header
        .validate(extended)
        .expect("embedded metadata must match the domain");
    let plan = tables
        .bind(domain)
        .expect("embedded FFT tables must match the domain");
    let twiddles = TwiddleTable::bind(record::TWIDDLES, packed)
        .expect("embedded packed twiddles must match the domain");
    const OPTIONS: ExecutionOptions =
        ExecutionOptions::DEFAULT.with_task_budget(TaskBudget::new(2).unwrap());
    let mut scratch = [PastaField::ZERO; record::SIZE];
    assert!(plan.scratch_requirements(OPTIONS).unwrap().field_elements <= scratch.len());
    let coefficients =
        core::array::from_fn::<_, { record::SIZE }, _>(|i| PastaField::from_u64(i as u64 + 1));
    let mut evaluations = [PastaField::ZERO; record::SIZE];
    plan.execute(
        TransformRequest {
            input_storage: InputStorage::Preserve,
            ..TransformRequest::new(Direction::Forward)
        },
        Some(coefficients.as_slice().into()),
        &mut evaluations,
        OPTIONS,
        &SerialExecutor,
        &mut scratch,
    )
    .unwrap();
    let mut recovered = [PastaField::ZERO; record::SIZE];
    plan.execute(
        TransformRequest {
            input_storage: InputStorage::Preserve,
            ..TransformRequest::new(Direction::Inverse)
        },
        Some(evaluations.as_slice().into()),
        &mut recovered,
        OPTIONS,
        &SerialExecutor,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(recovered, coefficients);
    FftPlan::new(
        plan,
        TransformRequest {
            input_storage: InputStorage::Preserve,
            ..TransformRequest::new(Direction::Forward)
        },
        StorageLayout::Fragments {
            length: core::num::NonZeroUsize::new(4).unwrap(),
            whole_bank: false,
        },
        OPTIONS,
    )
    .unwrap()
    .with_twiddles(twiddles)
    .execute(
        Some(&coefficients),
        &mut recovered,
        None,
        &mut scratch,
        &SerialExecutor,
    )
    .unwrap();
    assert_eq!(recovered, evaluations);
    let scales = ExpansionScales::bind(
        record::SIZE,
        extended,
        ExpansionScaleNormalization::UnscaledInverse,
        scales,
    )
    .expect("embedded residue scales must match the domain");
    let expansion = Expansion::new(plan, extended, None)
        .unwrap()
        .with_scales(scales)
        .unwrap();
    let mut output = [PastaField::ZERO; record::EXTENDED_SIZE];
    assert!(
        expansion
            .evaluation_scratch(OPTIONS)
            .unwrap()
            .field_elements
            <= scratch.len()
    );
    expansion
        .evaluations(
            &evaluations,
            &mut output,
            OPTIONS,
            &SerialExecutor,
            &mut scratch,
        )
        .unwrap();
    let view = EvaluationView::bind(
        &output,
        extended,
        EvaluationLayout::Residues(expansion.layout()),
    )
    .unwrap();
    let mut point = extended.shift();
    for row in 0..extended.size() {
        let expected = coefficients
            .iter()
            .rev()
            .fold(PastaField::ZERO, |sum, coefficient| {
                sum.mul(&point).add(coefficient)
            });
        assert_eq!(*view.get(row).unwrap(), expected);
        point = point.mul(&extended.domain().root());
    }
}
