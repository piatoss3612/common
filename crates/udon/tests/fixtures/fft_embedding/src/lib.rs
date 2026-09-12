//! Transforms directly from embedded tables using only stack-owned work buffers.
#![no_std]
#![forbid(unsafe_code)]
#![deny(warnings)]

use udon::{
    fft::{
        Direction, Domain, ExecutionOptions, Expansion, ExpansionOptions,
        ExpansionScaleNormalization, ExpansionScales, ResidueView, SerialExecutor, Strategy,
        Tables, TransformRequest, TwiddleTable,
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
        &FP_TABLES.factored,
    );
    exercise_field(
        FQ_TABLES.header,
        FQ_TABLES.tables(),
        &FQ_TABLES.residues,
        &FQ_TABLES.factored,
    );
}

fn exercise_field<M: PrimeModulus>(
    header: record::Header,
    tables: Tables<'_, M>,
    scales: &[PastaField<M>],
    factored: &[PastaField<M>],
) {
    let domain = Domain::for_size(record::SIZE).unwrap().subgroup();
    let extended = Domain::for_size(record::EXTENDED_SIZE)
        .unwrap()
        .coset(PastaField::zeta())
        .unwrap();
    header
        .validate(extended)
        .expect("embedded metadata must match the domain");
    let plan = tables
        .bind(domain)
        .unwrap()
        .validate()
        .expect("embedded FFT tables must match the domain")
        .plan();
    let twiddles = TwiddleTable::bind(record::TWIDDLES, factored)
        .unwrap()
        .validate()
        .expect("embedded factored twiddles must match the domain");
    const OPTIONS: ExecutionOptions = ExecutionOptions {
        tile_len: 4,
        columns_per_task: 2,
        max_tasks: 1,
    };
    const SCRATCH: usize = match OPTIONS.requirements(record::SIZE) {
        Ok(required) => required.field_elements,
        Err(_) => panic!("unsupported transform configuration"),
    };
    let mut scratch = [PastaField::ZERO; SCRATCH];
    assert_eq!(
        plan.scratch_requirements(OPTIONS).unwrap().field_elements,
        scratch.len()
    );
    let coefficients =
        core::array::from_fn::<_, { record::SIZE }, _>(|i| PastaField::from_u64(i as u64 + 1));
    let mut evaluations = [PastaField::ZERO; record::SIZE];
    plan.forward_into(
        &coefficients,
        &mut evaluations,
        OPTIONS,
        &SerialExecutor,
        &mut scratch,
    )
    .unwrap();
    let mut recovered = [PastaField::ZERO; record::SIZE];
    plan.inverse_into(
        &evaluations,
        &mut recovered,
        OPTIONS,
        &SerialExecutor,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(recovered, coefficients);
    plan.configure(
        TransformRequest::new(Direction::Forward),
        Strategy::serial(),
    )
    .unwrap()
    .with_twiddles(twiddles)
    .unwrap()
    .execute_into(&coefficients, &mut recovered, &SerialExecutor, &mut [])
    .unwrap();
    assert_eq!(recovered, evaluations);
    let scales = ExpansionScales::bind(
        record::SIZE,
        extended,
        ExpansionScaleNormalization::UnscaledInverse,
        scales,
    )
    .unwrap();
    let expansion = Expansion::new(plan, extended, None)
        .unwrap()
        .with_scales(scales)
        .unwrap();
    expansion
        .validate_scales()
        .expect("embedded residue scales must match the domain");
    let mut output = [PastaField::ZERO; record::EXTENDED_SIZE];
    const EXPANSION_OPTIONS: ExpansionOptions = ExpansionOptions {
        max_residue_tasks: 2,
        transform: OPTIONS,
    };
    const EXPANSION_SCRATCH: usize =
        match EXPANSION_OPTIONS.evaluation_requirements(record::SIZE, record::EXTENDED_SIZE) {
            Ok(required) => required.field_elements,
            Err(_) => panic!("unsupported expansion configuration"),
        };
    let mut scratch = [PastaField::ZERO; EXPANSION_SCRATCH];
    assert_eq!(
        expansion
            .evaluation_scratch(EXPANSION_OPTIONS)
            .unwrap()
            .field_elements,
        scratch.len()
    );
    expansion
        .evaluations(
            &evaluations,
            &mut output,
            EXPANSION_OPTIONS,
            &SerialExecutor,
            &mut scratch,
        )
        .unwrap();
    let view = ResidueView::new(&output, expansion.layout()).unwrap();
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
