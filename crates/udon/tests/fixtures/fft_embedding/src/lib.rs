//! Transforms directly from embedded tables using only stack-owned work buffers.
#![no_std]
#![forbid(unsafe_code)]
#![deny(warnings)]

use udon::{
    fft::{
        Domain, ExecutionOptions, Expansion, ExpansionOptions, Plan, ResidueView, SerialExecutor,
        Tables,
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
    exercise_field(FP_TABLES.tables(), &FP_TABLES.residues);
    exercise_field(FQ_TABLES.tables(), &FQ_TABLES.residues);
}

fn exercise_field<M: PrimeModulus>(tables: Tables<'_, M>, scales: &[PastaField<M>]) {
    let domain = Domain::for_size(record::SIZE).unwrap().subgroup();
    tables
        .validate(domain)
        .expect("embedded FFT tables must match the domain");
    let plan = Plan::new(domain, tables).unwrap();
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
    let extended = Domain::for_size(record::EXTENDED_SIZE)
        .unwrap()
        .coset(PastaField::zeta())
        .unwrap();
    let expansion = Expansion::new(plan, extended, Some(scales)).unwrap();
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
