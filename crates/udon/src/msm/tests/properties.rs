//! Generated full-width and short scalars over dense and indexed point sets.

use super::*;
use crate::field::pasta::test_support::arbitrary_field;
use proptest::{
    prelude::*,
    test_runner::{FileFailurePersistence, TestCaseResult},
};
use std::format;

fn scalar<M: crate::field::PrimeModulus>() -> impl Strategy<Value = PastaField<M>> {
    prop_oneof![
        3 => arbitrary_field::<M>(),
        1 => any::<u64>().prop_map(PastaField::from_u64),
    ]
}

fn terms<M: crate::field::PrimeModulus>() -> impl Strategy<Value = Vec<(PastaField<M>, u8)>> {
    prop_oneof![
        3 => 0usize..=65,
        1 => proptest::sample::select(vec![127, 128, 129, 191, 192, 193, 511, 512, 513]),
    ]
    .prop_flat_map(|len| proptest::collection::vec((scalar::<M>(), any::<u8>()), len))
}

fn check<C: PastaCurve>(
    base_scalars: Vec<PastaField<C::Scalar>>,
    terms: Vec<(PastaField<C::Scalar>, u8)>,
    tasks: usize,
) -> TestCaseResult {
    let mut bases: Vec<_> = base_scalars
        .iter()
        .map(|k| Point::<C>::GENERATOR.mul_projective(k).to_point())
        .collect();
    // Repeated points, inverse pairs, and identity share buckets with the
    // independently generated points. Indices may repeat or omit any base.
    bases.extend([bases[0], bases[0].neg(), Point::IDENTITY]);
    let scalars: Vec<_> = terms.iter().map(|(k, _)| *k).collect();
    let indices: Vec<_> = terms
        .iter()
        .map(|(_, i)| u32::from(*i) % bases.len() as u32)
        .collect();
    let dense: Vec<_> = indices.iter().map(|i| bases[*i as usize]).collect();
    let options = ExecutionOptions::default().with_task_budget(TaskBudget::new(tasks).unwrap());
    for input in [
        Input::indexed(Bases::Points(&bases), &indices, &scalars).unwrap(),
        Input::new(Bases::Points(&dense), &scalars),
    ] {
        let required = input.requirements(options).unwrap();
        let mut buffers = Buffers::new(required);
        let actual = input
            .execute(options, &SerialExecutor, buffers.borrow())
            .unwrap();
        prop_assert_eq!(oracle::check(&input, actual), Ok(()));
        buffers.tails(required);
        // The parallel call receives the first call's dirty storage.
        let actual = input.execute(options, &Pool, buffers.borrow()).unwrap();
        prop_assert_eq!(oracle::check(&input, actual), Ok(()));
        buffers.tails(required);
    }
    Ok(())
}

proptest! {
    #![proptest_config(ProptestConfig {
        cases: 32,
        failure_persistence: Some(std::boxed::Box::new(FileFailurePersistence::WithSource("regressions"))),
        .. ProptestConfig::default()
    })]

    #[test]
    fn pallas_msm(
        bases in proptest::collection::vec(arbitrary_field::<<Pallas as PastaCurve>::Scalar>(), 1..=8),
        terms in terms::<<Pallas as PastaCurve>::Scalar>(),
        tasks in 1usize..=4,
    ) {
        check::<Pallas>(bases, terms, tasks)?;
    }

    #[test]
    fn vesta_msm(
        bases in proptest::collection::vec(arbitrary_field::<<Vesta as PastaCurve>::Scalar>(), 1..=8),
        terms in terms::<<Vesta as PastaCurve>::Scalar>(),
        tasks in 1usize..=4,
    ) {
        check::<Vesta>(bases, terms, tasks)?;
    }
}

#[test]
fn ladder_oracle_rejects_a_corrupted_result_at_the_booth_boundary() {
    fn check<C: PastaCurve>() {
        for n in [BOOTH_MIN - 1, BOOTH_MIN, BOOTH_MIN + 1] {
            let bases = vec![Point::<C>::GENERATOR; n];
            let scalars: Vec<_> = field_samples::<C::Scalar>().take(n).collect();
            let input = Input::new(Bases::Points(&bases), &scalars);
            let options = ExecutionOptions::default();
            let mut buffers = Buffers::new(input.requirements(options).unwrap());
            let actual = input
                .execute(options, &SerialExecutor, buffers.borrow())
                .unwrap();
            assert_eq!(oracle::check(&input, actual), Ok(()));
            let sentinel = if n == BOOTH_MIN {
                actual.add(&ProjectivePoint::GENERATOR)
            } else {
                actual
            };
            assert_eq!(oracle::check(&input, sentinel).is_err(), n == BOOTH_MIN);
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}
