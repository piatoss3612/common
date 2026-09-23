//! Subgroups and fixed cosets, optional tables, and dirty transform scratch.

use super::*;
use crate::field::pasta::test_support::arbitrary_field;
use proptest::{
    prelude::*,
    test_runner::{FileFailurePersistence, TestCaseResult},
};

fn cases<M: PrimeModulus>()
-> impl proptest::strategy::Strategy<Value = (u32, Vec<PastaField<M>>, bool, u8)> {
    (0u32..=7).prop_flat_map(|log| {
        (
            Just(log),
            proptest::collection::vec(arbitrary_field::<M>(), 1 << log),
            any::<bool>(),
            any::<u8>(),
        )
    })
}

fn transforms<M: PrimeModulus>(
    log: u32,
    coefficients: Vec<PastaField<M>>,
    coset: bool,
    choices: u8,
) -> TestCaseResult {
    let domain = Domain::<PastaField<M>>::new(log).unwrap();
    let domain = if coset {
        domain.coset()
    } else {
        domain.subgroup()
    };
    let prepared = Prepared::new(domain);
    let plan = Tables {
        forward: (choices & 1 != 0).then_some(prepared.forward.as_slice()),
        inverse: (choices & 2 != 0).then_some(prepared.inverse.as_slice()),
        inverse_finish: (choices & 4 != 0).then_some(prepared.finish.as_slice()),
    }
    .bind(domain);
    let options = crate::fft::Strategy {
        tile_len: [1, 4, 64, 1024][usize::from(choices >> 4 & 3)],
        columns_per_task: 3,
        max_tasks: usize::from(choices >> 6) + 1,
    };
    let count = plan.scratch_requirements_with(options).unwrap();
    let sentinel = PastaField::from_u64(73);
    let mut scratch = vec![sentinel; count + 1];
    let mut output = coefficients.clone();
    for input in [
        coefficients.clone(),
        coefficients.into_iter().rev().collect(),
    ] {
        output.copy_from_slice(&input);
        plan.forward_with(&mut output, options, &SerialExecutor, &mut scratch)
            .unwrap();
        prop_assert_eq!(check_forward(&input, domain, &output), Ok(()));
        assert_loose_bound(&output);
        plan.inverse_with(&mut output, options, &SerialExecutor, &mut scratch)
            .unwrap();
        prop_assert_eq!(&output, &input);
        prop_assert_eq!(scratch[count], sentinel);
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
    fn fp_transforms((log, coefficients, coset, choices) in cases::<PallasBase>()) {
        transforms(log, coefficients, coset, choices)?;
    }

    #[test]
    fn fq_transforms((log, coefficients, coset, choices) in cases::<PallasScalar>()) {
        transforms(log, coefficients, coset, choices)?;
    }
}

#[test]
fn direct_oracle_rejects_a_corrupted_transform() {
    fn check<M: PrimeModulus>() {
        for log in [2, 3, 4] {
            let domain = Domain::<PastaField<M>>::new(log).unwrap();
            let coefficients = inputs(domain.size());
            let mut actual = coefficients.clone();
            Transform::new(domain.subgroup())
                .forward(
                    &mut actual,
                    crate::exec::ExecutionOptions::default(),
                    &SerialExecutor,
                    &mut [],
                )
                .unwrap();
            let coset = domain.subgroup();
            assert_eq!(check_forward(&coefficients, coset, &actual), Ok(()));
            if log == 3 {
                actual[0] = actual[0].add(&PastaField::<M>::ONE);
            }
            assert_eq!(
                check_forward(&coefficients, coset, &actual).is_err(),
                log == 3
            );
        }
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}
