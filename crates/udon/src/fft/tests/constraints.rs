//! FFT layouts and execution respect caller resource limits.
use crate::{
    exec::{
        ExecutionOptions, SerialExecutor, TaskBudget,
        execution::{Identity, TaskError, TaskStorage},
    },
    fft::{
        Direction, Domain, ElementOrder, Expansion, ExpansionOrder, ExpansionStorage, InputStorage,
        InputSupport, StorageLayout, Transform, TransformRequest,
        execution::{ExpansionPlan, ExpansionRun, FftPlan},
    },
    field::{PallasBase, PallasScalar, PastaField, PrimeModulus},
};
use core::num::NonZeroUsize;
use std::{vec, vec::Vec};

fn fft<M: PrimeModulus>() {
    for size in [64, 2048, 4096] {
        let domain = Domain::<PastaField<M>>::for_size(size).unwrap().coset();
        let transform = Transform::new(domain);
        for tasks in [1, 4] {
            for limit in [0, 64 * 32, size * 32] {
                let options = ExecutionOptions::default()
                    .with_task_budget(TaskBudget::new(tasks).unwrap())
                    .with_memory_limit(limit);
                for length in [0, 1, 17, size] {
                    let input: Vec<_> = (0..length)
                        .map(|i| PastaField::from_u64(i as u64 + 2))
                        .collect();
                    for order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                        let request = TransformRequest {
                            input_storage: InputStorage::Preserve,
                            support: InputSupport::Prefix(length),
                            output_order: order,
                            ..TransformRequest::new(Direction::Forward)
                        };
                        let plan =
                            FftPlan::new(transform, request, StorageLayout::Contiguous, options)
                                .unwrap();
                        assert!(plan.retained_fields() * 32 <= limit);
                        let mut scratch =
                            vec![PastaField::from_u64(99); plan.retained_fields() + 3];
                        let mut output = vec![PastaField::ZERO; size];
                        plan.execute(
                            Some(&input),
                            &mut output,
                            None,
                            &mut scratch,
                            &SerialExecutor,
                        );
                        assert!(
                            scratch[plan.retained_fields()..]
                                .iter()
                                .all(|v| v.reduce() == PastaField::from_u64(99))
                        );
                        for row in [0, 1, size / 3, size - 1] {
                            let point = domain
                                .shift()
                                .mul(&domain.domain().root().pow_u64(row as u64));
                            let expected = input
                                .iter()
                                .rev()
                                .fold(PastaField::ZERO, |acc, coefficient| {
                                    acc.mul(&point).add(coefficient)
                                });
                            let index = if order == ElementOrder::Natural {
                                row
                            } else {
                                row.reverse_bits() >> (usize::BITS - size.ilog2())
                            };
                            assert_eq!((output[index]).reduce(), (expected).reduce());
                        }
                        let mut direct = vec![PastaField::ZERO; size];
                        transform
                            .execute(
                                request,
                                Some(input.as_slice().into()),
                                &mut direct,
                                options,
                                &SerialExecutor,
                                &mut scratch,
                            )
                            .unwrap();
                        assert_eq!(
                            (direct)
                                .iter()
                                .map(|value| value.reduce())
                                .collect::<Vec<_>>(),
                            (output)
                                .iter()
                                .map(|value| value.reduce())
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
    let base = Transform::new(Domain::<PastaField<M>>::for_size(4096).unwrap().subgroup());
    let expansion = Expansion::new(base, Domain::for_size(8192).unwrap().subgroup(), None).unwrap();
    let plan = ExpansionPlan::new(
        expansion,
        ExpansionStorage::CoefficientWorkspace {
            scale: zakura_udon::fft::InverseScale::Normalized,
        },
        ExpansionOrder::Residues,
        InputSupport::Full,
        ElementOrder::Natural,
        StorageLayout::Fragments {
            length: NonZeroUsize::new(64).unwrap(),
            whole_bank: false,
        },
        ExecutionOptions::default().with_memory_limit(2 * 4096 * 32),
    )
    .unwrap();
    assert!(plan.snapshot_fields() > 0);
    let mut identities = core::array::from_fn::<_, 3, _>(|_| Identity::new());
    let mut tasks = [const { [const { TaskStorage::EMPTY }; 1] }; 3];
    assert!(matches!(
        ExpansionRun::new(plan, false, &mut identities, &mut tasks),
        Err(TaskError::Storage)
    ));
}

#[test]
fn transform_selection_obeys_constraints_and_mathematical_layouts() {
    fft::<PallasBase>();
    fft::<PallasScalar>();
}
