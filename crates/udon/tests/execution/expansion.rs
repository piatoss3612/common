use super::{fft_pipeline::Banks, run_pool};
use std::num::NonZeroUsize;
use zakura_udon::{
    exec::{
        ExecutionOptions, SerialExecutor, TaskBudget,
        run::{Identity, TaskStorage},
    },
    fft::{
        Domain, ElementOrder, Expansion, ExpansionOrder, ExpansionStorage, InputSupport,
        InverseScale, StorageLayout, Transform,
        run::{ExpansionBank, ExpansionPlan, ExpansionRun},
    },
    field::{PallasBase, PallasScalar, PastaField, PrimeModulus},
};

fn check<M: PrimeModulus>() {
    const SLOTS: usize = 3;
    for size in [8, 64] {
        let base = Transform::new(Domain::<M>::for_size(size).unwrap().subgroup());
        for (residues, coset) in [1, 2, 8].into_iter().flat_map(|n| [(n, false), (n, true)]) {
            let domain = Domain::for_size(size * residues).unwrap();
            let domain = if coset {
                domain.coset()
            } else {
                domain.subgroup()
            };
            let expansion = Expansion::new(base, domain, None).unwrap();
            for storage in [
                ExpansionStorage::Coefficients,
                ExpansionStorage::ReuseOutput,
                ExpansionStorage::CoefficientWorkspace {
                    scale: InverseScale::Normalized,
                },
                ExpansionStorage::CoefficientWorkspace {
                    scale: InverseScale::Unscaled,
                },
                ExpansionStorage::DisposableInput {
                    scale: InverseScale::Normalized,
                },
                ExpansionStorage::DisposableInput {
                    scale: InverseScale::Unscaled,
                },
            ] {
                for order in [ExpansionOrder::Residues, ExpansionOrder::BitReversed] {
                    for input_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                        let coefficients: Vec<_> = (0..size)
                            .map(|i| PastaField::from_u64((i * i + 2) as u64))
                            .collect();
                        let support = if storage == ExpansionStorage::Coefficients
                            && input_order == ElementOrder::Natural
                        {
                            InputSupport::Prefix(size / 2)
                        } else {
                            InputSupport::Full
                        };
                        let count = if let InputSupport::Prefix(n) = support {
                            n
                        } else {
                            size
                        };
                        let mut input = coefficients.clone();
                        if storage != ExpansionStorage::Coefficients {
                            base.forward(&mut input, Default::default(), &SerialExecutor, &mut [])
                                .unwrap();
                        }
                        if input_order == ElementOrder::BitReversed {
                            let original = input.clone();
                            for (index, value) in input.iter_mut().enumerate() {
                                *value =
                                    original[index.reverse_bits() >> (usize::BITS - size.ilog2())];
                            }
                        }
                        let mut expected = vec![PastaField::ZERO; size * residues];
                        expansion
                            .coefficients(
                                &coefficients[..count],
                                &mut expected,
                                ExecutionOptions::default(),
                                &SerialExecutor,
                                &mut [],
                            )
                            .unwrap();
                        if order == ExpansionOrder::BitReversed {
                            let original = expected.clone();
                            for block in 0..residues {
                                for row in 0..size {
                                    let residue = if residues == 1 {
                                        0
                                    } else {
                                        block.reverse_bits() >> (usize::BITS - residues.ilog2())
                                    };
                                    let index = row.reverse_bits() >> (usize::BITS - size.ilog2());
                                    expected[block * size + row] = original[residue * size + index];
                                }
                            }
                        }
                        let plan = ExpansionPlan::new(
                            expansion,
                            storage,
                            order,
                            support,
                            input_order,
                            StorageLayout::Fragments {
                                length: NonZeroUsize::new(8).unwrap(),
                                whole_bank: false,
                            },
                            ExecutionOptions::default()
                                .with_task_budget(TaskBudget::new(3).unwrap()),
                        )
                        .unwrap();
                        let mut sizes = vec![size; 2 + residues + SLOTS + residues];
                        sizes[0] = count;
                        sizes[2 + residues..2 + residues + SLOTS].fill(plan.snapshot_fields());
                        let banks = Banks::new(&sizes, 8, 2 + residues..2 + residues + SLOTS);
                        banks.write(0, &input[..count]);
                        let factor: Vec<_> = (0..size)
                            .map(|i| PastaField::from_u64((i + 1) as u64))
                            .collect();
                        for block in 0..residues {
                            banks.write(2 + residues + SLOTS + block, &factor);
                            for (v, f) in expected[block * size..(block + 1) * size]
                                .iter_mut()
                                .zip(&factor)
                            {
                                *v = v.mul(f);
                            }
                        }
                        // Synchronous execution shares the planned semantics and storage modes.
                        for tasks in [1, 3, 4] {
                            let plan = ExpansionPlan::new(
                                expansion,
                                storage,
                                order,
                                support,
                                input_order,
                                StorageLayout::Contiguous,
                                ExecutionOptions::default()
                                    .with_task_budget(TaskBudget::new(tasks).unwrap()),
                            )
                            .unwrap();
                            let mut output = vec![PastaField::ZERO; expected.len()];
                            let mut scratch = vec![PastaField::ZERO; plan.scratch_fields()];
                            let mut workspace = vec![PastaField::ZERO; plan.coefficient_fields()];
                            let mut disposable = input[..count].to_vec();
                            let factors = factor.repeat(residues);
                            let view =
                                if matches!(storage, ExpansionStorage::DisposableInput { .. }) {
                                    Some(plan.execute_disposable(
                                        &mut disposable,
                                        &mut output,
                                        Some(&factors),
                                        &mut scratch,
                                        &SerialExecutor,
                                    ))
                                } else {
                                    plan.execute(
                                        &input[..count],
                                        &mut output,
                                        &mut workspace,
                                        Some(&factors),
                                        &mut scratch,
                                        &SerialExecutor,
                                    )
                                };
                            if let Some(view) = view {
                                let normalized: Vec<_> = view
                                    .as_slice()
                                    .iter()
                                    .map(|c| c.mul(&view.normalization_factor()))
                                    .collect();
                                assert_eq!(
                                    (normalized)
                                        .iter()
                                        .map(|value| value.reduce())
                                        .collect::<Vec<_>>(),
                                    (coefficients)
                                        .iter()
                                        .map(|value| value.reduce())
                                        .collect::<Vec<_>>()
                                );
                            }
                            assert_eq!(
                                (output)
                                    .iter()
                                    .map(|value| value.reduce())
                                    .collect::<Vec<_>>(),
                                (expected)
                                    .iter()
                                    .map(|value| value.reduce())
                                    .collect::<Vec<_>>(),
                                "contiguous size={size}, residues={residues}, coset={coset}, storage={storage:?}, order={order:?}, input={input_order:?}"
                            );
                        }
                        let mut ids = core::array::from_fn(|_| Identity::new());
                        let mut metadata = [const { [const { TaskStorage::EMPTY }; 3] }; SLOTS];
                        let mut run =
                            ExpansionRun::new(plan, true, &mut ids, &mut metadata).unwrap();
                        let map = |bank| match bank {
                            ExpansionBank::Input => 0,
                            ExpansionBank::Coefficients => 1,
                            ExpansionBank::Output(block) => 2 + block,
                        };
                        let mut published = vec![false; residues];
                        run_pool::scoped(4, 3, |pool| {
                            while !run.is_complete() {
                                for slot in 0..SLOTS {
                                    let mut cursor = 0;
                                    loop {
                                        let mut ready = [None];
                                        if !pool.available()
                                            || run.ready_slot_from(slot, cursor, &mut ready) == 0
                                        {
                                            break;
                                        }
                                        let request = ready[0].take().unwrap();
                                        cursor = request.task.key.index() + 1;
                                        let factor =
                                            if let ExpansionBank::Output(block) = request.values {
                                                2 + residues + SLOTS + block
                                            } else {
                                                0
                                            };
                                        if let Some(task) = run
                                            .try_claim(request.clone(), || {
                                                banks.acquire(
                                                    &request.task,
                                                    map(request.input),
                                                    map(request.values),
                                                    2 + residues + slot,
                                                    factor,
                                                )
                                            })
                                            .unwrap()
                                        {
                                            assert!(pool.submit((slot, task)).is_ok());
                                        }
                                    }
                                }
                                let (slot, receipt) =
                                    pool.receive().expect("admitted expansion progress");
                                let completed = run.complete(slot, receipt).unwrap();
                                assert_eq!(completed.task.error, None);
                                if let Some(block) = completed.residue {
                                    assert!(!published[block]);
                                    published[block] = true;
                                }
                                drop(completed);
                            }
                        });
                        assert!(published.iter().all(|v| *v));
                        let result: Vec<_> = (0..residues)
                            .flat_map(|block| banks.read(2 + block))
                            .collect();
                        assert_eq!(
                            (result)
                                .iter()
                                .map(|value| value.reduce())
                                .collect::<Vec<_>>(),
                            (expected)
                                .iter()
                                .map(|value| value.reduce())
                                .collect::<Vec<_>>(),
                            "size={size}, residues={residues}, storage={storage:?}, order={order:?}, input={input_order:?}"
                        );
                        if matches!(
                            storage,
                            ExpansionStorage::CoefficientWorkspace { .. }
                                | ExpansionStorage::DisposableInput { .. }
                        ) {
                            let scale: PastaField<M> = if matches!(
                                storage,
                                ExpansionStorage::CoefficientWorkspace {
                                    scale: InverseScale::Unscaled
                                } | ExpansionStorage::DisposableInput {
                                    scale: InverseScale::Unscaled
                                }
                            ) {
                                PastaField::from_u64(size as u64)
                            } else {
                                PastaField::ONE
                            };
                            assert_eq!(
                                (banks.read(
                                    if matches!(storage, ExpansionStorage::DisposableInput { .. }) {
                                        0
                                    } else {
                                        1
                                    }
                                ))
                                .iter()
                                .map(|value| value.reduce())
                                .collect::<Vec<_>>(),
                                (coefficients
                                    .iter()
                                    .map(|c| c.mul(&scale))
                                    .collect::<Vec<_>>())
                                .iter()
                                .map(|value| value.reduce())
                                .collect::<Vec<_>>()
                            );
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn incremental_expansion_preserves_liveness_orders_scales_and_products() {
    check::<PallasBase>();
    check::<PallasScalar>();
}
