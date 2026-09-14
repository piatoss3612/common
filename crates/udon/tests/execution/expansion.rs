use super::{fft_pipeline::Banks, run_pool};
use std::num::NonZeroUsize;
use zakura_udon::{
    exec::{
        SerialExecutor,
        run::{Identity, TaskStorage},
    },
    fft::{
        Codelet, Domain, ElementOrder, Expansion, ExpansionOptions, ExpansionOrder,
        ExpansionStorage, InputSupport, InverseScale, Plan,
        run::{ExpansionBank, ExpansionPlan, ExpansionRun},
    },
    field::{PallasBase, PallasScalar, PastaField, PrimeModulus},
};

fn check<M: PrimeModulus>() {
    const SLOTS: usize = 3;
    for size in [8, 64] {
        let base = Plan::without_tables(Domain::<M>::for_size(size).unwrap().subgroup());
        for residues in [1, 2, 8] {
            let domain = Domain::for_size(size * residues)
                .unwrap()
                .coset(PastaField::from_u64(7))
                .unwrap();
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
                                ExpansionOptions::serial(),
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
                            NonZeroUsize::new(8).unwrap(),
                            Codelet::Radix4,
                        )
                        .unwrap();
                        let mut sizes = vec![size; 2 + residues + SLOTS + residues];
                        sizes[0] = count;
                        let banks = Banks::new(&sizes, 8);
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
                            result, expected,
                            "size={size}, residues={residues}, storage={storage:?}, order={order:?}, input={input_order:?}"
                        );
                        if matches!(
                            storage,
                            ExpansionStorage::CoefficientWorkspace { .. }
                                | ExpansionStorage::DisposableInput { .. }
                        ) {
                            let scale = if matches!(
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
                                banks.read(
                                    if matches!(storage, ExpansionStorage::DisposableInput { .. }) {
                                        0
                                    } else {
                                        1
                                    }
                                ),
                                coefficients
                                    .iter()
                                    .map(|c| c.mul(&scale))
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
