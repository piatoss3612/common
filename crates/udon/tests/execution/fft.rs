use super::{fft_run::Arena, run_pool};
use std::num::NonZeroUsize;
use zakura_udon::{
    exec::{
        SerialExecutor,
        run::{Identity, TaskStorage},
    },
    fft::{
        Codelet, Direction, Domain, ElementOrder, InputSupport, InverseScale, Plan, Strategy,
        TransformRequest,
        run::{FftPlan, FftRun},
    },
    field::{PallasBase, PallasScalar, PastaField, PrimeModulus},
};

fn check<M: PrimeModulus>() {
    for size in [1, 8, 64, 1024] {
        let plan = Plan::without_tables(
            Domain::<M>::for_size(size)
                .unwrap()
                .coset(PastaField::from_u64(7))
                .unwrap(),
        );
        let original: Vec<_> = (0..size)
            .map(|i| PastaField::from_u64((i * i + 3) as u64))
            .collect();
        for direction in [Direction::Forward, Direction::Inverse] {
            for input_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                for output_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                    for (tile, codelet, separate, workers) in [
                        (1, Codelet::Radix2, false, 1),
                        (8, Codelet::Radix4, true, 3),
                        (32, Codelet::Radix8, false, 4),
                    ] {
                        let support = if separate && input_order == ElementOrder::Natural {
                            InputSupport::Prefix(size / 2)
                        } else {
                            InputSupport::Full
                        };
                        let request = TransformRequest {
                            input_order,
                            output_order,
                            support,
                            inverse_scale: if direction == Direction::Inverse && tile == 8 {
                                InverseScale::Unscaled
                            } else {
                                InverseScale::Normalized
                            },
                            ..TransformRequest::new(direction)
                        };
                        let operation = plan.configure(request, Strategy::serial()).unwrap();
                        let input = if let InputSupport::Prefix(len) = support {
                            &original[..len]
                        } else {
                            &original
                        };
                        let mut expected = vec![PastaField::ZERO; size];
                        operation
                            .execute_into(input, &mut expected, &SerialExecutor, &mut [])
                            .unwrap();
                        let product = separate;
                        let factor: Vec<_> = (0..size)
                            .map(|i| PastaField::from_u64((i + 1) as u64))
                            .collect();
                        if product {
                            for (value, factor) in expected.iter_mut().zip(&factor) {
                                *value = value.mul(factor);
                            }
                        }
                        let arithmetic = FftPlan::new(
                            plan,
                            request,
                            NonZeroUsize::new(tile).unwrap(),
                            codelet,
                            separate,
                        )
                        .unwrap();
                        let arena = Arena::new(arithmetic);
                        arena.write(&original);
                        assert!(arena.bytes() < 256 * 1024);
                        let mut identity = Identity::new();
                        let mut slots = [const { TaskStorage::EMPTY }; 5];
                        let mut run =
                            FftRun::new(arithmetic, product, &mut identity, &mut slots).unwrap();
                        run_pool::scoped(workers, 3, |pool| {
                            let mut ready = core::array::from_fn::<_, 5, _>(|_| None);
                            while !run.is_complete() {
                                let count = run.ready(&mut ready);
                                for request in ready[..count].iter().flatten() {
                                    if !pool.available() {
                                        break;
                                    }
                                    if let Some(task) = run
                                        .try_claim(request.clone(), || {
                                            arena.acquire(request, input, &factor)
                                        })
                                        .unwrap()
                                    {
                                        assert!(pool.submit(task).is_ok());
                                    }
                                }
                                let completed = run
                                    .complete(pool.receive().expect("admitted FFT must progress"))
                                    .unwrap();
                                assert_eq!(completed.error, None);
                                assert!(!run.is_failed());
                                drop(completed);
                            }
                        });
                        let result: Vec<_> =
                            arena.values.iter().flat_map(|s| s.read().clone()).collect();
                        assert_eq!(
                            result, expected,
                            "size={size}, request={request:?}, tile={tile}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn fragmented_orders_prefixes_cosets_products_and_codelets_match_contiguous() {
    check::<PallasBase>();
    check::<PallasScalar>();
}

fn blocked<M: PrimeModulus>() {
    use super::fft_pipeline::Banks;
    for (size, tile) in [(64, 8), (1024, 32)] {
        let plan = Plan::without_tables(
            Domain::<M>::for_size(size)
                .unwrap()
                .coset(PastaField::from_u64(7))
                .unwrap(),
        );
        let original: Vec<_> = (0..size)
            .map(|i| PastaField::from_u64((i * i + 3) as u64))
            .collect();
        for direction in [Direction::Forward, Direction::Inverse] {
            for input_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                for output_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                    for panels in [1, 3, 8] {
                        let request = TransformRequest {
                            input_order,
                            output_order,
                            ..TransformRequest::new(direction)
                        };
                        let mut expected = vec![PastaField::ZERO; size];
                        plan.configure(request, Strategy::serial())
                            .unwrap()
                            .execute_into(&original, &mut expected, &SerialExecutor, &mut [])
                            .unwrap();
                        let arithmetic = FftPlan::new(
                            plan,
                            request,
                            NonZeroUsize::new(tile).unwrap(),
                            Codelet::Radix4,
                            false,
                        )
                        .unwrap()
                        .with_columns(NonZeroUsize::MIN, NonZeroUsize::new(panels).unwrap())
                        .unwrap();
                        let banks = Banks::new(&[size, size], tile);
                        banks.write(0, &original);
                        let mut id = Identity::new();
                        let mut slots = [const { TaskStorage::EMPTY }; 5];
                        let mut run = FftRun::new(arithmetic, false, &mut id, &mut slots).unwrap();
                        run_pool::scoped(4, 3, |pool| {
                            while !run.is_complete() {
                                let mut cursor = 0;
                                loop {
                                    let mut ready = [None];
                                    if !pool.available() || run.ready_from(cursor, &mut ready) == 0
                                    {
                                        break;
                                    }
                                    let request = ready[0].take().unwrap();
                                    cursor = request.key.index() + 1;
                                    if let Some(task) = run
                                        .try_claim(request.clone(), || {
                                            banks.acquire(&request, 0, 0, 1, 0)
                                        })
                                        .unwrap()
                                    {
                                        assert!(pool.submit(task).is_ok());
                                    }
                                }
                                let published = run
                                    .complete(pool.receive().expect("bounded panels must progress"))
                                    .unwrap();
                                assert_eq!(published.error, None);
                                drop(published);
                            }
                        });
                        assert_eq!(banks.read(0), expected, "{request:?}, panels={panels}");
                        let contiguous = arithmetic.with_contiguous_permutation();
                        let mut values = original.clone();
                        let mut scratch = vec![PastaField::ZERO; contiguous.retained_fields()];
                        contiguous
                            .execute(
                                None,
                                &mut values,
                                None,
                                &mut scratch,
                                NonZeroUsize::new(3).unwrap(),
                                &SerialExecutor,
                            )
                            .unwrap();
                        assert_eq!(values, expected, "structured {request:?}, panels={panels}");
                    }
                    let request = TransformRequest {
                        input_order,
                        output_order,
                        ..TransformRequest::new(direction)
                    };
                    let stage = FftPlan::new(
                        plan,
                        request,
                        NonZeroUsize::new(tile).unwrap(),
                        Codelet::Radix8,
                        false,
                    )
                    .unwrap()
                    .with_contiguous_permutation();
                    assert_eq!(stage.retained_fields(), 0);
                    let mut values = original.clone();
                    stage
                        .execute(
                            None,
                            &mut values,
                            None,
                            &mut [],
                            NonZeroUsize::new(4).unwrap(),
                            &SerialExecutor,
                        )
                        .unwrap();
                    let mut expected = original.clone();
                    plan.configure(request, Strategy::serial())
                        .unwrap()
                        .execute(&mut expected, &SerialExecutor, &mut [])
                        .unwrap();
                    assert_eq!(values, expected);
                }
            }
        }
    }
}

#[test]
fn bounded_column_bands_and_structured_driver_match_stage_transforms() {
    blocked::<PallasBase>();
    blocked::<PallasScalar>();
}

fn sparse_tables<M: PrimeModulus>() {
    use zakura_udon::fft::{TwiddleDescription, TwiddleStorage, TwiddleTable};
    let nz = |n| NonZeroUsize::new(n).unwrap();
    let size = 64;
    let plan = Plan::without_tables(
        Domain::<M>::for_size(size)
            .unwrap()
            .coset(PastaField::from_u64(7))
            .unwrap(),
    );
    let original: Vec<_> = (0..size)
        .map(|i| PastaField::from_u64((i * i + 3) as u64))
        .collect();
    for storage in [TwiddleStorage::Dense, TwiddleStorage::StagePacked] {
        for table_size in [8, 64, 128] {
            for inverse in [false, true] {
                let description = TwiddleDescription {
                    size: table_size,
                    inverse,
                    storage,
                };
                let mut entries = vec![PastaField::ZERO; description.requirements().unwrap()];
                let table = TwiddleTable::prepare(description, &mut entries).unwrap();
                for direction in [Direction::Forward, Direction::Inverse] {
                    for scale in [InverseScale::Normalized, InverseScale::Unscaled] {
                        if direction == Direction::Forward && scale == InverseScale::Unscaled {
                            continue;
                        }
                        for prefix in [0, 1, 3, 21, 64] {
                            let request = TransformRequest {
                                support: InputSupport::Prefix(prefix),
                                inverse_scale: scale,
                                ..TransformRequest::new(direction)
                            };
                            let mut expected = vec![PastaField::ZERO; size];
                            plan.configure(request, Strategy::serial())
                                .unwrap()
                                .execute_into(
                                    &original[..prefix],
                                    &mut expected,
                                    &SerialExecutor,
                                    &mut [],
                                )
                                .unwrap();
                            for (tile, columns) in [(8, 3), (8, 9), (64, 3)] {
                                let arithmetic =
                                    FftPlan::new(plan, request, nz(tile), Codelet::Radix4, true)
                                        .unwrap()
                                        .with_columns(nz(columns), nz(3))
                                        .unwrap()
                                        .with_twiddles(table)
                                        .unwrap();
                                let mut values = vec![PastaField::ZERO; size];
                                let mut scratch =
                                    vec![PastaField::ZERO; arithmetic.retained_fields()];
                                arithmetic
                                    .execute(
                                        Some(&original[..prefix]),
                                        &mut values,
                                        None,
                                        &mut scratch,
                                        nz(4),
                                        &SerialExecutor,
                                    )
                                    .unwrap();
                                assert_eq!(
                                    values, expected,
                                    "{description:?}, {request:?}, tile={tile}, columns={columns}"
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn sparse_initialization_and_partial_panels_use_either_twiddle_direction() {
    sparse_tables::<PallasBase>();
    sparse_tables::<PallasScalar>();
}

#[test]
fn scatter_initialization_reads_bounded_consecutive_input_tiles() {
    use core::cell::RefCell;
    use zakura_udon::{
        exec::run::ReadView,
        fft::{
            Initialization,
            run::{Bank, Buffers, WorkKind},
        },
        field::Fp,
    };

    struct Source {
        values: [Fp; 64],
        reads: RefCell<Vec<usize>>,
    }
    impl ReadView<Fp> for Source {
        fn len(&self) -> usize {
            self.values.len()
        }
        fn get(&self, index: usize) -> Option<&Fp> {
            self.reads.borrow_mut().push(index);
            self.values.get(index)
        }
    }
    let source = Source {
        values: core::array::from_fn(|i| Fp::from_u64(i as u64 + 1)),
        reads: RefCell::new(Vec::new()),
    };
    let plan = Plan::without_tables(Domain::for_size(64).unwrap().subgroup());
    let request = TransformRequest::new(Direction::Inverse);
    let arithmetic = FftPlan::new(
        plan,
        request,
        NonZeroUsize::new(8).unwrap(),
        Codelet::Radix2,
        true,
    )
    .unwrap()
    .with_scatter_initialization();
    let mut values = [Fp::ZERO; 64];
    let mut identity = Identity::new();
    let mut slots = [const { TaskStorage::EMPTY }; 3];
    let mut run = FftRun::new(arithmetic, false, &mut identity, &mut slots).unwrap();
    for tile in 0..8 {
        let mut ready = [None];
        assert_eq!(run.ready(&mut ready), 1);
        let ready = ready[0].take().unwrap();
        assert_eq!(ready.kind, WorkKind::InitializeScatter);
        assert_eq!(ready.write, (Bank::Values, 0..64));
        let mut task = run
            .try_claim(ready, || {
                Some(Buffers {
                    values: &mut values,
                    pair: &mut [],
                    source: &source,
                    factor: &[],
                })
            })
            .unwrap()
            .unwrap();
        task.execute().unwrap();
        assert_eq!(
            *source.reads.borrow(),
            (0..(tile + 1) * 8).collect::<Vec<_>>()
        );
        assert_eq!(run.complete(task.finish()).unwrap().error, None);
    }
    assert_eq!(run.inflight(), 0);
    let mut ready = [None];
    run.ready(&mut ready);
    assert_eq!(ready[0].as_ref().unwrap().kind, WorkKind::Local);

    // The compatibility strategy and direct run driver agree with gather for
    // both directions, every physical order, and the single-task fused case.
    for direction in [Direction::Forward, Direction::Inverse] {
        for input_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
            for output_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                let request = TransformRequest {
                    input_order,
                    output_order,
                    ..TransformRequest::new(direction)
                };
                let mut expected = [Fp::ZERO; 64];
                plan.configure(request, Strategy::serial())
                    .unwrap()
                    .execute_into(&source.values, &mut expected, &SerialExecutor, &mut [])
                    .unwrap();
                for tile in [8, 64] {
                    let mut strategy = Strategy::serial();
                    strategy.initialization = Initialization::Scatter;
                    strategy.execution.tile_len = tile;
                    plan.configure(request, strategy)
                        .unwrap()
                        .execute_into(&source.values, &mut values, &SerialExecutor, &mut [])
                        .unwrap();
                    assert_eq!(values, expected);
                }
            }
        }
    }
}

#[test]
fn failed_and_cancelled_fft_tasks_drain_before_banks_are_reused() {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    use zakura_udon::{
        exec::run::{Outcome, ReadView, TaskError},
        fft::run::{Buffers, Resources},
        field::Fp,
    };

    struct Source([Fp; 64]);
    impl ReadView<Fp> for Source {
        fn len(&self) -> usize {
            self.0.len()
        }
        fn get(&self, index: usize) -> Option<&Fp> {
            assert_ne!(index, 1, "injected failure after the first copied field");
            self.0.get(index)
        }
    }
    struct Lease<'a> {
        values: &'a mut [Fp],
        source: &'a Source,
    }
    impl Resources<PallasBase> for Lease<'_> {
        fn buffers(&mut self) -> Buffers<'_, PallasBase> {
            Buffers {
                values: self.values,
                pair: &mut [],
                source: self.source,
                factor: &[],
            }
        }
    }

    let plan = Plan::without_tables(Domain::for_size(64).unwrap().subgroup());
    let arithmetic = FftPlan::new(
        plan,
        TransformRequest::new(Direction::Forward),
        NonZeroUsize::new(8).unwrap(),
        Codelet::Radix2,
        true,
    )
    .unwrap();
    let source = Source(core::array::from_fn(|i| Fp::from_u64(i as u64 + 1)));
    for cancel in [false, true] {
        let mut values = [Fp::ZERO; 64];
        let mut id = Identity::new();
        let mut slots = [const { TaskStorage::EMPTY }; 2];
        let mut run = FftRun::new(arithmetic, false, &mut id, &mut slots).unwrap();
        let mut ready = [None, None];
        assert_eq!(run.ready(&mut ready), 2);
        let (left, right) = values.split_at_mut(8);
        let first_request = ready[0].take().unwrap();
        let mut first = run
            .try_claim(first_request.clone(), || {
                Some(Lease {
                    values: left,
                    source: &source,
                })
            })
            .unwrap()
            .unwrap();
        let mut second = run
            .try_claim(ready[1].take().unwrap(), || {
                Some(Lease {
                    values: &mut right[..8],
                    source: &source,
                })
            })
            .unwrap()
            .unwrap();
        if !cancel {
            assert!(catch_unwind(AssertUnwindSafe(|| first.execute())).is_err());
        }
        second.execute().unwrap();
        let failed = run.complete(first.finish()).unwrap();
        assert_eq!(
            failed.outcome,
            if cancel {
                Outcome::Cancelled
            } else {
                Outcome::Failed
            }
        );
        assert!(run.is_failed());
        assert_eq!(run.inflight(), 1);
        assert_eq!(run.ready(&mut ready), 0);
        assert!(matches!(
            run.try_claim::<Lease<'_>>(first_request, || panic!("failed run acquired storage")),
            Err(TaskError::Failed)
        ));
        let drained = run.complete(second.finish()).unwrap();
        assert_eq!(drained.outcome, Outcome::Success);
        assert_eq!(run.inflight(), 0);
        assert!(!run.is_complete());
        for value in values {
            assert_eq!(Fp::from_bytes(value.to_bytes()), Some(value));
        }

        // Returned leases permit refill and a fresh invocation of the same plan.
        values.fill(Fp::ZERO);
        let mut scratch = vec![Fp::ZERO; arithmetic.retained_fields()];
        arithmetic
            .execute(
                Some(&source.0),
                &mut values,
                None,
                &mut scratch,
                NonZeroUsize::MIN,
                &SerialExecutor,
            )
            .unwrap();
        assert_eq!(
            values[0],
            source.0.iter().fold(Fp::ZERO, |sum, value| sum.add(value))
        );
    }
}
