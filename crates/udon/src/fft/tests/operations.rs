use super::*;

fn ordered<M: PrimeModulus>(values: &[PastaField<M>], order: InputOrder) -> Vec<PastaField<M>> {
    (0..values.len())
        .map(|i| {
            values[if order == InputOrder::Natural {
                i
            } else {
                reverse(i, values.len().ilog2())
            }]
        })
        .collect()
}

fn inverse_direct<M: PrimeModulus>(
    evaluations: &[PastaField<M>],
    domain: CosetDomain<M>,
    normalized: bool,
) -> Vec<PastaField<M>> {
    (0..domain.size())
        .map(|degree| {
            let step = domain.domain().inverse_root().pow_u64(degree as u64);
            let mut power = PastaField::ONE;
            let mut sum = PastaField::ZERO;
            for value in evaluations {
                sum = sum.add(&value.mul(&power));
                power = power.mul(&step);
            }
            sum = sum.mul(&domain.inverse_shift().pow_u64(degree as u64));
            if normalized {
                sum.mul(&domain.domain().size_inverse())
            } else {
                sum
            }
        })
        .collect()
}

fn strategy(backend: Backend) -> Strategy {
    Strategy {
        backend,
        execution: ExecutionOptions {
            tile_len: 4,
            columns_per_task: 3,
            max_tasks: 3,
        },
        budget: ResourceBudget::for_tasks(3),
        ..Strategy::serial()
    }
}

fn operations<M: PrimeModulus>() {
    for log in 0..=6 {
        for shift in [PastaField::ONE, PastaField::zeta(), PastaField::from_u64(7)] {
            let domain = Domain::<M>::new(log).unwrap().coset(shift).unwrap();
            let plan = Plan::without_tables(domain);
            let input = inputs(domain.size());
            let expected_forward = direct(&input, domain);
            let expected_inverse = inverse_direct(&input, domain, true);
            for backend in [Backend::InPlace, Backend::Blocked] {
                for direction in [Direction::Forward, Direction::Inverse] {
                    let expected = if direction == Direction::Forward {
                        &expected_forward
                    } else {
                        &expected_inverse
                    };
                    for input_order in [InputOrder::Natural, InputOrder::BitReversed] {
                        for output_order in [InputOrder::Natural, InputOrder::BitReversed] {
                            let request = TransformRequest {
                                input_order,
                                output_order,
                                ..TransformRequest::new(direction)
                            };
                            let operation = plan.configure(request, strategy(backend)).unwrap();
                            let mut scratch =
                                vec![PastaField::ONE; operation.requirements().scratch_fields + 1];
                            let original = ordered(&input, input_order);
                            let mut output = original.clone();
                            operation
                                .execute(&mut output, &SerialExecutor, &mut scratch)
                                .unwrap();
                            assert_eq!(
                                output,
                                ordered(expected, output_order),
                                "log={log}, backend={backend:?}, request={request:?}"
                            );
                            assert_canonical(&scratch);
                            assert_eq!(scratch.last(), Some(&PastaField::ONE));
                            for initialization in [
                                Initialization::Scatter,
                                Initialization::Gather,
                                Initialization::Blocked,
                            ] {
                                plan.configure(
                                    request,
                                    Strategy {
                                        initialization,
                                        ..strategy(backend)
                                    },
                                )
                                .unwrap()
                                .execute_into(&original, &mut output, &SerialExecutor, &mut scratch)
                                .unwrap();
                                assert_eq!(output, ordered(expected, output_order));
                            }
                        }
                    }
                }
            }
            for codelet in [Codelet::Radix4, Codelet::Radix8] {
                for direction in [Direction::Forward, Direction::Inverse] {
                    for output_order in [InputOrder::Natural, InputOrder::BitReversed] {
                        let request = TransformRequest {
                            output_order,
                            ..TransformRequest::new(direction)
                        };
                        let operation = plan
                            .configure(
                                request,
                                Strategy {
                                    codelet,
                                    ..strategy(Backend::InPlace)
                                },
                            )
                            .unwrap();
                        let mut output = input.clone();
                        operation
                            .execute(&mut output, &SerialExecutor, &mut [])
                            .unwrap();
                        let expected = if direction == Direction::Forward {
                            &expected_forward
                        } else {
                            &expected_inverse
                        };
                        assert_eq!(
                            output,
                            ordered(expected, output_order),
                            "codelet={codelet:?}, request={request:?}, log={log}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn schedules_and_orders_match_independent_transforms() {
    operations::<PallasBase>();
    operations::<PallasScalar>();
}

fn interpreted_codelets<M: PrimeModulus>() {
    use super::super::stages::{RADIX4, RADIX8};
    for steps in [&RADIX4[..], &RADIX8[..]] {
        let size = if steps.len() == 4 { 4 } else { 8 };
        let domain = Domain::<M>::for_size(size).unwrap().subgroup();
        let input = inputs(size);
        for inverse in [false, true] {
            for dif in [false, true] {
                let mut interpreted = if dif {
                    input.clone()
                } else {
                    ordered(&input, InputOrder::BitReversed)
                };
                for index in 0..steps.len() {
                    let step = steps[if dif { steps.len() - 1 - index } else { index }];
                    assert_eq!(step.right - step.left, step.block / 2);
                    assert!(step.exponent < step.block / 2 && step.right < size);
                    let root = if inverse {
                        domain.domain().inverse_root()
                    } else {
                        domain.domain().root()
                    };
                    let power = root.pow_u64((size / step.block * step.exponent) as u64);
                    let left = interpreted[step.left];
                    let right = interpreted[step.right];
                    if dif {
                        interpreted[step.left] = left.add(&right);
                        interpreted[step.right] = left.sub(&right).mul(&power);
                    } else {
                        let product = right.mul(&power);
                        interpreted[step.left] = left.add(&product);
                        interpreted[step.right] = left.sub(&product);
                    }
                }
                let expected = if inverse {
                    inverse_direct(&input, domain, false)
                } else {
                    direct(&input, domain)
                };
                assert_eq!(
                    interpreted,
                    if dif {
                        ordered(&expected, InputOrder::BitReversed)
                    } else {
                        expected
                    }
                );
            }
        }
    }
}

#[test]
fn generated_codelet_schedule_interpreter_matches_direct_sums() {
    interpreted_codelets::<PallasBase>();
    interpreted_codelets::<PallasScalar>();
}

fn prefixes_and_products<M: PrimeModulus>() {
    let domain = Domain::<M>::new(5)
        .unwrap()
        .coset(PastaField::from_u64(7))
        .unwrap();
    let plan = Plan::without_tables(domain);
    let values = inputs(domain.size());
    let factors = direct(&values, domain);
    for len in [0, 1, 2, 3, 7, 16, 31, 32] {
        for backend in [Backend::InPlace, Backend::Blocked] {
            for direction in [Direction::Forward, Direction::Inverse] {
                for output_order in [InputOrder::Natural, InputOrder::BitReversed] {
                    for inverse_scale in [InverseScale::Normalized, InverseScale::Unscaled] {
                        if direction == Direction::Forward
                            && inverse_scale == InverseScale::Unscaled
                        {
                            continue;
                        }
                        let request = TransformRequest {
                            support: InputSupport::Prefix(len),
                            output_order,
                            inverse_scale,
                            ..TransformRequest::new(direction)
                        };
                        let operation = plan.configure(request, strategy(backend)).unwrap();
                        let mut output = vec![PastaField::ONE; domain.size()];
                        let mut scratch =
                            vec![PastaField::ZERO; operation.requirements().scratch_fields];
                        operation
                            .execute_into(
                                &values[..len],
                                &mut output,
                                &SerialExecutor,
                                &mut scratch,
                            )
                            .unwrap();
                        let expected = if direction == Direction::Forward {
                            direct(&values[..len], domain)
                        } else {
                            inverse_direct(
                                &values[..len],
                                domain,
                                inverse_scale == InverseScale::Normalized,
                            )
                        };
                        assert_eq!(
                            output,
                            ordered(&expected, output_order),
                            "len={len}, backend={backend:?}, request={request:?}"
                        );
                        let mut in_place = values.clone();
                        operation
                            .execute(&mut in_place, &SerialExecutor, &mut scratch)
                            .unwrap();
                        assert_eq!(in_place, output);
                        if direction == Direction::Forward {
                            let layout = if output_order == InputOrder::Natural {
                                EvaluationLayout::Natural
                            } else {
                                EvaluationLayout::BitReversed
                            };
                            let factor_values = ordered(&factors, output_order);
                            let factor =
                                EvaluationView::bind(&factor_values, domain, layout).unwrap();
                            operation
                                .execute_product_into(
                                    &values[..len],
                                    factor,
                                    &mut output,
                                    &SerialExecutor,
                                    &mut scratch,
                                )
                                .unwrap();
                            let product: Vec<_> = expected
                                .iter()
                                .zip(&factors)
                                .map(|(a, b)| a.mul(b))
                                .collect();
                            assert_eq!(output, ordered(&product, output_order));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn inverse_prefix_scale_and_terminal_products_match_direct_sums() {
    prefixes_and_products::<PallasBase>();
    prefixes_and_products::<PallasScalar>();
}

fn power_tables<M: PrimeModulus, E: Executor>(executor: &E) {
    let domain = Domain::<M>::new(6)
        .unwrap()
        .coset(PastaField::from_u64(7))
        .unwrap();
    let plan = Plan::without_tables(domain);
    let input = inputs(domain.size());
    for size in [1, 8, 64, 256] {
        for inverse in [false, true] {
            for storage in [TwiddleStorage::Dense, TwiddleStorage::StagePacked] {
                let description = TwiddleDescription {
                    size,
                    inverse,
                    storage,
                };
                let mut values = vec![PastaField::ZERO; description.requirements().unwrap()];
                let table = TwiddleTable::prepare(description, &mut values).unwrap();
                for direction in [Direction::Forward, Direction::Inverse] {
                    let operation = plan
                        .configure(TransformRequest::new(direction), strategy(Backend::InPlace))
                        .unwrap()
                        .with_twiddles(table)
                        .unwrap();
                    let mut output = input.clone();
                    let mut scratch =
                        vec![PastaField::ZERO; operation.requirements().scratch_fields];
                    operation
                        .execute(&mut output, executor, &mut scratch)
                        .unwrap();
                    let expected = if direction == Direction::Forward {
                        direct(&input, domain)
                    } else {
                        inverse_direct(&input, domain, true)
                    };
                    assert_eq!(
                        output, expected,
                        "table={description:?}, direction={direction:?}"
                    );
                }
                if let Some(value) = values.first_mut() {
                    *value = PastaField::ZERO;
                    assert!(matches!(
                        TwiddleTable::bind(description, &values),
                        Err(FftError::InvalidTables)
                    ));
                }
            }
        }
    }
    let mut scales = vec![PastaField::ZERO; domain.size()];
    let scales = PowerTable::prepare(PastaField::ONE, domain.shift(), &mut scales).unwrap();
    let operation = plan
        .configure(
            TransformRequest::new(Direction::Forward),
            strategy(Backend::InPlace),
        )
        .unwrap()
        .with_forward_scales(scales)
        .unwrap();
    let mut output = input.clone();
    operation.execute(&mut output, &Threads, &mut []).unwrap();
    assert_eq!(output, direct(&input, domain));
}

#[test]
fn twiddle_shapes_strides_directions_and_coset_powers_are_compatible() {
    power_tables::<PallasBase, _>(&SerialExecutor);
    power_tables::<PallasScalar, _>(&SerialExecutor);
    power_tables::<PallasBase, _>(&Threads);
    power_tables::<PallasScalar, _>(&Threads);
}

#[test]
fn prepared_validation_and_batch_resources_precede_mutation() {
    const DESCRIPTION: OperationDescription = OperationDescription {
        size: 256,
        request: TransformRequest::new(Direction::Forward),
        strategy: Strategy::budgeted(ResourceBudget {
            scratch_fields: 0,
            table_bytes: 0,
            max_tasks: 3,
        }),
    };
    const REQUIRED: OperationRequirements = match DESCRIPTION.requirements(0) {
        Ok(r) => r,
        Err(_) => panic!("invalid sizing"),
    };
    assert_eq!(REQUIRED.backend, Backend::InPlace);
    assert_eq!(REQUIRED.scratch_fields, 0);
    let domain = Domain::<PallasBase>::new(8).unwrap().subgroup();
    let plan = Plan::without_tables(domain);
    let operation = plan
        .configure(DESCRIPTION.request, DESCRIPTION.strategy)
        .unwrap();
    assert_eq!(operation.requirements(), REQUIRED);
    let polynomial = inputs(domain.size());
    let expected = direct(&polynomial, domain);
    for backend in [Backend::InPlace, Backend::Blocked] {
        let operation = plan
            .configure(DESCRIPTION.request, strategy(backend))
            .unwrap();
        for count in [0, 1, 2, 5] {
            let mut batch = polynomial.repeat(count);
            let mut scratch =
                vec![Fp::ZERO; operation.batch_requirements(count).unwrap().field_elements];
            operation
                .execute_batch(&mut batch, &Threads, &mut scratch)
                .unwrap();
            assert_eq!(batch, expected.repeat(count));
            assert_canonical(&scratch);
        }
    }
    let mut values = polynomial.clone();
    let preserving = plan
        .configure(
            TransformRequest {
                input_policy: InputPolicy::Preserve,
                ..DESCRIPTION.request
            },
            DESCRIPTION.strategy,
        )
        .unwrap();
    assert_eq!(
        preserving.execute(&mut values, &SerialExecutor, &mut []),
        Err(FftError::InvalidExecution)
    );
    assert_eq!(values, polynomial);
    let prefix = plan
        .configure(
            TransformRequest {
                support: InputSupport::Prefix(1),
                ..DESCRIPTION.request
            },
            DESCRIPTION.strategy,
        )
        .unwrap();
    for invalid in [preserving, prefix] {
        for count in [0, 2] {
            assert_eq!(
                invalid.batch_requirements(count),
                Err(FftError::InvalidExecution)
            );
            let mut batch = polynomial.repeat(count);
            assert_eq!(
                invalid.execute_batch(&mut batch, &SerialExecutor, &mut []),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(batch, polynomial.repeat(count));
        }
    }
    let blocked = plan
        .configure(DESCRIPTION.request, strategy(Backend::Blocked))
        .unwrap();
    assert!(matches!(
        blocked.execute(&mut values, &SerialExecutor, &mut []),
        Err(FftError::ScratchTooSmall { .. })
    ));
    assert_eq!(values, polynomial);
    assert!(matches!(
        plan.configure(
            DESCRIPTION.request,
            Strategy {
                budget: DESCRIPTION.strategy.budget,
                ..strategy(Backend::Blocked)
            }
        ),
        Err(FftError::ResourceLimit)
    ));
    let wrong = domain.domain().coset(Fp::from_u64(7)).unwrap();
    let factor = EvaluationView::bind(&polynomial, wrong, EvaluationLayout::Natural).unwrap();
    assert_eq!(
        operation.execute_product_into(&polynomial, factor, &mut values, &SerialExecutor, &mut []),
        Err(FftError::InvalidLayout)
    );
    assert_eq!(values, polynomial);
}

#[test]
fn prepared_table_budgets_and_auto_selection_preserve_blocked_partitions() {
    let domain = Domain::<PallasBase>::new(6)
        .unwrap()
        .coset(Fp::from_u64(7))
        .unwrap();
    let prepared = Prepared::new(domain);
    let plan = Plan::new(prepared.tables().bind(domain).unwrap());
    let request = TransformRequest::new(Direction::Forward);
    let table_bytes = 4 * (domain.size() / 2) * core::mem::size_of::<Fp>();
    let mut selection = strategy(Backend::Auto);
    selection.budget.table_bytes = table_bytes;
    selection.budget.scratch_fields = 96;
    let operation = plan.configure(request, selection).unwrap();
    let required = operation.requirements();
    assert_eq!(required.backend, Backend::Blocked);
    assert_eq!(required.retained_table_bytes, table_bytes);
    assert_eq!(required.scratch_fields, 96);
    assert_eq!(required.per_worker_scratch_fields, 48);
    assert_eq!(required.scratch_partitions, 2);
    assert_eq!(required.max_tasks, 3);

    let input = inputs(domain.size());
    let expected = direct(&input, domain);
    let mut output = input.clone();
    let mut scratch = vec![Fp::ONE; required.scratch_fields];
    assert!(matches!(
        operation.execute(&mut output, &Threads, &mut scratch[..95]),
        Err(FftError::ScratchTooSmall {
            required: 96,
            provided: 95
        })
    ));
    assert_eq!(output, input);
    assert!(scratch.iter().all(|value| *value == Fp::ONE));
    operation
        .execute(&mut output, &Threads, &mut scratch)
        .unwrap();
    assert_eq!(output, expected);

    selection.budget.scratch_fields = 95;
    let fallback = plan.configure(request, selection).unwrap();
    assert_eq!(fallback.requirements().backend, Backend::InPlace);
    assert_eq!(fallback.requirements().scratch_fields, 0);
    assert_eq!(fallback.requirements().per_worker_scratch_fields, 0);
    assert_eq!(fallback.requirements().scratch_partitions, 0);
    assert!(matches!(
        plan.configure(
            request,
            Strategy {
                backend: Backend::Blocked,
                ..selection
            }
        ),
        Err(FftError::ResourceLimit)
    ));
    selection.budget.table_bytes -= 1;
    assert!(matches!(
        plan.configure(request, selection),
        Err(FftError::ResourceLimit)
    ));

    // The dense provider aliases a plan table; both borrows still count.
    let dense = TwiddleTable::bind(
        TwiddleDescription {
            size: domain.size(),
            inverse: false,
            storage: TwiddleStorage::Dense,
        },
        &prepared.forward,
    )
    .unwrap()
    .validate()
    .unwrap();
    let mut packed = [Fp::ZERO; 7];
    let local = TwiddleTable::prepare(
        TwiddleDescription {
            size: 8,
            inverse: false,
            storage: TwiddleStorage::StagePacked,
        },
        &mut packed,
    )
    .unwrap();
    let mut scales = vec![Fp::ZERO; domain.size()];
    let scales = PowerTable::prepare(Fp::ONE, domain.shift(), &mut scales).unwrap();
    for table in [dense, local] {
        assert!(matches!(
            plan.configure(request, strategy(Backend::Blocked))
                .unwrap()
                .with_twiddles(table),
            Err(FftError::InvalidExecution)
        ));
        let bytes = table_bytes
            + core::mem::size_of_val(table.as_slice())
            + core::mem::size_of_val(scales.as_slice());
        let mut selection = strategy(Backend::Auto);
        selection.budget.table_bytes = bytes;
        let operation = plan
            .configure(request, selection)
            .unwrap()
            .with_twiddles(table)
            .unwrap()
            .with_forward_scales(scales)
            .unwrap();
        assert_eq!(operation.requirements().backend, Backend::InPlace);
        assert_eq!(operation.requirements().scratch_fields, 0);
        assert_eq!(operation.requirements().retained_table_bytes, bytes);
        let mut output = input.clone();
        operation.execute(&mut output, &Threads, &mut []).unwrap();
        assert_eq!(output, expected);
        selection.budget.table_bytes -= 1;
        assert!(matches!(
            plan.configure(request, selection)
                .unwrap()
                .with_twiddles(table)
                .unwrap()
                .with_forward_scales(scales),
            Err(FftError::ResourceLimit)
        ));
    }
}

#[test]
fn prepared_parallel_panics_restore_all_field_buffers() {
    let domain = Domain::<PallasScalar>::new(8)
        .unwrap()
        .coset(PastaField::from_u64(7))
        .unwrap();
    let plan = Plan::without_tables(domain);
    for backend in [Backend::InPlace, Backend::Blocked] {
        for codelet in [Codelet::Radix2, Codelet::Radix4, Codelet::Radix8] {
            if backend != Backend::InPlace && codelet != Codelet::Radix2 {
                continue;
            }
            for direction in [Direction::Forward, Direction::Inverse] {
                for output_order in [InputOrder::Natural, InputOrder::BitReversed] {
                    let operation = plan
                        .configure(
                            TransformRequest {
                                output_order,
                                ..TransformRequest::new(direction)
                            },
                            Strategy {
                                codelet,
                                ..strategy(backend)
                            },
                        )
                        .unwrap();
                    let mut values = inputs(domain.size());
                    let mut scratch =
                        vec![PastaField::ONE; operation.requirements().scratch_fields];
                    let joins = CountJoins::default();
                    operation
                        .execute(&mut values, &joins, &mut scratch)
                        .unwrap();
                    for index in 0..joins.take() {
                        let mut values = inputs(domain.size());
                        let failed = catch_unwind(AssertUnwindSafe(|| {
                            operation.execute(
                                &mut values,
                                &FailAt {
                                    calls: AtomicUsize::new(0),
                                    index,
                                },
                                &mut scratch,
                            )
                        }));
                        assert!(failed.is_err());
                        assert_canonical(&values);
                        assert_canonical(&scratch);
                    }
                }
            }
        }
    }
}
