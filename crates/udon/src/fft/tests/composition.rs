use super::*;

fn consume_coefficients<M: PrimeModulus>(view: CoefficientView<'_, M>, ordinary: &[PastaField<M>]) {
    for size in [ordinary.len(), ordinary.len() * 4] {
        for shift in [PastaField::ONE, PastaField::zeta(), PastaField::from_u64(7)] {
            let domain = Domain::<M>::for_size(size).unwrap().coset(shift).unwrap();
            let plan = Plan::without_tables(domain);
            let expected = direct(ordinary, domain);
            let mut output = vec![PastaField::ONE; size];
            plan.forward_prefix(
                view,
                &mut output,
                ExecutionOptions::serial(),
                &SerialExecutor,
                &mut [],
            )
            .unwrap();
            assert_eq!(output, expected);
            if size == ordinary.len() {
                plan.forward_into(
                    view,
                    &mut output,
                    ExecutionOptions::serial(),
                    &SerialExecutor,
                    &mut [],
                )
                .unwrap();
                assert_eq!(output, expected);
            }
            let mut scales = vec![PastaField::ZERO; size];
            let scales = PowerTable::prepare(PastaField::ONE, shift, &mut scales).unwrap();
            for backend in [Backend::InPlace, Backend::Blocked] {
                for initialization in [
                    Initialization::Scatter,
                    Initialization::Gather,
                    Initialization::Blocked,
                ] {
                    for order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                        let operation = plan
                            .configure(
                                TransformRequest {
                                    support: if size == ordinary.len() {
                                        InputSupport::Full
                                    } else {
                                        InputSupport::Prefix(ordinary.len())
                                    },
                                    output_order: order,
                                    ..TransformRequest::new(Direction::Forward)
                                },
                                Strategy {
                                    backend,
                                    initialization,
                                    execution: ExecutionOptions {
                                        tile_len: 4,
                                        columns_per_task: 2,
                                        max_tasks: 3,
                                    },
                                    budget: ResourceBudget::for_tasks(3),
                                    ..Strategy::serial()
                                },
                            )
                            .unwrap();
                        let factor_values = vec![PastaField::from_u64(11); size];
                        let layout = if order == ElementOrder::Natural {
                            EvaluationLayout::Natural
                        } else {
                            EvaluationLayout::BitReversed
                        };
                        let factor = EvaluationView::bind(&factor_values, domain, layout).unwrap();
                        for operation in [operation, operation.with_forward_scales(scales).unwrap()]
                        {
                            let mut scratch =
                                vec![PastaField::ONE; operation.requirements().scratch_fields + 1];
                            operation
                                .execute_coefficients(
                                    view,
                                    &mut output,
                                    &SerialExecutor,
                                    &mut scratch,
                                )
                                .unwrap();
                            let result = EvaluationView::bind(&output, domain, layout).unwrap();
                            for (row, value) in expected.iter().enumerate() {
                                assert_eq!(result.get(row), Some(value));
                            }
                            operation
                                .execute_coefficient_product(
                                    view,
                                    factor,
                                    &mut output,
                                    &SerialExecutor,
                                    &mut scratch,
                                )
                                .unwrap();
                            let result = EvaluationView::bind(&output, domain, layout).unwrap();
                            for (row, value) in expected.iter().enumerate() {
                                assert_eq!(result.get(row), Some(&value.mul(&factor_values[0])));
                            }
                            assert_eq!(scratch.last(), Some(&PastaField::ONE));
                        }
                    }
                }
            }

            let extended = Domain::for_size(size * 2).unwrap().coset(shift).unwrap();
            let base = Plan::without_tables(domain.domain().subgroup());
            let expected = direct(ordinary, extended);
            let mut output = vec![PastaField::ONE; extended.size()];
            for normalization in [
                None,
                Some(ExpansionScaleNormalization::Coefficients),
                Some(ExpansionScaleNormalization::UnscaledInverse),
            ] {
                let mut scale_values = vec![PastaField::ZERO; extended.size()];
                let scales = normalization.map(|normalization| {
                    ExpansionScales::prepare(size, extended, normalization, &mut scale_values)
                        .unwrap()
                });
                let expansion = Expansion::new(base, extended, scales).unwrap();
                expansion
                    .coefficients(
                        view,
                        &mut output,
                        ExpansionOptions::serial(),
                        &SerialExecutor,
                        &mut [],
                    )
                    .unwrap();
                let result = expansion.view(&output).unwrap();
                for (row, value) in expected.iter().enumerate() {
                    assert_eq!(result.get(row), Some(value));
                }
                let factor_values = vec![PastaField::from_u64(11); extended.size()];
                expansion
                    .short_product(
                        view,
                        expansion.view(&factor_values).unwrap(),
                        &mut output,
                        ExpansionOptions::serial(),
                        &SerialExecutor,
                        &mut [],
                    )
                    .unwrap();
                let result = expansion.view(&output).unwrap();
                for (row, value) in expected.iter().enumerate() {
                    assert_eq!(result.get(row), Some(&value.mul(&factor_values[0])));
                }
                for order in [ExpansionOrder::Residues, ExpansionOrder::BitReversed] {
                    let operation = expansion
                        .configure(
                            order,
                            ExpansionStorage::Coefficients,
                            ExpansionStrategy {
                                transform: ExecutionOptions {
                                    tile_len: 4,
                                    columns_per_task: 2,
                                    max_tasks: 3,
                                },
                                budget: ResourceBudget::for_tasks(3),
                            },
                        )
                        .unwrap();
                    let mut scratch =
                        vec![PastaField::ONE; operation.requirements().scratch_fields + 1];
                    operation
                        .execute_coefficients(view, &mut output, &SerialExecutor, &mut scratch)
                        .unwrap();
                    let result = operation.view(&output).unwrap();
                    for (row, value) in expected.iter().enumerate() {
                        assert_eq!(result.get(row), Some(value));
                    }
                    operation
                        .execute_product_into(
                            view,
                            operation.view(&factor_values).unwrap(),
                            &mut output,
                            &SerialExecutor,
                            &mut scratch,
                        )
                        .unwrap();
                    let result = operation.view(&output).unwrap();
                    for (row, value) in expected.iter().enumerate() {
                        assert_eq!(result.get(row), Some(&value.mul(&factor_values[0])));
                    }
                    assert_eq!(scratch.last(), Some(&PastaField::ONE));
                    let inner_order = if order == ExpansionOrder::Residues {
                        ElementOrder::Natural
                    } else {
                        ElementOrder::BitReversed
                    };
                    for residue in 0..expansion.layout().residues() {
                        let operation = expansion.residue(residue, inner_order).unwrap();
                        let mut output = vec![PastaField::ZERO; size];
                        operation
                            .coefficients(
                                view,
                                &mut output,
                                ExecutionOptions::serial(),
                                &SerialExecutor,
                                &mut [],
                            )
                            .unwrap();
                        for row in 0..size {
                            let index = if inner_order == ElementOrder::Natural {
                                row
                            } else {
                                reverse(row, size.ilog2())
                            };
                            assert_eq!(
                                output[index],
                                expected[residue + expansion.layout().residues() * row]
                            );
                        }
                    }
                }
            }
        }
    }
}

fn coefficient_composition<M: PrimeModulus>() {
    for size in [1, 8] {
        let domain = Domain::<M>::for_size(size).unwrap().subgroup();
        let mut ordinary = inputs(size);
        ordinary[0] = PastaField::from_u64(13);
        let evaluations = direct(&ordinary, domain);
        let expansion = Expansion::new(Plan::without_tables(domain), domain, None).unwrap();
        for scale in [InverseScale::Normalized, InverseScale::Unscaled] {
            let mut retained = evaluations.clone();
            let mut output = vec![PastaField::ZERO; size];
            let view = expansion
                .configure(
                    ExpansionOrder::Residues,
                    ExpansionStorage::DisposableInput { scale },
                    ExpansionStrategy::serial(),
                )
                .unwrap()
                .execute_disposable(&mut retained, &mut output, &SerialExecutor, &mut [])
                .unwrap();
            let before = view.as_slice().to_vec();
            consume_coefficients(view, &ordinary);
            assert_eq!(view.as_slice(), before);
        }
    }
}

#[test]
fn retained_views_feed_transforms_expansions_residues_and_products() {
    coefficient_composition::<PallasBase>();
    coefficient_composition::<PallasScalar>();
}

#[test]
fn coefficient_view_errors_preserve_buffers_and_skip_execution() {
    let domain = Domain::<PallasBase>::new(3).unwrap().subgroup();
    let plan = Plan::without_tables(domain);
    let coefficients = inputs(domain.size());
    let view = CoefficientView::normalized(&coefficients);
    let joins = CountJoins::default();
    let mut output = vec![PastaField::ONE; domain.size()];
    let mut scratch = vec![PastaField::ONE; 128];
    for request in [
        TransformRequest::new(Direction::Inverse),
        TransformRequest {
            input_order: ElementOrder::BitReversed,
            ..TransformRequest::new(Direction::Forward)
        },
    ] {
        let operation = plan.configure(request, Strategy::serial()).unwrap();
        assert_eq!(
            operation.execute_coefficients(view, &mut output, &joins, &mut scratch),
            Err(FftError::InvalidExecution)
        );
        assert_eq!(
            operation.execute_coefficient_product(
                view,
                EvaluationView::bind(&coefficients, domain, EvaluationLayout::Natural).unwrap(),
                &mut output,
                &joins,
                &mut scratch
            ),
            Err(FftError::InvalidExecution)
        );
    }
    let operation = plan
        .configure(
            TransformRequest::new(Direction::Forward),
            Strategy {
                backend: Backend::Blocked,
                execution: ExecutionOptions {
                    tile_len: 4,
                    columns_per_task: 2,
                    max_tasks: 2,
                },
                budget: ResourceBudget::for_tasks(2),
                ..Strategy::serial()
            },
        )
        .unwrap();
    assert!(matches!(
        operation.execute_coefficients(view, &mut output, &joins, &mut []),
        Err(FftError::ScratchTooSmall { .. })
    ));
    assert!(matches!(
        operation.execute_coefficients(
            CoefficientView::normalized(&coefficients[..1]),
            &mut output,
            &joins,
            &mut scratch
        ),
        Err(FftError::LengthMismatch { .. })
    ));
    let other = domain.domain().coset(PastaField::from_u64(7)).unwrap();
    let factor = EvaluationView::bind(&coefficients, other, EvaluationLayout::Natural).unwrap();
    assert_eq!(
        operation.execute_coefficient_product(view, factor, &mut output, &joins, &mut scratch),
        Err(FftError::InvalidLayout)
    );
    let expansion = Expansion::new(plan, domain, None).unwrap();
    let operation = expansion
        .configure(
            ExpansionOrder::Residues,
            ExpansionStorage::ReuseOutput,
            ExpansionStrategy::serial(),
        )
        .unwrap();
    assert_eq!(
        operation.execute_coefficients(view, &mut output, &joins, &mut scratch),
        Err(FftError::InvalidExecution)
    );
    let small = Plan::without_tables(Domain::new(2).unwrap().subgroup());
    assert!(matches!(
        small.forward_prefix(
            view,
            &mut output[..4],
            ExecutionOptions::serial(),
            &joins,
            &mut scratch
        ),
        Err(FftError::InvalidPrefix { .. })
    ));
    let small = Expansion::new(small, domain, None).unwrap();
    assert!(matches!(
        small.coefficients(
            view,
            &mut output,
            ExpansionOptions::serial(),
            &joins,
            &mut scratch
        ),
        Err(FftError::InvalidPrefix { .. })
    ));
    assert!(output.iter().chain(&scratch).all(|v| *v == PastaField::ONE));
    assert_eq!(joins.take(), 0);

    // Empty ordinary prefixes remain valid through the view conversion.
    plan.forward_prefix(
        CoefficientView::normalized(&[]),
        &mut output,
        ExecutionOptions::serial(),
        &joins,
        &mut scratch,
    )
    .unwrap();
    assert!(output.iter().all(|v| *v == PastaField::ZERO));
}
