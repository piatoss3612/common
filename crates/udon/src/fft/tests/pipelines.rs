use super::*;

fn expansion_strategy() -> ExpansionStrategy {
    ExpansionStrategy {
        transform: ExecutionOptions {
            tile_len: 4,
            columns_per_task: 2,
            max_tasks: 5,
        },
        budget: ResourceBudget::for_tasks(5),
    }
}

fn expansions<M: PrimeModulus>() {
    for log in [0, 2, 5] {
        let base = Plan::without_tables(Domain::<M>::new(log).unwrap().subgroup());
        let coefficients = inputs(base.domain().size());
        let evaluations = direct(&coefficients, base.domain());
        for extra in [0, 1, 3] {
            for shift in [PastaField::ONE, PastaField::zeta(), PastaField::from_u64(7)] {
                let domain = Domain::new(log + extra).unwrap().coset(shift).unwrap();
                let expansion = Expansion::new(base, domain, None).unwrap();
                let expected = direct(&coefficients, domain);
                for normalization in [
                    None,
                    Some(ExpansionScaleNormalization::Coefficients),
                    Some(ExpansionScaleNormalization::UnscaledInverse),
                ] {
                    let mut scale_values = vec![PastaField::ZERO; domain.size()];
                    let expansion = if let Some(normalization) = normalization {
                        let scales = ExpansionScales::prepare(
                            base.domain().size(),
                            domain,
                            normalization,
                            &mut scale_values,
                        )
                        .unwrap()
                        .validate()
                        .unwrap();
                        scales
                            .artifact()
                            .validate(base.domain().size(), domain, normalization)
                            .unwrap();
                        expansion.with_scales(scales).unwrap()
                    } else {
                        expansion
                    };
                    expansion.validate_scales().unwrap();
                    let mut legacy = vec![PastaField::ZERO; domain.size()];
                    expansion
                        .evaluations(
                            &evaluations,
                            &mut legacy,
                            ExpansionOptions::serial(),
                            &SerialExecutor,
                            &mut [],
                        )
                        .unwrap();
                    let legacy = ResidueView::new(&legacy, expansion.layout()).unwrap();
                    for (row, value) in expected.iter().enumerate() {
                        assert_eq!(legacy.get(row), Some(value));
                    }
                    for order in [ExpansionOrder::Residues, ExpansionOrder::BitReversed] {
                        for storage in [
                            ExpansionStorage::Coefficients,
                            ExpansionStorage::ReuseOutput,
                            ExpansionStorage::CoefficientWorkspace,
                            ExpansionStorage::DisposableInput,
                        ] {
                            let operation = expansion
                                .configure(order, storage, expansion_strategy())
                                .unwrap();
                            assert_eq!(
                                operation
                                    .description()
                                    .requirements(operation.requirements().retained_table_bytes)
                                    .unwrap(),
                                operation.requirements()
                            );
                            let mut output = vec![PastaField::ONE; domain.size()];
                            let mut scratch =
                                vec![PastaField::ONE; operation.requirements().scratch_fields + 1];
                            let mut working =
                                vec![PastaField::ONE; operation.requirements().coefficient_fields];
                            let mut disposable = evaluations.clone();
                            match storage {
                                ExpansionStorage::Coefficients => operation.execute_into(
                                    &coefficients,
                                    &mut output,
                                    &SerialExecutor,
                                    &mut scratch,
                                ),
                                ExpansionStorage::ReuseOutput => operation.execute_into(
                                    &evaluations,
                                    &mut output,
                                    &SerialExecutor,
                                    &mut scratch,
                                ),
                                ExpansionStorage::CoefficientWorkspace => operation
                                    .execute_with_workspace(
                                        &evaluations,
                                        &mut output,
                                        &mut working,
                                        &SerialExecutor,
                                        &mut scratch,
                                    ),
                                ExpansionStorage::DisposableInput => operation.execute_disposable(
                                    &mut disposable,
                                    &mut output,
                                    &SerialExecutor,
                                    &mut scratch,
                                ),
                            }
                            .unwrap();
                            let view = operation.view(&output).unwrap();
                            for (row, value) in expected.iter().enumerate() {
                                assert_eq!(
                                    view.get(row),
                                    Some(value),
                                    "log={log}, extra={extra}, normalization={normalization:?}, order={order:?}, storage={storage:?}, row={row}"
                                );
                            }
                            assert_canonical(&scratch);
                            assert_eq!(scratch.last(), Some(&PastaField::ONE));
                            if storage == ExpansionStorage::Coefficients {
                                let mut product = output.clone();
                                operation
                                    .execute_product_into(
                                        &coefficients,
                                        view,
                                        &mut product,
                                        &SerialExecutor,
                                        &mut scratch,
                                    )
                                    .unwrap();
                                for (product, value) in product.iter().zip(&output) {
                                    assert_eq!(*product, value.square());
                                }
                            }
                            if order == ExpansionOrder::BitReversed {
                                Plan::without_tables(domain)
                                    .inverse_bit_reversed(
                                        &mut output,
                                        ExecutionOptions::serial(),
                                        &SerialExecutor,
                                        &mut [],
                                    )
                                    .unwrap();
                                assert_eq!(&output[..coefficients.len()], coefficients);
                                assert!(
                                    output[coefficients.len()..]
                                        .iter()
                                        .all(|v| *v == PastaField::ZERO)
                                );
                            }
                        }
                    }
                    for order in [InputOrder::Natural, InputOrder::BitReversed] {
                        let mut output = vec![PastaField::ONE; base.domain().size()];
                        for index in 0..expansion.layout().residues() {
                            let residue = expansion.residue(index, order).unwrap();
                            let options = expansion_strategy().transform;
                            let mut scratch = vec![
                                PastaField::ONE;
                                residue
                                    .scratch_requirements(options)
                                    .unwrap()
                                    .field_elements
                            ];
                            residue
                                .coefficients(
                                    &coefficients,
                                    &mut output,
                                    options,
                                    &SerialExecutor,
                                    &mut scratch,
                                )
                                .unwrap();
                            let layout = if order == InputOrder::Natural {
                                EvaluationLayout::Natural
                            } else {
                                EvaluationLayout::BitReversed
                            };
                            let view =
                                EvaluationView::bind(&output, residue.domain(), layout).unwrap();
                            for row in 0..base.domain().size() {
                                assert_eq!(
                                    view.get(row),
                                    Some(&expected[index + expansion.layout().residues() * row])
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
fn expansion_storage_normalization_streaming_and_direct_bridge() {
    expansions::<PallasBase>();
    expansions::<PallasScalar>();
}

#[test]
fn expansion_metadata_storage_errors_and_panics() {
    let base = Plan::without_tables(Domain::<PallasBase>::new(5).unwrap().subgroup());
    let domain = Domain::new(7).unwrap().coset(Fp::from_u64(7)).unwrap();
    let expansion = Expansion::new(base, domain, None).unwrap();
    let coefficients = inputs(base.domain().size());
    assert!(matches!(
        expansion.configure(
            ExpansionOrder::BitReversed,
            ExpansionStorage::CoefficientWorkspace,
            ExpansionStrategy {
                budget: ResourceBudget {
                    scratch_fields: base.domain().size() - 1,
                    ..ResourceBudget::for_tasks(1)
                },
                ..ExpansionStrategy::serial()
            }
        ),
        Err(FftError::ResourceLimit)
    ));
    let evaluations = direct(&coefficients, base.domain());
    for order in [ExpansionOrder::Residues, ExpansionOrder::BitReversed] {
        for storage in [
            ExpansionStorage::Coefficients,
            ExpansionStorage::ReuseOutput,
            ExpansionStorage::CoefficientWorkspace,
            ExpansionStorage::DisposableInput,
        ] {
            let operation = expansion
                .configure(order, storage, expansion_strategy())
                .unwrap();
            let mut output = vec![Fp::ONE; domain.size()];
            let mut scratch = vec![Fp::ONE; operation.requirements().scratch_fields];
            let mut workspace = vec![Fp::ONE; operation.requirements().coefficient_fields];
            let mut disposable = evaluations.clone();
            let execute = |input: &mut [Fp],
                           output: &mut [Fp],
                           workspace: &mut [Fp],
                           scratch: &mut [Fp],
                           executor: &CountJoins| {
                match storage {
                    ExpansionStorage::Coefficients => {
                        operation.execute_into(&coefficients, output, executor, scratch)
                    }
                    ExpansionStorage::ReuseOutput => {
                        operation.execute_into(&evaluations, output, executor, scratch)
                    }
                    ExpansionStorage::CoefficientWorkspace => operation.execute_with_workspace(
                        &evaluations,
                        output,
                        workspace,
                        executor,
                        scratch,
                    ),
                    ExpansionStorage::DisposableInput => {
                        operation.execute_disposable(input, output, executor, scratch)
                    }
                }
            };
            let joins = CountJoins::default();
            execute(
                &mut disposable,
                &mut output,
                &mut workspace,
                &mut scratch,
                &joins,
            )
            .unwrap();
            for index in 0..joins.take() {
                let mut disposable = evaluations.clone();
                let executor = FailAt {
                    calls: AtomicUsize::new(0),
                    index,
                };
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        match storage {
                            ExpansionStorage::Coefficients => operation.execute_into(
                                &coefficients,
                                &mut output,
                                &executor,
                                &mut scratch,
                            ),
                            ExpansionStorage::ReuseOutput => operation.execute_into(
                                &evaluations,
                                &mut output,
                                &executor,
                                &mut scratch,
                            ),
                            ExpansionStorage::CoefficientWorkspace => operation
                                .execute_with_workspace(
                                    &evaluations,
                                    &mut output,
                                    &mut workspace,
                                    &executor,
                                    &mut scratch,
                                ),
                            ExpansionStorage::DisposableInput => operation.execute_disposable(
                                &mut disposable,
                                &mut output,
                                &executor,
                                &mut scratch,
                            ),
                        }
                    }))
                    .is_err()
                );
                for values in [&output, &workspace, &scratch, &disposable] {
                    assert_canonical(values);
                }
            }
            output.fill(Fp::ONE);
            if !scratch.is_empty() {
                let short = scratch.len() - 1;
                assert!(matches!(
                    execute(
                        &mut disposable,
                        &mut output,
                        &mut workspace,
                        &mut scratch[..short],
                        &joins
                    ),
                    Err(FftError::ScratchTooSmall { .. })
                ));
                assert!(output.iter().all(|v| *v == Fp::ONE));
            }
        }
    }
    let mut scales = vec![Fp::ZERO; domain.size()];
    let table = ExpansionScales::prepare(
        base.domain().size(),
        domain,
        ExpansionScaleNormalization::UnscaledInverse,
        &mut scales,
    )
    .unwrap();
    let artifact = table.artifact();
    assert_eq!(
        artifact.validate(
            base.domain().size(),
            domain,
            ExpansionScaleNormalization::Coefficients
        ),
        Err(FftError::InvalidTables)
    );
    assert!(
        expansion
            .with_scales(
                ExpansionScales::bind(
                    base.domain().size(),
                    domain.domain().subgroup(),
                    ExpansionScaleNormalization::UnscaledInverse,
                    &scales
                )
                .unwrap()
            )
            .is_err()
    );
    scales[1] = Fp::ONE;
    assert!(matches!(
        ExpansionScales::bind(
            base.domain().size(),
            domain,
            ExpansionScaleNormalization::UnscaledInverse,
            &scales
        )
        .unwrap()
        .validate(),
        Err(FftError::InvalidTables)
    ));
}

#[test]
fn interpolation_modes_validate_before_mutation_and_restore_fields_on_panic() {
    let plan = Plan::without_tables(Domain::<PallasBase>::new(7).unwrap().subgroup());
    let other = Plan::without_tables(Domain::new(7).unwrap().coset(Fp::from_u64(7)).unwrap());
    let original = inputs(plan.domain().size());
    let options = ExecutionOptions {
        tile_len: 8,
        columns_per_task: 3,
        max_tasks: 5,
    };
    let parallel = InterpolationOptions {
        transform: options,
        max_class_tasks: 3,
        max_tasks: 5,
    };
    let requirements = parallel
        .requirements(original.len(), &[original.len(); 2])
        .unwrap();
    assert!(requirements.scratch_partitions * requirements.transform_tasks <= parallel.max_tasks);
    for sum in [false, true] {
        let fields = if sum {
            options
                .interpolation_requirements(original.len(), &[original.len(); 2])
                .unwrap()
                .field_elements
        } else {
            requirements.scratch_fields
        };
        let mut output = original.clone();
        let mut a = original.clone();
        let mut b = original.clone();
        let mut scratch = vec![Fp::ONE; fields];
        let mut output_class = Class::new(plan, &mut output, InputOrder::Natural).unwrap();
        let mut lifts = [
            Class::new(plan, &mut a, InputOrder::Natural).unwrap(),
            Class::new(other, &mut b, InputOrder::Natural).unwrap(),
        ];
        let result = if sum {
            interpolate_sum(
                &mut output_class,
                &mut lifts,
                options,
                &SerialExecutor,
                &mut scratch[..fields - 1],
            )
        } else {
            interpolate_classes_parallel(
                &mut output_class,
                &mut lifts,
                parallel,
                &SerialExecutor,
                &mut scratch[..fields - 1],
            )
        };
        assert!(matches!(result, Err(FftError::ScratchTooSmall { .. })));
        assert_eq!(output_class.state(), ClassState::Evaluations);
        assert_eq!(output_class.values(), original);
        for class in &lifts {
            assert_eq!(class.state(), ClassState::Evaluations);
            assert_eq!(class.values(), original);
        }
        assert!(scratch.iter().all(|v| *v == Fp::ONE));
        let count = CountJoins::default();
        if sum {
            interpolate_sum(&mut output_class, &mut lifts, options, &count, &mut scratch)
        } else {
            interpolate_classes_parallel(
                &mut output_class,
                &mut lifts,
                parallel,
                &count,
                &mut scratch,
            )
        }
        .unwrap();
        for index in 0..count.take() {
            let mut output = original.clone();
            let mut a = original.clone();
            let mut b = original.clone();
            let mut output_class = Class::new(plan, &mut output, InputOrder::Natural).unwrap();
            let mut lifts = [
                Class::new(plan, &mut a, InputOrder::Natural).unwrap(),
                Class::new(other, &mut b, InputOrder::Natural).unwrap(),
            ];
            let executor = FailAt {
                calls: AtomicUsize::new(0),
                index,
            };
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    if sum {
                        interpolate_sum(
                            &mut output_class,
                            &mut lifts,
                            options,
                            &executor,
                            &mut scratch,
                        )
                    } else {
                        interpolate_classes_parallel(
                            &mut output_class,
                            &mut lifts,
                            parallel,
                            &executor,
                            &mut scratch,
                        )
                    }
                }))
                .is_err()
            );
            assert_ne!(output_class.state(), ClassState::Evaluations);
            for values in [&output, &a, &b, &scratch] {
                assert_canonical(values);
            }
        }
    }
}

fn interpolation<M: PrimeModulus>() {
    for log in [0, 3, 7] {
        let output_domain = Domain::<M>::new(log)
            .unwrap()
            .coset(PastaField::from_u64(7))
            .unwrap();
        let coefficients = inputs(output_domain.size());
        // Include equal-sized domains, different orders, and a pair of equal
        // smaller domains that the destructive path can combine separately.
        let sizes = [
            output_domain.size(),
            output_domain.size(),
            (output_domain.size() / 2).max(1),
            (output_domain.size() / 2).max(1),
        ];
        let domains = [
            output_domain,
            output_domain.domain().subgroup(),
            Domain::for_size(sizes[2]).unwrap().subgroup(),
            Domain::for_size(sizes[3]).unwrap().subgroup(),
        ];
        let lift_coefficients: Vec<_> = sizes.iter().map(|&size| inputs::<M>(size)).collect();
        let mut expected = coefficients.clone();
        for lift in &lift_coefficients {
            for (out, value) in expected.iter_mut().zip(lift) {
                *out = out.add(value);
            }
        }
        let options = ExecutionOptions {
            tile_len: 4,
            columns_per_task: 2,
            max_tasks: 3,
        };
        let parallel = InterpolationOptions {
            transform: options,
            max_class_tasks: 3,
            max_tasks: 5,
        };
        for mode in 0..3 {
            let mut output = direct(&coefficients, output_domain);
            let mut evaluations: Vec<_> = domains
                .iter()
                .zip(&lift_coefficients)
                .enumerate()
                .map(|(i, (&domain, coefficients))| {
                    let mut values = direct(coefficients, domain);
                    if i % 2 == 0 {
                        Plan::without_tables(domain).permute(&mut values);
                    }
                    values
                })
                .collect();
            let mut lifts: Vec<_> = evaluations
                .iter_mut()
                .zip(&domains)
                .enumerate()
                .map(|(i, (values, &domain))| {
                    Class::new(
                        Plan::without_tables(domain),
                        values,
                        if i % 2 == 0 {
                            InputOrder::BitReversed
                        } else {
                            InputOrder::Natural
                        },
                    )
                    .unwrap()
                })
                .collect();
            let mut class = Class::new(
                Plan::without_tables(output_domain),
                &mut output,
                InputOrder::Natural,
            )
            .unwrap();
            let fields = if mode == 1 {
                let required = parallel.scratch(&class, &lifts).unwrap();
                assert_eq!(
                    required,
                    parallel.requirements(output_domain.size(), &sizes).unwrap()
                );
                required.scratch_fields
            } else {
                interpolation_scratch(&class, &lifts, options)
                    .unwrap()
                    .field_elements
            };
            let mut scratch = vec![PastaField::ONE; fields + 1];
            match mode {
                0 => interpolate_classes(&mut class, &mut lifts, options, &Threads, &mut scratch),
                1 => interpolate_classes_parallel(
                    &mut class,
                    &mut lifts,
                    parallel,
                    &Threads,
                    &mut scratch,
                ),
                _ => interpolate_sum(&mut class, &mut lifts, options, &Threads, &mut scratch),
            }
            .unwrap();
            assert_eq!(class.values(), expected);
            assert_eq!(class.state(), ClassState::Coefficients);
            for (class, expected) in lifts.iter().zip(&lift_coefficients) {
                if mode < 2 {
                    assert_eq!(class.values(), expected);
                } else {
                    assert_eq!(class.state(), ClassState::Consumed);
                }
            }
            assert_eq!(scratch.last(), Some(&PastaField::ONE));
            assert_canonical(&scratch);
            assert_eq!(
                interpolate_sum(&mut class, &mut lifts, options, &Threads, &mut scratch),
                Err(FftError::InvalidClassState)
            );
        }
    }
}

#[test]
fn equal_size_parallel_and_destructive_interpolation_match_polynomial_sums() {
    interpolation::<PallasBase>();
    interpolation::<PallasScalar>();
}
