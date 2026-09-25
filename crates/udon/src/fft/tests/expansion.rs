use super::*;

fn expansions<M: PrimeModulus>() {
    for log in [0, 1, 3, 6] {
        let base_domain = Domain::<PastaField<M>>::new(log).unwrap();
        let base_tables = Prepared::new(base_domain.subgroup());
        let base = base_tables.tables().bind(base_domain.subgroup());
        let coefficients = inputs(base_domain.size());
        let evaluations = direct(&coefficients, base_domain.subgroup());
        for extra in [0, 1, 2, 3] {
            for coset in [false, true] {
                let domain = {
                    let subgroup = Domain::new(log + extra).unwrap();
                    if coset {
                        subgroup.coset()
                    } else {
                        subgroup.subgroup()
                    }
                };
                let mut scales =
                    vec![
                        PastaField::ZERO;
                        ExpansionScales::<M>::requirements(base.domain().size(), domain.size())
                            .unwrap()
                    ];
                let scales = ExpansionScales::prepare(
                    base.domain().size(),
                    domain,
                    ExpansionScaleNormalization::Coefficients,
                    &mut scales,
                )
                .unwrap();
                for scales in [None, Some(scales)] {
                    let expansion = Expansion::new(base, domain, scales).unwrap();
                    let expected = direct(&coefficients, domain);
                    for options in [
                        ExpansionStrategy::SERIAL,
                        ExpansionStrategy {
                            max_residue_tasks: 3,
                            ..ExpansionStrategy::SERIAL
                        },
                        ExpansionStrategy {
                            max_residue_tasks: 1,
                            transform: Strategy {
                                tile_len: 4,
                                columns_per_task: 3,
                                max_tasks: 3,
                            },
                        },
                        ExpansionStrategy {
                            max_residue_tasks: 3,
                            transform: Strategy {
                                tile_len: 4,
                                columns_per_task: 3,
                                max_tasks: 3,
                            },
                        },
                    ] {
                        let mut output = vec![PastaField::ZERO; domain.size()];
                        let count = expansion.coefficient_scratch_with(options).unwrap();
                        let mut scratch = vec![PastaField::ONE; count + 1];
                        expansion
                            .coefficients_with(
                                &coefficients,
                                &mut output,
                                options,
                                &SerialExecutor,
                                &mut scratch,
                            )
                            .unwrap();
                        assert_eq!((scratch[count]).reduce(), (PastaField::<_>::ONE).reduce());
                        let view = expansion.view(&output);
                        for (row, expected) in expected.iter().enumerate() {
                            assert_eq!(
                                (view.get(row)).map(|value| value.reduce()),
                                (Some(expected)).map(|value| value.reduce())
                            );
                        }
                        let count = expansion.evaluation_scratch_with(options).unwrap();
                        scratch.fill(PastaField::ONE);
                        expansion
                            .evaluations_with(
                                &evaluations,
                                &mut output,
                                options,
                                &SerialExecutor,
                                &mut scratch,
                            )
                            .unwrap();
                        assert!(
                            scratch[count..]
                                .iter()
                                .all(|value| value.reduce() == PastaField::ONE)
                        );
                        let view = expansion.view(&output);
                        for (row, expected) in expected.iter().enumerate() {
                            assert_eq!(
                                (view.get(row)).map(|value| value.reduce()),
                                (Some(expected)).map(|value| value.reduce())
                            );
                        }
                        for len in [1, coefficients.len().min(3), coefficients.len().min(10)] {
                            let short = &coefficients[..len];
                            let mut product = vec![PastaField::ZERO; domain.size()];
                            expansion
                                .short_product_with(
                                    short,
                                    view,
                                    &mut product,
                                    options,
                                    &SerialExecutor,
                                    &mut scratch,
                                )
                                .unwrap();
                            let short_values = direct(short, domain);
                            let product = expansion.view(&product);
                            for row in 0..domain.size() {
                                assert_eq!(
                                    (product.get(row)).map(|value| value.reduce()),
                                    (Some(&expected[row].mul(&short_values[row])))
                                        .map(|value| value.reduce())
                                );
                            }
                        }
                        expansion
                            .coefficients_with(
                                &[],
                                &mut output,
                                options,
                                &SerialExecutor,
                                &mut scratch,
                            )
                            .unwrap();
                        assert!(
                            output
                                .iter()
                                .all(|value| value.reduce() == PastaField::ZERO)
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn residue_expansion_and_fused_short_products_match_direct_evaluation() {
    expansions::<PallasBase>();
    expansions::<PallasScalar>();
}

fn large_expansion<M: PrimeModulus>() {
    let base_domain = Domain::<PastaField<M>>::new(11).unwrap();
    let prepared = Prepared::new(base_domain.subgroup());
    let base = prepared.tables().bind(base_domain.subgroup());
    let coefficients = inputs(base_domain.size());
    for extra in [1, 3] {
        let domain = Domain::new(11 + extra).unwrap().coset();
        let expansion = Expansion::new(base, domain, None).unwrap();
        let expected = reference_coset(&coefficients, domain);
        let mut output = vec![PastaField::ZERO; domain.size()];
        let options = ExpansionStrategy {
            max_residue_tasks: 3,
            transform: Strategy {
                tile_len: 256,
                columns_per_task: 127,
                max_tasks: 3,
            },
        };
        let mut scratch =
            vec![PastaField::ZERO; expansion.coefficient_scratch_with(options).unwrap()];
        expansion
            .coefficients_with(&coefficients, &mut output, options, &Threads, &mut scratch)
            .unwrap();
        let mut natural = vec![PastaField::ZERO; domain.size()];
        expansion.layout().copy_to_natural(&output, &mut natural);
        assert_eq!(reduced(&natural), reduced(&expected));
        let evaluations = reference_coset(&coefficients, base_domain.subgroup());
        expansion
            .evaluations_with(&evaluations, &mut output, options, &Threads, &mut scratch)
            .unwrap();
        expansion.layout().copy_to_natural(&output, &mut natural);
        assert_eq!(reduced(&natural), reduced(&expected));
        let factor = expansion.view(&output);
        let mut product = vec![PastaField::ZERO; domain.size()];
        expansion
            .short_product_with(
                &coefficients[..5],
                factor,
                &mut product,
                options,
                &Threads,
                &mut scratch,
            )
            .unwrap();
        let short_values = direct(&coefficients[..5], domain);
        let product = expansion.view(&product);
        for row in 0..domain.size() {
            assert_eq!(
                (product.get(row)).map(|value| value.reduce()),
                (Some(&expected[row].mul(&short_values[row]))).map(|value| value.reduce())
            );
        }
    }
}

#[test]
fn original_two_and_eight_residue_expansions_match_zero_padded_fft() {
    large_expansion::<PallasBase>();
    large_expansion::<PallasScalar>();
}

#[test]
fn every_expansion_transform_uses_the_callers_executor_and_options() {
    let base = Transform::new(Domain::<Fp>::new(6).unwrap().subgroup());
    let coefficients = inputs(base.domain().size());
    let options = ExpansionStrategy {
        // With no joins across residues, every observed join is intra-residue.
        max_residue_tasks: 1,
        transform: Strategy {
            tile_len: 8,
            columns_per_task: 3,
            max_tasks: 4,
        },
    };
    let mut scratch = vec![Fp::ZERO; base.scratch_requirements_with(options.transform).unwrap()];
    let executor = CountJoins::default();
    let mut evaluations = coefficients.clone();
    base.forward_with(&mut evaluations, options.transform, &executor, &mut scratch)
        .unwrap();
    let forward_joins = executor.take();
    let mut inverse = evaluations.clone();
    base.inverse_with(&mut inverse, options.transform, &executor, &mut scratch)
        .unwrap();
    let inverse_joins = executor.take();
    assert!(forward_joins > 0 && inverse_joins > 0);

    for extra in [0, 1, 3] {
        let domain = Domain::new(6 + extra).unwrap().coset();
        let expansion = Expansion::new(base, domain, None).unwrap();
        let residues = expansion.layout().residues();
        let mut output = vec![Fp::ZERO; domain.size()];
        expansion
            .coefficients_with(&coefficients, &mut output, options, &executor, &mut scratch)
            .unwrap();
        assert!(executor.take() >= residues);
        let expected = output.clone();
        expansion
            .evaluations_with(&evaluations, &mut output, options, &executor, &mut scratch)
            .unwrap();
        assert!(executor.take() > residues);
        assert_eq!(reduced(&output), reduced(&expected));
        let ones = vec![Fp::ONE; domain.size()];
        let factor = expansion.view(&ones);
        expansion
            .short_product_with(
                &coefficients[..5],
                factor,
                &mut output,
                options,
                &executor,
                &mut scratch,
            )
            .unwrap();
        let pruned_joins = executor.take();
        assert!(pruned_joins > 0);
        let expected_short = direct(&coefficients[..5], domain);
        let view = expansion.view(&output);
        for (row, expected) in expected_short.iter().enumerate() {
            assert_eq!(
                (view.get(row)).map(|value| value.reduce()),
                (Some(expected)).map(|value| value.reduce())
            );
        }

        // Whole transforms still allow callers to parallelize only residues
        // without reserving scratch for tiled transforms.
        let options = ExpansionStrategy {
            max_residue_tasks: 3,
            ..ExpansionStrategy::SERIAL
        };
        assert_eq!(expansion.coefficient_scratch_with(options).unwrap(), 0);
        expansion
            .coefficients_with(&coefficients, &mut output, options, &executor, &mut [])
            .unwrap();
        assert_eq!(executor.take(), residues.min(3) - 1);
        assert_eq!(reduced(&output), reduced(&expected));
    }
}

fn prepared_evaluation_expansions<M: PrimeModulus>() {
    let subgroup = Domain::<PastaField<M>>::new(6).unwrap().subgroup();
    let prepared = Prepared::new(subgroup);
    let coefficients = inputs::<M>(subgroup.size());
    let evaluations = direct(&coefficients, subgroup);
    let saved = evaluations.clone();
    for extra in [0, 1, 3] {
        for coset in [false, true] {
            let extended = {
                let subgroup = Domain::new(6 + extra).unwrap();
                if coset {
                    subgroup.coset()
                } else {
                    subgroup.subgroup()
                }
            };
            let expected = reference_coset(&coefficients, extended);
            let mut scales = vec![PastaField::ZERO; extended.size()];
            let scales = ExpansionScales::prepare(
                subgroup.size(),
                extended,
                ExpansionScaleNormalization::Coefficients,
                &mut scales,
            )
            .unwrap();
            for mask in 0..8 {
                let tables = Tables {
                    forward: (mask & 1 != 0).then_some(prepared.forward.as_slice()),
                    inverse: (mask & 2 != 0).then_some(prepared.inverse.as_slice()),
                    inverse_finish: (mask & 4 != 0).then_some(prepared.finish.as_slice()),
                };
                let expansion =
                    Expansion::new(tables.bind(subgroup), extended, Some(scales)).unwrap();
                for transform in [
                    Strategy::SERIAL,
                    Strategy {
                        tile_len: 8,
                        columns_per_task: 3,
                        max_tasks: 3,
                    },
                ] {
                    let options = ExpansionStrategy {
                        max_residue_tasks: 3,
                        transform,
                    };
                    let count = expansion.evaluation_scratch_with(options).unwrap();
                    let mut scratch = vec![PastaField::ONE; count + 1];
                    let mut output = vec![PastaField::ZERO; extended.size()];
                    expansion
                        .evaluations_with(
                            &evaluations,
                            &mut output,
                            options,
                            &SerialExecutor,
                            &mut scratch,
                        )
                        .unwrap();
                    let view = expansion.view(&output);
                    for (row, expected) in expected.iter().enumerate() {
                        assert_eq!(
                            (view.get(row)).map(|value| value.reduce()),
                            (Some(expected)).map(|value| value.reduce())
                        );
                    }
                    assert_eq!(bytes_of_slice(&evaluations), bytes_of_slice(&saved));
                    assert_eq!((scratch[count]).reduce(), (PastaField::<_>::ONE).reduce());
                    assert_loose_bound(&output);
                    assert_loose_bound(&scratch);
                }
            }
        }
    }
}

#[test]
fn prepared_evaluation_expansion_supports_every_base_table_subset() {
    prepared_evaluation_expansions::<PallasBase>();
    prepared_evaluation_expansions::<PallasScalar>();
}

fn constant_prefixes<M: PrimeModulus>() {
    let base = Transform::new(Domain::<PastaField<M>>::new(6).unwrap().subgroup());
    let transform = Strategy {
        tile_len: 8,
        columns_per_task: 3,
        max_tasks: 4,
    };
    let options = ExpansionStrategy {
        max_residue_tasks: 3,
        transform,
    };
    let executor = CountJoins::default();
    let constant = [PastaField::from_u64(19)];
    for extra in [0, 1, 3] {
        for coset in [false, true] {
            let domain = {
                let subgroup = Domain::new(6 + extra).unwrap();
                if coset {
                    subgroup.coset()
                } else {
                    subgroup.subgroup()
                }
            };
            let plan = Transform::new(domain);
            let expansion = Expansion::new(base, domain, None).unwrap();
            let required = expansion
                .coefficient_scratch_with(options)
                .unwrap()
                .max(plan.scratch_requirements_with(transform).unwrap());
            let mut scratch = vec![PastaField::ONE; required + 1];
            let mut output = vec![PastaField::ZERO; domain.size()];
            for coefficients in [&constant[..0], &constant[..]] {
                let expected = coefficients.first().copied().unwrap_or(PastaField::ZERO);
                for expanded in [false, true] {
                    if expanded {
                        expansion
                            .coefficients_with(
                                coefficients,
                                &mut output,
                                options,
                                &executor,
                                &mut scratch,
                            )
                            .unwrap();
                    } else {
                        plan.forward_prefix_with(
                            coefficients,
                            &mut output,
                            transform,
                            &executor,
                            &mut scratch,
                        )
                        .unwrap();
                    }
                    assert_eq!(executor.take(), 0);
                    assert!(
                        output
                            .iter()
                            .all(|&value| value.reduce() == expected.reduce())
                    );
                    assert!(
                        scratch
                            .iter()
                            .all(|&value| value.reduce() == PastaField::ONE)
                    );
                }
            }
            let factor = inputs::<M>(domain.size());
            expansion
                .short_product_with(
                    &constant,
                    expansion.view(&factor),
                    &mut output,
                    options,
                    &executor,
                    &mut scratch,
                )
                .unwrap();
            assert_eq!(
                executor.take(),
                expansion.layout().residues().min(options.max_residue_tasks) - 1
            );
            for (actual, factor) in output.iter().zip(&factor) {
                assert_eq!((*actual).reduce(), (constant[0].mul(factor)).reduce());
            }
            assert!(
                scratch
                    .iter()
                    .all(|&value| value.reduce() == PastaField::ONE)
            );
            if extra == 0 && !coset {
                let input = inputs::<M>(base.domain().size());
                expansion
                    .evaluations_with(&input, &mut output, options, &executor, &mut scratch)
                    .unwrap();
                assert_eq!(reduced(&output), reduced(&input));
                assert_eq!(executor.take(), 0);
                assert!(
                    scratch
                        .iter()
                        .all(|&value| value.reduce() == PastaField::ONE)
                );
            }
        }
    }
}

#[test]
fn constant_prefixes_and_subgroup_copies_skip_transform_scheduling() {
    constant_prefixes::<PallasBase>();
    constant_prefixes::<PallasScalar>();
}
