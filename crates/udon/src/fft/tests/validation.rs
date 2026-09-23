use super::*;

#[test]
fn expansion_validates_options_and_partitioned_scratch_before_mutation() {
    let base = Transform::new(Domain::<Fp>::new(6).unwrap().subgroup());
    let input = inputs(base.domain().size());
    let transform = Strategy {
        tile_len: 8,
        columns_per_task: 3,
        max_tasks: 3,
    };
    let per_transform = base.scratch_requirements_with(transform).unwrap();
    for extra in [0, 1, 3] {
        let expansion =
            Expansion::new(base, Domain::new(6 + extra).unwrap().subgroup(), None).unwrap();
        let mut output = vec![Fp::ONE; expansion.layout().size()];
        let factors = output.clone();
        let factor = expansion.view(&factors);
        for tasks in [1, 2, 3, usize::MAX] {
            let options = ExpansionStrategy {
                max_residue_tasks: tasks,
                transform,
            };
            let count = expansion.coefficient_scratch_with(options).unwrap();
            let eval_count = expansion.evaluation_scratch_with(options).unwrap();
            assert_eq!(
                count,
                per_transform * tasks.min(expansion.layout().residues())
            );
            assert_eq!(
                eval_count,
                per_transform * tasks.min(expansion.layout().residues() - 1).max(1)
            );
            let mut scratch = vec![Fp::ONE; count - 1];
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = expansion.coefficients_with(
                        &input,
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    );
                }))
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = expansion.coefficients_with(
                        &[],
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    );
                }))
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = expansion.short_product_with(
                        &input[..5],
                        factor,
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    );
                }))
                .is_err()
            );
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = expansion.evaluations_with(
                        &input,
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch[..eval_count - 1],
                    );
                }))
                .is_err()
            );
            assert_eq!(reduced(&output), reduced(&factors));
            assert!(scratch.iter().all(|value| value.reduce() == Fp::ONE));
        }
        let valid = ExpansionStrategy {
            max_residue_tasks: 2,
            transform,
        };
        for options in [
            ExpansionStrategy {
                max_residue_tasks: 0,
                ..valid
            },
            ExpansionStrategy {
                transform: Strategy {
                    tile_len: 3,
                    ..transform
                },
                ..valid
            },
            ExpansionStrategy {
                transform: Strategy {
                    columns_per_task: 0,
                    ..transform
                },
                ..valid
            },
            ExpansionStrategy {
                transform: Strategy {
                    max_tasks: 0,
                    ..transform
                },
                ..valid
            },
        ] {
            let mut scratch = vec![Fp::ONE; 512];
            assert_eq!(
                expansion.coefficient_scratch_with(options),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(
                expansion.evaluation_scratch_with(options),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(
                expansion.coefficients_with(
                    &input,
                    &mut output,
                    options,
                    &SerialExecutor,
                    &mut scratch
                ),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(
                expansion.evaluations_with(
                    &input,
                    &mut output,
                    options,
                    &SerialExecutor,
                    &mut scratch
                ),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(
                expansion.short_product_with(
                    &input[..5],
                    factor,
                    &mut output,
                    options,
                    &SerialExecutor,
                    &mut scratch
                ),
                Err(FftError::InvalidExecution)
            );
            assert_eq!(reduced(&output), reduced(&factors));
            assert!(scratch.iter().all(|value| value.reduce() == Fp::ONE));
        }
    }
}

#[test]
fn size_queries_validate_without_constructing_domains() {
    const _: () = {
        let serial = Strategy::SERIAL;
        let expansion = ExpansionStrategy::SERIAL;
        let tables = match TableRequirements::for_size(1) {
            Ok(required) => required,
            Err(_) => panic!("singleton rejected"),
        };
        assert!(tables.twiddles == 0);
        assert!(matches!(serial.requirements(1), Ok(0)));
        assert!(matches!(expansion.coefficient_requirements(1, 1), Ok(0)));
        assert!(matches!(expansion.evaluation_requirements(1, 1), Ok(0)));
        let invalid_sizes = [0, 3, usize::MAX];
        let mut index = 0;
        while index < invalid_sizes.len() {
            let size = invalid_sizes[index];
            assert!(matches!(
                TableRequirements::for_size(size),
                Err(FftError::InvalidSize)
            ));
            assert!(matches!(
                serial.requirements(size),
                Err(FftError::InvalidSize)
            ));
            assert!(matches!(
                expansion.coefficient_requirements(size, 16),
                Err(FftError::InvalidSize)
            ));
            assert!(matches!(
                expansion.evaluation_requirements(1, size),
                Err(FftError::InvalidSize)
            ));
            index += 1;
        }
        let invalid_options = [
            Strategy {
                tile_len: 0,
                ..serial
            },
            Strategy {
                tile_len: 3,
                ..serial
            },
            Strategy {
                columns_per_task: 0,
                ..serial
            },
            Strategy {
                max_tasks: 0,
                ..serial
            },
        ];
        index = 0;
        while index < invalid_options.len() {
            let options = invalid_options[index];
            assert!(matches!(
                options.requirements(1),
                Err(FftError::InvalidExecution)
            ));
            let expansion = ExpansionStrategy {
                transform: options,
                ..expansion
            };
            assert!(matches!(
                expansion.coefficient_requirements(1, 1),
                Err(FftError::InvalidExecution)
            ));
            assert!(matches!(
                expansion.evaluation_requirements(1, 1),
                Err(FftError::InvalidExecution)
            ));
            index += 1;
        }
        let invalid = ExpansionStrategy {
            max_residue_tasks: 0,
            ..expansion
        };
        assert!(matches!(
            invalid.coefficient_requirements(1, 1),
            Err(FftError::InvalidExecution)
        ));
        assert!(matches!(
            invalid.evaluation_requirements(1, 1),
            Err(FftError::InvalidExecution)
        ));
        assert!(matches!(
            expansion.coefficient_requirements(8, 4),
            Err(FftError::InvalidLayout)
        ));
        assert!(matches!(
            expansion.evaluation_requirements(8, 4),
            Err(FftError::InvalidLayout)
        ));
    };

    for log in 0..usize::BITS {
        let size = 1usize << log;
        let expected = Domain::<Fp>::for_size(size).map(|_| ());
        assert_eq!(TableRequirements::for_size(size).map(|_| ()), expected);
        assert_eq!(Strategy::SERIAL.requirements(size).map(|_| ()), expected);
        assert_eq!(
            ExpansionStrategy::SERIAL
                .coefficient_requirements(size, size)
                .map(|_| ()),
            expected
        );
        assert_eq!(
            ExpansionStrategy::SERIAL
                .evaluation_requirements(1, size)
                .map(|_| ()),
            expected
        );
    }
    let saturated = Strategy {
        tile_len: 4,
        columns_per_task: usize::MAX,
        max_tasks: usize::MAX,
    };
    assert!(saturated.requirements(16).is_ok());
    let expansion = ExpansionStrategy {
        max_residue_tasks: usize::MAX,
        transform: saturated,
    };
    assert!(expansion.coefficient_requirements(16, 64).is_ok());
    assert!(expansion.evaluation_requirements(16, 64).is_ok());
}

#[test]
fn invalid_descriptions_and_short_scratch_do_not_mutate_buffers() {
    assert!(matches!(Domain::<Fp>::new(33), Err(FftError::InvalidSize)));
    assert!(matches!(
        Domain::<Fp>::new(u32::MAX),
        Err(FftError::InvalidSize)
    ));
    for size in [0, 3, 7, usize::MAX] {
        assert!(Domain::<Fp>::for_size(size).is_err());
    }
    let domain = Domain::new(6).unwrap().subgroup();
    if usize::BITS == 32 {
        assert!(matches!(Domain::<Fp>::new(32), Err(FftError::SizeOverflow)));
    } else {
        assert_eq!(Domain::<Fp>::new(32).unwrap().size() as u64, 1u64 << 32);
    }
    let plan = Transform::new(domain);
    let options = Strategy {
        tile_len: 8,
        columns_per_task: 3,
        max_tasks: 3,
    };
    let required = plan.scratch_requirements_with(options).unwrap();
    let original = inputs::<PallasBase>(64);
    let mut output = original.clone();
    let mut scratch = vec![Fp::ONE; required - 1];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = plan.forward_with(&mut output, options, &SerialExecutor, &mut scratch);
        }))
        .is_err()
    );
    assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    assert_eq!(
        bytes_of_slice(&scratch),
        bytes_of_slice(&vec![<Fp>::ONE; required - 1])
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ =
                plan.inverse_bit_reversed_with(&mut output, options, &SerialExecutor, &mut scratch);
        }))
        .is_err()
    );
    assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = plan.forward_into_with(
                &original,
                &mut output,
                options,
                &SerialExecutor,
                &mut scratch,
            );
        }))
        .is_err()
    );
    assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    assert_eq!(
        bytes_of_slice(&scratch),
        bytes_of_slice(&vec![<Fp>::ONE; required - 1])
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = plan.inverse_into_with(
                &original,
                &mut output,
                options,
                &SerialExecutor,
                &mut scratch,
            );
        }))
        .is_err()
    );
    assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    assert_eq!(
        bytes_of_slice(&scratch),
        bytes_of_slice(&vec![<Fp>::ONE; required - 1])
    );
    for (input_len, output_len) in [(63, 64), (65, 64), (64, 63), (64, 65)] {
        let input = vec![Fp::ONE; input_len];
        let mut output = vec![Fp::ZERO; output_len];
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = plan.forward_into_with(
                    &input,
                    &mut output,
                    Strategy::SERIAL,
                    &SerialExecutor,
                    &mut [],
                );
            }))
            .is_err()
        );
        assert_eq!(
            bytes_of_slice(&output),
            bytes_of_slice(&vec![<Fp>::ZERO; output_len])
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = plan.inverse_into_with(
                    &input,
                    &mut output,
                    Strategy::SERIAL,
                    &SerialExecutor,
                    &mut [],
                );
            }))
            .is_err()
        );
        assert_eq!(
            bytes_of_slice(&output),
            bytes_of_slice(&vec![<Fp>::ZERO; output_len])
        );
        if output_len != 64 {
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = plan.inverse_bit_reversed_with(
                        &mut output,
                        Strategy::SERIAL,
                        &SerialExecutor,
                        &mut [],
                    );
                }))
                .is_err()
            );
            assert_eq!(
                bytes_of_slice(&output),
                bytes_of_slice(&vec![<Fp>::ZERO; output_len])
            );
        }
    }
    for options in [
        Strategy {
            tile_len: 3,
            ..options
        },
        Strategy {
            columns_per_task: 0,
            ..options
        },
        Strategy {
            max_tasks: 0,
            ..options
        },
    ] {
        assert_eq!(
            plan.forward_with(&mut output, options, &SerialExecutor, &mut []),
            Err(FftError::InvalidExecution)
        );
        assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    }
    assert!(matches!(
        execution::InterpolationPlan::new(
            [
                (Transform::new(domain), ElementOrder::Natural),
                (
                    Transform::new(Domain::new(7).unwrap().subgroup()),
                    ElementOrder::Natural,
                ),
            ],
            false,
            StorageLayout::Contiguous,
            crate::exec::ExecutionOptions::default(),
        ),
        Err(FftError::InvalidLayout)
    ));
    assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    let expansion = Expansion::new(plan, Domain::new(7).unwrap().subgroup(), None).unwrap();
    let options = ExpansionStrategy {
        max_residue_tasks: 2,
        transform: options,
    };
    let mut expanded = [Fp::ONE; 128];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = expansion.evaluations_with(
                &original,
                &mut expanded,
                options,
                &SerialExecutor,
                &mut [],
            );
        }))
        .is_err()
    );
    assert_eq!(bytes_of_slice(&expanded), bytes_of_slice(&[<Fp>::ONE; 128]));
    let too_long = FftError::InvalidPrefix {
        min: 0,
        max: 64,
        actual: 65,
    };
    assert_eq!(
        expansion.coefficients_with(
            &[Fp::ONE; 65],
            &mut expanded,
            options,
            &SerialExecutor,
            &mut []
        ),
        Err(too_long)
    );
    assert_eq!(bytes_of_slice(&expanded), bytes_of_slice(&[<Fp>::ONE; 128]));
    assert_eq!(
        plan.forward_prefix_with(
            &[Fp::ONE; 65],
            &mut output,
            Strategy::SERIAL,
            &SerialExecutor,
            &mut []
        ),
        Err(too_long)
    );
    assert_eq!(bytes_of_slice(&output), bytes_of_slice(&original));
    assert_eq!(
        std::format!("{too_long}"),
        "input prefix must contain 0..=64 elements, received 65"
    );
    let factor_values = [Fp::ONE; 128];
    let factor = expansion.view(&factor_values);
    for len in [0, 65] {
        let error = expansion
            .short_product_with(
                &vec![Fp::ONE; len],
                factor,
                &mut expanded,
                ExpansionStrategy::SERIAL,
                &SerialExecutor,
                &mut [],
            )
            .unwrap_err();
        assert_eq!(
            error,
            FftError::InvalidPrefix {
                min: 1,
                max: 64,
                actual: len
            }
        );
        assert_eq!(bytes_of_slice(&expanded), bytes_of_slice(&[<Fp>::ONE; 128]));
    }
}

#[test]
fn table_preparation_checks_all_lengths_before_writing() {
    let domain = Domain::<PastaField<PallasBase>>::new(3).unwrap().coset();
    let mut valid = [Fp::from_u64(17); 4];
    let mut wrong = [Fp::ONE; 3];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = TablesMut {
                inverse: Some(&mut valid),
                forward: Some(&mut wrong),
                ..TablesMut::default()
            }
            .prepare(domain);
        }))
        .is_err()
    );
    assert_eq!(
        bytes_of_slice(&valid),
        bytes_of_slice(&[<Fp>::from_u64(17); 4])
    );
    assert_eq!(bytes_of_slice(&wrong), bytes_of_slice(&[<Fp>::ONE; 3]));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = TablesMut {
                forward: Some(&mut valid),
                inverse_finish: Some(&mut wrong),
                ..TablesMut::default()
            }
            .prepare(domain);
        }))
        .is_err()
    );
    assert_eq!(
        bytes_of_slice(&valid),
        bytes_of_slice(&[<Fp>::from_u64(17); 4])
    );
    assert_eq!(bytes_of_slice(&wrong), bytes_of_slice(&[<Fp>::ONE; 3]));
}
