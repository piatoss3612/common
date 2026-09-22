use super::*;

use crate::fft::run::{ExpansionPlan, FftPlan, InterpolationPlan};
use core::num::NonZeroUsize;

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

const OPTIONS: Strategy = Strategy {
    tile_len: 4,
    columns_per_task: 2,
    max_tasks: 5,
};

fn check_coefficients<M: PrimeModulus>(
    view: CoefficientView<'_, M>,
    coefficients: &[PastaField<M>],
    scale: InverseScale,
) {
    assert_eq!(view.scale(), scale);
    let multiplier: PastaField<M> = match scale {
        InverseScale::Normalized => PastaField::ONE,
        InverseScale::Unscaled => PastaField::from_u64(coefficients.len() as u64),
    };
    assert_eq!(view.as_slice().len(), coefficients.len());
    for (actual, coefficient) in view.as_slice().iter().zip(coefficients) {
        assert_eq!((*actual).reduce(), (coefficient.mul(&multiplier)).reduce());
        assert_eq!(
            (actual.mul(&view.normalization_factor())).reduce(),
            (*coefficient).reduce()
        );
    }
}

fn expansions<M: PrimeModulus>() {
    for log in [0, 2, 5] {
        let base = Transform::new(Domain::<M>::new(log).unwrap().subgroup());
        let coefficients = inputs(base.domain().size());
        let evaluations = direct(&coefficients, base.domain());
        for extra in [0, 1, 3] {
            for shift in [PastaField::ONE, PastaField::ZETA, PastaField::from_u64(7)] {
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
                        .unwrap();
                        expansion.with_scales(scales)
                    } else {
                        expansion
                    };
                    let mut contiguous = vec![PastaField::ZERO; domain.size()];
                    expansion
                        .evaluations_with(
                            &evaluations,
                            &mut contiguous,
                            ExpansionStrategy::SERIAL,
                            &SerialExecutor,
                            &mut [],
                        )
                        .unwrap();
                    let contiguous = EvaluationView::bind(
                        &contiguous,
                        domain,
                        EvaluationLayout::Residues(expansion.layout()),
                    );
                    for (row, value) in expected.iter().enumerate() {
                        assert_eq!(
                            (contiguous.get(row)).map(|value| value.reduce()),
                            (Some(value)).map(|value| value.reduce())
                        );
                    }
                    for order in [ExpansionOrder::Residues, ExpansionOrder::BitReversed] {
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
                            let operation = ExpansionPlan::with_strategy(
                                expansion,
                                storage,
                                order,
                                InputSupport::Full,
                                ElementOrder::Natural,
                                nz(4),
                                Codelet::Radix2,
                            );
                            let scale = match storage {
                                ExpansionStorage::CoefficientWorkspace { scale }
                                | ExpansionStorage::DisposableInput { scale } => Some(scale),
                                _ => None,
                            };
                            let operation = operation.unwrap();
                            let mut output = vec![PastaField::ONE; domain.size()];
                            let mut scratch =
                                vec![PastaField::ONE; operation.scratch_fields_with(nz(5)) + 1];
                            let mut working = vec![PastaField::ONE; operation.coefficient_fields()];
                            let mut disposable = evaluations.clone();
                            let retained =
                                if matches!(storage, ExpansionStorage::DisposableInput { .. }) {
                                    Some(operation.execute_disposable_with(
                                        &mut disposable,
                                        &mut output,
                                        None,
                                        &mut scratch,
                                        nz(5),
                                        &SerialExecutor,
                                    ))
                                } else {
                                    let input = if storage == ExpansionStorage::Coefficients {
                                        &coefficients
                                    } else {
                                        &evaluations
                                    };
                                    operation.execute_with(
                                        input,
                                        &mut output,
                                        &mut working,
                                        None,
                                        &mut scratch,
                                        nz(5),
                                        &SerialExecutor,
                                    )
                                };
                            if let Some(view) = retained {
                                check_coefficients(view, &coefficients, scale.unwrap());
                            }
                            let layout = if order == ExpansionOrder::Residues {
                                EvaluationLayout::Residues(expansion.layout())
                            } else {
                                EvaluationLayout::BitReversed
                            };
                            let view = EvaluationView::bind(&output, domain, layout);
                            for (row, value) in expected.iter().enumerate() {
                                assert_eq!(
                                    (view.get(row)).map(|value| value.reduce()),
                                    (Some(value)).map(|value| value.reduce()),
                                    "log={log}, extra={extra}, normalization={normalization:?}, order={order:?}, storage={storage:?}, row={row}"
                                );
                            }
                            assert_loose_bound(&scratch);
                            assert_eq!(
                                (scratch.last()).map(|value| value.reduce()),
                                (Some(&PastaField::<_>::ONE)).map(|value| value.reduce())
                            );
                            if storage == ExpansionStorage::Coefficients {
                                let mut product = output.clone();
                                operation.execute_with(
                                    &coefficients,
                                    &mut product,
                                    &mut [],
                                    Some(&output),
                                    &mut scratch,
                                    nz(5),
                                    &SerialExecutor,
                                );
                                for (product, value) in product.iter().zip(&output) {
                                    assert_eq!((*product).reduce(), (value.square()).reduce());
                                }
                            }
                            if order == ExpansionOrder::BitReversed {
                                Transform::new(domain)
                                    .inverse_bit_reversed_with(
                                        &mut output,
                                        Strategy::SERIAL,
                                        &SerialExecutor,
                                        &mut [],
                                    )
                                    .unwrap();
                                assert_eq!(
                                    reduced(&output[..coefficients.len()]),
                                    reduced(&coefficients)
                                );
                                assert!(
                                    output[coefficients.len()..]
                                        .iter()
                                        .all(|v| v.reduce() == PastaField::ZERO)
                                );
                            }
                        }
                    }
                    for order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                        let mut output = vec![PastaField::ONE; base.domain().size()];
                        for index in 0..expansion.layout().residues() {
                            let residue = expansion.residue(index, order).unwrap();
                            let options = OPTIONS;
                            let mut scratch = vec![
                                PastaField::ONE;
                                residue
                                    .scratch_requirements_with(options)
                                    .unwrap()
                                    .field_elements
                            ];
                            residue
                                .coefficients_with(
                                    &coefficients,
                                    &mut output,
                                    options,
                                    &SerialExecutor,
                                    &mut scratch,
                                )
                                .unwrap();
                            let layout = if order == ElementOrder::Natural {
                                EvaluationLayout::Natural
                            } else {
                                EvaluationLayout::BitReversed
                            };
                            let view = EvaluationView::bind(&output, residue.domain(), layout);
                            for row in 0..base.domain().size() {
                                assert_eq!(
                                    (view.get(row)).map(|value| value.reduce()),
                                    (Some(&expected[index + expansion.layout().residues() * row]))
                                        .map(|value| value.reduce())
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
fn residue_domains_retain_inverse_shifts_and_extended_rows() {
    fn check<M: PrimeModulus>() {
        for log in [0, 1, 5] {
            let base = Transform::new(Domain::<M>::new(log).unwrap().subgroup());
            for extra in [0, 1, 4] {
                for shift in [PastaField::ONE, PastaField::ZETA, PastaField::from_u64(7)] {
                    let extended = Domain::new(log + extra).unwrap().coset(shift).unwrap();
                    let expansion = Expansion::new(base, extended, None).unwrap();
                    for residue in 0..expansion.layout().residues() {
                        let domain = expansion
                            .residue(residue, ElementOrder::Natural)
                            .unwrap()
                            .domain();
                        assert_eq!(
                            domain.inverse_shift().reduce(),
                            domain.shift().invert().unwrap().reduce()
                        );
                        for row in 0..base.domain().size() {
                            let extended_row = residue + expansion.layout().residues() * row;
                            let expected =
                                shift.mul(&extended.domain().root().pow_u64(extended_row as u64));
                            let actual = domain
                                .shift()
                                .mul(&domain.domain().root().pow_u64(row as u64));
                            assert_eq!(actual.reduce(), expected.reduce());
                            assert_eq!(
                                domain.inverse_scale(row).reduce(),
                                base.domain()
                                    .domain()
                                    .size_inverse()
                                    .mul(&domain.shift().pow_u64(row as u64).invert().unwrap())
                                    .reduce()
                            );
                        }
                    }
                }
            }
        }
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}

#[test]
fn resolved_expansion_counts_cover_all_storage_modes_under_memory_limits() {
    // Calling these queries from a const function checks that workspace sizing
    // remains available during constant evaluation.
    const fn counts<M: PrimeModulus>(plan: &ExpansionPlan<'_, M>) -> (usize, usize, usize) {
        (
            plan.coefficient_fields(),
            plan.snapshot_fields(),
            plan.scratch_fields(),
        )
    }

    fn check<M: PrimeModulus>() {
        let base = Transform::new(Domain::<M>::new(5).unwrap().subgroup());
        let domain = Domain::new(8)
            .unwrap()
            .coset(PastaField::from_u64(7))
            .unwrap();
        let expansion = Expansion::new(base, domain, None).unwrap();
        let coefficients = inputs(base.domain().size());
        let expected = direct(&coefficients, domain);
        let evaluations = direct(&coefficients, base.domain());
        let mut accepted = 0;
        let mut rejected = 0;
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
                    for limit in [0, 32 * 32, 32 * 512] {
                        let plan = ExpansionPlan::new(
                            expansion,
                            storage,
                            order,
                            InputSupport::Full,
                            input_order,
                            StorageLayout::Fragments {
                                length: nz(4),
                                whole_bank: false,
                            },
                            crate::exec::ExecutionOptions::default()
                                .with_task_budget(crate::exec::TaskBudget::new(5).unwrap())
                                .with_memory_limit(limit),
                        );
                        let plan = match plan {
                            Ok(plan) => plan,
                            Err(FftError::MemoryLimit { .. }) => {
                                rejected += 1;
                                continue;
                            }
                            Err(error) => panic!("unexpected planning error: {error:?}"),
                        };
                        accepted += 1;
                        let (workspace, snapshot, scratch) = counts(&plan);
                        assert!(
                            (workspace + scratch) * core::mem::size_of::<PastaField<M>>() <= limit
                        );
                        assert!(snapshot <= scratch);
                        let mut input = if storage == ExpansionStorage::Coefficients {
                            coefficients.clone()
                        } else {
                            evaluations.clone()
                        };
                        if input_order == ElementOrder::BitReversed {
                            let natural = input.clone();
                            for (index, value) in input.iter_mut().enumerate() {
                                *value = natural[reverse(index, base.domain().size().ilog2())];
                            }
                        }
                        let mut output = vec![PastaField::ONE; domain.size()];
                        let mut workspace = vec![PastaField::ONE; workspace];
                        let mut scratch = vec![PastaField::ONE; scratch];
                        if matches!(storage, ExpansionStorage::DisposableInput { .. }) {
                            plan.execute_disposable(
                                &mut input,
                                &mut output,
                                None,
                                &mut scratch,
                                &SerialExecutor,
                            );
                        } else {
                            plan.execute(
                                &input,
                                &mut output,
                                &mut workspace,
                                None,
                                &mut scratch,
                                &SerialExecutor,
                            );
                        }
                        let layout = if order == ExpansionOrder::Residues {
                            EvaluationLayout::Residues(expansion.layout())
                        } else {
                            EvaluationLayout::BitReversed
                        };
                        let view = EvaluationView::bind(&output, domain, layout);
                        for (row, value) in expected.iter().enumerate() {
                            assert_eq!(view.get(row).unwrap().reduce(), value.reduce());
                        }
                        assert_eq!(counts(&plan), (workspace.len(), snapshot, scratch.len()));
                    }
                }
            }
        }
        assert!(accepted > 0 && rejected > 0);
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}

fn short_bit_reversed_expansions<M: PrimeModulus, E: Executor>(executor: &E) {
    for log in [0, 4, 8, 11] {
        let subgroup = Domain::<M>::new(log).unwrap().subgroup();
        let domain = Domain::new(log + 3)
            .unwrap()
            .coset(PastaField::ZETA)
            .unwrap();
        let prepared = Prepared::new(subgroup);
        let mut coefficients = inputs(subgroup.size());
        coefficients[0] = PastaField::from_u64(9);
        let factors = inputs(domain.size());
        for len in [0, 1, 10, subgroup.size() / 16, subgroup.size() / 16 + 1] {
            if len > subgroup.size() {
                continue;
            }
            let expected = reference_coset(&coefficients[..len], domain);
            for with_tables in [false, true] {
                let base = if with_tables {
                    // Exercise reconstruction from the opposite table direction.
                    Tables {
                        inverse: Some(&prepared.inverse),
                        ..Tables::default()
                    }
                    .bind(subgroup)
                } else {
                    Transform::new(subgroup)
                };
                for normalization in [
                    None,
                    Some(ExpansionScaleNormalization::Coefficients),
                    Some(ExpansionScaleNormalization::UnscaledInverse),
                ] {
                    let mut scales = vec![PastaField::ZERO; domain.size()];
                    let expansion = Expansion::new(base, domain, None).unwrap();
                    let expansion = if let Some(normalization) = normalization {
                        expansion.with_scales(
                            ExpansionScales::prepare(
                                subgroup.size(),
                                domain,
                                normalization,
                                &mut scales,
                            )
                            .unwrap(),
                        )
                    } else {
                        expansion
                    };
                    let operation = ExpansionPlan::with_strategy(
                        expansion,
                        ExpansionStorage::Coefficients,
                        ExpansionOrder::BitReversed,
                        InputSupport::Prefix(len),
                        ElementOrder::Natural,
                        nz(4),
                        Codelet::Radix2,
                    )
                    .unwrap();
                    let mut ordered_factors = vec![PastaField::ZERO; domain.size()];
                    for (row, factor) in factors.iter().enumerate() {
                        ordered_factors[EvaluationLayout::BitReversed
                            .index(row, domain.size())
                            .unwrap()] = *factor;
                    }
                    let mut output = vec![PastaField::ONE; domain.size()];
                    let mut scratch =
                        vec![PastaField::ONE; operation.scratch_fields_with(nz(5)) + 1];
                    operation.execute_with(
                        &coefficients[..len],
                        &mut output,
                        &mut [],
                        None,
                        &mut scratch,
                        nz(5),
                        executor,
                    );
                    let view = EvaluationView::bind(&output, domain, EvaluationLayout::BitReversed);
                    for (row, expected) in expected.iter().enumerate() {
                        assert_eq!(
                            (view.get(row)).map(|value| value.reduce()),
                            (Some(expected)).map(|value| value.reduce())
                        );
                    }
                    operation.execute_with(
                        &coefficients[..len],
                        &mut output,
                        &mut [],
                        Some(&ordered_factors),
                        &mut scratch,
                        nz(5),
                        executor,
                    );
                    let view = EvaluationView::bind(&output, domain, EvaluationLayout::BitReversed);
                    for (row, expected) in expected.iter().enumerate() {
                        assert_eq!(
                            (view.get(row)).map(|value| value.reduce()),
                            (Some(&expected.mul(&factors[row]))).map(|value| value.reduce())
                        );
                    }
                    // A retained inverse can also be a short prefix of this base.
                    if len.is_power_of_two() {
                        let raw: Vec<_> = coefficients[..len]
                            .iter()
                            .map(|value| value.mul(&PastaField::<_>::from_u64(len as u64)))
                            .collect();
                        let mut product = vec![PastaField::ZERO; domain.size()];
                        let retained = CoefficientView::new(&raw, InverseScale::Unscaled);
                        operation
                            .with_coefficient_scale(retained.normalization_factor())
                            .execute_with(
                                retained.as_slice(),
                                &mut product,
                                &mut [],
                                Some(&ordered_factors),
                                &mut scratch,
                                nz(5),
                                executor,
                            );
                        assert_eq!(reduced(&product), reduced(&output));
                    }
                    assert_loose_bound(&scratch);
                    assert_eq!(
                        (scratch.last()).map(|value| value.reduce()),
                        (Some(&PastaField::<_>::ONE)).map(|value| value.reduce())
                    );
                }
            }
        }
    }
}

#[test]
fn short_bit_reversed_products_match_reference_at_pruning_boundary() {
    short_bit_reversed_expansions::<PallasBase, _>(&SerialExecutor);
    short_bit_reversed_expansions::<PallasScalar, _>(&Threads);
}

#[test]
fn short_bit_reversed_residues_preserve_loose_bounds_on_panic() {
    let base = Transform::new(Domain::<PallasBase>::new(8).unwrap().subgroup());
    let domain = Domain::new(11).unwrap().coset(Fp::ZETA).unwrap();
    let expansion = Expansion::new(base, domain, None).unwrap();
    let residue = expansion.residue(5, ElementOrder::BitReversed).unwrap();
    let input = inputs(10);
    let expected = direct(&input, residue.domain());
    let options = OPTIONS;
    let mut output = vec![Fp::ONE; base.domain().size()];
    let mut scratch = vec![
        Fp::ONE;
        residue
            .scratch_requirements_with(options)
            .unwrap()
            .field_elements
            + 1
    ];
    let joins = CountJoins::default();
    residue
        .coefficients_with(&input, &mut output, options, &joins, &mut scratch)
        .unwrap();
    let view = EvaluationView::bind(&output, residue.domain(), EvaluationLayout::BitReversed);
    for (row, expected) in expected.iter().enumerate() {
        assert_eq!(
            (view.get(row)).map(|value| value.reduce()),
            (Some(expected)).map(|value| value.reduce())
        );
    }
    let count = joins.take();
    assert!(count > 0);
    for index in 0..count {
        let executor = FailAt {
            calls: AtomicUsize::new(0),
            index,
        };
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                residue.coefficients_with(&input, &mut output, options, &executor, &mut scratch)
            }))
            .is_err()
        );
        assert_loose_bound(&output);
        assert_loose_bound(&scratch);
        assert_eq!(
            (scratch.last()).map(|value| value.reduce()),
            (Some(&<Fp>::ONE)).map(|value| value.reduce())
        );
    }
}

#[test]
fn expansion_metadata_storage_errors_and_panics() {
    let base = Transform::new(Domain::<PallasBase>::new(5).unwrap().subgroup());
    let domain = Domain::new(7).unwrap().coset(Fp::from_u64(7)).unwrap();
    let expansion = Expansion::new(base, domain, None).unwrap();
    let coefficients = inputs(base.domain().size());
    let evaluations = direct(&coefficients, base.domain());
    for order in [ExpansionOrder::Residues, ExpansionOrder::BitReversed] {
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
            let operation = ExpansionPlan::with_strategy(
                expansion,
                storage,
                order,
                InputSupport::Full,
                ElementOrder::Natural,
                nz(4),
                Codelet::Radix2,
            )
            .unwrap();
            let mut output = vec![Fp::ONE; domain.size()];
            let mut scratch = vec![Fp::ONE; operation.scratch_fields_with(nz(5))];
            let mut workspace = vec![Fp::ONE; operation.coefficient_fields()];
            let mut disposable = evaluations.clone();
            let execute = |input: &mut [Fp],
                           output: &mut [Fp],
                           workspace: &mut [Fp],
                           scratch: &mut [Fp],
                           executor: &CountJoins| {
                if matches!(storage, ExpansionStorage::DisposableInput { .. }) {
                    operation.execute_disposable_with(
                        input,
                        output,
                        None,
                        scratch,
                        nz(5),
                        executor,
                    );
                } else {
                    let input = if storage == ExpansionStorage::Coefficients {
                        &coefficients
                    } else {
                        &evaluations
                    };
                    operation.execute_with(
                        input,
                        output,
                        workspace,
                        None,
                        scratch,
                        nz(5),
                        executor,
                    );
                }
            };
            let joins = CountJoins::default();
            execute(
                &mut disposable,
                &mut output,
                &mut workspace,
                &mut scratch,
                &joins,
            );
            for index in 0..joins.take() {
                let mut disposable = evaluations.clone();
                let executor = FailAt {
                    calls: AtomicUsize::new(0),
                    index,
                };
                assert!(
                    catch_unwind(AssertUnwindSafe(|| {
                        if matches!(storage, ExpansionStorage::DisposableInput { .. }) {
                            operation.execute_disposable_with(
                                &mut disposable,
                                &mut output,
                                None,
                                &mut scratch,
                                nz(5),
                                &executor,
                            );
                        } else {
                            let input = if storage == ExpansionStorage::Coefficients {
                                &coefficients
                            } else {
                                &evaluations
                            };
                            operation.execute_with(
                                input,
                                &mut output,
                                &mut workspace,
                                None,
                                &mut scratch,
                                nz(5),
                                &executor,
                            );
                        }
                    }))
                    .is_err()
                );
                for values in [&output, &workspace, &scratch, &disposable] {
                    assert_loose_bound(values);
                }
            }
            if !workspace.is_empty() {
                let short = workspace.len() - 1;
                let before = (
                    disposable.clone(),
                    output.clone(),
                    workspace.clone(),
                    scratch.clone(),
                );
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        execute(
                            &mut disposable,
                            &mut output,
                            &mut workspace[..short],
                            &mut scratch,
                            &joins,
                        );
                    }))
                    .is_err()
                );
                for (actual, expected) in [
                    (&disposable, &before.0),
                    (&output, &before.1),
                    (&workspace, &before.2),
                    (&scratch, &before.3),
                ] {
                    assert_eq!(
                        bento::bytes_of_slice(actual),
                        bento::bytes_of_slice(expected)
                    );
                }
                assert_eq!(joins.take(), 0);
            }
            output.fill(Fp::ONE);
            if !scratch.is_empty() {
                let short = scratch.len() - 1;
                let before = (
                    disposable.clone(),
                    output.clone(),
                    workspace.clone(),
                    scratch.clone(),
                );
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        execute(
                            &mut disposable,
                            &mut output,
                            &mut workspace,
                            &mut scratch[..short],
                            &joins,
                        );
                    }))
                    .is_err()
                );
                assert_eq!(bytes_of_slice(&disposable), bytes_of_slice(&before.0));
                assert_eq!(bytes_of_slice(&output), bytes_of_slice(&before.1));
                assert_eq!(bytes_of_slice(&workspace), bytes_of_slice(&before.2));
                assert_eq!(bytes_of_slice(&scratch), bytes_of_slice(&before.3));
                assert_eq!(joins.take(), 0);
            }
        }
    }
    let mut scales = vec![Fp::ZERO; domain.size()];
    ExpansionScales::prepare(
        base.domain().size(),
        domain,
        ExpansionScaleNormalization::UnscaledInverse,
        &mut scales,
    )
    .unwrap();
    let mut other_values = vec![Fp::ZERO; domain.size()];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = expansion.with_scales(
                ExpansionScales::prepare(
                    base.domain().size(),
                    domain.domain().subgroup(),
                    ExpansionScaleNormalization::UnscaledInverse,
                    &mut other_values,
                )
                .unwrap(),
            );
        }))
        .is_err()
    );
}

#[test]
fn interpolation_checks_lengths_before_mutation_and_preserves_loose_bounds_on_panic() {
    let domain = Domain::<PallasBase>::new(7).unwrap().subgroup();
    let other = domain.domain().coset(Fp::from_u64(7)).unwrap();
    let original = inputs(domain.size());
    for consume in [false, true] {
        let transforms = [domain, domain, other].map(|domain| {
            FftPlan::with_strategy(
                Transform::new(domain),
                TransformRequest::new(Direction::Inverse),
                nz(8),
                Codelet::Radix2,
            )
            .unwrap()
            .with_columns(nz(3), nz(2))
            .unwrap()
        });
        let plan = InterpolationPlan::with_transforms(transforms, consume);
        let mut values: [_; 3] = core::array::from_fn(|_| original.clone());
        let mut scratch: [_; 3] =
            core::array::from_fn(|i| vec![Fp::ONE; plan.snapshot_fields(i).unwrap() + 1]);
        let count = CountJoins::default();
        let short = plan.snapshot_fields(2).unwrap() - 1;
        let [a, b, c] = &mut scratch;
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                plan.execute_with(
                    values.each_mut().map(Vec::as_mut_slice),
                    [a, b, &mut c[..short]],
                    nz(5),
                    &count,
                );
            }))
            .is_err()
        );
        assert!(
            values
                .iter()
                .all(|v| bento::bytes_of_slice(v) == bento::bytes_of_slice(&original))
        );
        assert!(scratch.iter().flatten().all(|v| v.reduce() == Fp::ONE));
        assert_eq!(count.take(), 0);
        plan.execute_with(
            values.each_mut().map(Vec::as_mut_slice),
            scratch.each_mut().map(Vec::as_mut_slice),
            nz(5),
            &count,
        );
        let joins = count.take();
        assert!(joins > 0);
        for index in 0..joins {
            values.iter_mut().for_each(|v| v.copy_from_slice(&original));
            let executor = FailAt {
                calls: AtomicUsize::new(0),
                index,
            };
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    plan.execute_with(
                        values.each_mut().map(Vec::as_mut_slice),
                        scratch.each_mut().map(Vec::as_mut_slice),
                        nz(5),
                        &executor,
                    )
                }))
                .is_err()
            );
            for buffer in values.iter().chain(&scratch) {
                assert_loose_bound(buffer);
            }
            assert!(
                scratch
                    .iter()
                    .all(|s| s.last().map(|value| value.reduce()) == Some(Fp::ONE))
            );
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
        for consume in [false, true] {
            // One task exercises fused interpolation; multiple tasks exercise
            // separate class transforms with the same independent polynomial sum.
            for tasks in [1, 5] {
                let mut values: [_; 5] = core::array::from_fn(|i| {
                    if i == 0 {
                        direct(&coefficients, output_domain)
                    } else {
                        let natural = direct(&lift_coefficients[i - 1], domains[i - 1]);
                        if i % 2 == 1 {
                            (0..natural.len())
                                .map(|j| natural[reverse(j, natural.len().ilog2())])
                                .collect()
                        } else {
                            natural
                        }
                    }
                });
                let transforms = core::array::from_fn(|i| {
                    FftPlan::with_strategy(
                        Transform::new(if i == 0 {
                            output_domain
                        } else {
                            domains[i - 1]
                        }),
                        TransformRequest {
                            input_order: if i % 2 == 1 {
                                ElementOrder::BitReversed
                            } else {
                                ElementOrder::Natural
                            },
                            ..TransformRequest::new(Direction::Inverse)
                        },
                        nz(4),
                        Codelet::Radix2,
                    )
                    .unwrap()
                });
                let plan = InterpolationPlan::<_, 5>::with_transforms(transforms, consume);
                let mut scratch: [_; 5] = core::array::from_fn(|i| {
                    vec![PastaField::ONE; plan.snapshot_fields(i).unwrap() + 1]
                });
                plan.execute_with(
                    values.each_mut().map(Vec::as_mut_slice),
                    scratch.each_mut().map(Vec::as_mut_slice),
                    nz(tasks),
                    &Threads,
                );
                assert_eq!(reduced(&values[0]), reduced(&expected));
                if !consume {
                    for (actual, expected) in values[1..].iter().zip(&lift_coefficients) {
                        assert_eq!(reduced(actual), reduced(expected));
                    }
                }
                for buffer in &scratch {
                    assert_eq!(
                        (buffer.last()).map(|value| value.reduce()),
                        (Some(&PastaField::<_>::ONE)).map(|value| value.reduce())
                    );
                    assert_loose_bound(buffer);
                }
            }
        }
    }
}

#[test]
fn equal_size_parallel_and_destructive_interpolation_match_polynomial_sums() {
    interpolation::<PallasBase>();
    interpolation::<PallasScalar>();
}

#[test]
fn prepared_subgroup_copy_preserves_validation_and_skips_scheduling() {
    fn check<M: PrimeModulus>() {
        let base = Transform::new(Domain::<M>::new(5).unwrap().subgroup());
        let input = inputs(base.domain().size());
        for normalization in [
            ExpansionScaleNormalization::Coefficients,
            ExpansionScaleNormalization::UnscaledInverse,
        ] {
            let mut scales = vec![PastaField::ZERO; input.len()];
            let scales =
                ExpansionScales::prepare(input.len(), base.domain(), normalization, &mut scales)
                    .unwrap();
            let expansion = Expansion::new(base, base.domain(), None)
                .unwrap()
                .with_scales(scales);
            for order in [ExpansionOrder::Residues, ExpansionOrder::BitReversed] {
                let operation = ExpansionPlan::with_strategy(
                    expansion,
                    ExpansionStorage::ReuseOutput,
                    order,
                    InputSupport::Full,
                    ElementOrder::Natural,
                    nz(4),
                    Codelet::Radix2,
                )
                .unwrap();
                let count = operation.scratch_fields_with(nz(5));
                let mut scratch = vec![PastaField::ONE; count + 1];
                let mut output = vec![PastaField::ONE; input.len()];
                let joins = CountJoins::default();
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _ = operation.execute_with(
                            &input[..input.len() - 1],
                            &mut output,
                            &mut [],
                            None,
                            &mut scratch,
                            nz(5),
                            &joins,
                        );
                    }))
                    .is_err()
                );
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _ = operation.execute_with(
                            &input,
                            &mut output[..input.len() - 1],
                            &mut [],
                            None,
                            &mut scratch,
                            nz(5),
                            &joins,
                        );
                    }))
                    .is_err()
                );
                if count > 0 {
                    assert!(
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let _ = operation.execute_with(
                                &input,
                                &mut output,
                                &mut [],
                                None,
                                &mut scratch[..count - 1],
                                nz(5),
                                &joins,
                            );
                        }))
                        .is_err()
                    );
                }
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _ = operation.execute_with(
                            &input,
                            &mut output,
                            &mut [],
                            Some(&input[..input.len() - 1]),
                            &mut scratch,
                            nz(5),
                            &joins,
                        );
                    }))
                    .is_err()
                );
                assert!(output.iter().all(|v| v.reduce() == PastaField::ONE));
                operation.execute_with(
                    &input,
                    &mut output,
                    &mut [],
                    None,
                    &mut scratch,
                    nz(5),
                    &joins,
                );
                let layout = if order == ExpansionOrder::Residues {
                    EvaluationLayout::Residues(expansion.layout())
                } else {
                    EvaluationLayout::BitReversed
                };
                let view = EvaluationView::bind(&output, base.domain(), layout);
                for (i, value) in input.iter().enumerate() {
                    assert_eq!(
                        (view.get(i)).map(|value| value.reduce()),
                        (Some(value)).map(|value| value.reduce())
                    );
                }
                assert_eq!(joins.take(), 0);
                assert!(scratch.iter().all(|v| v.reduce() == PastaField::ONE));
            }
        }
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}
