use super::*;
use crate::fft::run::{ExpansionPlan, FftPlan};
use core::num::NonZeroUsize;

fn nz(value: usize) -> NonZeroUsize {
    NonZeroUsize::new(value).unwrap()
}

fn consume_coefficients<M: PrimeModulus>(view: CoefficientView<'_, M>, ordinary: &[PastaField<M>]) {
    for size in [ordinary.len(), ordinary.len() * 4] {
        for shift in [PastaField::ONE, PastaField::ZETA, PastaField::from_u64(7)] {
            let domain = Domain::<M>::for_size(size).unwrap().coset(shift).unwrap();
            let plan = Transform::new(domain);
            let expected = direct(ordinary, domain);
            let mut output = vec![PastaField::ONE; size];
            plan.forward_prefix_with(
                view,
                &mut output,
                Strategy::SERIAL,
                &SerialExecutor,
                &mut [],
            )
            .unwrap();
            assert_eq!(output, expected);
            if size == ordinary.len() {
                plan.forward_into_with(
                    view,
                    &mut output,
                    Strategy::SERIAL,
                    &SerialExecutor,
                    &mut [],
                )
                .unwrap();
                assert_eq!(output, expected);
            }
            let mut scales = vec![PastaField::ZERO; size];
            let scales = PowerTable::prepare(PastaField::ONE, shift, &mut scales).unwrap();
            for columns in [false, true] {
                for scatter in [false, true] {
                    for order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                        let mut operation = FftPlan::with_strategy(
                            plan,
                            TransformRequest {
                                support: if size == ordinary.len() {
                                    InputSupport::Full
                                } else {
                                    InputSupport::Prefix(ordinary.len())
                                },
                                input_storage: InputStorage::Preserve,
                                output_order: order,
                                ..TransformRequest::new(Direction::Forward)
                            },
                            nz(4),
                            Codelet::Radix2,
                        )
                        .unwrap()
                        .with_input_scale(view.normalization_factor())
                        .unwrap();
                        if columns {
                            operation = operation.with_columns(nz(2), nz(3)).unwrap();
                        }
                        if scatter {
                            operation = operation.with_scatter_initialization();
                        }
                        let factor_values = vec![PastaField::from_u64(11); size];
                        let layout = if order == ElementOrder::Natural {
                            EvaluationLayout::Natural
                        } else {
                            EvaluationLayout::BitReversed
                        };
                        for operation in [operation, operation.with_forward_scales(scales).unwrap()]
                        {
                            let mut scratch =
                                vec![PastaField::ONE; operation.retained_fields() + 1];
                            for factor in [None, Some(factor_values.as_slice())] {
                                operation
                                    .execute_with(
                                        Some(view.as_slice()),
                                        &mut output,
                                        factor,
                                        &mut scratch,
                                        nz(3),
                                        &SerialExecutor,
                                    )
                                    .unwrap();
                                let result = EvaluationView::bind(&output, domain, layout).unwrap();
                                for (row, value) in expected.iter().enumerate() {
                                    let expected = if factor.is_some() {
                                        value.mul(&factor_values[0])
                                    } else {
                                        *value
                                    };
                                    assert_eq!(result.get(row), Some(&expected));
                                }
                                assert_eq!(scratch.last(), Some(&PastaField::ONE));
                            }
                        }
                    }
                }
            }

            let extended = Domain::for_size(size * 2).unwrap().coset(shift).unwrap();
            let base = Transform::new(domain.domain().subgroup());
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
                    .coefficients_with(
                        view,
                        &mut output,
                        ExpansionStrategy::SERIAL,
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
                    .short_product_with(
                        view,
                        expansion.view(&factor_values).unwrap(),
                        &mut output,
                        ExpansionStrategy::SERIAL,
                        &SerialExecutor,
                        &mut [],
                    )
                    .unwrap();
                let result = expansion.view(&output).unwrap();
                for (row, value) in expected.iter().enumerate() {
                    assert_eq!(result.get(row), Some(&value.mul(&factor_values[0])));
                }
                for order in [ExpansionOrder::Residues, ExpansionOrder::BitReversed] {
                    let operation = ExpansionPlan::with_strategy(
                        expansion,
                        ExpansionStorage::Coefficients,
                        order,
                        InputSupport::Prefix(view.as_slice().len()),
                        ElementOrder::Natural,
                        nz(4),
                        Codelet::Radix2,
                    )
                    .unwrap()
                    .with_coefficient_scale(view.normalization_factor())
                    .unwrap();
                    let layout = if order == ExpansionOrder::Residues {
                        EvaluationLayout::Residues(expansion.layout())
                    } else {
                        EvaluationLayout::BitReversed
                    };
                    let mut scratch =
                        vec![PastaField::ONE; operation.scratch_fields_with(nz(3)).unwrap() + 1];
                    for factor in [None, Some(factor_values.as_slice())] {
                        operation
                            .execute_with(
                                view.as_slice(),
                                &mut output,
                                &mut [],
                                factor,
                                &mut scratch,
                                nz(3),
                                &SerialExecutor,
                            )
                            .unwrap();
                        let result = EvaluationView::bind(&output, extended, layout).unwrap();
                        for (row, value) in expected.iter().enumerate() {
                            let expected = if factor.is_some() {
                                value.mul(&factor_values[0])
                            } else {
                                *value
                            };
                            assert_eq!(result.get(row), Some(&expected));
                        }
                        assert_eq!(scratch.last(), Some(&PastaField::ONE));
                    }
                    let inner_order = if order == ExpansionOrder::Residues {
                        ElementOrder::Natural
                    } else {
                        ElementOrder::BitReversed
                    };
                    for residue in 0..expansion.layout().residues() {
                        let operation = expansion.residue(residue, inner_order).unwrap();
                        let mut output = vec![PastaField::ZERO; size];
                        operation
                            .coefficients_with(
                                view,
                                &mut output,
                                Strategy::SERIAL,
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
        let expansion = Expansion::new(Transform::new(domain), domain, None).unwrap();
        for scale in [InverseScale::Normalized, InverseScale::Unscaled] {
            let mut retained = evaluations.clone();
            let mut output = vec![PastaField::ZERO; size];
            let view = ExpansionPlan::with_strategy(
                expansion,
                ExpansionStorage::DisposableInput { scale },
                ExpansionOrder::Residues,
                InputSupport::Full,
                ElementOrder::Natural,
                nz(size),
                Codelet::Radix2,
            )
            .unwrap()
            .execute_disposable_with(
                &mut retained,
                &mut output,
                None,
                &mut [],
                nz(1),
                &SerialExecutor,
            )
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
    let plan = Transform::new(domain);
    let coefficients = inputs(domain.size());
    let view = CoefficientView::normalized(&coefficients);
    let joins = CountJoins::default();
    let mut output = vec![PastaField::ONE; domain.size()];
    let mut scratch = vec![PastaField::ONE; 128];
    assert!(matches!(
        plan.forward_into_with(
            CoefficientView::normalized(&coefficients[..1]),
            &mut output,
            Strategy::SERIAL,
            &joins,
            &mut scratch,
        ),
        Err(FftError::LengthMismatch { .. })
    ));
    let other = domain.domain().coset(PastaField::from_u64(7)).unwrap();
    let factor = EvaluationView::bind(&coefficients, other, EvaluationLayout::Natural).unwrap();
    let expansion = Expansion::new(plan, domain, None).unwrap();
    assert_eq!(
        expansion.short_product_with(
            view,
            factor,
            &mut output,
            ExpansionStrategy::SERIAL,
            &joins,
            &mut scratch
        ),
        Err(FftError::InvalidLayout)
    );
    let small = Transform::new(Domain::new(2).unwrap().subgroup());
    assert!(matches!(
        small.forward_prefix_with(
            view,
            &mut output[..4],
            Strategy::SERIAL,
            &joins,
            &mut scratch
        ),
        Err(FftError::InvalidPrefix { .. })
    ));
    let small = Expansion::new(small, domain, None).unwrap();
    assert!(matches!(
        small.coefficients_with(
            view,
            &mut output,
            ExpansionStrategy::SERIAL,
            &joins,
            &mut scratch
        ),
        Err(FftError::InvalidPrefix { .. })
    ));
    assert!(output.iter().chain(&scratch).all(|v| *v == PastaField::ONE));
    assert_eq!(joins.take(), 0);

    // Empty ordinary prefixes remain valid through the view conversion.
    plan.forward_prefix_with(
        CoefficientView::normalized(&[]),
        &mut output,
        Strategy::SERIAL,
        &joins,
        &mut scratch,
    )
    .unwrap();
    assert!(output.iter().all(|v| *v == PastaField::ZERO));
}
