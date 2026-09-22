use super::*;
use crate::fft::run::FftPlan;
use core::num::NonZeroUsize;

fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn ordered<M: PrimeModulus>(values: &[PastaField<M>], order: ElementOrder) -> Vec<PastaField<M>> {
    (0..values.len())
        .map(|i| {
            values[if order == ElementOrder::Natural {
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

fn operations<M: PrimeModulus>() {
    for log in 0..=6 {
        for shift in [PastaField::ONE, PastaField::ZETA, PastaField::from_u64(7)] {
            let domain = Domain::<M>::new(log).unwrap().coset(shift).unwrap();
            let plan = Transform::new(domain);
            let input = inputs(domain.size());
            let forward = direct(&input, domain);
            let inverse = inverse_direct(&input, domain, true);
            for columns in [false, true] {
                for direction in [Direction::Forward, Direction::Inverse] {
                    let expected = if direction == Direction::Forward {
                        &forward
                    } else {
                        &inverse
                    };
                    for input_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                        for output_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                            for codelet in [Codelet::Radix2, Codelet::Radix4, Codelet::Radix8] {
                                for input_storage in [InputStorage::InPlace, InputStorage::Preserve]
                                {
                                    let request = TransformRequest {
                                        input_storage,
                                        input_order,
                                        output_order,
                                        ..TransformRequest::new(direction)
                                    };
                                    let mut operation =
                                        FftPlan::with_strategy(plan, request, nz(4), codelet)
                                            .unwrap();
                                    if columns {
                                        operation = operation.with_columns(nz(3), nz(3)).unwrap();
                                    }
                                    let original = ordered(&input, input_order);
                                    for scatter in [false, true] {
                                        let operation = if scatter {
                                            operation.with_scatter_initialization()
                                        } else {
                                            operation
                                        };
                                        let mut output = original.clone();
                                        let mut scratch =
                                            vec![PastaField::ONE; operation.retained_fields() + 1];
                                        operation
                                            .execute_with(
                                                (input_storage == InputStorage::Preserve)
                                                    .then_some(original.as_slice()),
                                                &mut output,
                                                None,
                                                &mut scratch,
                                                nz(3),
                                                &SerialExecutor,
                                            )
                                            .unwrap();
                                        assert_eq!(
                                            output,
                                            ordered(expected, output_order),
                                            "log={log}, columns={columns}, codelet={codelet:?}, request={request:?}"
                                        );
                                        assert_canonical(&scratch);
                                        assert_eq!(scratch.last(), Some(&PastaField::ONE));
                                    }
                                }
                            }
                        }
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
                    ordered(&input, ElementOrder::BitReversed)
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
                        ordered(&expected, ElementOrder::BitReversed)
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
    let plan = Transform::new(domain);
    let values = inputs(domain.size());
    let factors = direct(&values, domain);
    for len in [0, 1, 2, 3, 7, 16, 31, 32] {
        for columns in [false, true] {
            for direction in [Direction::Forward, Direction::Inverse] {
                for output_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                    for inverse_scale in [InverseScale::Normalized, InverseScale::Unscaled] {
                        if direction == Direction::Forward
                            && inverse_scale == InverseScale::Unscaled
                        {
                            continue;
                        }
                        let expected = if direction == Direction::Forward {
                            direct(&values[..len], domain)
                        } else {
                            inverse_direct(
                                &values[..len],
                                domain,
                                inverse_scale == InverseScale::Normalized,
                            )
                        };
                        for input_storage in [InputStorage::InPlace, InputStorage::Preserve] {
                            let request = TransformRequest {
                                input_storage,
                                support: InputSupport::Prefix(len),
                                output_order,
                                inverse_scale,
                                ..TransformRequest::new(direction)
                            };
                            let mut operation =
                                FftPlan::with_strategy(plan, request, nz(4), Codelet::Radix2)
                                    .unwrap();
                            if columns {
                                operation = operation.with_columns(nz(3), nz(3)).unwrap();
                            }
                            let input =
                                (input_storage == InputStorage::Preserve).then_some(&values[..len]);
                            let mut scratch =
                                vec![PastaField::ONE; operation.retained_fields() + 1];
                            let mut output = values.clone();
                            operation
                                .execute_with(
                                    input,
                                    &mut output,
                                    None,
                                    &mut scratch,
                                    nz(3),
                                    &SerialExecutor,
                                )
                                .unwrap();
                            assert_eq!(
                                output,
                                ordered(&expected, output_order),
                                "len={len}, request={request:?}"
                            );
                            let factor = ordered(&factors, output_order);
                            output.copy_from_slice(&values);
                            operation
                                .execute_with(
                                    input,
                                    &mut output,
                                    Some(&factor),
                                    &mut scratch,
                                    nz(3),
                                    &SerialExecutor,
                                )
                                .unwrap();
                            let product: Vec<_> = expected
                                .iter()
                                .zip(&factors)
                                .map(|(a, b)| a.mul(b))
                                .collect();
                            assert_eq!(output, ordered(&product, output_order));
                            assert_eq!(scratch.last(), Some(&PastaField::ONE));
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
    let plan = Transform::new(domain);
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
                    let operation = FftPlan::with_strategy(
                        plan,
                        TransformRequest::new(direction),
                        nz(4),
                        Codelet::Radix2,
                    )
                    .unwrap()
                    .with_contiguous_permutation()
                    .with_twiddles(table);
                    let mut output = input.clone();
                    let mut scratch = vec![PastaField::ZERO; operation.retained_fields()];
                    operation
                        .execute_with(None, &mut output, None, &mut scratch, nz(3), executor)
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
    let operation = FftPlan::with_strategy(
        plan,
        TransformRequest::new(Direction::Forward),
        nz(4),
        Codelet::Radix2,
    )
    .unwrap()
    .with_forward_scales(scales)
    .unwrap()
    .with_contiguous_permutation();
    let mut output = input.clone();
    operation
        .execute_with(None, &mut output, None, &mut [], nz(3), &Threads)
        .unwrap();
    assert_eq!(output, direct(&input, domain));
}

#[test]
fn twiddle_shapes_strides_directions_and_coset_powers_are_compatible() {
    power_tables::<PallasBase, _>(&SerialExecutor);
    power_tables::<PallasScalar, _>(&SerialExecutor);
    power_tables::<PallasBase, _>(&Threads);
    power_tables::<PallasScalar, _>(&Threads);
}

fn bound_plan_tables<M: PrimeModulus>() {
    for log in [0, 1, 2, 3, 8] {
        for shift in [
            PastaField::ONE,
            PastaField::ZETA,
            PastaField::ZETA_INVERSE,
            PastaField::from_u64(7),
        ] {
            let domain = Domain::<M>::new(log).unwrap().coset(shift).unwrap();
            let prepared = Prepared::new(domain);
            let input = inputs(domain.size());
            let forward = direct(&input, domain);
            let inverse = inverse_direct(&input, domain, true);
            let raw = inverse_direct(&input, domain, false);
            for mask in 0..16 {
                let plan = Tables {
                    forward: (mask & 1 != 0).then_some(prepared.forward.as_slice()),
                    inverse: (mask & 2 != 0).then_some(prepared.inverse.as_slice()),
                    inverse_finish: (mask & 4 != 0).then_some(prepared.finish.as_slice()),
                    inverse_scales: (mask & 8 != 0).then_some(prepared.scales.as_slice()),
                }
                .bind(domain)
                .unwrap();
                for (direction, inverse_scale, expected) in [
                    (Direction::Forward, InverseScale::Normalized, &forward),
                    (Direction::Inverse, InverseScale::Normalized, &inverse),
                    (Direction::Inverse, InverseScale::Unscaled, &raw),
                ] {
                    for input_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                        for output_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                            for codelet in [Codelet::Radix2, Codelet::Radix4, Codelet::Radix8] {
                                for tasks in [1, 3, 8] {
                                    let request = TransformRequest {
                                        input_order,
                                        output_order,
                                        inverse_scale,
                                        ..TransformRequest::new(direction)
                                    };
                                    let operation =
                                        FftPlan::with_strategy(plan, request, nz(128), codelet)
                                            .unwrap()
                                            .with_contiguous_permutation();
                                    let mut output = ordered(&input, input_order);
                                    // Serial joins still exercise every task region, including
                                    // the paired terminal split at size 256.
                                    operation
                                        .execute_with(
                                            None,
                                            &mut output,
                                            None,
                                            &mut [],
                                            nz(tasks),
                                            &SerialExecutor,
                                        )
                                        .unwrap();
                                    assert_eq!(
                                        output,
                                        ordered(expected, output_order),
                                        "log={log}, mask={mask}, codelet={codelet:?}, tasks={tasks}, request={request:?}"
                                    );
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn bound_plan_table_directions_and_inverse_scales_match_direct_sums() {
    bound_plan_tables::<PallasBase>();
    bound_plan_tables::<PallasScalar>();
}

#[test]
fn transform_configuration_and_scratch_are_checked_before_mutation() {
    let plan = Transform::<PallasBase>::new(Domain::new(4).unwrap().subgroup());
    let request = TransformRequest::new(Direction::Forward);
    assert!(matches!(
        FftPlan::with_strategy(plan, request, nz(3), Codelet::Radix2),
        Err(FftError::InvalidExecution)
    ));
    for request in [
        TransformRequest {
            support: InputSupport::Prefix(17),
            ..request
        },
        TransformRequest {
            support: InputSupport::Prefix(3),
            input_order: ElementOrder::BitReversed,
            ..request
        },
        TransformRequest {
            inverse_scale: InverseScale::Unscaled,
            ..request
        },
    ] {
        assert!(FftPlan::with_strategy(plan, request, nz(4), Codelet::Radix2).is_err());
    }
    let operation = FftPlan::with_strategy(plan, request, nz(4), Codelet::Radix2)
        .unwrap()
        .with_columns(nz(3), nz(2))
        .unwrap();
    let mut values = inputs(16);
    let original = values.clone();
    let mut scratch = vec![Fp::ONE; operation.retained_fields() - 1];
    let joins = CountJoins::default();
    assert!(matches!(
        operation.execute_with(None, &mut values, None, &mut scratch, nz(3), &joins),
        Err(FftError::ScratchTooSmall { .. })
    ));
    assert_eq!(values, original);
    assert!(scratch.iter().all(|v| *v == Fp::ONE));
    assert_eq!(joins.take(), 0);
}

#[test]
fn parallel_panics_restore_all_field_buffers() {
    let domain = Domain::<PallasScalar>::new(8)
        .unwrap()
        .coset(PastaField::from_u64(7))
        .unwrap();
    let prepared = Prepared::new(domain);
    for tables in [Tables::default(), prepared.tables()] {
        let plan = tables.bind(domain).unwrap();
        for columns in [false, true] {
            for codelet in [Codelet::Radix2, Codelet::Radix4, Codelet::Radix8] {
                for direction in [Direction::Forward, Direction::Inverse] {
                    for output_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                        let mut operation = FftPlan::with_strategy(
                            plan,
                            TransformRequest {
                                output_order,
                                ..TransformRequest::new(direction)
                            },
                            nz(4),
                            codelet,
                        )
                        .unwrap();
                        if columns {
                            operation = operation.with_columns(nz(3), nz(3)).unwrap();
                        }
                        let mut values = inputs(domain.size());
                        let mut scratch = vec![PastaField::ONE; operation.retained_fields()];
                        let joins = CountJoins::default();
                        operation
                            .execute_with(None, &mut values, None, &mut scratch, nz(3), &joins)
                            .unwrap();
                        for index in 0..joins.take() {
                            let mut values = inputs(domain.size());
                            let failed = catch_unwind(AssertUnwindSafe(|| {
                                operation.execute_with(
                                    None,
                                    &mut values,
                                    None,
                                    &mut scratch,
                                    nz(3),
                                    &FailAt {
                                        calls: AtomicUsize::new(0),
                                        index,
                                    },
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
}

#[test]
fn finish_tables_preserve_prefix_and_codelet_results() {
    for log in [3, 7] {
        for shift in [Fp::ONE, Fp::ZETA, Fp::ZETA_INVERSE, Fp::from_u64(7)] {
            let domain = Domain::new(log).unwrap().coset(shift).unwrap();
            let prepared = Prepared::new(domain);
            let plan = prepared.tables().bind(domain).unwrap();
            let dense = TwiddleTable::bind(
                TwiddleDescription {
                    size: domain.size(),
                    inverse: true,
                    storage: TwiddleStorage::Dense,
                },
                &prepared.inverse,
            )
            .unwrap();
            for len in [0, 1, 2, 3, domain.size()] {
                let input = inputs(len);
                let expected = inverse_direct(&input, domain, true);
                for (columns, codelet) in [(false, Codelet::Radix8), (true, Codelet::Radix2)] {
                    for input_storage in [InputStorage::Preserve, InputStorage::InPlace] {
                        let request = TransformRequest {
                            input_storage,
                            support: InputSupport::Prefix(len),
                            ..TransformRequest::new(Direction::Inverse)
                        };
                        let mut operation =
                            FftPlan::with_strategy(plan, request, nz(4), codelet).unwrap();
                        if columns {
                            operation = operation.with_columns(nz(3), nz(3)).unwrap();
                        }
                        for operation in [operation, operation.with_twiddles(dense)] {
                            let mut scratch = vec![Fp::ONE; operation.retained_fields() + 1];
                            let mut output = input.clone();
                            output.resize(domain.size(), Fp::ONE);
                            operation
                                .execute_with(
                                    (input_storage == InputStorage::Preserve)
                                        .then_some(input.as_slice()),
                                    &mut output,
                                    None,
                                    &mut scratch,
                                    nz(3),
                                    &SerialExecutor,
                                )
                                .unwrap();
                            assert_eq!(output, expected);
                            assert_eq!(scratch.last(), Some(&Fp::ONE));
                        }
                    }
                }
            }
        }
    }
}
