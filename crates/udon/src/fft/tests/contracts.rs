use super::*;
use crate::fft::run::ExpansionPlan;
use core::num::NonZeroUsize;

fn bindings<M: PrimeModulus>() {
    let subgroup = Domain::<M>::new(4).unwrap();
    let domain = subgroup.coset(PastaField::from_u64(7)).unwrap();
    let other = subgroup.coset(PastaField::from_u64(9)).unwrap();
    let prepared = Prepared::new(domain);
    let plan = prepared.tables().bind(domain);
    assert!(plan.domain().same_domain(domain));
    assert_eq!(
        plan.tables.forward.unwrap().as_ptr(),
        prepared.forward.as_ptr()
    );
    assert_eq!(
        plan.tables.inverse.unwrap().as_ptr(),
        prepared.inverse.as_ptr()
    );
    assert_eq!(
        plan.tables.inverse_finish.unwrap().as_ptr(),
        prepared.finish.as_ptr()
    );
    assert_eq!(
        plan.tables.inverse_scales.unwrap().as_ptr(),
        prepared.scales.as_ptr()
    );
    // Ordinary twiddles have no shift, so sharing them across cosets is valid.
    let twiddles = Tables {
        forward: Some(&prepared.forward),
        inverse: Some(&prepared.inverse),
        ..Tables::default()
    };
    let plan = twiddles.bind(other);
    let coefficients = inputs(other.size());
    let mut output = coefficients.clone();
    plan.forward_with(&mut output, Strategy::SERIAL, &SerialExecutor, &mut [])
        .unwrap();
    assert_eq!(
        output.iter().map(|v| v.reduce()).collect::<Vec<_>>(),
        direct(&coefficients, other)
            .iter()
            .map(|v| v.reduce())
            .collect::<Vec<_>>()
    );

    let short = Tables {
        forward: Some(&prepared.forward[..3]),
        ..Tables::default()
    };
    assert!(std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| short.bind(domain))).is_err());

    for first in [PastaField::ZERO, PastaField::ONE, PastaField::from_u64(3)] {
        for step in [PastaField::ZERO, PastaField::ONE, domain.shift()] {
            let mut values = [PastaField::ZERO; 17];
            let generated = PowerTable::prepare(first, step, &mut values);
            for (i, value) in generated.as_slice().iter().enumerate() {
                assert_eq!(value.reduce(), first.mul(&step.pow_u64(i as u64)).reduce());
            }
            let bound = PowerTable::bind(first, step, generated.as_slice());
            assert_eq!(bound.as_slice().as_ptr(), generated.as_slice().as_ptr());
        }
    }
    // Both representatives of zero are valid fields, and neither is a coset shift.
    for zero in [
        PastaField::ZERO,
        PastaField::from_montgomery_limbs(M::MODULUS),
    ] {
        assert!(matches!(subgroup.coset(zero), Err(FftError::ZeroShift)));
    }
    for base_size in [1, 4, 16] {
        for normalization in [
            ExpansionScaleNormalization::Coefficients,
            ExpansionScaleNormalization::UnscaledInverse,
        ] {
            let mut values = vec![PastaField::ZERO; domain.size()];
            let generated =
                ExpansionScales::prepare(base_size, domain, normalization, &mut values).unwrap();
            let first = match normalization {
                ExpansionScaleNormalization::Coefficients => PastaField::<M>::ONE,
                ExpansionScaleNormalization::UnscaledInverse => {
                    PastaField::power_of_two_inverse(base_size.ilog2())
                }
            };
            for (index, value) in generated.as_slice().iter().enumerate() {
                let point = domain
                    .shift()
                    .mul(&subgroup.root().pow_u64((index / base_size) as u64));
                assert_eq!(
                    value.reduce(),
                    first
                        .mul(&point.pow_u64((index % base_size) as u64))
                        .reduce()
                );
            }
            let bound =
                ExpansionScales::bind(base_size, domain, normalization, generated.as_slice())
                    .unwrap();
            assert_eq!(bound.as_slice().as_ptr(), generated.as_slice().as_ptr());
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = ExpansionScales::bind(base_size, domain, normalization, &values[..3]);
                }))
                .is_err()
            );
        }
    }
}

#[test]
fn bindings_borrow_trusted_tables_and_check_shapes() {
    bindings::<PallasBase>();
    bindings::<PallasScalar>();
}

fn twiddle_oracle<M: PrimeModulus>() {
    for log in 0..=9 {
        let domain = Domain::<M>::new(log).unwrap();
        for inverse in [false, true] {
            let root = if inverse {
                domain.inverse_root()
            } else {
                domain.root()
            };
            for storage in [TwiddleStorage::Dense, TwiddleStorage::StagePacked] {
                let description = TwiddleDescription {
                    size: domain.size(),
                    inverse,
                    storage,
                };
                // Independent entry exponentiation checks production recurrences
                // and the packed stage offsets against the documented formula.
                let exponents: Vec<_> = match storage {
                    TwiddleStorage::Dense => (0..domain.size() / 2).collect(),
                    TwiddleStorage::StagePacked => (1..=log)
                        .flat_map(|stage| {
                            let size = 1 << stage;
                            (0..size / 2).map(move |i| i * domain.size() / size)
                        })
                        .collect(),
                };
                let expected: Vec<_> = exponents.iter().map(|&i| root.pow_u64(i as u64)).collect();
                let mut values = vec![PastaField::ZERO; description.requirements().unwrap()];
                assert_eq!(
                    reduced(
                        TwiddleTable::prepare(description, &mut values)
                            .unwrap()
                            .as_slice()
                    ),
                    reduced(&expected)
                );
                TwiddleTable::bind(description, &expected).unwrap();
                if !values.is_empty() {
                    assert!(
                        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                            let _ = TwiddleTable::bind(description, &values[1..]);
                        }))
                        .is_err()
                    );
                }
            }
        }
    }
}

#[test]
fn twiddle_recurrences_match_independent_exponentiation() {
    twiddle_oracle::<PallasBase>();
    twiddle_oracle::<PallasScalar>();
}

fn product_domain<M: PrimeModulus>() {
    let base = Transform::new(Domain::<M>::new(3).unwrap().subgroup());
    let domain = Domain::new(5)
        .unwrap()
        .coset(PastaField::from_u64(7))
        .unwrap();
    let expansion = Expansion::new(base, domain, None).unwrap();
    let coefficients = inputs(base.domain().size());
    let values = inputs(domain.size());
    let options = ExpansionStrategy {
        max_residue_tasks: 2,
        transform: Strategy {
            tile_len: 2,
            columns_per_task: 2,
            max_tasks: 4,
        },
    };
    let required = expansion
        .coefficient_scratch_with(options)
        .unwrap()
        .field_elements;
    let mut scratch = vec![PastaField::ONE; required + 1];
    let mut output = vec![PastaField::ONE; domain.size()];
    let joins = CountJoins::default();
    // A different coset and a rotation of the same point set both have the
    // wrong ordered evaluation domain. Matching dimensions cannot establish it.
    for other in [
        domain.domain().coset(PastaField::from_u64(9)).unwrap(),
        domain
            .domain()
            .coset(domain.shift().mul(&domain.domain().root()))
            .unwrap(),
    ] {
        let factor = EvaluationView::bind(
            &values,
            other,
            EvaluationLayout::Residues(expansion.layout()),
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = expansion.short_product_with(
                    &coefficients,
                    factor,
                    &mut output,
                    options,
                    &joins,
                    &mut scratch,
                );
            }))
            .is_err()
        );
        assert!(
            output
                .iter()
                .chain(&scratch)
                .all(|v| v.reduce() == PastaField::ONE)
        );
        assert_eq!(joins.take(), 0);
    }
    let factor = EvaluationView::bind(&values, domain, EvaluationLayout::Natural);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = expansion.short_product_with(
                &coefficients,
                factor,
                &mut output,
                options,
                &joins,
                &mut scratch,
            );
        }))
        .is_err()
    );
    assert!(
        output
            .iter()
            .chain(&scratch)
            .all(|v| v.reduce() == PastaField::ONE)
    );
    assert_eq!(joins.take(), 0);
    expansion
        .short_product_with(
            &coefficients,
            expansion.view(&values),
            &mut output,
            options,
            &joins,
            &mut scratch,
        )
        .unwrap();
    let result = expansion.view(&output);
    let factor = expansion.view(&values);
    for (row, evaluation) in direct(&coefficients, domain).iter().enumerate() {
        assert_eq!(
            (result.get(row)).map(|value| value.reduce()),
            (Some(&evaluation.mul(factor.get(row).unwrap()))).map(|value| value.reduce())
        );
    }
    assert_eq!(
        (scratch.last()).map(|value| value.reduce()),
        (Some(&PastaField::<_>::ONE)).map(|value| value.reduce())
    );
}

#[test]
fn short_products_require_the_same_ordered_coset_before_execution() {
    product_domain::<PallasBase>();
    product_domain::<PallasScalar>();
}

fn retained_with_base_tables<M: PrimeModulus>() {
    let domain = Domain::<M>::new(3).unwrap().subgroup();
    let extended = Domain::new(5)
        .unwrap()
        .coset(PastaField::from_u64(7))
        .unwrap();
    let prepared = Prepared::new(domain);
    let coefficients = inputs(domain.size());
    let evaluations = direct(&coefficients, domain);
    let expected = direct(&coefficients, extended);
    for mask in 0..16 {
        let base = Tables {
            forward: (mask & 1 != 0).then_some(prepared.forward.as_slice()),
            inverse: (mask & 2 != 0).then_some(prepared.inverse.as_slice()),
            inverse_finish: (mask & 4 != 0).then_some(prepared.finish.as_slice()),
            inverse_scales: (mask & 8 != 0).then_some(prepared.scales.as_slice()),
        }
        .bind(domain);
        let expansion = Expansion::new(base, extended, None).unwrap();
        for scale in [InverseScale::Normalized, InverseScale::Unscaled] {
            for order in [ExpansionOrder::Residues, ExpansionOrder::BitReversed] {
                let mut retained = vec![PastaField::ONE; domain.size()];
                let view = {
                    let operation = ExpansionPlan::with_strategy(
                        expansion,
                        ExpansionStorage::CoefficientWorkspace { scale },
                        order,
                        InputSupport::Full,
                        ElementOrder::Natural,
                        NonZeroUsize::new(domain.size()).unwrap(),
                        Codelet::Radix2,
                    )
                    .unwrap();
                    let layout = if order == ExpansionOrder::Residues {
                        EvaluationLayout::Residues(expansion.layout())
                    } else {
                        EvaluationLayout::BitReversed
                    };
                    let mut output = vec![PastaField::ONE; extended.size()];
                    let mut scratch = [PastaField::ONE];
                    let view = operation
                        .execute_with(
                            &evaluations,
                            &mut output,
                            &mut retained,
                            None,
                            &mut scratch,
                            NonZeroUsize::MIN,
                            &SerialExecutor,
                        )
                        .unwrap();
                    for (row, value) in expected.iter().enumerate() {
                        assert_eq!(
                            (EvaluationView::bind(&output, extended, layout).get(row))
                                .map(|value| value.reduce()),
                            (Some(value)).map(|value| value.reduce())
                        );
                    }
                    assert_eq!(
                        bytes_of_slice(&scratch),
                        bytes_of_slice(&[PastaField::<M>::ONE])
                    );
                    // Output and scratch can be reused while the view lives.
                    output.fill(PastaField::ZERO);
                    scratch.fill(PastaField::ZERO);
                    view
                };
                assert_eq!(view.scale(), scale);
                for (actual, coefficient) in view.as_slice().iter().zip(&coefficients) {
                    assert_eq!(
                        (actual.mul(&view.normalization_factor())).reduce(),
                        (*coefficient).reduce()
                    );
                }
            }
        }
    }
}

#[test]
fn retained_coefficients_work_with_every_base_table_subset_and_outlive_output() {
    retained_with_base_tables::<PallasBase>();
    retained_with_base_tables::<PallasScalar>();
}

fn reused_cosets<M: PrimeModulus>() {
    for log in [0, 4] {
        let subgroup = Domain::<M>::new(log).unwrap();
        let original = subgroup.coset(PastaField::from_u64(7)).unwrap();
        let prepared = Prepared::new(original);
        for mask in 0..16 {
            let bound = Tables {
                forward: (mask & 1 != 0).then_some(prepared.forward.as_slice()),
                inverse: (mask & 2 != 0).then_some(prepared.inverse.as_slice()),
                inverse_finish: (mask & 4 != 0).then_some(prepared.finish.as_slice()),
                inverse_scales: (mask & 8 != 0).then_some(prepared.scales.as_slice()),
            }
            .bind(original);
            for shift in [PastaField::from_u64(7), PastaField::ONE, PastaField::ZETA] {
                let domain = subgroup.coset(shift).unwrap();
                let rebound = bound.for_coset(domain);
                let tables = rebound.tables;
                assert_eq!(
                    tables.forward.map(|s| s.as_ptr()),
                    bound.tables.forward.map(|s| s.as_ptr())
                );
                assert_eq!(
                    tables.inverse.map(|s| s.as_ptr()),
                    bound.tables.inverse.map(|s| s.as_ptr())
                );
                assert_eq!(
                    tables.inverse_finish.is_some(),
                    mask & 4 != 0 && shift.reduce() == original.shift().reduce()
                );
                assert_eq!(
                    tables.inverse_scales.is_some(),
                    mask & 8 != 0 && shift.reduce() == original.shift().reduce()
                );
                let coefficients = inputs(domain.size());
                let mut output = coefficients.clone();
                rebound
                    .forward_with(&mut output, Strategy::SERIAL, &SerialExecutor, &mut [])
                    .unwrap();
                assert_eq!(reduced(&output), reduced(&direct(&coefficients, domain)));
                rebound
                    .inverse_with(&mut output, Strategy::SERIAL, &SerialExecutor, &mut [])
                    .unwrap();
                assert_eq!(reduced(&output), reduced(&coefficients));
            }
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    let _ = bound.for_coset(Domain::new(log + 1).unwrap().subgroup());
                }))
                .is_err()
            );
        }
    }
}

#[test]
fn tables_reuse_borrows_across_cosets_and_drop_shift_dependent_entries() {
    reused_cosets::<PallasBase>();
    reused_cosets::<PallasScalar>();
}
