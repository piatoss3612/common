use super::*;
use crate::fft::run::ExpansionPlan;
use core::num::NonZeroUsize;

fn imports<M: PrimeModulus>() {
    let subgroup = Domain::<M>::new(4).unwrap();
    let domain = subgroup.coset(PastaField::from_u64(7)).unwrap();
    let other = subgroup.coset(PastaField::from_u64(9)).unwrap();
    let prepared = Prepared::new(domain);
    assert!(
        prepared
            .tables()
            .bind(domain)
            .unwrap()
            .domain()
            .same_domain(domain)
    );
    assert!(matches!(
        prepared.tables().bind(other),
        Err(FftError::InvalidTables)
    ));
    // Ordinary twiddles have no shift, so sharing them across cosets is valid.
    let twiddles = Tables {
        forward: Some(&prepared.forward),
        inverse: Some(&prepared.inverse),
        ..Tables::default()
    };
    let plan = twiddles.bind(other).unwrap();
    let coefficients = inputs(other.size());
    let mut output = coefficients.clone();
    plan.forward_with(&mut output, Strategy::serial(), &SerialExecutor, &mut [])
        .unwrap();
    assert_eq!(output, direct(&coefficients, other));

    let invalid: &PastaField<M> = bento::AlignedBytes([0xff; 32]).as_value();
    for family in 0..4 {
        let mut prepared = Prepared::new(domain);
        let values = match family {
            0 => &mut prepared.forward,
            1 => &mut prepared.inverse,
            2 => &mut prepared.finish,
            _ => &mut prepared.scales,
        };
        values[3] = *invalid;
        assert!(matches!(
            prepared.tables().bind(domain),
            Err(FftError::InvalidTables)
        ));
        // The explicit trust path checks shape but intentionally skips contents.
        let trusted = prepared.tables().bind_trusted(domain).unwrap();
        assert!(matches!(trusted.validate(), Err(FftError::InvalidTables)));
    }
    let short = Tables {
        forward: Some(&prepared.forward[..3]),
        ..Tables::default()
    };
    assert!(matches!(
        short.bind(domain),
        Err(FftError::LengthMismatch { .. })
    ));
    assert!(matches!(
        short.bind_trusted(domain),
        Err(FftError::LengthMismatch { .. })
    ));

    for first in [PastaField::ZERO, PastaField::ONE, PastaField::from_u64(3)] {
        for step in [PastaField::ZERO, PastaField::ONE, domain.shift()] {
            let mut values = [PastaField::ZERO; 17];
            let generated = PowerTable::prepare(first, step, &mut values).unwrap();
            for (i, value) in generated.as_slice().iter().enumerate() {
                assert_eq!(*value, first.mul(&step.pow_u64(i as u64)));
            }
            PowerTable::bind(first, step, generated.as_slice()).unwrap();
            values[8] = *invalid;
            assert!(matches!(
                PowerTable::bind(first, step, &values),
                Err(FftError::InvalidTables)
            ));
            assert!(PowerTable::bind_trusted(first, step, &values).is_ok());
        }
    }
    for (first, step) in [(*invalid, PastaField::ONE), (PastaField::ONE, *invalid)] {
        for len in [0, 3] {
            let mut values = vec![PastaField::ONE; len];
            assert!(matches!(
                PowerTable::bind(first, step, &values),
                Err(FftError::InvalidTables)
            ));
            assert!(matches!(
                PowerTable::bind_trusted(first, step, &values),
                Err(FftError::InvalidTables)
            ));
            assert!(matches!(
                PowerTable::prepare(first, step, &mut values),
                Err(FftError::InvalidTables)
            ));
            assert!(values.iter().all(|v| *v == PastaField::ONE));
        }
    }
    assert!(matches!(
        subgroup.coset(*invalid),
        Err(FftError::InvalidShift)
    ));
    let modulus: &PastaField<M> = const {
        let mut bytes = [0; 32];
        let mut i = 0;
        while i < 32 {
            bytes[i] = M::MODULUS[i / 8].to_ne_bytes()[i % 8];
            i += 1;
        }
        bento::AlignedBytes(bytes)
    }
    .as_value();
    assert!(matches!(
        subgroup.coset(*modulus),
        Err(FftError::InvalidShift)
    ));
    assert!(matches!(
        subgroup.coset(PastaField::ZERO),
        Err(FftError::ZeroShift)
    ));

    for base_size in [1, 4, 16] {
        for normalization in [
            ExpansionScaleNormalization::Coefficients,
            ExpansionScaleNormalization::UnscaledInverse,
        ] {
            let mut values = vec![PastaField::ZERO; domain.size()];
            let generated =
                ExpansionScales::prepare(base_size, domain, normalization, &mut values).unwrap();
            let first = match normalization {
                ExpansionScaleNormalization::Coefficients => PastaField::ONE,
                ExpansionScaleNormalization::UnscaledInverse => {
                    PastaField::power_of_two_inverse(base_size.ilog2())
                }
            };
            for (index, value) in generated.as_slice().iter().enumerate() {
                let point = domain
                    .shift()
                    .mul(&subgroup.root().pow_u64((index / base_size) as u64));
                assert_eq!(
                    *value,
                    first.mul(&point.pow_u64((index % base_size) as u64))
                );
            }
            ExpansionScales::bind(base_size, domain, normalization, generated.as_slice()).unwrap();
            values[3] = *invalid;
            assert!(matches!(
                ExpansionScales::bind(base_size, domain, normalization, &values),
                Err(FftError::InvalidTables)
            ));
            assert!(
                ExpansionScales::bind_trusted(base_size, domain, normalization, &values).is_ok()
            );
            assert!(matches!(
                ExpansionScales::bind_trusted(base_size, domain, normalization, &values[..3]),
                Err(FftError::LengthMismatch { .. })
            ));
        }
    }
}

#[test]
fn imports_check_contents_and_domain_while_trusted_bindings_check_shape() {
    imports::<PallasBase>();
    imports::<PallasScalar>();
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
                    TwiddleTable::prepare(description, &mut values)
                        .unwrap()
                        .as_slice(),
                    expected
                );
                TwiddleTable::bind(description, &expected).unwrap();
                for index in [0, values.len() / 2, values.len().saturating_sub(1)] {
                    if let Some(value) = values.get_mut(index) {
                        *value = value.add(&PastaField::ONE);
                        assert!(matches!(
                            TwiddleTable::bind(description, &values),
                            Err(FftError::InvalidTables)
                        ));
                        assert!(TwiddleTable::bind_trusted(description, &values).is_ok());
                        values[index] = expected[index];
                    }
                }
                if !values.is_empty() {
                    let invalid: &PastaField<M> = bento::AlignedBytes([0xff; 32]).as_value();
                    values[0] = *invalid;
                    assert!(matches!(
                        TwiddleTable::bind(description, &values),
                        Err(FftError::InvalidTables)
                    ));
                    assert!(matches!(
                        TwiddleTable::bind_trusted(description, &values[1..]),
                        Err(FftError::LengthMismatch { .. })
                    ));
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
        )
        .unwrap();
        assert_eq!(
            expansion.short_product_with(
                &coefficients,
                factor,
                &mut output,
                options,
                &joins,
                &mut scratch
            ),
            Err(FftError::InvalidLayout)
        );
        assert!(output.iter().chain(&scratch).all(|v| *v == PastaField::ONE));
        assert_eq!(joins.take(), 0);
    }
    let factor = EvaluationView::bind(&values, domain, EvaluationLayout::Natural).unwrap();
    assert_eq!(
        expansion.short_product_with(
            &coefficients,
            factor,
            &mut output,
            options,
            &joins,
            &mut scratch
        ),
        Err(FftError::InvalidLayout)
    );
    assert!(output.iter().chain(&scratch).all(|v| *v == PastaField::ONE));
    assert_eq!(joins.take(), 0);
    expansion
        .short_product_with(
            &coefficients,
            expansion.view(&values).unwrap(),
            &mut output,
            options,
            &joins,
            &mut scratch,
        )
        .unwrap();
    let result = expansion.view(&output).unwrap();
    let factor = expansion.view(&values).unwrap();
    for (row, evaluation) in direct(&coefficients, domain).iter().enumerate() {
        assert_eq!(
            result.get(row),
            Some(&evaluation.mul(factor.get(row).unwrap()))
        );
    }
    assert_eq!(scratch.last(), Some(&PastaField::ONE));
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
        .bind(domain)
        .unwrap();
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
                        .unwrap()
                        .unwrap();
                    for (row, value) in expected.iter().enumerate() {
                        assert_eq!(
                            EvaluationView::bind(&output, extended, layout)
                                .unwrap()
                                .get(row),
                            Some(value)
                        );
                    }
                    assert_eq!(scratch, [PastaField::ONE]);
                    // Output and scratch can be reused while the view lives.
                    output.fill(PastaField::ZERO);
                    scratch.fill(PastaField::ZERO);
                    view
                };
                assert_eq!(view.scale(), scale);
                for (actual, coefficient) in view.as_slice().iter().zip(&coefficients) {
                    assert_eq!(actual.mul(&view.normalization_factor()), *coefficient);
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
            .bind(original)
            .unwrap();
            for shift in [PastaField::from_u64(7), PastaField::ONE, PastaField::zeta()] {
                let domain = subgroup.coset(shift).unwrap();
                let rebound = bound.for_coset(domain).unwrap();
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
                    mask & 4 != 0 && shift == original.shift()
                );
                assert_eq!(
                    tables.inverse_scales.is_some(),
                    mask & 8 != 0 && shift == original.shift()
                );
                let coefficients = inputs(domain.size());
                let mut output = coefficients.clone();
                rebound
                    .forward_with(&mut output, Strategy::serial(), &SerialExecutor, &mut [])
                    .unwrap();
                assert_eq!(output, direct(&coefficients, domain));
                rebound
                    .inverse_with(&mut output, Strategy::serial(), &SerialExecutor, &mut [])
                    .unwrap();
                assert_eq!(output, coefficients);
            }
            assert!(matches!(
                bound.for_coset(Domain::new(log + 1).unwrap().subgroup()),
                Err(FftError::InvalidTables)
            ));
        }
    }
}

#[test]
fn validated_tables_reuse_borrows_across_cosets_and_drop_shift_dependent_entries() {
    reused_cosets::<PallasBase>();
    reused_cosets::<PallasScalar>();
}
