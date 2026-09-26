//! Domain metadata and optional generic evaluation against direct transforms
//! and the interpolated Lagrange basis.

use super::*;
use crate::field::Fq;

#[test]
fn domain_scalars_are_consistent() {
    for log_size in 0..=8 {
        let domain = Domain::<Fp>::new(log_size).unwrap();
        let size = domain.size() as u64;
        assert_eq!(domain.log_size(), log_size);
        assert_eq!(size, 1 << log_size);
        assert_eq!(domain.root() * domain.inverse_root(), <Fp>::ONE);
        assert_eq!(domain.size_inverse() * <Fp>::from_u64(size), <Fp>::ONE);
        assert_eq!(domain.root().pow_u64(size), <Fp>::ONE);
        if log_size > 0 {
            assert_ne!(domain.root().pow_u64(size / 2), <Fp>::ONE);
        }
        assert_eq!(domain, Domain::for_size(domain.size()).unwrap());
    }
    let trivial = Domain::<Fq>::new(0).unwrap();
    assert_eq!(trivial.size(), 1);
    assert_eq!(trivial.root(), <Fq>::ONE);
    assert_eq!(trivial.size_inverse(), <Fq>::ONE);
}

#[cfg(feature = "traits")]
mod consumer {
    use super::*;
    use crate::field::FftField;

    #[test]
    fn field_domain_factory_uses_native_parameters() {
        fn check<M: PrimeModulus>() {
            for log_size in [0, 1, 4, 8, 16, 32, 33, u32::MAX] {
                let native = Domain::<PastaField<M>>::new(log_size);
                let generic = <PastaField<M> as FftField>::domain(log_size);
                assert_eq!(generic, native);
                if let (Ok(generic), Ok(native)) = (generic, native) {
                    assert_eq!(generic.root(), native.root());
                    assert_eq!(generic.inverse_root(), native.inverse_root());
                    assert_eq!(generic.size_inverse(), native.size_inverse());
                }
            }
        }
        check::<PallasBase>();
        check::<PallasScalar>();
    }

    /// Horner evaluation: the quadratic oracle the transforms are checked against.
    fn evaluate<M: PrimeModulus>(
        coefficients: &[PastaField<M>],
        x: PastaField<M>,
    ) -> PastaField<M> {
        coefficients
            .iter()
            .rev()
            .fold(PastaField::ZERO, |accumulator, coefficient| {
                accumulator.mul(&x).add(coefficient)
            })
    }

    fn transforms<M: PrimeModulus>() {
        for log_size in 0..=8 {
            let domain = Domain::<PastaField<M>>::new(log_size).unwrap();
            let coefficients: Vec<PastaField<M>> = (0..domain.size())
                .map(|index| PastaField::<M>::DELTA.pow_u64(index as u64 + 1))
                .collect();

            let mut values = coefficients.clone();
            assert_eq!(
                reference::count_transforms(|| domain.transform(&mut values)),
                0
            );
            let elements = domain.elements();
            assert_eq!(elements.len(), domain.size());
            for (value, element) in values.iter().zip(elements) {
                assert_eq!(*value, evaluate(&coefficients, element));
            }

            assert_eq!(
                reference::count_transforms(|| domain.inverse_transform(&mut values)),
                0
            );
            assert_eq!(values, coefficients);
        }
    }

    #[test]
    fn domain_transforms_evaluate_at_every_element_and_invert() {
        transforms::<PallasBase>();
        transforms::<PallasScalar>();
    }

    fn large_transforms<M: PrimeModulus>() {
        for log_size in 9..=13 {
            let domain = Domain::<PastaField<M>>::new(log_size).unwrap();
            let original = inputs::<M>(domain.size());
            let mut expected = original.clone();
            reference::transform(&mut expected, &domain.root());
            let mut actual = original.clone();
            assert_eq!(
                reference::count_transforms(|| domain.transform(&mut actual)),
                0
            );
            assert_eq!(actual, expected);

            // Arbitrary evaluation vectors exercise the inverse independently of
            // the forward transform, including loose representation boundaries.
            actual.copy_from_slice(&original);
            expected.copy_from_slice(&original);
            reference::inverse_transform(
                &mut expected,
                &domain.inverse_root(),
                &domain.size_inverse(),
            );
            assert_eq!(
                reference::count_transforms(|| domain.inverse_transform(&mut actual)),
                0
            );
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn domain_transforms_use_field_kernels_across_plan_sizes() {
        large_transforms::<PallasBase>();
        large_transforms::<PallasScalar>();
    }

    #[test]
    fn field_transform_hooks_reject_wrong_lengths_before_writes() {
        let domain = Domain::<Fp>::new(3).unwrap();
        for transform in [<Fp as FftField>::fft, <Fp as FftField>::ifft] {
            for length in [0, 3, 4, 7, 9] {
                let mut values = vec![Fp::DELTA; length];
                let original = values.clone();
                assert!(catch_unwind(AssertUnwindSafe(|| transform(domain, &mut values))).is_err());
                assert_eq!(values, original);
            }
        }
    }

    #[test]
    fn domain_elements_are_consistent() {
        for log_size in 0..=8 {
            let domain = Domain::<Fp>::new(log_size).unwrap();
            let size = domain.size() as u64;
            assert_eq!(
                domain.root(),
                <Fp as FftField>::root_of_unity(log_size).unwrap()
            );

            let mut power = <Fp>::ONE;
            for element in domain.elements() {
                assert_eq!(element, power);
                assert!(domain.contains(element));
                assert!(domain.vanishing(element).is_zero());
                power *= domain.root();
            }
            for outsider in [<Fp>::from_u64(2), <Fp>::from_u64(3), <Fp>::DELTA] {
                let expected = outsider.pow_u64(size) - <Fp>::ONE;
                assert_eq!(domain.vanishing(outsider), expected);
                assert_eq!(domain.contains(outsider), expected.is_zero());
            }
        }

        let trivial = Domain::<Fq>::new(0).unwrap();
        assert!(trivial.contains(<Fq>::ONE));
        assert!(!trivial.contains(<Fq>::ONE.neg()));
    }

    /// Lagrange evaluations match the basis polynomials interpolated through the
    /// inverse transform, report a domain element by its index, and truncate to
    /// a prefix without touching the scratch tail.
    #[test]
    fn lagrange_evaluations_match_the_interpolated_basis() {
        let domain = Domain::<Fp>::new(5).unwrap();
        let size = domain.size();

        let mut basis = Vec::with_capacity(size);
        for (index, element) in domain.elements().enumerate() {
            // A domain element yields its unit vector directly, truncated to the
            // requested prefix, and reports its index even beyond that prefix.
            let mut evaluations = vec![<Fp>::DELTA; size];
            assert_eq!(
                domain.lagrange_evaluations(element, &mut evaluations, &mut vec![<Fp>::ZERO; size]),
                Some(index)
            );
            for (position, value) in evaluations.iter().enumerate() {
                let expected = if position == index {
                    <Fp>::ONE
                } else {
                    <Fp>::ZERO
                };
                assert_eq!(*value, expected);
            }
            let mut prefix = vec![<Fp>::DELTA; 3];
            assert_eq!(
                domain.lagrange_evaluations(element, &mut prefix, &mut [<Fp>::ZERO; 3]),
                Some(index)
            );
            assert_eq!(prefix, evaluations[..3]);

            // The basis polynomial as coefficients: the inverse transform of a
            // unit vector.
            let mut coefficients = vec![<Fp>::ZERO; size];
            coefficients[index] = <Fp>::ONE;
            domain.inverse_transform(&mut coefficients);
            for (other_index, other) in domain.elements().enumerate() {
                let expected = if index == other_index {
                    <Fp>::ONE
                } else {
                    <Fp>::ZERO
                };
                assert_eq!(evaluate(&coefficients, other), expected);
            }
            basis.push(coefficients);
        }

        let x = <Fp>::DELTA;
        let mut evaluations = vec![<Fp>::ZERO; size];
        let mut scratch = vec![<Fp>::DELTA; size + 2];
        assert_eq!(
            domain.lagrange_evaluations(x, &mut evaluations, &mut scratch),
            None
        );
        for (coefficients, evaluation) in basis.iter().zip(&evaluations) {
            assert_eq!(evaluate(coefficients, x), *evaluation);
        }
        assert!(scratch[size..].iter().all(|value| *value == <Fp>::DELTA));

        let mut prefix = vec![<Fp>::ZERO; 7];
        assert_eq!(
            domain.lagrange_evaluations(x, &mut prefix, &mut scratch),
            None
        );
        assert_eq!(prefix, evaluations[..7]);
        assert_eq!(domain.lagrange_evaluations(x, &mut [], &mut []), None);
    }

    #[test]
    #[should_panic(expected = "exceed the domain size")]
    fn lagrange_evaluations_reject_more_than_the_domain_size() {
        let domain = Domain::<Fp>::new(2).unwrap();
        let _ =
            domain.lagrange_evaluations(<Fp>::DELTA, &mut [<Fp>::ZERO; 5], &mut [<Fp>::ZERO; 5]);
    }

    #[test]
    fn lagrange_evaluations_use_native_ranges_with_bounded_scratch() {
        fn check<M: PrimeModulus>() {
            for log_size in [0, 1, 3, 5] {
                let domain = Domain::<PastaField<M>>::new(log_size).unwrap();
                let nodes: Vec<_> = domain.elements().collect();
                for point in nodes
                    .iter()
                    .copied()
                    .chain([PastaField::ZERO, PastaField::DELTA])
                {
                    for count in [0, 1, domain.size() / 2, domain.size()] {
                        let mut expected = vec![PastaField::ZERO; count];
                        domain
                            .subgroup()
                            .evaluate_lagrange(&point, 0..count, &mut expected, &mut [])
                            .unwrap();
                        for capacity in [0, 1, count / 2, count, count + 2] {
                            let mut values = vec![PastaField::DELTA; count];
                            let mut scratch = vec![PastaField::DELTA; capacity];
                            assert_eq!(
                                crate::fft::domain::count_size_powers(|| {
                                    assert_eq!(
                                        crate::fft::generic::count_lagrange_evaluations(|| {
                                            assert_eq!(
                                                domain.lagrange_evaluations(
                                                    point,
                                                    &mut values,
                                                    &mut scratch
                                                ),
                                                nodes.iter().position(|node| *node == point),
                                            );
                                        }),
                                        0
                                    );
                                }),
                                1
                            );
                            assert_eq!(values, expected);
                            assert!(
                                scratch
                                    .iter()
                                    .skip(count)
                                    .all(|value| *value == PastaField::DELTA)
                            );
                        }
                    }
                }
            }
        }
        check::<PallasBase>();
        check::<PallasScalar>();
    }

    #[test]
    fn field_lagrange_hook_rejects_long_output_before_writes() {
        let domain = Domain::<Fp>::new(2).unwrap();
        let mut values = [Fp::DELTA; 5];
        let mut scratch = [Fp::DELTA; 5];
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                <Fp as FftField>::lagrange_evaluations(domain, Fp::ZERO, &mut values, &mut scratch)
            }))
            .is_err()
        );
        assert_eq!(values, [Fp::DELTA; 5]);
        assert_eq!(scratch, [Fp::DELTA; 5]);
    }

    #[test]
    #[should_panic(expected = "transform input length")]
    fn domain_transform_rejects_the_wrong_length() {
        let domain = Domain::<Fp>::new(3).unwrap();
        domain.transform(&mut [<Fp>::ONE; 4]);
    }
}

#[test]
fn bit_reverse_matches_known_values_and_is_an_involution() {
    assert_eq!(bit_reverse(0b001, 3), 0b100);
    assert_eq!(bit_reverse(0b110, 3), 0b011);
    assert_eq!(bit_reverse(0b0001, 4), 0b1000);
    assert_eq!(bit_reverse(1, 1), 1);
    assert_eq!(bit_reverse(0, 0), 0);
    assert_eq!(bit_reverse(usize::MAX, 0), 0);
    assert_eq!(bit_reverse(1, usize::BITS), 1 << (usize::BITS - 1));
    assert_eq!(bit_reverse(1, usize::BITS + 5), 1 << (usize::BITS - 1));
    // Bits above the width are discarded.
    assert_eq!(bit_reverse(0b1_0110, 4), 0b0110);
    for bits in 1..=11 {
        for index in 0..(1usize << bits) {
            let reversed = bit_reverse(index, bits);
            assert!(reversed < (1 << bits));
            assert_eq!(bit_reverse(reversed, bits), index);
        }
    }
}

#[test]
fn layouts_and_subdomain_rows_are_distinct_from_coefficient_tiles() {
    for (size, count) in [(0, 1), (3, 1), (8, 0), (8, 3), (8, 16)] {
        assert_eq!(
            ResidueLayout::new(size, count),
            Err(FftError::InvalidLayout)
        );
    }
    for residues in [1, 2, 4, 8] {
        let layout = ResidueLayout::new(32, residues).unwrap();
        let input = inputs::<PallasBase>(32);
        let mut stored = [Fp::ZERO; 32];
        let mut natural = [Fp::ZERO; 32];
        layout.copy_from_natural(&input, &mut stored);
        layout.copy_to_natural(&stored, &mut natural);
        assert_eq!(reduced(&input), reduced(&natural));
        let view = EvaluationView::bind(
            &stored,
            Domain::for_size(32).unwrap().subgroup(),
            EvaluationLayout::Residues(layout),
        );
        for (row, expected) in input.iter().enumerate() {
            assert_eq!(
                (view.get(row)).map(|value| value.reduce()),
                (Some(expected)).map(|value| value.reduce())
            );
            assert_eq!(layout.natural_row(layout.index(row).unwrap()), Some(row));
            assert_eq!(
                (view.get_extended_row(row * 4, Domain::for_size(128).unwrap().subgroup()))
                    .map(|value| value.reduce()),
                (Some(expected)).map(|value| value.reduce())
            );
            assert_eq!(
                (view.get_extended_row(row * 4 + 1, Domain::for_size(128).unwrap().subgroup()))
                    .map(|value| value.reduce()),
                None
            );
        }
        assert_eq!((view.get(32)).map(|value| value.reduce()), None);
        assert!(
            view.get_extended_row(0, Domain::for_size(128).unwrap().coset())
                .is_none()
        );
        assert_eq!(
            (view.get_extended_row(0, Domain::for_size(16).unwrap().subgroup()))
                .map(|value| value.reduce()),
            None
        );
        for (input_len, output_len) in [(31, 32), (33, 32), (32, 31), (32, 33)] {
            let source = vec![<Fp>::ONE; input_len];
            let mut destination = vec![<Fp>::ZERO; output_len];
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    layout.copy_from_natural(&source, &mut destination);
                }))
                .is_err()
            );
            assert_eq!(destination, vec![<Fp>::ZERO; output_len]);
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    layout.copy_to_natural(&source, &mut destination);
                }))
                .is_err()
            );
            assert_eq!(destination, vec![<Fp>::ZERO; output_len]);
        }
    }
}
