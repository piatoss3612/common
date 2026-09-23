//! The generic evaluation domain against direct evaluation and the
//! interpolated Lagrange basis.

use super::*;
use crate::curve::{Pallas, PastaCurve, ProjectivePoint, Vesta};
use crate::field::{FftField, Fq};

/// Horner evaluation: the quadratic oracle the transforms are checked against.
fn evaluate<M: PrimeModulus>(coefficients: &[PastaField<M>], x: PastaField<M>) -> PastaField<M> {
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
        domain.transform(&mut values);
        let elements = domain.elements();
        assert_eq!(elements.len(), domain.size());
        for (value, element) in values.iter().zip(elements) {
            assert_eq!(*value, evaluate(&coefficients, element));
        }

        domain.inverse_transform(&mut values);
        assert_eq!(values, coefficients);
    }
}

#[test]
fn domain_transforms_evaluate_at_every_element_and_invert() {
    transforms::<PallasBase>();
    transforms::<PallasScalar>();
}

#[test]
fn domain_scalars_and_elements_are_consistent() {
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
        assert_eq!(
            domain.root(),
            <Fp as FftField>::root_of_unity(log_size).unwrap()
        );
        assert_eq!(domain, Domain::for_size(domain.size()).unwrap());

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
    assert_eq!(trivial.size(), 1);
    assert_eq!(trivial.root(), <Fq>::ONE);
    assert_eq!(trivial.size_inverse(), <Fq>::ONE);
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
    let _ = domain.lagrange_evaluations(<Fp>::DELTA, &mut [<Fp>::ZERO; 5], &mut [<Fp>::ZERO; 5]);
}

#[test]
#[should_panic(expected = "scratch must cover every evaluation")]
fn lagrange_evaluations_reject_short_scratch() {
    let domain = Domain::<Fp>::new(2).unwrap();
    let _ = domain.lagrange_evaluations(<Fp>::DELTA, &mut [<Fp>::ZERO; 4], &mut [<Fp>::ZERO; 3]);
}

#[test]
#[should_panic(expected = "transform input length")]
fn domain_transform_rejects_the_wrong_length() {
    let domain = Domain::<Fp>::new(3).unwrap();
    domain.transform(&mut [<Fp>::ONE; 4]);
}

/// Group-valued transforms through the domain agree with the direct sum.
fn group_transforms<C: PastaCurve>() {
    for log_size in 0..=4 {
        let domain = Domain::<PastaField<C::Scalar>>::new(log_size).unwrap();
        let original: Vec<_> = (0..domain.size())
            .map(|i| match i % 4 {
                0 => ProjectivePoint::<C>::IDENTITY,
                1 => ProjectivePoint::GENERATOR,
                2 => ProjectivePoint::GENERATOR.neg(),
                _ => ProjectivePoint::GENERATOR.mul(&PastaField::<_>::from_u64(i as u64 + 3)),
            })
            .collect();
        let mut actual = original.clone();
        domain.transform(&mut actual);
        for (output, element) in actual.iter().zip(domain.elements()) {
            let mut power = PastaField::ONE;
            let mut expected = ProjectivePoint::IDENTITY;
            for input in &original {
                expected = expected.add(&input.mul(&power));
                power = power.mul(&element);
            }
            assert_eq!(*output, expected);
        }
        domain.inverse_transform(&mut actual);
        assert_eq!(actual, original);
    }
}

#[test]
fn group_transforms_through_the_domain_match_the_direct_sum() {
    group_transforms::<Pallas>();
    group_transforms::<Vesta>();
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
