//! The optional domain interface also works with an independent non-Pasta field.

use super::field_model::Small;
use std::panic::{AssertUnwindSafe, catch_unwind};
use zakura_udon::{
    fft::FftError,
    field::{FftField, Field},
};

#[test]
fn generic_domains_use_the_fields_roots_and_transforms() {
    for log_size in 0..=Small::TWO_ADICITY {
        let domain = Small::domain(log_size).unwrap();
        let size = 1 << log_size;
        assert_eq!(domain.size(), size);
        assert_eq!(domain.root() * domain.inverse_root(), Small::ONE);
        assert_eq!(domain.root().pow_u64(size as u64), Small::ONE);
        if size > 1 {
            assert_ne!(domain.root().pow_u64((size / 2) as u64), Small::ONE);
        }
        assert_eq!(domain.size_inverse() * Small::from(size as u64), Small::ONE);
        let mut values: Vec<_> = (0..size as u64).map(|i| Small::from(i * i + 7)).collect();
        let original = values.clone();
        domain.transform(&mut values);
        for (value, point) in values.iter().zip(domain.elements()) {
            let expected = original
                .iter()
                .rev()
                .fold(Small::ZERO, |acc, coefficient| acc * point + coefficient);
            assert_eq!(*value, expected);
        }
        domain.inverse_transform(&mut values);
        assert_eq!(values, original);
    }
    assert_eq!(Small::domain(5), Err(FftError::InvalidSize));
    assert_eq!(Small::domain(u32::MAX), Err(FftError::InvalidSize));
}

#[test]
fn generic_lagrange_matches_direct_interpolation_over_the_entire_field() {
    for log_size in 0..=Small::TWO_ADICITY {
        let domain = Small::domain(log_size).unwrap();
        let nodes: Vec<_> = domain.elements().collect();
        for integer in 0..17 {
            let point = Small::from(integer);
            let expected: Vec<_> = nodes
                .iter()
                .enumerate()
                .map(|(i, node)| {
                    nodes
                        .iter()
                        .enumerate()
                        .filter(|(j, _)| i != *j)
                        .map(|(_, other)| (point - other) * (*node - other).invert().unwrap())
                        .product::<Small>()
                })
                .collect();
            assert_eq!(expected.iter().sum::<Small>(), Small::ONE);
            for count in [0, 1, nodes.len() / 2, nodes.len()] {
                for capacity in [0, 1, count / 2, count, count + 2] {
                    let mut values = vec![Small::from(13); count];
                    let mut scratch = vec![Small::from(7); capacity];
                    assert_eq!(
                        domain.lagrange_evaluations(point, &mut values, &mut scratch),
                        nodes.iter().position(|node| *node == point),
                    );
                    assert_eq!(values, expected[..count]);
                    assert!(
                        scratch
                            .iter()
                            .skip(count)
                            .all(|value| *value == Small::from(7))
                    );
                }
            }
        }
    }
}

#[test]
fn generic_lagrange_hook_validates_before_mutation() {
    let domain = Small::domain(2).unwrap();
    let mut values = [Small::from(11); 5];
    let mut scratch = [Small::from(7); 5];
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            Small::lagrange_evaluations(domain, Small::ZERO, &mut values, &mut scratch)
        }))
        .is_err()
    );
    assert_eq!(values, [Small::from(11); 5]);
    assert_eq!(scratch, [Small::from(7); 5]);
}
