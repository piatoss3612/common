use super::*;
use crate::curve::{Pallas, PastaCurve, ProjectivePoint, Vesta};

fn exercise<C: PastaCurve>() {
    for size in [1, 2, 4, 8, 16] {
        let domain = Domain::<C::Scalar>::for_size(size).unwrap();
        let original: Vec<_> = (0..size)
            .map(|i| match i % 4 {
                0 => ProjectivePoint::<C>::IDENTITY,
                1 => ProjectivePoint::GENERATOR,
                2 => ProjectivePoint::GENERATOR.neg(),
                _ => ProjectivePoint::GENERATOR.mul(&PastaField::from_u64(i as u64 + 3)),
            })
            .collect();
        let mut actual = original.clone();
        reference::transform(&mut actual, &domain.root());
        let mut step = PastaField::ONE;
        for output in &actual {
            let mut power = PastaField::ONE;
            let mut expected = ProjectivePoint::IDENTITY;
            for input in &original {
                expected = expected.add(&input.mul(&power));
                power = power.mul(&step);
            }
            assert_eq!(output.to_point(), expected.to_point());
            step = step.mul(&domain.root());
        }
        reference::inverse_transform(&mut actual, &domain.inverse_root(), &domain.size_inverse());
        for (actual, expected) in actual.iter().zip(original) {
            assert_eq!(actual.to_point(), expected.to_point());
        }
    }
}

#[test]
fn projective_transforms_match_dft_and_round_trip() {
    exercise::<Pallas>();
    exercise::<Vesta>();
}
