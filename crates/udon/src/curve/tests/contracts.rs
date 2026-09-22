use super::*;

fn batches<C: PastaCurve>() {
    let generator = Point::<C>::GENERATOR;
    for size in [0, 1, 2, 3, 7, 8, 17, 32] {
        for pattern in 0..6 {
            let expected: Vec<_> = (0..size)
                .map(|i| {
                    if match pattern {
                        0 => false,
                        1 => i % 3 != 0,
                        2 => i % 2 == 0,
                        3 => i % 2 != 0,
                        4 => i + 1 != size,
                        _ => i != 1,
                    } {
                        Point::IDENTITY
                    } else {
                        generator
                            .mul_projective(&PastaField::<_>::from_u64(i as u64 + 1))
                            .to_point()
                    }
                })
                .collect();
            let points: Vec<_> = expected
                .iter()
                .enumerate()
                .map(|(i, point)| scaled(point, i as u64 + 2))
                .collect();
            let mut output = vec![generator; size];
            let sentinel = PastaField::from_u64(987);
            let mut scratch = vec![sentinel; size + 3];
            for _ in 0..2 {
                batch_normalize(&points, &mut output, &mut scratch);
                assert_eq!(output, expected);
                assert_eq!(
                    scratch[size..]
                        .iter()
                        .map(|value| value.reduce())
                        .collect::<Vec<_>>(),
                    [sentinel; 3]
                        .iter()
                        .map(|value| value.reduce())
                        .collect::<Vec<_>>()
                );
            }
            let identities = vec![ProjectivePoint::IDENTITY; size];
            batch_normalize(&identities, &mut output, &mut scratch);
            assert!(output.iter().all(Point::is_identity));
            assert_eq!(
                scratch[size..]
                    .iter()
                    .map(|value| value.reduce())
                    .collect::<Vec<_>>(),
                [sentinel; 3]
                    .iter()
                    .map(|value| value.reduce())
                    .collect::<Vec<_>>()
            );

            let old_scratch = scratch.clone();
            let mut wrong = vec![generator; size + 1];
            let old_output = wrong.clone();
            assert!(
                std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                    batch_normalize(&points, &mut wrong, &mut scratch);
                }))
                .is_err()
            );
            assert_eq!(wrong, old_output);
            assert_eq!(
                (scratch)
                    .iter()
                    .map(|value| value.reduce())
                    .collect::<Vec<_>>(),
                (old_scratch)
                    .iter()
                    .map(|value| value.reduce())
                    .collect::<Vec<_>>()
            );
            for capacity in [0, size / 2, size.saturating_sub(1)] {
                batch_normalize(&points, &mut output, &mut scratch[..capacity]);
                assert_eq!(output, expected);
                assert_eq!(
                    scratch[size..]
                        .iter()
                        .map(|value| value.reduce())
                        .collect::<Vec<_>>(),
                    [sentinel; 3]
                        .iter()
                        .map(|value| value.reduce())
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

#[test]
fn normalization_preserves_order_identity_and_scratch_contracts() {
    batches::<Pallas>();
    batches::<Vesta>();
}

#[test]
fn coordinate_construction_checks_the_curve_equation() {
    fn check<C: PastaCurve>() {
        let generator = AffinePoint::<C>::GENERATOR;
        let (x, y) = generator.coordinates();
        assert_eq!(AffinePoint::<C>::from_xy(*x, *y), Some(generator));
        assert_eq!(Point::<C>::from_xy(*x, *y), Some(generator.to_point()));
        for (x, y) in [(PastaField::ZERO, *y), (*x, PastaField::ZERO)] {
            assert_eq!(AffinePoint::<C>::from_xy(x, y), None);
            assert_eq!(Point::<C>::from_xy(x, y), None);
        }
        assert_eq!(
            AffinePoint::<C>::from_xy(PastaField::ZERO, PastaField::ZERO),
            None
        );
        assert_eq!(
            Point::<C>::from_xy(PastaField::ZERO, PastaField::ZERO),
            Some(Point::IDENTITY)
        );
    }
    check::<Pallas>();
    check::<Vesta>();
}
