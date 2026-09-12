use super::*;

fn batches<C: PastaCurve>() {
    let generator = Point::<C>::GENERATOR;
    for size in [0, 1, 2, 3, 7, 8, 17, 32] {
        for identities in [false, true] {
            let expected: Vec<_> = (0..size)
                .map(|i| {
                    if identities && i % 3 != 0 {
                        Point::IDENTITY
                    } else {
                        generator
                            .mul_projective(&PastaField::from_u64(i as u64))
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
                batch_normalize(&points, &mut output, &mut scratch).unwrap();
                assert_eq!(output, expected);
                assert_eq!(&scratch[size..], &[sentinel; 3]);
            }
            let identities = vec![ProjectivePoint::IDENTITY; size];
            batch_normalize(&identities, &mut output, &mut scratch).unwrap();
            assert!(output.iter().all(Point::is_identity));
            assert_eq!(&scratch[size..], &[sentinel; 3]);

            let old_scratch = scratch.clone();
            let mut wrong = vec![generator; size + 1];
            let old_output = wrong.clone();
            assert_eq!(
                batch_normalize(&points, &mut wrong, &mut scratch),
                Err(CurveError::LengthMismatch {
                    buffer: "output",
                    expected: size,
                    actual: size + 1,
                })
            );
            assert_eq!(wrong, old_output);
            assert_eq!(scratch, old_scratch);
            if size != 0 {
                let old_output = output.clone();
                assert_eq!(
                    batch_normalize(&points, &mut output, &mut scratch[..size - 1]),
                    Err(CurveError::ScratchTooSmall {
                        buffer: "field",
                        required: size,
                        provided: size - 1,
                    })
                );
                assert_eq!(output, old_output);
                assert_eq!(scratch, old_scratch);
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
fn checked_coordinates_reject_unreduced_storage_before_arithmetic() {
    fn check<C: PastaCurve>() {
        let invalid = invalid_field();
        let zero = PastaField::ZERO;
        let generator = AffinePoint::<C>::GENERATOR;
        for (x, y) in [
            (invalid, generator.y),
            (generator.x, invalid),
            (invalid, zero),
            (zero, invalid),
        ] {
            assert_eq!(AffinePoint::<C>::from_xy(x, y), None);
            assert_eq!(Point::<C>::from_xy(x, y), None);
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}
