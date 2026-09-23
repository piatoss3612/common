use super::*;
use crate::field::count_inversions;

fn trait_msm<C: PastaCurve>() {
    let generator = Point::<C>::GENERATOR;
    let points = [
        Point::IDENTITY,
        generator,
        generator.neg(),
        generator.to_projective().double().to_point(),
        generator.endomorphism(),
    ];
    let weights = [
        PastaField::<C::Scalar>::ZERO,
        PastaField::ONE,
        PastaField::<C::Scalar>::ONE.neg(),
        PastaField::from_u64(2),
        PastaField::ZETA,
    ];
    let corpus = scalar_corpus::<C>();
    let modulus = crate::test_support::modulus::<C::Base>();
    let reference = reference::Reference::generator(&modulus);
    for size in [
        0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 63, 64, 65, 127, 128, 129, 191, 192, 255, 256, 257, 511,
        512, 513, 1025, 8193,
    ] {
        let bases: Vec<_> = points.iter().copied().cycle().take(size).collect();
        let scalars: Vec<_> = corpus.iter().copied().cycle().take(size).collect();
        let exponent = scalars
            .iter()
            .zip(weights.iter().cycle())
            .fold(PastaField::ZERO, |sum, (scalar, weight)| {
                sum.add(&scalar.mul(weight))
            });
        let expected = reference.mul(
            &num_bigint::BigUint::from_bytes_le(&exponent.to_bytes()),
            &modulus,
        );
        let actual = <Point<C> as Affine>::msm(&scalars, &bases);
        expected.assert_point(&actual.to_point());

        assert!(<Point<C> as Affine>::msm(&vec![PastaField::ZERO; size], &bases).is_identity());
        assert!(<Point<C> as Affine>::msm(&scalars, &vec![Point::IDENTITY; size]).is_identity());
    }
}

#[test]
fn trait_msm_supports_bounded_scratch_and_chunk_boundaries() {
    trait_msm::<Pallas>();
    trait_msm::<Vesta>();
}

#[test]
#[should_panic(expected = "msm operands must have equal length")]
fn trait_msm_rejects_mismatched_lengths() {
    <Point<Pallas> as Affine>::msm(&[PastaField::ONE], &[]);
}

fn normalization_inversions<C: PastaCurve>() {
    let generator = Point::<C>::GENERATOR;
    let negative = generator.neg();
    let rotated = generator.endomorphism();
    let zero = PastaField::<C::Base>::from_montgomery_limbs(C::Base::MODULUS);
    let affine = negative.as_affine().unwrap();
    let loose = ProjectivePoint {
        x: affine.x.add(&zero),
        y: affine.y.add(&zero),
        z: PastaField::<C::Base>::ONE.add(&zero),
        marker: PhantomData,
    };
    let identity = ProjectivePoint { z: zero, ..loose };
    let cases = [
        (ProjectivePoint::IDENTITY, Point::IDENTITY, false),
        (generator.to_projective(), generator, false),
        (identity, Point::IDENTITY, false),
        (loose, negative, false),
        (scaled(&generator, 2), generator, true),
        (rotated.to_projective(), rotated, false),
        (scaled(&negative, 3), negative, true),
        (scaled(&rotated, 5), rotated, true),
    ];
    for (point, expected, needs_inverse) in cases {
        let inversions = count_inversions(|| assert_eq!(point.to_point(), expected));
        assert_eq!(inversions, usize::from(needs_inverse));
    }

    // Rotate and truncate to put skipped entries at both lane endpoints,
    // between factors, and in batches without any nontrivial denominator.
    for offset in 0..cases.len() {
        for size in 0..=cases.len() {
            let cases: Vec<_> = cases.iter().cycle().skip(offset).take(size).collect();
            let points: Vec<_> = cases.iter().map(|case| case.0).collect();
            let expected: Vec<_> = cases.iter().map(|case| case.1).collect();
            for capacity in 0..=size + 1 {
                let mut output = vec![generator; size];
                let sentinel = PastaField::from_u64(987);
                let mut scratch = vec![sentinel; capacity + 2];
                let inversions = count_inversions(|| {
                    batch_normalize(&points, &mut output, &mut scratch[..capacity]);
                });
                let expected_inversions = cases
                    .chunks(capacity.max(1))
                    .filter(|chunk| chunk.iter().any(|case| case.2))
                    .count();
                assert_eq!(
                    inversions, expected_inversions,
                    "offset={offset}, size={size}, capacity={capacity}"
                );
                assert_eq!(output, expected);
                assert!(
                    scratch[capacity.min(size)..]
                        .iter()
                        .all(|value| value.montgomery_limbs() == sentinel.montgomery_limbs())
                );
            }
        }
    }
}

#[test]
fn normalization_inverts_only_nontrivial_denominators() {
    normalization_inversions::<Pallas>();
    normalization_inversions::<Vesta>();
}

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
