//! Fused points and slopes checked against independent affine integer arithmetic.

use num_bigint::BigUint;

use super::{reference::Reference, *};
use crate::field::pasta::test_support::modulus;

fn check_step<C: PastaCurve>(
    a: &ProjectivePoint<C>,
    b: &AffinePoint<C>,
) -> Option<ProjectivePoint<C>> {
    let p = modulus::<C::Base>();
    let ra = Reference::from_point(&a.to_point());
    let rb = Reference::from_point(&b.to_point());
    let step = a.incomplete_double_and_add(b);
    let Some((ax, ay)) = &ra.coordinates else {
        assert!(step.is_none());
        return None;
    };
    let (bx, by) = rb.coordinates.as_ref().unwrap();
    if ax == bx {
        assert!(step.is_none());
        return None;
    }
    let intermediate = ra.add(&rb, &p);
    let (rx, ry) = intermediate.coordinates.as_ref().unwrap();
    if ax == rx {
        assert!(step.is_none());
        return None;
    }
    let step = step.expect("both incomplete additions have distinct x-coordinates");
    let expected = ra.add(&intermediate, &p);
    expected.assert_point(&step.point.to_point());
    assert!(!step.point.is_identity());

    let denominator = BigUint::from_bytes_le(&step.point.coordinates().2.to_bytes());
    for ((x, y), numerator) in [(bx, by), (rx, ry)].into_iter().zip(step.slope_numerators) {
        let slope = (y + &p - ay) * (x + &p - ax).modpow(&(&p - 2_u32), &p) % &p;
        assert_eq!(
            BigUint::from_bytes_le(&numerator.to_bytes()),
            slope * &denominator % &p
        );
    }
    Some(step.point)
}

fn incomplete_double_and_add<C: PastaCurve>() {
    let generator = AffinePoint::<C>::GENERATOR;
    // Small positive and negative multiples include both exceptional first
    // additions, A = ±B, and the exceptional second addition, B = -2A.
    for a in 0..10 {
        let a = generator
            .mul_projective(&PastaField::<_>::from_u64(a))
            .to_point();
        for b in 1..9 {
            let b = generator
                .mul_projective(&PastaField::<_>::from_u64(b))
                .to_point();
            for b in [*b.as_affine().unwrap(), b.as_affine().unwrap().neg()] {
                check_step(&a.to_projective(), &b);
                check_step(&scaled(&a, 13), &b);
            }
        }
    }

    // Full-width inputs with independent projective scales exercise carries.
    let mut scalars = field_samples::<C::Scalar>().filter(|s| !s.is_zero());
    for z in field_samples::<C::Base>().filter(|z| !z.is_zero()).take(24) {
        let a = generator
            .mul_projective(&scalars.next().unwrap())
            .to_point();
        let b = generator
            .mul_projective(&scalars.next().unwrap())
            .to_point();
        let a = a.as_affine().unwrap();
        let a = ProjectivePoint {
            x: a.x.mul(&z.square()),
            y: a.y.mul(&z.square()).mul(&z),
            z,
            marker: PhantomData,
        };
        check_step(&a, b.as_affine().unwrap());
    }

    // The endomorphism preserves y and changes x. Horizontal chords in either
    // addition must succeed even though the corresponding slope is zero.
    let rotated = generator.endomorphism();
    let b = rotated
        .to_projective()
        .sub(&generator.to_projective())
        .to_point();
    for a in [generator.to_projective(), scaled(&generator.to_point(), 13)] {
        check_step(&a, &rotated).unwrap();
        check_step(&a, b.as_affine().unwrap()).unwrap();
    }

    // Returned points remain usable in later fused and complete operations.
    let mut a = generator.to_projective().double();
    for _ in 0..32 {
        a = check_step(&a, &generator).unwrap();
        a = check_step(&a, &generator.neg()).unwrap();
    }
    let p = modulus::<C::Base>();
    let ra = Reference::from_point(&a.to_point());
    ra.add(&ra, &p).assert_point(&a.double().to_point());
    ra.add(&Reference::from_point(&generator.to_point()), &p)
        .assert_point(&a.add_mixed(&generator).to_point());
}

#[test]
fn pallas_incomplete_double_and_add_matches_integer_arithmetic() {
    incomplete_double_and_add::<Pallas>();
}

#[test]
fn vesta_incomplete_double_and_add_matches_integer_arithmetic() {
    incomplete_double_and_add::<Vesta>();
}
