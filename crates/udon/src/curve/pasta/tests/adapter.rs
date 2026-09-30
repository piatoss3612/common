//! Consumer adapter operators and borrowed views of native Pasta curve storage.

use crate::curve::{
    Affine, AffineAdapter, Pallas, PastaCurve, Point, Projective, ProjectiveAdapter,
    ProjectivePoint, Vesta,
};
use crate::field::{Field, FieldAdapter, PastaField};

#[test]
#[allow(
    clippy::op_ref,
    reason = "Every owned and borrowed operator form is part of the consumer API."
)]
fn operators_preserve_native_curve_arithmetic() {
    fn check<C: PastaCurve>() {
        let g = Point::<C>::GENERATOR;
        for a in [Point::IDENTITY, g, g.neg()] {
            let x = AffineAdapter::new(a);
            assert_eq!((-x).into_inner(), a.neg());
            assert_eq!((-&x).into_inner(), a.neg());
            for scalar in [
                FieldAdapter::ZERO,
                FieldAdapter::ONE,
                -FieldAdapter::ONE,
                FieldAdapter::from(7),
            ] {
                let expected = a.mul_projective(scalar.as_inner());
                assert_eq!((x * scalar).into_inner(), expected);
                assert_eq!((x * &scalar).into_inner(), expected);
                assert_eq!((&x * scalar).into_inner(), expected);
                assert_eq!((&x * &scalar).into_inner(), expected);
                let p = x.to_projective();
                assert_eq!((p * scalar).into_inner(), expected);
                assert_eq!((p * &scalar).into_inner(), expected);
                assert_eq!((&p * scalar).into_inner(), expected);
                assert_eq!((&p * &scalar).into_inner(), expected);
            }
            for b in [Point::IDENTITY, g, g.neg()] {
                let (a, b) = (a.to_projective(), b.to_projective());
                let (x, y) = (ProjectiveAdapter::new(a), ProjectiveAdapter::new(b));
                macro_rules! binary {
                    ($op:tt, $assign:tt, $method:ident) => {
                        let expected = ProjectiveAdapter::new(a.$method(&b));
                        assert_eq!(x $op y, expected);
                        assert_eq!(x $op &y, expected);
                        assert_eq!(&x $op y, expected);
                        assert_eq!(&x $op &y, expected);
                        let mut actual = x;
                        actual $assign y;
                        assert_eq!(actual, expected);
                        actual = x;
                        actual $assign &y;
                        assert_eq!(actual, expected);
                    }
                }
                binary!(+, +=, add);
                binary!(-, -=, sub);
                assert_eq!((-x).into_inner(), a.neg());
                assert_eq!((-&x).into_inner(), a.neg());
            }
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}

#[test]
fn transparent_views_preserve_curve_storage() {
    fn check<C: PastaCurve>() {
        let g = Point::<C>::GENERATOR;
        let mut native = [Point::IDENTITY, g, g.neg()];
        let mut projective = native.map(|p| p.to_projective());
        for range in [0..0, 1..1, 0..3, 1..3] {
            let slice = &native[range.clone()];
            let wrapped = AffineAdapter::from_slice(slice);
            assert_eq!(wrapped.as_ptr().cast::<Point<C>>(), slice.as_ptr());
            assert_eq!(AffineAdapter::as_slice(wrapped), slice);
            let slice = &projective[range];
            let wrapped = ProjectiveAdapter::from_slice(slice);
            assert_eq!(
                wrapped.as_ptr().cast::<ProjectivePoint<C>>(),
                slice.as_ptr()
            );
            assert_eq!(ProjectiveAdapter::as_slice(wrapped), slice);
        }
        assert!(core::ptr::eq(
            AffineAdapter::from_ref(&native[1]).as_inner(),
            &native[1]
        ));
        assert!(core::ptr::eq(
            ProjectiveAdapter::from_ref(&projective[1]).as_inner(),
            &projective[1]
        ));
        let wrapped = AffineAdapter::from_slice_mut(&mut native);
        wrapped[0] = AffineAdapter::new(g);
        *wrapped[1].as_inner_mut() = Point::IDENTITY;
        AffineAdapter::as_slice_mut(wrapped)[2] = g;
        assert_eq!(native, [g, Point::IDENTITY, g]);
        let wrapped = ProjectiveAdapter::from_slice_mut(&mut projective);
        wrapped[0] = ProjectiveAdapter::new(g.to_projective());
        *wrapped[1].as_inner_mut() = ProjectivePoint::IDENTITY;
        ProjectiveAdapter::as_slice_mut(wrapped)[2] = g.to_projective();
        assert_eq!(projective, native.map(|p| p.to_projective()));
        let mut output = [AffineAdapter::<C>::identity(); 3];
        AffineAdapter::batch_to_affine(ProjectiveAdapter::from_slice(&projective), &mut output);
        assert_eq!(AffineAdapter::as_slice(&output), &native);
        let mut owned = projective.map(ProjectiveAdapter::new);
        ProjectiveAdapter::as_slice_mut(&mut owned)[0] = g.mul_projective(&PastaField::from_u64(2));
        assert_eq!(owned[0], AffineAdapter::new(g).to_projective().double());
        AffineAdapter::as_slice_mut(&mut output)[0] = Point::IDENTITY;
        assert!(output[0].is_identity());
    }
    check::<Pallas>();
    check::<Vesta>();
}
