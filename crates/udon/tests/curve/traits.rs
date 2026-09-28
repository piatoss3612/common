use super::{
    edwards::Edwards,
    field_model::{Small, SmallScalar},
};
use zakura_udon::{
    curve::{Affine, Projective},
    field::Field,
};

fn generic_curve<A: Affine>() {
    let identity = A::identity();
    let g = A::generator();
    for point in [identity, g, -g] {
        let encoding = point.to_bytes();
        assert_eq!(A::from_bytes(encoding), Some(point));
        assert_eq!(point.to_projective().to_affine(), point);
    }
    let bases = [identity, g, -g];
    let scalars = [A::Scalar::from(4), A::Scalar::from(3), A::Scalar::ONE];
    assert_eq!(A::msm(&scalars, &bases), g.to_projective().double());
    let points = [
        g.to_projective(),
        A::Projective::identity(),
        (-g).to_projective(),
    ];
    let mut out = [identity; 3];
    A::batch_to_affine(&points, &mut out);
    assert_eq!(out, [g, identity, -g]);
}

#[test]
fn edwards_without_fft_or_endomorphism_and_with_wide_encoding() {
    generic_curve::<Edwards>();
    let identity = <Edwards as Affine>::identity();
    assert_eq!(Edwards::from_xy(Small::ZERO, Small::ONE), Some(identity));
    assert_eq!(Edwards::from_xy(Small::ZERO, -Small::ONE), None);
    assert_eq!(Edwards::from_xy(Small::ZERO, Small::ZERO), None);
    assert_ne!(identity.to_bytes(), [0; 48]);
    let mut bad_padding = identity.to_bytes();
    bad_padding[47] = 1;
    assert_eq!(Edwards::from_bytes(bad_padding), None);
    // Check every subgroup pair using the Edwards law in the coordinate field.
    for i in 0..5 {
        let a = Edwards(SmallScalar::from(i));
        let (x, y) = a.xy();
        assert_eq!(
            -x.square() + y.square(),
            Small::ONE + Small::from(6) * x.square() * y.square()
        );
        for j in 0..5 {
            let b = Edwards(SmallScalar::from(j));
            let (u, v) = b.xy();
            let t = Small::from(6) * x * u * y * v;
            let expected = (
                (x * v + y * u) * (Small::ONE + t).invert().unwrap(),
                (y * v + x * u) * (Small::ONE - t).invert().unwrap(),
            );
            assert_eq!((a + b).xy(), expected);
            assert_eq!(a.add_mixed(&b).xy(), expected);
        }
    }
}
