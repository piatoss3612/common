//! Representation widths and optional capabilities in the public traits.

#[path = "support/field_reference.rs"]
mod field_reference;

use core::{iter::Sum, ops};
use field_reference::{Small, SmallScalar};
use zakura_udon::{
    curve::{Affine, Projective},
    field::{Field, PrimeField},
};

// The order-five subgroup of -x² + y² = 1 + 6x²y² over F_17, generated
// by (5, 7). Store the discrete logarithm to make the test model small;
// the test below independently checks the Edwards addition law.
const COORDINATES: [(u64, u64); 5] = [(0, 1), (5, 7), (10, 9), (7, 9), (12, 7)];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
struct Edwards(SmallScalar);

impl Edwards {
    fn xy(self) -> (Small, Small) {
        let (x, y) = COORDINATES[self.0.to_bytes()[0] as usize];
        (Small::from(x), Small::from(y))
    }
}

macro_rules! binary {
    ($trait:ident, $method:ident, $assign:ident, $assign_method:ident, $op:tt) => {
        impl ops::$trait<&Self> for Edwards {
            type Output = Self;
            fn $method(self, rhs: &Self) -> Self {
                Self(self.0 $op rhs.0)
            }
        }
        impl ops::$trait for Edwards {
            type Output = Self;
            fn $method(self, rhs: Self) -> Self {
                ops::$trait::$method(self, &rhs)
            }
        }
        impl ops::$assign<&Self> for Edwards {
            fn $assign_method(&mut self, rhs: &Self) {
                *self = ops::$trait::$method(*self, rhs);
            }
        }
        impl ops::$assign for Edwards {
            fn $assign_method(&mut self, rhs: Self) {
                ops::$assign::$assign_method(self, &rhs);
            }
        }
    };
}
binary!(Add, add, AddAssign, add_assign, +);
binary!(Sub, sub, SubAssign, sub_assign, -);

impl ops::Neg for Edwards {
    type Output = Self;
    fn neg(self) -> Self {
        Self(-self.0)
    }
}

impl ops::Mul<SmallScalar> for Edwards {
    type Output = Self;
    fn mul(self, scalar: SmallScalar) -> Self {
        Self(self.0 * scalar)
    }
}

impl ops::Mul<&SmallScalar> for Edwards {
    type Output = Self;
    fn mul(self, scalar: &SmallScalar) -> Self {
        self * *scalar
    }
}

impl Sum for Edwards {
    fn sum<I: Iterator<Item = Self>>(points: I) -> Self {
        Self(points.map(|point| point.0).sum())
    }
}

impl<'a> Sum<&'a Self> for Edwards {
    fn sum<I: Iterator<Item = &'a Self>>(points: I) -> Self {
        points.copied().sum()
    }
}

impl Affine for Edwards {
    type Base = Small;
    type Scalar = SmallScalar;
    type Projective = Self;
    // Deliberate padding checks that generic consumers allow encodings wider
    // than 32 bytes. All padding must be zero for a canonical encoding.
    type Repr = [u8; 48];

    fn identity() -> Self {
        Self(SmallScalar::ZERO)
    }
    fn generator() -> Self {
        Self(SmallScalar::ONE)
    }
    fn is_identity(&self) -> bool {
        self.0.is_zero()
    }
    fn from_xy(x: Small, y: Small) -> Option<Self> {
        COORDINATES
            .iter()
            .position(|&(a, b)| x == Small::from(a) && y == Small::from(b))
            .map(|index| Self(SmallScalar::from(index as u64)))
    }
    fn coordinates(&self) -> Option<(Small, Small)> {
        (!Affine::is_identity(self)).then(|| self.xy())
    }
    fn to_projective(&self) -> Self {
        *self
    }
    fn negate(&self) -> Self {
        -*self
    }
    fn to_bytes(&self) -> Self::Repr {
        let (x, y) = self.xy();
        let mut bytes = [0; 48];
        bytes[0] = y.to_bytes()[0] | (u8::from(x.is_odd()) << 7);
        bytes
    }
    fn from_bytes(bytes: Self::Repr) -> Option<Self> {
        (0..5)
            .map(|index| Self(SmallScalar::from(index)))
            .find(|point| point.to_bytes() == bytes)
    }
}

impl Projective for Edwards {
    type Base = Small;
    type Scalar = SmallScalar;
    type Affine = Self;

    fn identity() -> Self {
        <Self as Affine>::identity()
    }
    fn generator() -> Self {
        <Self as Affine>::generator()
    }
    fn is_identity(&self) -> bool {
        Affine::is_identity(self)
    }
    fn double(&self) -> Self {
        *self + self
    }
    fn add_mixed(&self, rhs: &Self) -> Self {
        *self + rhs
    }
    fn to_affine(&self) -> Self {
        *self
    }
}

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
fn pasta_cycle_exposes_compatible_endomorphisms() {
    use zakura_udon::{
        curve::{EndomorphismAffine, EndomorphismProjective},
        cycle::{Cycle, Pasta},
        field::CubeRootField,
    };

    fn check<A: EndomorphismAffine>() {
        for scalar in [A::Scalar::ZERO, A::Scalar::ONE, A::Scalar::from(13)] {
            let point = A::generator() * scalar;
            let affine = point.to_affine();
            assert_eq!(point.endomorphism(), point * A::Scalar::ZETA);
            assert_eq!(affine.endomorphism().to_projective(), point.endomorphism());
            if let Some((x, y)) = affine.coordinates() {
                assert_eq!(y.square(), x.square() * x + A::B);
                assert_eq!(
                    affine.endomorphism().coordinates(),
                    Some((A::Base::ZETA * x, y))
                );
            } else {
                assert!(affine.endomorphism().is_identity());
            }
        }
    }
    fn cycle<C: Cycle>() {
        check::<C::HostCurve>();
        check::<C::NestedCurve>();
    }
    cycle::<Pasta>();
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

#[test]
fn generic_representations_support_bls_and_jubjub_fields() {
    use num_bigint::BigUint;
    use zakura_udon::field::{PrimeField, batch_invert, low_u64, random};

    fn check<F: PrimeField>()
    where
        F::Limbs: AsMut<[u64]>,
    {
        let modulus = BigUint::from_bytes_le(
            &F::MODULUS
                .as_ref()
                .iter()
                .flat_map(|limb| limb.to_le_bytes())
                .collect::<Vec<_>>(),
        );
        let integers = [
            BigUint::from(0u8),
            BigUint::from(1u8),
            BigUint::from(u64::MAX),
            (BigUint::from(1u8) << 256usize) + 7u8,
            (BigUint::from(1u8) << 320usize) + 11u8,
            &modulus - 1u8,
        ];
        for integer in integers {
            let integer = integer % &modulus;
            let mut limbs = F::MODULUS;
            limbs.as_mut().fill(0);
            let digits = integer.to_u64_digits();
            limbs.as_mut()[..digits.len()].copy_from_slice(&digits);
            let value = F::from_limbs(limbs).unwrap();
            let bytes = value.to_bytes();
            assert_eq!(BigUint::from_bytes_le(bytes.as_ref()), integer);
            assert_eq!(F::from_bytes(bytes), Some(value));
            let bits = value.to_le_bits();
            assert_eq!(bits.as_ref().len(), bytes.as_ref().len() * 8);
            for (index, bit) in bits.as_ref().iter().enumerate() {
                assert_eq!(*bit, integer.bit(index as u64));
            }
            assert_eq!(low_u64(&value), digits.first().copied().unwrap_or(0));
            assert_eq!(value.square().sqrt().unwrap().square(), value.square());
            if !value.is_zero() {
                assert_eq!(value * value.invert().unwrap(), F::ONE);
            }
        }
        assert!(F::from_limbs(F::MODULUS).is_none());
        let mut draws = 0;
        let sample = random::<F>(|bytes| {
            draws += 1;
            bytes.fill(0xa5);
        });
        assert_eq!(draws, 1);
        assert_eq!(
            BigUint::from_bytes_le(sample.to_bytes().as_ref()),
            BigUint::from_bytes_le(&[0xa5; 64]) % &modulus
        );
        let mut values = [F::ZERO, F::from(2), F::from(3)];
        batch_invert(&mut values, &mut [F::ZERO; 1]);
        assert_eq!(values[0], F::ZERO);
        assert_eq!(values[1] * F::from(2), F::ONE);
        assert_eq!(values[2] * F::from(3), F::ONE);
        assert_eq!(
            zakura_udon::poly::evaluate(&[F::from(2), F::from(3)], F::from(4)),
            F::from(14)
        );
    }

    check::<field_reference::BlsBase>();
    check::<field_reference::BlsScalar>();
    check::<field_reference::JubjubScalar>();
    check::<field_reference::Small>();
    // A high limb outside a short encoding must be rejected, not discarded.
    assert!(field_reference::Small::from_limbs([1 << 32]).is_none());
}
