//! Small Edwards subgroup used only by the trait consumers.

use super::field_model::{Small, SmallScalar};
use core::{iter::Sum, ops};
use zakura_udon::{
    curve::{Affine, Projective},
    field::{Field, PrimeField},
};

// The order-five subgroup of -x² + y² = 1 + 6x²y² over F_17, generated
// by (5, 7). Store the discrete logarithm to make the test model small;
// the curve test independently checks the Edwards addition law.
const COORDINATES: [(u64, u64); 5] = [(0, 1), (5, 7), (10, 9), (7, 9), (12, 7)];

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct Edwards(pub(crate) SmallScalar);

impl Edwards {
    pub(crate) fn xy(self) -> (Small, Small) {
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
