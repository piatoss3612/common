//! Operator forms of the native group law.
//!
//! Every operator forwards to an inherent method, so nothing here adds an
//! arithmetic path. Addition and subtraction apply to [`ProjectivePoint`],
//! whose formulas are complete; scalar multiplication is available on all
//! three representations and returns a projective point, as the inherent
//! ladders do.

use core::{iter::Sum, ops};

use super::{AffinePoint, PastaCurve, Point, ProjectivePoint};
use crate::field::PastaField;

macro_rules! forward_point_operator {
    ($trait:ident, $method:ident, $inherent:ident) => {
        impl<C: PastaCurve> ops::$trait for ProjectivePoint<C> {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: Self) -> Self {
                ProjectivePoint::$inherent(&self, &rhs)
            }
        }

        impl<C: PastaCurve> ops::$trait<&ProjectivePoint<C>> for ProjectivePoint<C> {
            type Output = Self;

            #[inline]
            fn $method(self, rhs: &Self) -> Self {
                ProjectivePoint::$inherent(&self, rhs)
            }
        }

        impl<C: PastaCurve> ops::$trait<ProjectivePoint<C>> for &ProjectivePoint<C> {
            type Output = ProjectivePoint<C>;

            #[inline]
            fn $method(self, rhs: ProjectivePoint<C>) -> ProjectivePoint<C> {
                ProjectivePoint::$inherent(self, &rhs)
            }
        }

        impl<C: PastaCurve> ops::$trait<&ProjectivePoint<C>> for &ProjectivePoint<C> {
            type Output = ProjectivePoint<C>;

            #[inline]
            fn $method(self, rhs: &ProjectivePoint<C>) -> ProjectivePoint<C> {
                ProjectivePoint::$inherent(self, rhs)
            }
        }
    };
}

macro_rules! forward_point_assign_operator {
    ($trait:ident, $method:ident, $inherent:ident) => {
        impl<C: PastaCurve> ops::$trait for ProjectivePoint<C> {
            #[inline]
            fn $method(&mut self, rhs: Self) {
                *self = ProjectivePoint::$inherent(self, &rhs);
            }
        }

        impl<C: PastaCurve> ops::$trait<&ProjectivePoint<C>> for ProjectivePoint<C> {
            #[inline]
            fn $method(&mut self, rhs: &Self) {
                *self = ProjectivePoint::$inherent(self, rhs);
            }
        }
    };
}

forward_point_operator!(Add, add, add);
forward_point_operator!(Sub, sub, sub);
forward_point_assign_operator!(AddAssign, add_assign, add);
forward_point_assign_operator!(SubAssign, sub_assign, sub);

impl<C: PastaCurve> Sum for ProjectivePoint<C> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.fold(Self::IDENTITY, |sum, point| {
            ProjectivePoint::add(&sum, &point)
        })
    }
}

impl<'a, C: PastaCurve> Sum<&'a ProjectivePoint<C>> for ProjectivePoint<C> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        iter.fold(Self::IDENTITY, |sum, point| {
            ProjectivePoint::add(&sum, point)
        })
    }
}

macro_rules! forward_negation {
    ($point:ident) => {
        impl<C: PastaCurve> ops::Neg for $point<C> {
            type Output = Self;

            #[inline]
            fn neg(self) -> Self {
                $point::neg(&self)
            }
        }

        impl<C: PastaCurve> ops::Neg for &$point<C> {
            type Output = $point<C>;

            #[inline]
            fn neg(self) -> $point<C> {
                $point::neg(self)
            }
        }
    };
}

forward_negation!(AffinePoint);
forward_negation!(Point);
forward_negation!(ProjectivePoint);

macro_rules! forward_scalar_multiplication {
    ($point:ident, $inherent:ident) => {
        impl<C: PastaCurve> ops::Mul<PastaField<C::Scalar>> for $point<C> {
            type Output = ProjectivePoint<C>;

            #[inline]
            fn mul(self, scalar: PastaField<C::Scalar>) -> ProjectivePoint<C> {
                $point::$inherent(&self, &scalar)
            }
        }

        impl<C: PastaCurve> ops::Mul<&PastaField<C::Scalar>> for $point<C> {
            type Output = ProjectivePoint<C>;

            #[inline]
            fn mul(self, scalar: &PastaField<C::Scalar>) -> ProjectivePoint<C> {
                $point::$inherent(&self, scalar)
            }
        }

        impl<C: PastaCurve> ops::Mul<PastaField<C::Scalar>> for &$point<C> {
            type Output = ProjectivePoint<C>;

            #[inline]
            fn mul(self, scalar: PastaField<C::Scalar>) -> ProjectivePoint<C> {
                $point::$inherent(self, &scalar)
            }
        }

        impl<C: PastaCurve> ops::Mul<&PastaField<C::Scalar>> for &$point<C> {
            type Output = ProjectivePoint<C>;

            #[inline]
            fn mul(self, scalar: &PastaField<C::Scalar>) -> ProjectivePoint<C> {
                $point::$inherent(self, scalar)
            }
        }
    };
}

forward_scalar_multiplication!(AffinePoint, mul_projective);
forward_scalar_multiplication!(Point, mul_projective);
forward_scalar_multiplication!(ProjectivePoint, mul);

impl<C: PastaCurve> From<Point<C>> for ProjectivePoint<C> {
    /// Lifts the point without inversion; see [`ProjectivePoint::from_point`].
    #[inline]
    fn from(point: Point<C>) -> Self {
        ProjectivePoint::from_point(&point)
    }
}

impl<C: PastaCurve> From<ProjectivePoint<C>> for Point<C> {
    /// Normalizes the point; see [`ProjectivePoint::to_point`].
    fn from(point: ProjectivePoint<C>) -> Self {
        point.to_point()
    }
}
