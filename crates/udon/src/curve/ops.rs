//! Operator forms of the group law and the curve trait instances.
//!
//! Every operator forwards to an inherent method, so nothing here adds an
//! arithmetic path. Addition and subtraction apply to [`ProjectivePoint`],
//! whose formulas are complete; scalar multiplication is available on all
//! three representations and returns a projective point, as the inherent
//! ladders do. The [`Affine`] and [`Projective`] instances for [`Point`] and
//! [`ProjectivePoint`] delegate the same way, running the planned multiscalar
//! multiplication and batch normalization over bounded stack scratch.

use core::{iter::Sum, ops};

use super::{
    Affine, AffinePoint, EndomorphismAffine, EndomorphismProjective, PastaCurve, Point, Projective,
    ProjectivePoint, batch_normalize,
    msm::{Bases, Input, ScalarStorage, Scratch},
};
use crate::{
    exec::{ExecutionOptions, SerialExecutor},
    field::PastaField,
};

/// Field elements of stack scratch shared by one inversion in
/// [`Affine::batch_to_affine`].
const NORMALIZATION_BATCH: usize = 64;

// Stack scratch for [`Affine::msm`]. The planner streams any input length
// through these capacities with a windowed bucket kernel; the small affine,
// field, and index buffers admit its layouts for very short inputs.
const MSM_SCALARS: usize = 64;
const MSM_DIGITS: usize = 4096;
const MSM_AFFINE: usize = 16;
const MSM_PROJECTIVE: usize = 64;
const MSM_FIELD: usize = 16;
const MSM_INDICES: usize = 16;

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

impl<C: PastaCurve> Affine for Point<C> {
    type Base = PastaField<C::Base>;
    type Scalar = PastaField<C::Scalar>;
    type Projective = ProjectivePoint<C>;
    type Repr = [u8; 32];

    fn identity() -> Self {
        Self::IDENTITY
    }

    fn generator() -> Self {
        Self::GENERATOR
    }

    fn is_identity(&self) -> bool {
        Point::is_identity(self)
    }

    fn from_xy(x: PastaField<C::Base>, y: PastaField<C::Base>) -> Option<Self> {
        Point::from_xy(x.reduce(), y.reduce())
    }

    fn coordinates(&self) -> Option<(PastaField<C::Base>, PastaField<C::Base>)> {
        Point::coordinates(self).map(|(x, y)| (x.into_loose(), y.into_loose()))
    }

    fn to_projective(&self) -> ProjectivePoint<C> {
        Point::to_projective(self)
    }

    fn negate(&self) -> Self {
        Point::neg(self)
    }

    fn to_bytes(&self) -> [u8; 32] {
        Point::to_bytes(self)
    }

    fn from_bytes(bytes: [u8; 32]) -> Option<Self> {
        Point::from_bytes(bytes)
    }

    fn msm(scalars: &[PastaField<C::Scalar>], bases: &[Self]) -> ProjectivePoint<C> {
        assert_eq!(
            scalars.len(),
            bases.len(),
            "msm operands must have equal length"
        );
        if bases.is_empty() {
            return ProjectivePoint::IDENTITY;
        }
        let mut scalar_storage = [ScalarStorage::<C>::ZERO; MSM_SCALARS];
        let mut digits = [0u8; MSM_DIGITS];
        let mut affine = [AffinePoint::<C>::GENERATOR; MSM_AFFINE];
        let mut projective = [ProjectivePoint::<C>::IDENTITY; MSM_PROJECTIVE];
        let mut field = [PastaField::<C::Base>::ZERO; MSM_FIELD];
        let mut indices = [0usize; MSM_INDICES];
        let scratch = Scratch::new(
            &mut scalar_storage,
            &mut digits,
            &mut affine,
            &mut projective,
            &mut field,
            &mut indices,
        );
        let input = Input::new(Bases::Points(bases), scalars);
        // The capacities admit a one-term joint layout and width-four
        // projective buckets. The planner can shrink any nonempty input to fit.
        input
            .execute(ExecutionOptions::default(), &SerialExecutor, scratch)
            .expect("MSM stack scratch supports a bounded plan")
    }

    fn batch_to_affine(points: &[ProjectivePoint<C>], out: &mut [Self]) {
        // Bounded stack scratch shares one inversion per batch of points
        // without requiring caller storage; `batch_normalize` checks lengths.
        let mut scratch = [PastaField::<C::Base>::ZERO; NORMALIZATION_BATCH];
        batch_normalize(points, out, &mut scratch);
    }
}

impl<C: PastaCurve> Projective for ProjectivePoint<C> {
    type Base = PastaField<C::Base>;
    type Scalar = PastaField<C::Scalar>;
    type Affine = Point<C>;

    fn identity() -> Self {
        Self::IDENTITY
    }

    fn generator() -> Self {
        Self::GENERATOR
    }

    fn is_identity(&self) -> bool {
        ProjectivePoint::is_identity(self)
    }

    fn double(&self) -> Self {
        ProjectivePoint::double(self)
    }

    fn add_mixed(&self, rhs: &Point<C>) -> Self {
        match rhs.as_affine() {
            Some(point) => ProjectivePoint::add_mixed(self, point),
            None => *self,
        }
    }

    fn to_affine(&self) -> Point<C> {
        ProjectivePoint::to_point(self)
    }
}

impl<C: PastaCurve> EndomorphismAffine for Point<C> {
    const B: PastaField<C::Base> = AffinePoint::<C>::B;

    fn endomorphism(&self) -> Self {
        Point::endomorphism(self)
    }
}

impl<C: PastaCurve> EndomorphismProjective for ProjectivePoint<C> {
    fn endomorphism(&self) -> Self {
        ProjectivePoint::endomorphism(self)
    }
}

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
