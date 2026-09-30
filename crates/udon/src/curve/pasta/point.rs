//! Identity-capable affine operations.

use core::fmt;

use super::{AffinePoint, PastaCurve, Point, ProjectivePoint};
use crate::field::{PastaField, Reduced};

impl<C: PastaCurve> fmt::Debug for Point<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("Point").field(&self.0).finish()
    }
}

impl<C: PastaCurve> Default for Point<C> {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<C: PastaCurve> Point<C> {
    /// The additive identity.
    pub const IDENTITY: Self = Self(None);
    /// The generator `(-1, 2)`.
    pub const GENERATOR: Self = AffinePoint::GENERATOR.to_point();

    /// Constructs identity for `(0, 0)` or a checked nonidentity point.
    ///
    /// Returns `None` for other coordinates rejected by [`AffinePoint::from_xy`].
    pub fn from_xy(
        x: PastaField<C::Base, Reduced>,
        y: PastaField<C::Base, Reduced>,
    ) -> Option<Self> {
        if x.is_zero() && y.is_zero() {
            return Some(Self::IDENTITY);
        }
        AffinePoint::from_xy(x, y).map(|point| point.to_point())
    }

    /// Returns whether this point is identity.
    pub const fn is_identity(&self) -> bool {
        self.0.is_none()
    }

    /// Borrows the nonidentity point, or returns `None` for identity.
    pub const fn as_affine(&self) -> Option<&AffinePoint<C>> {
        self.0.as_ref()
    }

    /// Borrows `(x, y)`, or returns `None` for identity.
    #[expect(
        clippy::type_complexity,
        reason = "Expose coordinates as a borrowed pair."
    )]
    pub const fn coordinates(
        &self,
    ) -> Option<(&PastaField<C::Base, Reduced>, &PastaField<C::Base, Reduced>)> {
        match &self.0 {
            Some(point) => Some(point.coordinates()),
            None => None,
        }
    }

    /// Lifts this point to Jacobian coordinates.
    pub const fn to_projective(&self) -> ProjectivePoint<C> {
        ProjectivePoint::from_point(self)
    }

    /// Applies [`AffinePoint::endomorphism`], preserving identity.
    pub fn endomorphism(&self) -> Self {
        Self(self.0.map(|point| point.endomorphism()))
    }

    /// Returns the additive inverse, preserving identity.
    pub fn neg(&self) -> Self {
        Self(self.0.map(|point| point.neg()))
    }

    /// Returns `self + rhs` in projective coordinates, without inversion.
    ///
    /// Use [`super::batch_normalize`] to share an inversion across several
    /// results, or [`ProjectivePoint::to_point`] to normalize a single result.
    pub fn add(&self, rhs: &Self) -> ProjectivePoint<C> {
        match (self.as_affine(), rhs.as_affine()) {
            (None, _) => rhs.to_projective(),
            (_, None) => self.to_projective(),
            (Some(lhs), Some(rhs)) => lhs.to_projective().add_mixed(rhs),
        }
    }

    /// Returns `self - rhs` in projective coordinates, without inversion.
    pub fn sub(&self, rhs: &Self) -> ProjectivePoint<C> {
        self.add(&rhs.neg())
    }

    /// Returns `2 * self` in projective coordinates, without inversion.
    pub fn double(&self) -> ProjectivePoint<C> {
        self.to_projective().double()
    }

    /// Multiplies by a scalar using variable-time doubling and mixed addition.
    ///
    /// Uses [`AffinePoint::mul_projective`] for a nonidentity base, with the
    /// same scalar requirements, internal stack storage, and preparation costs.
    /// Multiplying identity returns identity without preparation.
    pub fn mul_projective(&self, scalar: &PastaField<C::Scalar>) -> ProjectivePoint<C> {
        match self.as_affine() {
            Some(point) => point.mul_projective(scalar),
            None => ProjectivePoint::IDENTITY,
        }
    }
}
