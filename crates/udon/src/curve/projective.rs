//! Jacobian formulas specialized to the Pasta equation's zero linear term.

use core::{fmt, marker::PhantomData};

use super::{AffinePoint, PastaCurve, Point, ProjectivePoint};
use crate::field::PastaField;

impl<C: PastaCurve> fmt::Debug for ProjectivePoint<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProjectivePoint")
            .field("x", &self.x)
            .field("y", &self.y)
            .field("z", &self.z)
            .finish()
    }
}

impl<C: PastaCurve> Default for ProjectivePoint<C> {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl<C: PastaCurve> PartialEq for ProjectivePoint<C> {
    fn eq(&self, rhs: &Self) -> bool {
        if self.is_identity() || rhs.is_identity() {
            return self.is_identity() == rhs.is_identity();
        }
        // Cross-multiplication compares affine x and y without dividing by z.
        let z1_squared = self.z.square();
        let z2_squared = rhs.z.square();
        self.x.mul(&z2_squared) == rhs.x.mul(&z1_squared)
            && self.y.mul(&z2_squared).mul(&rhs.z) == rhs.y.mul(&z1_squared).mul(&self.z)
    }
}

impl<C: PastaCurve> Eq for ProjectivePoint<C> {}

impl<C: PastaCurve> ProjectivePoint<C> {
    /// The additive identity, represented by zero coordinates.
    pub const IDENTITY: Self = Self {
        x: PastaField::ZERO,
        y: PastaField::ZERO,
        z: PastaField::ZERO,
        marker: PhantomData,
    };
    /// The generator `(-1, 2)` with `z = 1`.
    pub const GENERATOR: Self = AffinePoint::GENERATOR.to_projective();

    /// Returns the additive identity.
    pub const fn identity() -> Self {
        Self::IDENTITY
    }

    /// Returns the generator `(-1, 2)` with `z = 1`.
    pub const fn generator() -> Self {
        Self::GENERATOR
    }

    /// Lifts a nonidentity affine point with `z = 1`.
    pub const fn from_affine(point: &AffinePoint<C>) -> Self {
        Self {
            x: point.x,
            y: point.y,
            z: PastaField::ONE,
            marker: PhantomData,
        }
    }

    /// Lifts an affine point, preserving identity.
    pub const fn from_point(point: &Point<C>) -> Self {
        match point.as_affine() {
            Some(point) => Self::from_affine(point),
            None => Self::IDENTITY,
        }
    }

    /// Returns whether `z = 0`.
    pub const fn is_identity(&self) -> bool {
        self.z.is_zero()
    }

    /// Borrows the Jacobian coordinates `(x, y, z)`.
    #[expect(
        clippy::type_complexity,
        reason = "Expose coordinates in Jacobian order."
    )]
    pub const fn coordinates(
        &self,
    ) -> (
        &PastaField<C::Base>,
        &PastaField<C::Base>,
        &PastaField<C::Base>,
    ) {
        (&self.x, &self.y, &self.z)
    }

    /// Recovers affine coordinates with one inversion for a nonidentity point.
    ///
    /// Identity maps to [`Point::IDENTITY`].
    /// Use [`super::batch_normalize`] to share the inversion across a slice.
    pub fn to_point(&self) -> Point<C> {
        match self.z.invert() {
            Some(inverse) => self.normalize_with_inverse(&inverse).to_point(),
            None => Point::IDENTITY,
        }
    }

    // Callers establish z != 0 and supply z^-1; this helper cannot represent
    // identity. The squared and cubed inverse undo Jacobian scaling.
    pub(super) fn normalize_with_inverse(&self, inverse: &PastaField<C::Base>) -> AffinePoint<C> {
        let squared = inverse.square();
        AffinePoint {
            x: self.x.mul(&squared),
            y: self.y.mul(&squared).mul(inverse),
            marker: PhantomData,
        }
    }

    /// Applies [`AffinePoint::endomorphism`] in projective coordinates.
    ///
    /// Multiplies `x` by the coordinate field's [`PastaField::zeta`] value,
    /// leaving `y` and `z` unchanged. Preserves identity and projective scaling.
    pub fn endomorphism(&self) -> Self {
        Self {
            x: self.x.mul(&PastaField::zeta()),
            ..*self
        }
    }

    /// Returns the additive inverse `(x, -y, z)`.
    pub fn neg(&self) -> Self {
        Self {
            y: self.y.neg(),
            ..*self
        }
    }

    /// Returns `2 * self`, without inversion.
    pub fn double(&self) -> Self {
        if self.is_identity() {
            return Self::IDENTITY;
        }
        // For y^2 = x^3 + 5, use tangent numerator (3/2)*x^2 and z' = y*z.
        // Scaling this output by (4, 8, 2) gives the usual Jacobian doubling
        // coordinates, so omitting those factors preserves the affine point.
        let b = self.y.square();
        let c = b.square();
        let d = self.x.mul(&b);
        let e = self.x.square().triple().half();
        let f = e.square();
        let z = self.z.mul(&self.y);
        let x = f.sub(&d.double());
        let y = e.mul_sub(&d.sub(&x), &c);
        Self {
            x,
            y,
            z,
            marker: PhantomData,
        }
    }

    /// Returns `self + rhs`, handling identity, equal points, and inverse pairs
    /// without inversion.
    pub fn add(&self, rhs: &Self) -> Self {
        if self.is_identity() {
            return *rhs;
        }
        if rhs.is_identity() {
            return *self;
        }
        let z1_squared = self.z.square();
        let z2_squared = rhs.z.square();
        let u1 = self.x.mul(&z2_squared);
        let u2 = rhs.x.mul(&z1_squared);
        let s1 = self.y.mul(&z2_squared).mul(&rhs.z);
        let s2 = rhs.y.mul(&z1_squared).mul(&self.z);
        // u1/u2 and s1/s2 use the common scale z1*z2. Equal x coordinates
        // require doubling or identity; the addition formula needs h != 0.
        if u1 == u2 {
            return if s1 == s2 {
                self.double()
            } else {
                Self::IDENTITY
            };
        }
        let h = u2.sub(&u1);
        // Unscaled differences give z' = z1*z2*h. The common doubled-difference
        // formula scales this output by (4, 8, 2), preserving the affine point.
        let i = h.square();
        let j = h.mul(&i);
        let r = s2.sub(&s1);
        let v = u1.mul(&i);
        let x = r.square().sub(&j).sub(&v.double());
        let y = r.mul_sub_product(&v.sub(&x), &s1, &j);
        let z = self.z.mul(&rhs.z).mul(&h);
        Self {
            x,
            y,
            z,
            marker: PhantomData,
        }
    }

    /// Adds a nonidentity affine point without inversion.
    ///
    /// Handles identity, equal points, and inverse pairs.
    pub fn add_mixed(&self, rhs: &AffinePoint<C>) -> Self {
        if self.is_identity() {
            return Self::from_affine(rhs);
        }
        let z1_squared = self.z.square();
        let u2 = rhs.x.mul(&z1_squared);
        let s2 = rhs.y.mul(&z1_squared).mul(&self.z);
        // This is the general addition formula with the affine operand's z = 1.
        if self.x == u2 {
            return if self.y == s2 {
                self.double()
            } else {
                Self::IDENTITY
            };
        }
        let h = u2.sub(&self.x);
        // The unscaled formula in add specializes to z2 = 1 here.
        let i = h.square();
        let j = h.mul(&i);
        let r = s2.sub(&self.y);
        let v = self.x.mul(&i);
        let x = r.square().sub(&j).sub(&v.double());
        let y = r.mul_sub_product(&v.sub(&x), &self.y, &j);
        let z = self.z.mul(&h);
        Self {
            x,
            y,
            z,
            marker: PhantomData,
        }
    }

    /// Returns `self - rhs`, without inversion.
    pub fn sub(&self, rhs: &Self) -> Self {
        self.add(&rhs.neg())
    }

    /// Multiplies by a scalar using variable-time doubling and addition.
    ///
    /// Processes the full canonical scalar; zero returns identity. The scalar
    /// must satisfy [`PastaField`]'s reduced-residue invariant. Uses bounded
    /// internal stack storage, with no caller table, scratch, or allocation.
    ///
    /// The current implementation uses an inversion-free binary ladder for
    /// scalars below `2^64`. Larger scalars with a nonidentity base prepare eight
    /// cached affine entries on the stack, using one field inversion per call,
    /// then run a GLV/Eisenstein ladder. Identity skips preparation. The strategy
    /// is selected internally; stack frame sizes depend on the compiler and target.
    /// Use [`EisensteinTable`](super::EisensteinTable) or
    /// [`FixedBaseTable`](super::FixedBaseTable) to retain preparation for repeated
    /// multiplication of the same base.
    pub fn mul(&self, scalar: &PastaField<C::Scalar>) -> Self {
        if self.is_identity() || scalar.is_zero() {
            return Self::IDENTITY;
        }
        // Short scalars use the binary ladder without paying for table setup.
        if scalar
            .to_canonical_uint()
            .highest_set_bit()
            .is_some_and(|high| high < 64)
        {
            super::scalar::multiply(scalar, |point| point.add(self))
        } else {
            super::eisenstein::multiply_once(self, scalar)
        }
    }
}
