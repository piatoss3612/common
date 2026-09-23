//! Traits for code that is generic over a prime-order curve group.
//!
//! [`Affine`] is the protocol-boundary representation: identity is admitted
//! and the canonical compressed encoding is available. [`Projective`] is the
//! representation the group law runs in. Each names the other, so generic
//! code holds one type parameter and reaches both. [`Point`](super::Point)
//! and [`ProjectivePoint`](super::ProjectivePoint) implement the traits for
//! both Pasta curves by delegating to their inherent methods; the traits add
//! no algorithm of their own.
//!
//! [`Affine::msm`] and [`Affine::batch_to_affine`] have reference defaults, so
//! every implementation is complete from the start. The Pasta implementation
//! overrides both with the kernels in [`msm`](super::msm) and
//! [`batch_normalize`](super::batch_normalize), run serially over bounded
//! stack scratch of about 14 KiB. Callers that want parallel execution or
//! reusable scratch use those modules directly.
//!
//! All operations are variable-time, like the arithmetic they delegate to.

use super::{AffinePoint, PastaCurve, Point, ProjectivePoint, batch_normalize};
use crate::{
    exec::{ExecutionOptions, SerialExecutor},
    field::PastaField,
    msm::{Bases, Input, ScalarStorage, Scratch},
};

use core::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign},
};

use crate::field::{CubeRootField, Field, PrimeField};

/// A curve point in affine coordinates, including identity, with the canonical
/// compressed encoding.
///
/// [`Default`] is the identity. Scalar multiplication returns the projective
/// representation, as the inherent ladders do.
pub trait Affine:
    Copy
    + Eq
    + Default
    + Debug
    + Send
    + Sync
    + 'static
    + Neg<Output = Self>
    + Mul<Self::Scalar, Output = Self::Projective>
    + for<'a> Mul<&'a Self::Scalar, Output = Self::Projective>
    + From<Self::Projective>
{
    /// The field containing the coordinates.
    type Base: Field;

    /// The field of scalars, whose modulus is the group order.
    type Scalar: PrimeField;

    /// The curve's fixed-size canonical compressed encoding, including identity.
    type Repr: AsRef<[u8]> + AsMut<[u8]> + Copy + Debug + Eq + Send + Sync + 'static;

    /// The same curve in the coordinates the group law runs in.
    type Projective: Projective<Affine = Self, Base = Self::Base, Scalar = Self::Scalar>;

    /// Returns the group identity.
    fn identity() -> Self;

    /// Returns the fixed generator.
    fn generator() -> Self;

    /// Returns whether this point is identity.
    fn is_identity(&self) -> bool;

    /// Constructs a group element with the given coordinates, or `None` if
    /// they are invalid for this group. Implementations define whether a
    /// coordinate pair also denotes identity; Pasta reserves `(0, 0)`.
    /// For curves with a nontrivial cofactor, validation includes membership
    /// in the prime-order subgroup.
    fn from_xy(x: Self::Base, y: Self::Base) -> Option<Self>;

    /// Returns the coordinates, or `None` for identity.
    fn coordinates(&self) -> Option<(Self::Base, Self::Base)>;

    /// Lifts this point to the projective representation.
    fn to_projective(&self) -> Self::Projective;

    /// Returns the additive inverse.
    fn negate(&self) -> Self;

    /// Encodes this group element, including identity, in canonical compressed bytes.
    fn to_bytes(&self) -> Self::Repr;

    /// Decodes a canonical group encoding, rejecting every other input,
    /// including points outside the prime-order subgroup.
    fn from_bytes(bytes: Self::Repr) -> Option<Self>;

    /// Returns `sum(scalars[i] * bases[i])`.
    ///
    /// The default multiplies each base separately and sums the results. It
    /// requires no scratch and serves as the oracle for optimized
    /// implementations, which are checked against it. The Pasta points run the
    /// planned kernel from [`msm`](super::msm) over bounded stack scratch with
    /// a serial executor.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    fn msm(scalars: &[Self::Scalar], bases: &[Self]) -> Self::Projective {
        assert_eq!(
            scalars.len(),
            bases.len(),
            "msm operands must have equal length"
        );
        let mut sum = Self::Projective::identity();
        for (scalar, base) in scalars.iter().zip(bases) {
            sum += *base * scalar;
        }
        sum
    }

    /// Normalizes `points` into `out`, preserving order and identity positions.
    ///
    /// The default normalizes each point separately, with one inversion each.
    /// Implementations may share inversions and are checked against this.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    fn batch_to_affine(points: &[Self::Projective], out: &mut [Self]) {
        assert_eq!(
            points.len(),
            out.len(),
            "batch normalization operands must have equal length"
        );
        for (point, slot) in points.iter().zip(out.iter_mut()) {
            *slot = point.to_affine();
        }
    }
}

/// A curve point in the coordinates the group law runs in.
///
/// The operators are the complete group law, including identity, and
/// multiplication by a scalar. Equality compares group elements, not
/// representations. Iterator sums accept owned or borrowed points; empty
/// iterators return [`identity`](Self::identity).
pub trait Projective:
    Copy
    + Eq
    + Default
    + Debug
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Neg<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + AddAssign
    + SubAssign
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + Mul<Self::Scalar, Output = Self>
    + for<'a> Mul<&'a Self::Scalar, Output = Self>
    + From<Self::Affine>
    + Sum
    + for<'a> Sum<&'a Self>
{
    /// The field containing the coordinates.
    type Base: Field;

    /// The field of scalars, whose modulus is the group order.
    type Scalar: PrimeField;

    /// The same curve at the protocol boundary.
    type Affine: Affine<Projective = Self, Base = Self::Base, Scalar = Self::Scalar>;

    /// Returns the group identity.
    fn identity() -> Self;

    /// Returns the fixed generator.
    fn generator() -> Self;

    /// Returns whether this point is identity.
    fn is_identity(&self) -> bool;

    /// Returns `2 * self`.
    fn double(&self) -> Self;

    /// Adds an affine point without inversion, admitting identity in either
    /// operand and handling equal points and inverse pairs.
    ///
    /// For Pasta, nonidentity affine operands use
    /// [`ProjectivePoint::add_mixed`](super::ProjectivePoint::add_mixed).
    fn add_mixed(&self, rhs: &Self::Affine) -> Self;

    /// Normalizes this point to affine coordinates.
    fn to_affine(&self) -> Self::Affine;
}

/// An affine curve `y² = x³ + B` with a compatible order-three endomorphism.
///
/// The coordinate map `(x, y) -> (Base::ZETA * x, y)` must equal
/// multiplication by `Scalar::ZETA`. Identity is preserved. This optional
/// capability describes the Pasta curves; it is not required by [`Affine`].
pub trait EndomorphismAffine:
    Affine<Base: CubeRootField, Scalar: CubeRootField, Projective: EndomorphismProjective>
{
    /// The constant term of the curve equation `y² = x³ + B`.
    const B: Self::Base;

    /// Applies the coordinate map described by this trait.
    fn endomorphism(&self) -> Self;
}

/// The projective form of an [`EndomorphismAffine`] curve.
pub trait EndomorphismProjective: Projective<Base: CubeRootField, Scalar: CubeRootField> {
    /// Applies the same group endomorphism as the affine representation,
    /// without normalization.
    fn endomorphism(&self) -> Self;
}

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
