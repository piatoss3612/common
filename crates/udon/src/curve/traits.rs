//! Traits for code that is generic over a Pasta curve group.
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

use core::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign},
};

use crate::field::FftField;

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
    type Base: FftField;

    /// The field of scalars, whose modulus is the group order.
    type Scalar: FftField;

    /// The same curve in the coordinates the group law runs in.
    type Projective: Projective<Affine = Self, Base = Self::Base, Scalar = Self::Scalar>;

    /// The constant term of the curve equation `y² = x³ + B`.
    const B: Self::Base;

    /// Returns the group identity.
    fn identity() -> Self;

    /// Returns the fixed generator.
    fn generator() -> Self;

    /// Returns whether this point is identity.
    fn is_identity(&self) -> bool;

    /// Constructs the point with the given coordinates, or `None` if they are
    /// not on the curve. `(0, 0)` denotes identity.
    fn from_xy(x: Self::Base, y: Self::Base) -> Option<Self>;

    /// Returns the coordinates, or `None` for identity.
    fn coordinates(&self) -> Option<(Self::Base, Self::Base)>;

    /// Lifts this point to the projective representation.
    fn to_projective(&self) -> Self::Projective;

    /// Returns the additive inverse.
    fn negate(&self) -> Self;

    /// Applies the curve endomorphism `(x, y) -> (zeta * x, y)`, which equals
    /// multiplication by the scalar field's cube root of unity.
    fn endomorphism(&self) -> Self;

    /// Encodes this point in 32 canonical compressed bytes; identity encodes
    /// as zeros.
    fn to_bytes(&self) -> [u8; 32];

    /// Decodes a canonical compressed encoding, rejecting every other input.
    fn from_bytes(bytes: [u8; 32]) -> Option<Self>;

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
    type Base: FftField;

    /// The field of scalars, whose modulus is the group order.
    type Scalar: FftField;

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

    /// Applies the curve endomorphism; see [`Affine::endomorphism`].
    fn endomorphism(&self) -> Self;

    /// Normalizes this point to affine coordinates.
    fn to_affine(&self) -> Self::Affine;
}
