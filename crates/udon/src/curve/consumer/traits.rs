//! Traits for code that is generic over a prime-order curve group.
//!
//! [`Affine`] is the protocol-boundary representation: identity is admitted
//! and the canonical compressed encoding is available. [`Projective`] is the
//! representation the group law runs in. Each names the other, so generic
//! code holds one type parameter and reaches both.
//! [`AffineAdapter`](super::AffineAdapter) and
//! [`ProjectiveAdapter`](super::ProjectiveAdapter) implement the traits for
//! both Pasta curves by delegating to native methods. Every method is required,
//! so the traits add no algorithm of their own.
//!
//! The Pasta implementations of [`Affine::msm`] and [`Affine::batch_to_affine`]
//! use the kernels in [`msm`](crate::msm) and
//! [`batch_normalize`](crate::curve::batch_normalize), run serially over bounded
//! stack scratch of about 14 KiB. Callers that want parallel execution or
//! reusable scratch use those modules directly.
//!
//! All operations are variable-time, like the arithmetic they delegate to.

use core::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, AddAssign, Mul, Neg, Sub, SubAssign},
};

use crate::field::Field;

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
    type Scalar: Field;

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
    /// Empty input returns identity. Pasta adapters run the planned kernel
    /// from [`msm`](crate::msm) over bounded stack scratch with a serial executor.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    fn msm(scalars: &[Self::Scalar], bases: &[Self]) -> Self::Projective;

    /// Normalizes `points` into `out`, preserving order and identity positions.
    ///
    /// Pasta adapters use native batch normalization with shared inversions.
    ///
    /// # Panics
    ///
    /// Panics before mutation if the lengths differ.
    fn batch_to_affine(points: &[Self::Projective], out: &mut [Self]);
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
    type Scalar: Field;

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
    /// [`ProjectivePoint::add_mixed`](crate::curve::ProjectivePoint::add_mixed).
    fn add_mixed(&self, rhs: &Self::Affine) -> Self;

    /// Normalizes this point to affine coordinates.
    fn to_affine(&self) -> Self::Affine;
}

/// An affine curve `y² = x³ + B` with a compatible order-three endomorphism.
///
/// The coordinate map `(x, y) -> (Base::ZETA * x, y)` must equal
/// multiplication by `Scalar::ZETA`. Both roots must have order three.
/// Identity is preserved. This optional capability describes the Pasta curves;
/// it is not required by [`Affine`].
pub trait EndomorphismAffine: Affine<Projective: EndomorphismProjective> {
    /// The constant term of the curve equation `y² = x³ + B`.
    const B: Self::Base;

    /// Applies the coordinate map described by this trait.
    fn endomorphism(&self) -> Self;
}

/// The projective form of an [`EndomorphismAffine`] curve.
pub trait EndomorphismProjective: Projective {
    /// Applies the same group endomorphism as the affine representation,
    /// without normalization.
    fn endomorphism(&self) -> Self;
}
