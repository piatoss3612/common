//! Curve elements for consumers of the optional group traits.

use super::{Affine, EndomorphismAffine, EndomorphismProjective, Projective};
use crate::{
    curve::{AffinePoint, PastaCurve, Point, ProjectivePoint, batch_normalize},
    exec::{ExecutionOptions, SerialExecutor},
    field::{FieldAdapter, PastaField},
    msm::{Bases, Input, ScalarStorage, Scratch},
};
use core::{iter::Sum, ops};

/// An identity-capable affine point implementing [`super::Affine`].
///
/// Operators and the consumer traits live on this wrapper; native [`Point`]
/// arithmetic stays explicit. Borrowed views preserve the underlying storage.
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(transparent)]
pub struct AffineAdapter<C: PastaCurve>(pub(crate) Point<C>);

/// A projective point implementing [`super::Projective`].
///
/// Operators delegate to native [`ProjectivePoint`] methods. Slice views allow
/// batch kernels to borrow consumer buffers without copying or allocation.
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(transparent)]
pub struct ProjectiveAdapter<C: PastaCurve>(pub(crate) ProjectivePoint<C>);

impl<C: PastaCurve> AffineAdapter<C> {
    /// Borrows a native value as a consumer element without copying.
    pub fn from_ref(value: &Point<C>) -> &Self {
        // SAFETY: Self has the same layout and validity as its sole native
        // field. The returned shared reference preserves the input lifetime.
        unsafe { &*(core::ptr::from_ref(value).cast::<Self>()) }
    }

    /// Wraps a native value without changing its representation.
    pub const fn new(value: Point<C>) -> Self {
        Self(value)
    }

    /// Returns the native value.
    pub const fn into_inner(self) -> Point<C> {
        self.0
    }

    /// Borrows the native value.
    pub const fn as_inner(&self) -> &Point<C> {
        &self.0
    }

    /// Mutably borrows the native value.
    pub fn as_inner_mut(&mut self) -> &mut Point<C> {
        &mut self.0
    }

    /// Borrows native values as consumer elements without copying.
    pub fn from_slice(values: &[Point<C>]) -> &[Self] {
        // SAFETY: Self is transparent over Point<C> with identical
        // validity and alignment. The slice keeps its length and borrow; the
        // original reference exclusively controls mutation when mutable.
        unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<Self>(), values.len()) }
    }

    /// Mutably borrows native values as consumer elements without copying.
    pub fn from_slice_mut(values: &mut [Point<C>]) -> &mut [Self] {
        // SAFETY: Self is transparent over Point<C> with identical
        // validity and alignment. The slice keeps its length and borrow; the
        // original reference exclusively controls mutation when mutable.
        unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<Self>(), values.len()) }
    }

    /// Borrows the native buffer beneath consumer elements without copying.
    pub fn as_slice(values: &[Self]) -> &[Point<C>] {
        // SAFETY: Self is transparent over Point<C> with identical
        // validity and alignment. The slice keeps its length and borrow; the
        // original reference exclusively controls mutation when mutable.
        unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<Point<C>>(), values.len()) }
    }

    /// Mutably borrows the native buffer beneath consumer elements without copying.
    pub fn as_slice_mut(values: &mut [Self]) -> &mut [Point<C>] {
        // SAFETY: Self is transparent over Point<C> with identical
        // validity and alignment. The slice keeps its length and borrow; the
        // original reference exclusively controls mutation when mutable.
        unsafe {
            core::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<Point<C>>(), values.len())
        }
    }
}

impl<C: PastaCurve> From<Point<C>> for AffineAdapter<C> {
    fn from(value: Point<C>) -> Self {
        Self(value)
    }
}

impl<C: PastaCurve> From<AffineAdapter<C>> for Point<C> {
    fn from(value: AffineAdapter<C>) -> Self {
        value.0
    }
}
impl<C: PastaCurve> Default for AffineAdapter<C> {
    fn default() -> Self {
        Self(Point::IDENTITY)
    }
}
impl<C: PastaCurve> ops::Neg for AffineAdapter<C> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(self.0.neg())
    }
}
impl<C: PastaCurve> ops::Neg for &AffineAdapter<C> {
    type Output = AffineAdapter<C>;
    #[inline]
    fn neg(self) -> Self::Output {
        AffineAdapter(self.0.neg())
    }
}
impl<C: PastaCurve> ops::Mul<FieldAdapter<C::Scalar>> for AffineAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn mul(self, scalar: FieldAdapter<C::Scalar>) -> Self::Output {
        ProjectiveAdapter(self.0.mul_projective(&scalar.0))
    }
}
impl<C: PastaCurve> ops::Mul<&FieldAdapter<C::Scalar>> for AffineAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn mul(self, scalar: &FieldAdapter<C::Scalar>) -> Self::Output {
        ProjectiveAdapter(self.0.mul_projective(&scalar.0))
    }
}
impl<C: PastaCurve> ops::Mul<FieldAdapter<C::Scalar>> for &AffineAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn mul(self, scalar: FieldAdapter<C::Scalar>) -> Self::Output {
        ProjectiveAdapter(self.0.mul_projective(&scalar.0))
    }
}
impl<C: PastaCurve> ops::Mul<&FieldAdapter<C::Scalar>> for &AffineAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn mul(self, scalar: &FieldAdapter<C::Scalar>) -> Self::Output {
        ProjectiveAdapter(self.0.mul_projective(&scalar.0))
    }
}
impl<C: PastaCurve> ProjectiveAdapter<C> {
    /// Borrows a native value as a consumer element without copying.
    pub fn from_ref(value: &ProjectivePoint<C>) -> &Self {
        // SAFETY: Self has the same layout and validity as its sole native
        // field. The returned shared reference preserves the input lifetime.
        unsafe { &*(core::ptr::from_ref(value).cast::<Self>()) }
    }

    /// Wraps a native value without changing its representation.
    pub const fn new(value: ProjectivePoint<C>) -> Self {
        Self(value)
    }

    /// Returns the native value.
    pub const fn into_inner(self) -> ProjectivePoint<C> {
        self.0
    }

    /// Borrows the native value.
    pub const fn as_inner(&self) -> &ProjectivePoint<C> {
        &self.0
    }

    /// Mutably borrows the native value.
    pub fn as_inner_mut(&mut self) -> &mut ProjectivePoint<C> {
        &mut self.0
    }

    /// Borrows native values as consumer elements without copying.
    pub fn from_slice(values: &[ProjectivePoint<C>]) -> &[Self] {
        // SAFETY: Self is transparent over ProjectivePoint<C> with identical
        // validity and alignment. The slice keeps its length and borrow; the
        // original reference exclusively controls mutation when mutable.
        unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<Self>(), values.len()) }
    }

    /// Mutably borrows native values as consumer elements without copying.
    pub fn from_slice_mut(values: &mut [ProjectivePoint<C>]) -> &mut [Self] {
        // SAFETY: Self is transparent over ProjectivePoint<C> with identical
        // validity and alignment. The slice keeps its length and borrow; the
        // original reference exclusively controls mutation when mutable.
        unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<Self>(), values.len()) }
    }

    /// Borrows the native buffer beneath consumer elements without copying.
    pub fn as_slice(values: &[Self]) -> &[ProjectivePoint<C>] {
        // SAFETY: Self is transparent over ProjectivePoint<C> with identical
        // validity and alignment. The slice keeps its length and borrow; the
        // original reference exclusively controls mutation when mutable.
        unsafe {
            core::slice::from_raw_parts(values.as_ptr().cast::<ProjectivePoint<C>>(), values.len())
        }
    }

    /// Mutably borrows the native buffer beneath consumer elements without copying.
    pub fn as_slice_mut(values: &mut [Self]) -> &mut [ProjectivePoint<C>] {
        // SAFETY: Self is transparent over ProjectivePoint<C> with identical
        // validity and alignment. The slice keeps its length and borrow; the
        // original reference exclusively controls mutation when mutable.
        unsafe {
            core::slice::from_raw_parts_mut(
                values.as_mut_ptr().cast::<ProjectivePoint<C>>(),
                values.len(),
            )
        }
    }
}

impl<C: PastaCurve> From<ProjectivePoint<C>> for ProjectiveAdapter<C> {
    fn from(value: ProjectivePoint<C>) -> Self {
        Self(value)
    }
}

impl<C: PastaCurve> From<ProjectiveAdapter<C>> for ProjectivePoint<C> {
    fn from(value: ProjectiveAdapter<C>) -> Self {
        value.0
    }
}
impl<C: PastaCurve> Default for ProjectiveAdapter<C> {
    fn default() -> Self {
        Self(ProjectivePoint::IDENTITY)
    }
}
impl<C: PastaCurve> ops::Neg for ProjectiveAdapter<C> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(self.0.neg())
    }
}
impl<C: PastaCurve> ops::Neg for &ProjectiveAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn neg(self) -> Self::Output {
        ProjectiveAdapter(self.0.neg())
    }
}
impl<C: PastaCurve> ops::Mul<FieldAdapter<C::Scalar>> for ProjectiveAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn mul(self, scalar: FieldAdapter<C::Scalar>) -> Self::Output {
        ProjectiveAdapter(self.0.mul(&scalar.0))
    }
}
impl<C: PastaCurve> ops::Mul<&FieldAdapter<C::Scalar>> for ProjectiveAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn mul(self, scalar: &FieldAdapter<C::Scalar>) -> Self::Output {
        ProjectiveAdapter(self.0.mul(&scalar.0))
    }
}
impl<C: PastaCurve> ops::Mul<FieldAdapter<C::Scalar>> for &ProjectiveAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn mul(self, scalar: FieldAdapter<C::Scalar>) -> Self::Output {
        ProjectiveAdapter(self.0.mul(&scalar.0))
    }
}
impl<C: PastaCurve> ops::Mul<&FieldAdapter<C::Scalar>> for &ProjectiveAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn mul(self, scalar: &FieldAdapter<C::Scalar>) -> Self::Output {
        ProjectiveAdapter(self.0.mul(&scalar.0))
    }
}
impl<C: PastaCurve> ops::Add<ProjectiveAdapter<C>> for ProjectiveAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn add(self, rhs: ProjectiveAdapter<C>) -> Self::Output {
        ProjectiveAdapter(self.0.add(&rhs.0))
    }
}
impl<C: PastaCurve> ops::Add<&ProjectiveAdapter<C>> for ProjectiveAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn add(self, rhs: &ProjectiveAdapter<C>) -> Self::Output {
        ProjectiveAdapter(self.0.add(&rhs.0))
    }
}
impl<C: PastaCurve> ops::Add<ProjectiveAdapter<C>> for &ProjectiveAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn add(self, rhs: ProjectiveAdapter<C>) -> Self::Output {
        ProjectiveAdapter(self.0.add(&rhs.0))
    }
}
impl<C: PastaCurve> ops::Add<&ProjectiveAdapter<C>> for &ProjectiveAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn add(self, rhs: &ProjectiveAdapter<C>) -> Self::Output {
        ProjectiveAdapter(self.0.add(&rhs.0))
    }
}
impl<C: PastaCurve> ops::AddAssign<ProjectiveAdapter<C>> for ProjectiveAdapter<C> {
    #[inline]
    fn add_assign(&mut self, rhs: ProjectiveAdapter<C>) {
        self.0 = self.0.add(&rhs.0);
    }
}
impl<C: PastaCurve> ops::AddAssign<&ProjectiveAdapter<C>> for ProjectiveAdapter<C> {
    #[inline]
    fn add_assign(&mut self, rhs: &ProjectiveAdapter<C>) {
        self.0 = self.0.add(&rhs.0);
    }
}
impl<C: PastaCurve> ops::Sub<ProjectiveAdapter<C>> for ProjectiveAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn sub(self, rhs: ProjectiveAdapter<C>) -> Self::Output {
        ProjectiveAdapter(self.0.sub(&rhs.0))
    }
}
impl<C: PastaCurve> ops::Sub<&ProjectiveAdapter<C>> for ProjectiveAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn sub(self, rhs: &ProjectiveAdapter<C>) -> Self::Output {
        ProjectiveAdapter(self.0.sub(&rhs.0))
    }
}
impl<C: PastaCurve> ops::Sub<ProjectiveAdapter<C>> for &ProjectiveAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn sub(self, rhs: ProjectiveAdapter<C>) -> Self::Output {
        ProjectiveAdapter(self.0.sub(&rhs.0))
    }
}
impl<C: PastaCurve> ops::Sub<&ProjectiveAdapter<C>> for &ProjectiveAdapter<C> {
    type Output = ProjectiveAdapter<C>;
    #[inline]
    fn sub(self, rhs: &ProjectiveAdapter<C>) -> Self::Output {
        ProjectiveAdapter(self.0.sub(&rhs.0))
    }
}
impl<C: PastaCurve> ops::SubAssign<ProjectiveAdapter<C>> for ProjectiveAdapter<C> {
    #[inline]
    fn sub_assign(&mut self, rhs: ProjectiveAdapter<C>) {
        self.0 = self.0.sub(&rhs.0);
    }
}
impl<C: PastaCurve> ops::SubAssign<&ProjectiveAdapter<C>> for ProjectiveAdapter<C> {
    #[inline]
    fn sub_assign(&mut self, rhs: &ProjectiveAdapter<C>) {
        self.0 = self.0.sub(&rhs.0);
    }
}

impl<C: PastaCurve> From<AffineAdapter<C>> for ProjectiveAdapter<C> {
    fn from(value: AffineAdapter<C>) -> Self {
        Self(value.0.to_projective())
    }
}
impl<C: PastaCurve> From<ProjectiveAdapter<C>> for AffineAdapter<C> {
    fn from(value: ProjectiveAdapter<C>) -> Self {
        Self(value.0.to_point())
    }
}
impl<C: PastaCurve> Sum for ProjectiveAdapter<C> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        Self(iter.fold(ProjectivePoint::IDENTITY, |acc, value| acc.add(&value.0)))
    }
}
impl<'a, C: PastaCurve> Sum<&'a Self> for ProjectiveAdapter<C> {
    fn sum<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
        Self(iter.fold(ProjectivePoint::IDENTITY, |acc, value| acc.add(&value.0)))
    }
}

impl<C: PastaCurve> core::fmt::Debug for AffineAdapter<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

impl<C: PastaCurve> core::fmt::Debug for ProjectiveAdapter<C> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
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

impl<C: PastaCurve> Affine for AffineAdapter<C> {
    type Base = FieldAdapter<C::Base>;
    type Scalar = FieldAdapter<C::Scalar>;
    type Projective = ProjectiveAdapter<C>;
    type Repr = [u8; 32];

    fn identity() -> Self {
        Self(Point::IDENTITY)
    }

    fn generator() -> Self {
        Self(Point::GENERATOR)
    }

    fn is_identity(&self) -> bool {
        self.0.is_identity()
    }

    fn from_xy(x: Self::Base, y: Self::Base) -> Option<Self> {
        Point::from_xy(x.0.reduce(), y.0.reduce()).map(Self)
    }

    fn coordinates(&self) -> Option<(Self::Base, Self::Base)> {
        self.0
            .coordinates()
            .map(|(x, y)| (FieldAdapter(x.into_loose()), FieldAdapter(y.into_loose())))
    }

    fn to_projective(&self) -> ProjectiveAdapter<C> {
        ProjectiveAdapter(self.0.to_projective())
    }

    fn negate(&self) -> Self {
        Self(self.0.neg())
    }

    fn to_bytes(&self) -> [u8; 32] {
        self.0.to_bytes()
    }

    fn from_bytes(bytes: [u8; 32]) -> Option<Self> {
        Point::from_bytes(bytes).map(Self)
    }

    fn msm(scalars: &[Self::Scalar], bases: &[Self]) -> ProjectiveAdapter<C> {
        assert_eq!(
            scalars.len(),
            bases.len(),
            "msm operands must have equal length"
        );
        if bases.is_empty() {
            return ProjectiveAdapter(ProjectivePoint::IDENTITY);
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
        let input = Input::new(
            Bases::Points(Self::as_slice(bases)),
            FieldAdapter::as_slice(scalars),
        );
        // The capacities admit a one-term joint layout and width-four
        // projective buckets. The planner can shrink any nonempty input to fit.
        ProjectiveAdapter(
            input
                .execute(ExecutionOptions::default(), &SerialExecutor, scratch)
                .expect("MSM stack scratch supports a bounded plan"),
        )
    }

    fn batch_to_affine(points: &[ProjectiveAdapter<C>], out: &mut [Self]) {
        // Bounded stack scratch shares one inversion per batch of points
        // without requiring caller storage; `batch_normalize` checks lengths.
        let mut scratch = [PastaField::<C::Base>::ZERO; NORMALIZATION_BATCH];
        batch_normalize(
            ProjectiveAdapter::as_slice(points),
            Self::as_slice_mut(out),
            &mut scratch,
        );
    }
}

impl<C: PastaCurve> Projective for ProjectiveAdapter<C> {
    type Base = FieldAdapter<C::Base>;
    type Scalar = FieldAdapter<C::Scalar>;
    type Affine = AffineAdapter<C>;

    fn identity() -> Self {
        Self(ProjectivePoint::IDENTITY)
    }

    fn generator() -> Self {
        Self(ProjectivePoint::GENERATOR)
    }

    fn is_identity(&self) -> bool {
        self.0.is_identity()
    }

    fn double(&self) -> Self {
        Self(self.0.double())
    }

    fn add_mixed(&self, rhs: &AffineAdapter<C>) -> Self {
        match rhs.0.as_affine() {
            Some(point) => Self(self.0.add_mixed(point)),
            None => *self,
        }
    }

    fn to_affine(&self) -> AffineAdapter<C> {
        AffineAdapter(self.0.to_point())
    }
}

impl<C: PastaCurve> EndomorphismAffine for AffineAdapter<C> {
    const B: FieldAdapter<C::Base> = FieldAdapter(AffinePoint::<C>::B);

    fn endomorphism(&self) -> Self {
        Self(self.0.endomorphism())
    }
}

impl<C: PastaCurve> EndomorphismProjective for ProjectiveAdapter<C> {
    fn endomorphism(&self) -> Self {
        Self(self.0.endomorphism())
    }
}
