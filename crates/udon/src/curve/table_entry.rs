//! Storage choices shared by compact and expanded multiplication tables.

use core::{fmt, marker::PhantomData};

use super::{AffinePoint, CurveError, PastaCurve, ProjectivePoint, is_reduced};
use crate::field::PastaField;

/// Exact table length and minimum scratch lengths for curve table preparation.
///
/// All lengths count elements, not bytes. Preparation leaves scratch tails
/// beyond these lengths untouched. Batch multiplication reports its scratch
/// through [`EisensteinTableBatch::multiplication_scratch`][batch_scratch].
///
/// [batch_scratch]: super::EisensteinTableBatch::multiplication_scratch
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CurveTableRequirements {
    /// Number of entries, in the caller-selected representation.
    pub table_entries: usize,
    /// Minimum number of projective scratch elements.
    pub projective_scratch: usize,
    /// Minimum number of base-field scratch elements.
    pub field_scratch: usize,
}

/// A nonidentity affine point with its endomorphism x-coordinate cached.
///
/// Stores `(x, zeta * x, y)` in [`PastaField`]'s Montgomery representation:
/// 96 bytes with alignment 8, where `zeta` is the coordinate field's
/// [`PastaField::zeta`] value. All coordinates must be reduced, `(x, y)` must
/// satisfy the curve equation, and the cached coordinate must equal `zeta * x`.
/// [`bento::Pod`] checks memory layout only and requires little endian.
/// Invalid stored values remain memory-safe but arithmetic can panic or give
/// incorrect results. [`EisensteinTable::bind`](super::EisensteinTable::bind)
/// and [`FixedBaseTable::bind`](super::FixedBaseTable::bind) establish these
/// invariants.
///
/// Use [`AffinePoint`] entries to save storage, or this type to avoid field
/// multiplication when a table lookup applies the endomorphism.
// SAFETY: The derive checks padding and field layouts. Every bit pattern is
// safe to read and share; mathematical invariants do not affect memory safety.
#[derive(Clone, Copy, Eq, PartialEq, bento::Pod)]
#[repr(C)]
pub struct PreparedAffinePoint<C: PastaCurve> {
    x: PastaField<C::Base>,
    endomorphism_x: PastaField<C::Base>,
    y: PastaField<C::Base>,
    marker: PhantomData<C>,
}

impl<C: PastaCurve> fmt::Debug for PreparedAffinePoint<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PreparedAffinePoint")
            .field("affine", &self.to_affine())
            .field("endomorphism_x", &self.endomorphism_x)
            .finish()
    }
}

impl<C: PastaCurve> PreparedAffinePoint<C> {
    /// Caches the endomorphism coordinate of a valid affine point.
    ///
    /// Assumes [`AffinePoint`]'s mathematical invariants, as do point operations.
    pub fn from_affine(point: &AffinePoint<C>) -> Self {
        Self {
            x: point.x,
            endomorphism_x: point.x.mul(&PastaField::zeta()),
            y: point.y,
            marker: PhantomData,
        }
    }

    /// Returns the underlying affine point without checking stored coordinates.
    ///
    /// Copies only `x` and `y`; the cached coordinate is ignored. The result
    /// needs [`AffinePoint`]'s invariants before arithmetic or encoding. Use
    /// [`AffinePoint::from_xy`] to validate untrusted coordinates.
    pub const fn to_affine(&self) -> AffinePoint<C> {
        AffinePoint {
            x: self.x,
            y: self.y,
            marker: PhantomData,
        }
    }
}

pub(super) mod sealed {
    pub trait Entry {}
}

/// Selects the stored representation of a curve multiplication table entry.
///
/// This trait is sealed to [`AffinePoint`] (64 bytes) and
/// [`PreparedAffinePoint`] (96 bytes, cached endomorphism). Both implement
/// [`bento::Pod`]. Select the entry type through the table's generic parameter;
/// table preparation and validation enforce the same mathematical layout.
/// Generic callers can initialize entry buffers with [`Self::from_affine`].
///
/// ```
/// use zakura_udon::curve::{
///     AffinePoint, CurveTableEntry, Pallas, PastaCurve, PreparedAffinePoint,
/// };
///
/// fn buffer<C: PastaCurve, E: CurveTableEntry<C>>(base: &AffinePoint<C>) -> [E; 8] {
///     [E::from_affine(base); 8]
/// }
/// let base = AffinePoint::<Pallas>::GENERATOR;
/// let entries = buffer::<Pallas, PreparedAffinePoint<Pallas>>(&base);
/// assert_eq!(entries[0].affine(), base);
/// assert_eq!(entries[0].rotated(1), base.endomorphism());
/// assert!(entries[0].valid_cache());
/// ```
pub trait CurveTableEntry<C: PastaCurve>: sealed::Entry + Copy + fmt::Debug + Send + Sync {
    /// Constructs an entry from a point satisfying [`AffinePoint`]'s invariants.
    ///
    /// Copies affine coordinates and computes any cached endomorphism coordinate.
    fn from_affine(point: &AffinePoint<C>) -> Self;

    /// Copies affine coordinates without validating them or any cached value.
    ///
    /// The result needs [`AffinePoint`]'s invariants before arithmetic or encoding.
    /// Use [`AffinePoint::from_xy`] to validate untrusted coordinates.
    fn affine(&self) -> AffinePoint<C>;

    /// Applies [`AffinePoint::endomorphism`] `rotation` times.
    ///
    /// Assumes the entry satisfies its type's mathematical invariants, including
    /// cache consistency for [`PreparedAffinePoint`].
    ///
    /// # Panics
    ///
    /// Panics unless `rotation` is in `0..3`.
    fn rotated(&self, rotation: usize) -> AffinePoint<C>;

    /// Checks cached coordinates, assuming the affine coordinates are reduced.
    ///
    /// Always returns `true` for [`AffinePoint`], which has no cache. For
    /// [`PreparedAffinePoint`], checks that the cached coordinate is reduced and
    /// equals `zeta * x`. Does not check the curve equation or table membership;
    /// use checked table binding to establish the full table contract.
    fn valid_cache(&self) -> bool;
}

impl<C: PastaCurve> sealed::Entry for AffinePoint<C> {}
impl<C: PastaCurve> sealed::Entry for PreparedAffinePoint<C> {}

impl<C: PastaCurve> CurveTableEntry<C> for AffinePoint<C> {
    fn from_affine(point: &AffinePoint<C>) -> Self {
        *point
    }
    fn affine(&self) -> AffinePoint<C> {
        *self
    }
    fn rotated(&self, rotation: usize) -> AffinePoint<C> {
        match rotation {
            0 => *self,
            1 => self.endomorphism(),
            2 => Self {
                x: self.x.mul(&PastaField::zeta_inverse()),
                ..*self
            },
            _ => unreachable!("a cube root has three rotations"),
        }
    }
    fn valid_cache(&self) -> bool {
        true
    }
}

impl<C: PastaCurve> CurveTableEntry<C> for PreparedAffinePoint<C> {
    fn from_affine(point: &AffinePoint<C>) -> Self {
        Self::from_affine(point)
    }
    fn affine(&self) -> AffinePoint<C> {
        self.to_affine()
    }
    fn rotated(&self, rotation: usize) -> AffinePoint<C> {
        // zeta² + zeta + 1 = 0 gives the second rotation without a product.
        let x = match rotation {
            0 => self.x,
            1 => self.endomorphism_x,
            2 => self.x.add(&self.endomorphism_x).neg(),
            _ => unreachable!("a cube root has three rotations"),
        };
        AffinePoint {
            x,
            y: self.y,
            marker: PhantomData,
        }
    }
    fn valid_cache(&self) -> bool {
        is_reduced(&self.endomorphism_x) && self.endomorphism_x == self.x.mul(&PastaField::zeta())
    }
}

pub(super) fn check_entry<C: PastaCurve, E: CurveTableEntry<C>>(
    expected: &ProjectivePoint<C>,
    entry: &E,
) -> Result<(), CurveError> {
    let affine = entry.affine();
    // Reject raw residues before any arithmetic, including cached-coordinate
    // validation. Equality with a valid multiple establishes curve membership.
    if !is_reduced(&affine.x)
        || !is_reduced(&affine.y)
        || !entry.valid_cache()
        || *expected != affine.to_projective()
    {
        return Err(CurveError::InvalidTable);
    }
    Ok(())
}
