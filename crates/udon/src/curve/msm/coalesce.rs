//! Explicit, retained grouping of repeated bases.

use super::{Bases, CurveError, Input, PastaCurve, Point, Selection};
use crate::curve::{assert_length, assert_scratch};
use crate::field::{PastaField, ReductionState};

/// Caller-owned storage for one point's coalescing key and original position.
///
/// Initialize buffers with [`Self::EMPTY`]. [`CoalescingPlan::prepare`] overwrites
/// the required prefix; records carry no independently bindable point mapping.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct CoalescingKey {
    encoding: [u8; 32],
    position: usize,
}

impl CoalescingKey {
    /// Initial storage value, overwritten during preparation.
    pub const EMPTY: Self = Self {
        encoding: [0; 32],
        position: 0,
    };

    fn x(&self) -> [u8; 32] {
        let mut x = self.encoding;
        x[31] &= 0x7f;
        x
    }

    fn negative(&self) -> bool {
        self.encoding[31] & 0x80 != 0
    }
}

/// Retained grouping of equal and opposite identity-capable affine points.
///
/// Preparation encodes each point once and sorts caller-owned keys. Subsequent
/// [`Self::with_scalars`] calls sum coefficients modulo the scalar field,
/// negating coefficients for odd-y points. Output representatives have even
/// canonical y and are ordered lexicographically by the 32 little-endian bytes
/// of canonical x (not by its integer value). Identities and zero group sums
/// contribute no output. Equal x coordinates on these curves identify equal or
/// opposite points; identity has a distinct all-zero key.
///
/// The plan borrows the original point order and sorted key storage. Preparation
/// takes O(n log n) work, n key records and no other caller scratch; aggregation
/// is linear in n. Both are serial, allocation-free and variable-time. Choosing
/// this preprocessing is explicit: unrelated bases may not repay its cost.
/// Keys, original bases and aggregate outputs are outside
/// [`crate::exec::ExecutionOptions`]'s execution workspace ceiling. The returned
/// [`Input`] uses ordinary MSM scratch and worker contracts.
///
/// ```
/// use zakura_udon::{
///     curve::{Pallas, Point, msm::{CoalescingKey, CoalescingPlan}},
///     field::Fq,
/// };
/// let g = Point::<Pallas>::GENERATOR;
/// let bases = [g, g.neg(), Point::IDENTITY];
/// let mut keys = [CoalescingKey::EMPTY; 3];
/// let plan = CoalescingPlan::prepare(&bases, &mut keys);
/// assert_eq!(plan.groups(), 1);
/// let row = [<Fq>::from_u64(7), <Fq>::from_u64(2), Fq::ONE];
/// let mut points = [Point::IDENTITY];
/// let mut scalars = [Fq::ZERO];
/// let input = plan.with_scalars(&row, &mut points, &mut scalars);
/// assert_eq!(input.len(), 1);
/// assert_eq!(points[0].mul_projective(&scalars[0]),
///     g.mul_projective(&<Fq>::from_u64(5)));
/// ```
#[derive(Clone, Copy, Debug)]
pub struct CoalescingPlan<'a, C: PastaCurve> {
    bases: &'a [Point<C>],
    keys: &'a [CoalescingKey],
    groups: usize,
}

impl<'a, C: PastaCurve> CoalescingPlan<'a, C> {
    /// Encodes and groups the supplied points, retaining their original order.
    ///
    /// `keys` needs at least `bases.len()` records. A short buffer panics before
    /// mutation; only that prefix is changed, regardless of its initial contents.
    /// Empty bases change nothing. Rust's borrows keep storage disjoint and bind
    /// prepared keys to these immutable bases for the plan's lifetime.
    pub fn prepare(bases: &'a [Point<C>], keys: &'a mut [CoalescingKey]) -> Self {
        assert_scratch("coalescing keys", bases.len(), keys.len());
        let keys = &mut keys[..bases.len()];
        for (position, (base, key)) in bases.iter().zip(keys.iter_mut()).enumerate() {
            *key = CoalescingKey {
                encoding: base.to_bytes(),
                position,
            };
        }
        // The original position makes ties deterministic without an allocating
        // stable sort. Keep the encoded sign for coefficient negation later.
        keys.sort_unstable_by(|a, b| a.x().cmp(&b.x()).then(a.position.cmp(&b.position)));
        let groups = keys
            .chunk_by(|a, b| a.x() == b.x())
            .filter(|group| group[0].x() != [0; 32])
            .count();
        Self {
            bases,
            keys,
            groups,
        }
    }

    /// Number of input coefficients required for each row.
    pub const fn len(&self) -> usize {
        self.bases.len()
    }

    /// Whether there are no input terms.
    pub const fn is_empty(&self) -> bool {
        self.bases.is_empty()
    }

    /// Distinct nonidentity groups, bounding each output buffer's required size.
    pub const fn groups(&self) -> usize {
        self.groups
    }

    /// Combines a scalar row and returns its nonzero groups as an ordinary MSM.
    ///
    /// Accepts both scalar reduction states. `coefficients` needs exactly
    /// [`Self::len`] entries; `points` and `scalars` each need at least
    /// [`Self::groups`], even if this row cancels. Mismatches panic before any
    /// writes. Only the returned input's live prefix is changed, leaving every
    /// later output slot untouched. Initial output contents do not matter.
    ///
    /// The returned input borrows only the outputs, which contain canonical-sign
    /// representatives and loose scalar sums. The original coefficients and
    /// plan can be released after aggregation. Rust's borrows keep all input and
    /// writable storage disjoint. An empty or entirely cancelling row returns
    /// an empty MSM and changes no output.
    pub fn with_scalars<'s, S: ReductionState>(
        &self,
        coefficients: &[PastaField<C::Scalar, S>],
        points: &'s mut [Point<C>],
        scalars: &'s mut [PastaField<C::Scalar>],
    ) -> Input<'s, C> {
        assert_length("coefficients", self.len(), coefficients.len());
        assert_scratch("coalesced points", self.groups, points.len());
        assert_scratch("coalesced scalars", self.groups, scalars.len());
        let mut live = 0;
        for group in self.keys.chunk_by(|a, b| a.x() == b.x()) {
            let first = group[0];
            if first.x() == [0; 32] {
                continue;
            }
            let mut sum = PastaField::<C::Scalar>::ZERO;
            for key in group {
                sum = if key.negative() {
                    sum.sub(&coefficients[key.position])
                } else {
                    sum.add(&coefficients[key.position])
                };
            }
            if !sum.is_zero() {
                let base = self.bases[first.position];
                points[live] = if first.negative() { base.neg() } else { base };
                scalars[live] = sum;
                live += 1;
            }
        }
        Input::new(Bases::Points(&points[..live]), &scalars[..live])
    }
}

/// Retained grouping of repeated indices into borrowed base storage.
///
/// Only equal indices merge. Different indices remain separate even when their
/// bases are equal, opposite, or identity. Output indices are ascending; zero
/// scalar sums disappear. Use [`CoalescingPlan`] to recognize point equality.
///
/// Preparation sorts n original positions in caller-owned `usize` storage,
/// taking O(n log n) work. Aggregation is linear in n. Both are serial,
/// allocation-free and variable-time, using no additional caller scratch.
/// The original bases, indices, retained order and aggregate outputs are outside
/// [`crate::exec::ExecutionOptions`]'s workspace ceiling. Returned [`Input`]
/// values use the original base layout and ordinary MSM execution contracts.
#[derive(Clone, Copy, Debug)]
pub struct IndexedCoalescingPlan<'a, C: PastaCurve> {
    selection: Selection<'a, C>,
    indices: &'a [u32],
    order: &'a [usize],
    groups: usize,
}

impl<'a, C: PastaCurve> IndexedCoalescingPlan<'a, C> {
    /// Validates indices and retains a sorted permutation of their positions.
    ///
    /// `order` needs at least `indices.len()` entries or this panics. Invalid
    /// indices return [`CurveError::BaseIndexOutOfBounds`], even if a later row
    /// would give them zero coefficients. All checks precede writes. Only the
    /// required prefix of `order` changes; its initial contents do not matter.
    /// Immutable borrows preserve the validated mapping. Empty indices change
    /// nothing and are valid even with empty bases.
    pub fn prepare(
        bases: Bases<'a, C>,
        indices: &'a [u32],
        order: &'a mut [usize],
    ) -> Result<Self, CurveError> {
        assert_scratch("coalescing order", indices.len(), order.len());
        let selection = Selection::indexed(bases, indices)?;
        let order = &mut order[..indices.len()];
        for (position, entry) in order.iter_mut().enumerate() {
            *entry = position;
        }
        order.sort_unstable_by_key(|&position| (indices[position], position));
        let groups = order.chunk_by(|&a, &b| indices[a] == indices[b]).count();
        Ok(Self {
            selection,
            indices,
            order,
            groups,
        })
    }

    /// Number of input coefficients required for each row.
    pub const fn len(&self) -> usize {
        self.indices.len()
    }

    /// Whether there are no input terms.
    pub const fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Distinct indices, bounding each output buffer's required size.
    pub const fn groups(&self) -> usize {
        self.groups
    }

    /// Combines each index's coefficients modulo the scalar field.
    ///
    /// Accepts both reduction states. `coefficients` needs exactly [`Self::len`]
    /// entries and both outputs need at least [`Self::groups`], even for a row
    /// that cancels. Mismatches panic before writes. Only the returned input's
    /// live prefix changes; all later output entries remain untouched.
    ///
    /// The returned input borrows the original bases and aggregate outputs; the
    /// plan's mapping remains borrowed for that lifetime. The original coefficient
    /// row can be released. Rust's borrows keep input and writable storage disjoint.
    /// Entirely zero rows return empty MSMs.
    pub fn with_scalars<'s, S: ReductionState>(
        &self,
        coefficients: &[PastaField<C::Scalar, S>],
        indices: &'s mut [u32],
        scalars: &'s mut [PastaField<C::Scalar>],
    ) -> Input<'s, C>
    where
        'a: 's,
    {
        assert_length("coefficients", self.len(), coefficients.len());
        assert_scratch("coalesced indices", self.groups, indices.len());
        assert_scratch("coalesced scalars", self.groups, scalars.len());
        let mut live = 0;
        for group in self
            .order
            .chunk_by(|&a, &b| self.indices[a] == self.indices[b])
        {
            let mut sum = PastaField::<C::Scalar>::ZERO;
            for &position in group {
                sum = sum.add(&coefficients[position]);
            }
            if !sum.is_zero() {
                indices[live] = self.indices[group[0]];
                scalars[live] = sum;
                live += 1;
            }
        }
        // Preparation checked every index and aggregation only selects from
        // that immutable mapping, so no second bounds scan is needed.
        Selection::from_validated(self.selection.bases, &indices[..live])
            .with_scalars(&scalars[..live])
    }
}

#[cfg(test)]
mod tests;
