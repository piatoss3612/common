//! Constant regions with explicit additive corrections.

use super::{Bases, CurveError, Input, PastaCurve, Point, ProjectivePoint};
use crate::checks::{assert_length, assert_scratch};
use crate::field::{ConstantPrefix, PastaField, ReductionState};

/// One retained sum of a borrowed, ordered region of bases.
///
/// For bases `G`, a constant coefficient `a` and sparse differences `delta`,
/// compute `a * sum(G) + sum(delta[j] * G[indices[j]])`. Prepare the region once,
/// multiply [`Self::sum`] by `a`, and add the result of executing
/// [`Self::corrections`]. Corrections are additive differences, not replacement
/// values; repeated indices contribute separately. A slice selects a contiguous
/// region of a larger basis, and indices are relative to that slice. Multiple
/// regions compose by adding their results.
///
/// The handle retains one [`Point`] and immutably borrows the original basis
/// order. Preparation is serial, with linear work, constant auxiliary storage,
/// and no caller scratch. Its cost may outweigh savings for small or dense
/// rows. Preparation and execution are allocation-free and variable-time.
/// The retained sum, bases and correction buffers are inputs, outside
/// [`crate::exec::ExecutionOptions`]'s workspace ceiling. Corrections use
/// ordinary [`Input`] scratch, worker and incremental execution contracts.
///
/// A known linear-combination result can similarly be updated by adding an
/// indexed MSM over coefficient differences. The caller establishes that the
/// prior result belongs to the intended ordered basis and coefficients. Add
/// every extra term explicitly; this handle infers no application policy or
/// omitted terms.
///
/// ```
/// use zakura_udon::{
///     curve::{Pallas, Point},
///     field::{ConstantPrefix, Fq},
///     msm::BasisSum,
/// };
/// let bases = [Point::<Pallas>::GENERATOR; 4];
/// let basis = BasisSum::prepare(&bases);
/// let tail = [<Fq>::from_u64(7), <Fq>::from_u64(2)];
/// let row = ConstantPrefix::new(4, &<Fq>::from_u64(5), &tail)?;
/// let mut differences = [Fq::ZERO; 2];
/// let corrections = basis.tail_corrections(row, &mut differences);
/// assert_eq!(corrections.len(), 2);
/// assert_eq!(differences.map(Fq::reduce),
///     [<Fq>::from_u64(2).reduce(), <Fq>::from_i64(-3).reduce()]);
/// // Add the executed correction input to this constant contribution.
/// let constant = basis.sum().mul_projective(&row.constant());
/// assert_eq!(constant, Point::<Pallas>::GENERATOR.mul_projective(&<Fq>::from_u64(20)));
/// # Ok::<(), zakura_udon::field::ConstantPrefixError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct BasisSum<'a, C: PastaCurve> {
    original: &'a [Point<C>],
    sum: Point<C>,
}

impl<'a, C: PastaCurve> BasisSum<'a, C> {
    /// Sums the supplied region, preserving identities and cancellation.
    ///
    /// Empty input has the identity sum. Repeated bases are counted each time.
    /// The borrowed slice fixes the order used by subsequent correction rows.
    pub fn prepare(bases: &'a [Point<C>]) -> Self {
        let mut sum = ProjectivePoint::IDENTITY;
        for base in bases {
            if let Some(base) = base.as_affine() {
                sum = sum.add_mixed(base);
            }
        }
        Self {
            original: bases,
            sum: sum.to_point(),
        }
    }

    /// Original bases in the order addressed by correction indices and tails.
    pub const fn original(&self) -> &'a [Point<C>] {
        self.original
    }

    /// Sum of every base in this region, including repeated entries.
    pub const fn sum(&self) -> Point<C> {
        self.sum
    }

    /// Binds sparse additive differences to positions within this region.
    ///
    /// The returned input contains only the corrections. Add its executed
    /// result to the constant contribution or caller-supplied prior result,
    /// along with every explicit extra term. Zero differences, repeated or
    /// unordered indices, and empty corrections are accepted.
    ///
    /// Panics unless the index and difference counts match. Returns
    /// [`CurveError::BaseIndexOutOfBounds`] for the first invalid index, even
    /// when its difference is zero. Validation changes no storage. Immutable
    /// borrows preserve the basis, indices and differences during execution.
    pub fn corrections<'s>(
        &self,
        indices: &'s [u32],
        differences: &'s [PastaField<C::Scalar>],
    ) -> Result<Input<'s, C>, CurveError>
    where
        'a: 's,
    {
        Input::indexed(Bases::Points(self.original), indices, differences)
    }

    /// Writes actual tail values minus the constant and binds their bases.
    ///
    /// [`ConstantPrefix`] describes actual values after a constant prefix. This
    /// method writes `tail[j] - constant` and selects the corresponding suffix
    /// of the region. Add the executed result to `sum() * row.constant()` to
    /// recover the complete row. With a full tail that combination is independent
    /// of the constant; an empty tail needs no differences or correction work.
    ///
    /// `row.len()` must equal the region length, and `differences` needs at least
    /// `row.tail().len()` entries. A mismatch panics before mutation. Only that
    /// output prefix is written; surplus tails are untouched. The returned input
    /// borrows the bases and differences, not the descriptor's original tail.
    /// Rust's borrows keep input and output disjoint. Conversion is linear in the
    /// tail length, accepts both reduction states and needs no other scratch.
    pub fn tail_corrections<'s, S: ReductionState>(
        &self,
        row: ConstantPrefix<'_, C::Scalar, S>,
        differences: &'s mut [PastaField<C::Scalar>],
    ) -> Input<'s, C>
    where
        'a: 's,
    {
        assert_length("constant row", self.original.len(), row.len());
        assert_scratch("differences", row.tail().len(), differences.len());
        let differences = &mut differences[..row.tail().len()];
        for (difference, value) in differences.iter_mut().zip(row.tail()) {
            *difference = value.sub(&row.constant());
        }
        Input::new(
            Bases::Points(&self.original[row.prefix_len()..]),
            differences,
        )
    }
}

#[cfg(test)]
mod tests;
