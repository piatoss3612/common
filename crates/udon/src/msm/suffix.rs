//! Ordered suffix sums and explicit coefficient differences.

use super::{Bases, CurveError, Input, PastaCurve, Point, ProjectivePoint, Selection};
use crate::checks::{assert_length, assert_scratch};
use crate::curve::batch_normalize;
use crate::field::PastaField;

/// Borrowed suffix sums of an ordered basis, including identity sums.
///
/// For original bases `G`, stores `H[i] = G[i] + ... + G[n - 1]`. A coefficient
/// row `x` over `G` has the same sum as the row `d` over `H`, where `d[0] = x[0]`
/// and `d[i] = x[i] - x[i - 1]`. Repeated coefficients therefore become zero
/// terms. Preparation is explicit; whether its cost and storage pay off depends
/// on coefficient structure and basis reuse.
///
/// The handle borrows only the prepared sums; original basis storage can be
/// reused after preparation.
/// [`Self::with_scalars`] and [`Self::with_monotone_unsigned`] write differences
/// into caller-owned storage and return ordinary [`Input`] values. Their MSM
/// scratch and execution contracts are unchanged. Basis storage and difference
/// buffers are inputs, outside [`crate::exec::ExecutionOptions`]'s workspace ceiling;
/// preparation scratch can be reused after [`Self::prepare`] returns.
/// All operations are allocation-free and variable-time.
///
/// ```
/// use zakura_udon::{
///     curve::{Pallas, Point, ProjectivePoint},
///     field::Fp,
///     msm::SuffixBasis,
/// };
/// let bases = [Point::<Pallas>::GENERATOR; 3];
/// let mut sums = [Point::IDENTITY; 3];
/// let basis = SuffixBasis::prepare(&bases, &mut sums,
///     &mut [ProjectivePoint::IDENTITY; 2], &mut [Fp::ZERO; 2]);
/// assert_eq!(basis.suffix()[0].to_projective(),
///     ProjectivePoint::GENERATOR.double().add(&ProjectivePoint::GENERATOR));
/// let mut differences = [0; 3];
/// let input = basis.with_monotone_unsigned(&[5, 5, 9], &mut differences)?;
/// assert_eq!(input.len(), 3);
/// assert_eq!(differences, [5, 0, 4]);
/// # Ok::<(), zakura_udon::curve::CurveError>(())
/// ```
#[derive(Clone, Copy, Debug)]
pub struct SuffixBasis<'a, C: PastaCurve> {
    suffix: &'a [Point<C>],
}

impl<'a, C: PastaCurve> SuffixBasis<'a, C> {
    /// Writes one suffix sum per base and borrows the required output prefix.
    ///
    /// `output` needs at least `bases.len()` entries. Nonempty bases need at least
    /// one projective scratch entry; its length bounds the accumulation batches.
    /// Field scratch follows [`batch_normalize`]: smaller buffers subdivide
    /// normalization, and empty scratch normalizes individually. Initial output
    /// and scratch contents do not matter. Unused tails are untouched. Empty
    /// bases require no storage and change nothing.
    ///
    /// Insufficient output or projective scratch panics before any buffer is
    /// changed. Rust's borrows keep bases, output and scratch disjoint.
    pub fn prepare(
        bases: &[Point<C>],
        output: &'a mut [Point<C>],
        projective: &mut [ProjectivePoint<C>],
        field: &mut [PastaField<C::Base>],
    ) -> Self {
        assert_scratch("suffix output", bases.len(), output.len());
        assert_scratch(
            "projective",
            usize::from(!bases.is_empty()),
            projective.len(),
        );
        let suffix = &mut output[..bases.len()];
        let mut sum = ProjectivePoint::IDENTITY;
        let mut end = bases.len();
        while end != 0 {
            let start = end.saturating_sub(projective.len());
            for index in (start..end).rev() {
                if let Some(base) = bases[index].as_affine() {
                    sum = sum.add_mixed(base);
                }
                projective[index - start] = sum;
            }
            batch_normalize(&projective[..end - start], &mut suffix[start..end], field);
            end = start;
        }
        Self { suffix }
    }

    /// Prepared suffix sums in the original index order.
    pub const fn suffix(&self) -> &'a [Point<C>] {
        self.suffix
    }

    /// Number of bases and required coefficients per row.
    pub const fn len(&self) -> usize {
        self.suffix.len()
    }

    /// Whether the basis is empty.
    pub const fn is_empty(&self) -> bool {
        self.suffix.is_empty()
    }

    /// Writes field differences and binds them to the prepared suffix sums.
    ///
    /// Accepts arbitrary loose field coefficients, including decreases and
    /// modular wraparound. `coefficients` needs exactly [`Self::len`] entries;
    /// `differences` needs at least that many. A mismatch panics before mutation.
    /// Writes only the required prefix, leaving the tail untouched. The returned
    /// input borrows the differences and suffix sums, not the coefficients.
    pub fn with_scalars<'s>(
        &self,
        coefficients: &[PastaField<C::Scalar>],
        differences: &'s mut [PastaField<C::Scalar>],
    ) -> Input<'s, C>
    where
        'a: 's,
    {
        assert_length("coefficients", self.len(), coefficients.len());
        assert_scratch("differences", self.len(), differences.len());
        let differences = &mut differences[..self.len()];
        let mut previous = PastaField::ZERO;
        for (coefficient, difference) in coefficients.iter().zip(differences.iter_mut()) {
            *difference = coefficient.sub(&previous);
            previous = *coefficient;
        }
        Input::new(Bases::Points(self.suffix), differences)
    }

    /// Validates a nondecreasing integer row, then writes unsigned differences.
    ///
    /// Includes zero, repeated values and `u128::MAX`; all differences fit
    /// `u128`. Returns [`CurveError::InvalidScalar`] at the second position of
    /// the first decreasing pair, before changing any differences.
    ///
    /// `coefficients` needs exactly [`Self::len`] entries and `differences` at
    /// least that many. Length mismatches panic before mutation. Only the
    /// required prefix is written; the tail is untouched. The returned input
    /// borrows the differences and suffix sums, not the coefficients.
    pub fn with_monotone_unsigned<'s>(
        &self,
        coefficients: &[u128],
        differences: &'s mut [u128],
    ) -> Result<Input<'s, C>, CurveError>
    where
        'a: 's,
    {
        assert_length("coefficients", self.len(), coefficients.len());
        assert_scratch("differences", self.len(), differences.len());
        for (position, pair) in coefficients.windows(2).enumerate() {
            if pair[0] > pair[1] {
                return Err(CurveError::InvalidScalar {
                    position: position + 1,
                });
            }
        }
        let differences = &mut differences[..self.len()];
        let mut previous = 0;
        for (coefficient, difference) in coefficients.iter().zip(differences.iter_mut()) {
            *difference = coefficient - previous;
            previous = *coefficient;
        }
        Ok(Selection::new(Bases::Points(self.suffix)).with_unsigned(differences))
    }
}

#[cfg(test)]
mod tests;
