//! Explicit scalar compaction over a validated base selection.

use super::{CurveError, Input, PastaCurve, Selection, assert_length, assert_scratch};
use crate::field::{PastaField, ReductionState};

impl<'a, C: PastaCurve> Selection<'a, C> {
    /// Copies nonzero terms into caller storage without gathering bases.
    ///
    /// `coefficients` must have exactly [`Self::len`] entries. Both reduction
    /// states are accepted; loose zero represented by the modulus is removed.
    /// Each output buffer needs at least as many entries as there are nonzero
    /// coefficients. Length or capacity mismatches panic before writes. For a
    /// dense selection, a nonzero position exceeding `u32::MAX` returns
    /// [`CurveError::SizeOverflow`] before writes.
    ///
    /// The live output prefixes preserve term order and this selection's base
    /// indices, including duplicates. Identity bases with nonzero coefficients
    /// remain selected. Scalars are copied in loose representation. All slots
    /// beyond the live prefixes remain untouched, regardless of initial contents.
    /// Empty and all-zero rows return empty MSMs and change no output.
    ///
    /// The returned [`Input`] borrows the original bases and output buffers;
    /// the original selection's mapping stays borrowed for that lifetime, but
    /// `coefficients` can be released. Rust's borrows keep writable storage
    /// disjoint. Use [`Input::selection`] to retain the compact mapping for later
    /// rows supplied in the same compact order. This does not establish that
    /// omitted coefficients of another dense row are zero. If support is already
    /// known, use [`Self::indexed`] and [`Self::with_scalars`] directly.
    ///
    /// Compaction takes two serial linear passes, no allocation or additional
    /// caller scratch, and is variable-time. Output buffers are input storage
    /// outside [`crate::exec::ExecutionOptions`]'s workspace ceiling; execution
    /// uses ordinary MSM scratch. Choose compaction explicitly: its scanning and
    /// copying costs need not pay off for dense rows. Add all extra terms to the
    /// executed result, including when this input is empty.
    ///
    /// ```
    /// use zakura_udon::{
    ///     curve::{Pallas, Point, msm::{Bases, Selection}},
    ///     field::Fq,
    /// };
    /// let bases = [Point::<Pallas>::GENERATOR; 3];
    /// let row = [Fq::ZERO, <Fq>::from_u64(7), Fq::ZERO];
    /// let mut indices = [0; 1];
    /// let mut scalars = [Fq::ZERO; 1];
    /// let input = Selection::new(Bases::Points(&bases))
    ///     .with_nonzero_scalars(&row, &mut indices, &mut scalars)?;
    /// assert_eq!(input.len(), 1);
    /// let retained = input.selection();
    /// let next = [<Fq>::from_u64(9)];
    /// assert_eq!(retained.with_scalars(&next).len(), 1);
    /// # Ok::<(), zakura_udon::curve::CurveError>(())
    /// ```
    pub fn with_nonzero_scalars<'s, S: ReductionState>(
        &self,
        coefficients: &[PastaField<C::Scalar, S>],
        indices: &'s mut [u32],
        scalars: &'s mut [PastaField<C::Scalar>],
    ) -> Result<Input<'s, C>, CurveError>
    where
        'a: 's,
    {
        assert_length("coefficients", self.len(), coefficients.len());
        let mut live = 0;
        let mut last = None;
        for (position, coefficient) in coefficients.iter().enumerate() {
            if !coefficient.is_zero() {
                live += 1;
                last = Some(position);
            }
        }
        if self.indices.is_none() {
            check_dense_index(last)?;
        }
        assert_scratch("nonzero indices", live, indices.len());
        assert_scratch("nonzero scalars", live, scalars.len());
        let mut output = 0;
        for (position, coefficient) in coefficients.iter().enumerate() {
            if !coefficient.is_zero() {
                // The first pass checked the largest retained dense position;
                // indexed selections already carry validated immutable indices.
                indices[output] = self.indices.map_or(position as u32, |i| i[position]);
                scalars[output] = coefficient.into_loose();
                output += 1;
            }
        }
        Ok(Selection {
            bases: self.bases,
            indices: Some(&indices[..live]),
        }
        .with_scalars(&scalars[..live]))
    }
}

fn check_dense_index(last: Option<usize>) -> Result<(), CurveError> {
    if last.is_some_and(|position| u32::try_from(position).is_err()) {
        Err(CurveError::SizeOverflow)
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests;
