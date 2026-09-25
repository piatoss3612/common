use super::*;
use crate::curve::{assert_length, checked_count, reduce};

impl<C: PastaCurve, E: CurveTableEntry<C>> FixedBaseTable<'_, C, E> {
    /// Bounds each scratch length needed to sum all selected entries together.
    ///
    /// The returned count is a scalar-independent upper bound: two signed
    /// digits per window and one possible final carry for each width-2 table.
    /// Provide this many affine points and base-field elements to [`Self::sum`]
    /// to avoid splitting its reduction into batches. Smaller nonempty buffers
    /// are also accepted. Empty support returns zero.
    ///
    /// Returns [`CurveError::SizeOverflow`] if the count or either slice's byte
    /// length cannot be represented. This does not prevent executing the same
    /// support with bounded scratch.
    pub fn sum_scratch_len(tables: &[Self]) -> Result<usize, CurveError> {
        let mut count = 0_usize;
        for table in tables {
            let width = table.description.window_bits as usize;
            count = add_selection_count::<C>(count, width)?;
        }
        Ok(count)
    }

    /// Sums scalar products over an ordered, borrowed set of expanded tables.
    ///
    /// Computes `sum(scalars[i] * tables[i].base())`. The table handles bind
    /// each position to its curve, base and window layout under [`Self::bind`]'s
    /// trusted-entry contract. Mixed widths, repeated tables and opposite bases
    /// are accepted. Zero scalars, empty support and cancelling sums produce
    /// identity as appropriate. Every term is explicit; support discovery and
    /// any additional application terms belong to the caller.
    ///
    /// Both scratch buffers need at least one element for nonempty support.
    /// Their shorter length bounds each reduction batch;
    /// [`Self::sum_scratch_len`] gives an upper bound sufficient for a single
    /// batch. Initial scratch contents do not matter. Only used prefixes are
    /// written, and surplus tails are untouched.
    /// Rust's borrows keep scratch disjoint from tables and scalars.
    ///
    /// Panics before any writes unless `scalars.len() == tables.len()` and both
    /// buffers meet the minimum length. Execution is serial, allocation-free
    /// and variable-time, with no constant-time guarantee for secret inputs.
    /// Scratch is supplied directly, independently of MSM execution planning:
    /// each capacity element uses 64 affine bytes and 32 field bytes, in addition
    /// to bounded stack storage. Retained tables are inputs; their preparation
    /// cost and storage may outweigh the savings for infrequent use.
    ///
    /// ```
    /// use zakura_udon::{
    ///     curve::{FixedBaseDescription, FixedBaseTable, Pallas, PallasAffine,
    ///             PallasProjective},
    ///     field::{Fp, Fq},
    /// };
    /// let base = PallasAffine::GENERATOR;
    /// let mut entries = [base; 256];
    /// let table = FixedBaseTable::<Pallas>::prepare_with(
    ///     FixedBaseDescription::default(), &base, &mut entries,
    ///     &mut [PallasProjective::IDENTITY; 8], &mut [Fp::ZERO; 8],
    /// )?;
    /// let scalars = [Fq::from_u64(9), <Fq>::from_u64(4).neg()];
    /// // A smaller scratch allowance streams the same selected entries.
    /// let sum = FixedBaseTable::sum(&[table, table], &scalars,
    ///     &mut [base; 32], &mut [Fp::ZERO; 32]);
    /// assert_eq!(sum, table.mul(&Fq::from_u64(5)));
    /// # Ok::<(), zakura_udon::curve::CurveError>(())
    /// ```
    pub fn sum(
        tables: &[Self],
        scalars: &[PastaField<C::Scalar>],
        affine: &mut [AffinePoint<C>],
        field: &mut [PastaField<C::Base>],
    ) -> ProjectivePoint<C> {
        assert_length("scalars", tables.len(), scalars.len());
        let capacity = affine.len().min(field.len());
        assert_scratch("sum scratch", usize::from(!tables.is_empty()), capacity);
        let affine = &mut affine[..capacity];
        let field = &mut field[..capacity];
        let mut used = 0;
        let mut sum = ProjectivePoint::IDENTITY;
        for (table, scalar) in tables.iter().zip(scalars) {
            table.for_each_selected(scalar, |point| {
                affine[used] = point;
                used += 1;
                if used == capacity {
                    sum = sum.add(&reduce::sum(affine, field));
                    used = 0;
                }
            });
        }
        sum.add(&reduce::sum(&mut affine[..used], field))
    }
}

fn add_selection_count<C: PastaCurve>(count: usize, width: usize) -> Result<usize, CurveError> {
    // Pasta's lattice bounds permit a final carry only in the second half
    // at width two; both halves otherwise fit the stored data windows.
    let leaves = 2 * 128_usize.div_ceil(width) + usize::from(width == 2);
    let count = count.checked_add(leaves).ok_or(CurveError::SizeOverflow)?;
    checked_count::<AffinePoint<C>>(count, 1)
}

#[cfg(test)]
#[test]
fn expanded_sum_sizing_rejects_overflow() {
    use crate::curve::Pallas;
    for width in 2..=8 {
        for count in [
            usize::MAX,
            isize::MAX as usize / size_of::<AffinePoint<Pallas>>(),
        ] {
            assert_eq!(
                add_selection_count::<Pallas>(count, width),
                Err(CurveError::SizeOverflow)
            );
        }
    }
}
