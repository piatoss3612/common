//! Pasta denominator inversion and record visitors.

use super::{PastaField, PrimeModulus};
use crate::field::batch::{
    InversionLanes, NonzeroInversionLanes, batch_invert_groups_inner, inverse_seed,
};
use core::fmt;

/// Inverts a slice of nonzero field elements in place with one inversion.
///
/// Every value must be reduced and nonzero. `prefix` must have at least
/// `values.len()` elements; its initial contents do not matter, and its unused
/// tail is untouched. Empty input performs no inversion. Callers using chord
/// denominators must remove exceptional pairs before calling.
pub(crate) fn invert_nonzero<M: PrimeModulus>(
    values: &mut [PastaField<M>],
    prefix: &mut [PastaField<M>],
) {
    if values.is_empty() {
        return;
    }
    if values.len() == 1 {
        values[0] = values[0].invert().expect("nonzero denominators");
        return;
    }
    // Seed each lane with its first value. The reverse pass leaves those two
    // inverses directly, avoiding multiplication by one and unused updates.
    // Including the lane merge, n = values.len() needs 3*(n-1) multiplications
    // outside the inversion.
    let prefix = &mut prefix[..values.len()];
    let mut products = InversionLanes([values[0], values[1]]);
    for (i, (value, prefix)) in values.iter().zip(prefix.iter_mut()).enumerate().skip(2) {
        *prefix = products.push(i, value);
    }
    let mut inverses = products.invert(None);
    for (i, (value, prefix)) in values
        .iter_mut()
        .zip(prefix.iter())
        .enumerate()
        .skip(2)
        .rev()
    {
        *value = inverses.pop(i, value, prefix);
    }
    values[..2].copy_from_slice(&inverses.0);
}

/// Rejected input for batch inversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchInversionError {
    /// A record denominator was zero; no visitor has run.
    ZeroDenominator {
        /// Position of the zero denominator in the record slice.
        index: usize,
    },
}

impl fmt::Display for BatchInversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDenominator { index } => write!(f, "zero denominator at record {index}"),
        }
    }
}

impl core::error::Error for BatchInversionError {}

/// Inverts denominators read from immutable records and visits their inverses.
///
/// `denominator` must return the same field value for a record on every
/// call; the operation may evaluate it more than once.
/// Every denominator is checked for zero before the first visitor call;
/// a zero returns [`BatchInversionError::ZeroDenominator`]
/// without changing scratch or invoking `visit`. Empty input does nothing.
///
/// With one scratch field per record, the operation shares one field inversion.
/// Smaller scratch bounds batches, and empty scratch uses individual inversions.
/// No allocation or record copying occurs. Initial scratch contents do not matter;
/// surplus entries are untouched. Arithmetic is variable-time.
///
/// `visit` receives the original index, record, and inverse. Visit order is
/// unspecified. A visitor error stops immediately; earlier visitor effects and
/// scratch writes remain. The same partial-progress rule applies to a panic.
/// `E` converts validation failures and can also represent visitor failures.
///
/// ```
/// use zakura_udon::field::{BatchInversionError, Fp, try_batch_invert_by};
///
/// let records: [(Fp, Fp); 2] = [(Fp::from_u64(6), Fp::from_u64(2)),
///                (Fp::from_u64(20), Fp::from_u64(5))];
/// let mut quotients = [Fp::ZERO; 2];
/// try_batch_invert_by(&records, |record| record.1, &mut [Fp::ZERO; 2],
///     |index, record, inverse| {
///         quotients[index] = record.0.mul(&inverse);
///         Ok::<_, BatchInversionError>(())
///     }).unwrap();
/// assert_eq!(quotients.map(|value| value.reduce()), [Fp::from_u64(3), Fp::from_u64(4)]);
/// ```
pub fn try_batch_invert_by<R, M: PrimeModulus, E: From<BatchInversionError>>(
    records: &[R],
    denominator: impl Fn(&R) -> PastaField<M>,
    scratch: &mut [PastaField<M>],
    visit: impl FnMut(usize, &R, PastaField<M>) -> Result<(), E>,
) -> Result<(), E> {
    try_batch_invert_by_inner(records, denominator, None, scratch, visit)
}

/// Visits `scale / denominator(record)` for each immutable record.
///
/// This is [`try_batch_invert_by`] with a shared scale applied to each batch's
/// inverse seed. The scale may be zero, but every denominator must still be
/// nonzero: validation finishes before any scratch writes or visitor calls.
/// The denominator must remain stable across repeated calls. `visit` receives
/// the original index, record, and scaled inverse. Empty input does nothing.
/// Scratch sizing, untouched tails, unspecified visit order, and partial
/// progress on visitor errors or panics follow [`try_batch_invert_by`].
/// No allocation or record copying occurs; arithmetic is variable-time.
pub fn try_batch_invert_scaled_by<R, M: PrimeModulus, E: From<BatchInversionError>>(
    records: &[R],
    denominator: impl Fn(&R) -> PastaField<M>,
    scale: &PastaField<M>,
    scratch: &mut [PastaField<M>],
    visit: impl FnMut(usize, &R, PastaField<M>) -> Result<(), E>,
) -> Result<(), E> {
    try_batch_invert_by_inner(records, denominator, Some(scale), scratch, visit)
}

fn try_batch_invert_by_inner<R, M: PrimeModulus, E: From<BatchInversionError>>(
    records: &[R],
    denominator: impl Fn(&R) -> PastaField<M>,
    scale: Option<&PastaField<M>>,
    scratch: &mut [PastaField<M>],
    mut visit: impl FnMut(usize, &R, PastaField<M>) -> Result<(), E>,
) -> Result<(), E> {
    for (index, record) in records.iter().enumerate() {
        if denominator(record).is_zero() {
            return Err(BatchInversionError::ZeroDenominator { index }.into());
        }
    }
    if scratch.is_empty() {
        for (index, record) in records.iter().enumerate() {
            visit(
                index,
                record,
                inverse_seed(&denominator(record), scale).expect("stable nonzero denominator"),
            )?;
        }
        return Ok(());
    }
    for (chunk, records) in records.chunks(scratch.len()).enumerate() {
        let mut products = NonzeroInversionLanes::new();
        for (index, (record, prefix)) in records.iter().zip(scratch.iter_mut()).enumerate() {
            if let Some(product) = products.push(index, &denominator(record)) {
                *prefix = product;
            }
        }
        let mut inverses = products
            .invert_scaled(scale)
            .expect("nonempty nonzero batch");
        for (index, record) in records.iter().enumerate().rev() {
            let inverse = inverses.pop(index, &denominator(record), &scratch[index]);
            visit(chunk * scratch.len() + index, record, inverse)?;
        }
    }
    Ok(())
}

/// Replaces each nonzero value `x` by `scale / x`, preserving zeros.
///
/// The scale may be zero. Scratch sizing, untouched scratch tails, allocation,
/// and variable-time behavior are the same as [`batch_invert`](crate::field::batch_invert). Scaling the
/// shared inverse seed costs one extra multiplication per nonzero batch,
/// including when scratch is empty and each value forms its own batch.
/// Empty or all-zero inputs perform no inversion.
///
/// ```
/// use zakura_udon::field::{Fp, batch_invert_scaled};
/// let mut values = [Fp::from_u64(2), Fp::ZERO, Fp::from_u64(3)];
/// batch_invert_scaled(&mut values, &Fp::from_u64(6), &mut [Fp::ZERO; 3]);
/// assert_eq!(values.map(|value| value.reduce()),
///            [Fp::from_u64(3), Fp::ZERO, Fp::from_u64(2)]);
/// ```
pub fn batch_invert_scaled<M: PrimeModulus>(
    values: &mut [PastaField<M>],
    scale: &PastaField<M>,
    scratch: &mut [PastaField<M>],
) {
    batch_invert_groups_scaled(&mut [values], scale, scratch)
}

/// Replaces nonzero values across disjoint slices by `scale / x`.
///
/// This is [`batch_invert_scaled`] over the concatenation of `groups`, without
/// flattening or copying their contents. Zeros stay zero, and the scale may be
/// zero. Scratch sizing, inversion counts, untouched scratch tails, and the
/// requirement for stable [`AsMut::as_mut`] views are the same as
/// [`batch_invert_groups`](crate::field::batch_invert_groups). No allocation is performed; arithmetic is
/// variable-time. Each nonzero batch scales its shared inverse seed once.
pub fn batch_invert_groups_scaled<M: PrimeModulus>(
    groups: &mut [impl AsMut<[PastaField<M>]>],
    scale: &PastaField<M>,
    scratch: &mut [PastaField<M>],
) {
    batch_invert_groups_inner(groups, Some(scale), scratch)
}
