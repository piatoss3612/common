use core::fmt;

use super::{PastaField, PrimeModulus};

fn inverse_seed<M: PrimeModulus>(
    value: &PastaField<M>,
    scale: Option<&PastaField<M>>,
) -> Option<PastaField<M>> {
    value
        .invert()
        .map(|inverse| scale.map_or(inverse, |scale| inverse.mul(scale)))
}

/// Two product chains sharing one inversion.
///
/// Push nonzero factors in index order, invert the lane products, then pop in
/// reverse order with the saved prefixes. Separate even and odd lanes shorten
/// multiplication dependencies. Callers can seed the lanes with their first
/// factors to avoid multiplication by one and unused final updates.
struct InversionLanes<M: PrimeModulus>([PastaField<M>; 2]);

impl<M: PrimeModulus> InversionLanes<M> {
    fn push(&mut self, index: usize, value: &PastaField<M>) -> PastaField<M> {
        let product = &mut self.0[index & 1];
        let prefix = *product;
        *product = product.mul(value);
        prefix
    }

    fn invert(self, scale: Option<&PastaField<M>>) -> Self {
        // For lane products a and b, multiplying (a*b)^-1 by the opposite
        // product recovers each lane's inverse with one field inversion.
        // Applying the scale to this seed carries it into both recovery lanes.
        let inverse = inverse_seed(&self.0[0].mul(&self.0[1]), scale)
            .expect("a product of nonzero field elements is nonzero");
        Self([inverse.mul(&self.0[1]), inverse.mul(&self.0[0])])
    }

    fn pop(
        &mut self,
        index: usize,
        value: &PastaField<M>,
        prefix: &PastaField<M>,
    ) -> PastaField<M> {
        let inverse = &mut self.0[index & 1];
        // The prefix cancels earlier factors; multiplying by this value then
        // removes it from the lane inverse for the next step.
        let result = inverse.mul(prefix);
        *inverse = inverse.mul(value);
        result
    }
}

/// Two inversion lanes that track their first nonzero factors.
///
/// Callers skip zeros and push nonzero factors in increasing input-index order;
/// index parity selects the lane. Save each returned prefix, call `invert` once,
/// then pop the same factors and indices in reverse order. `invert` returns
/// `None` if no factors were pushed.
///
/// A lane's first push returns `None`, and its matching pop ignores the prefix.
/// Seeding these endpoints avoids multiplication by one; a single occupied lane
/// also avoids the product merge needed to share an inversion across two lanes.
pub(crate) struct NonzeroInversionLanes<M: PrimeModulus> {
    products: InversionLanes<M>,
    first: [Option<usize>; 2],
}

impl<M: PrimeModulus> NonzeroInversionLanes<M> {
    pub(crate) fn new() -> Self {
        Self {
            products: InversionLanes([PastaField::ONE; 2]),
            first: [None; 2],
        }
    }

    pub(crate) fn push(&mut self, index: usize, value: &PastaField<M>) -> Option<PastaField<M>> {
        let lane = index & 1;
        if self.first[lane].is_some() {
            Some(self.products.push(index, value))
        } else {
            self.first[lane] = Some(index);
            self.products.0[lane] = *value;
            None
        }
    }

    pub(crate) fn invert(self) -> Option<Self> {
        self.invert_scaled(None)
    }

    fn invert_scaled(mut self, scale: Option<&PastaField<M>>) -> Option<Self> {
        let lane = match self.first {
            [None, None] => return None,
            [Some(_), Some(_)] => {
                self.products = self.products.invert(scale);
                return Some(self);
            }
            [Some(_), None] => 0,
            [None, Some(_)] => 1,
        };
        self.products.0[lane] =
            inverse_seed(&self.products.0[lane], scale).expect("nonzero product");
        Some(self)
    }

    pub(crate) fn pop(
        &mut self,
        index: usize,
        value: &PastaField<M>,
        prefix: &PastaField<M>,
    ) -> PastaField<M> {
        if self.first[index & 1] == Some(index) {
            self.products.0[index & 1]
        } else {
            self.products.pop(index, value, prefix)
        }
    }
}

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

/// Replaces each nonzero value by its inverse, preserving zeros.
///
/// Uses one inversion per nonzero batch. With at least `values.len()` scratch
/// fields the entire input is one batch; smaller buffers bound the batch size,
/// and empty scratch inverts values individually. Initial scratch contents do
/// not matter and the unused tail is untouched. No allocation is performed.
/// Arithmetic is variable-time, including the locations of zeros.
///
/// ```
/// use zakura_udon::field::{Fp, batch_invert};
/// let mut values = [Fp::from_u64(7), Fp::ZERO, Fp::from_u64(3)];
/// batch_invert(&mut values, &mut [Fp::ZERO; 3]);
/// assert_eq!(values[0].mul(&<Fp>::from_u64(7)).reduce(), Fp::ONE);
/// assert!(values[1].is_zero());
/// ```
pub fn batch_invert<M: PrimeModulus>(values: &mut [PastaField<M>], scratch: &mut [PastaField<M>]) {
    batch_invert_groups(&mut [values], scratch)
}

/// Replaces each nonzero value `x` by `scale / x`, preserving zeros.
///
/// The scale may be zero. Scratch sizing, untouched scratch tails, allocation,
/// and variable-time behavior are the same as [`batch_invert`]. Scaling the
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

/// Inverts nonzero values across disjoint slices with shared inversions.
///
/// This is [`batch_invert`] over the concatenation of `groups`, without
/// flattening or copying their contents. Empty groups are allowed. With one
/// scratch field per input element, all groups share one inversion. Smaller
/// scratch bounds batches across group boundaries, with one inversion per
/// batch containing a nonzero value; empty scratch uses individual inversions.
/// Initial scratch contents do not matter and any unused tail is untouched.
/// Empty or all-zero inputs perform no inversion. No allocation is performed.
/// Arithmetic is variable-time, including the locations of zeros.
///
/// Each group's [`AsMut::as_mut`] must expose the same slice throughout the
/// call. Arrays, mutable slices, and vectors satisfy this requirement. A custom
/// implementation that changes its view can produce incorrect results or panic.
pub fn batch_invert_groups<M: PrimeModulus>(
    groups: &mut [impl AsMut<[PastaField<M>]>],
    scratch: &mut [PastaField<M>],
) {
    batch_invert_groups_inner(groups, None, scratch)
}

/// Replaces nonzero values across disjoint slices by `scale / x`.
///
/// This is [`batch_invert_scaled`] over the concatenation of `groups`, without
/// flattening or copying their contents. Zeros stay zero, and the scale may be
/// zero. Scratch sizing, inversion counts, untouched scratch tails, and the
/// requirement for stable [`AsMut::as_mut`] views are the same as
/// [`batch_invert_groups`]. No allocation is performed; arithmetic is
/// variable-time. Each nonzero batch scales its shared inverse seed once.
pub fn batch_invert_groups_scaled<M: PrimeModulus>(
    groups: &mut [impl AsMut<[PastaField<M>]>],
    scale: &PastaField<M>,
    scratch: &mut [PastaField<M>],
) {
    batch_invert_groups_inner(groups, Some(scale), scratch)
}

fn batch_invert_groups_inner<M: PrimeModulus>(
    groups: &mut [impl AsMut<[PastaField<M>]>],
    scale: Option<&PastaField<M>>,
    scratch: &mut [PastaField<M>],
) {
    if scratch.is_empty() {
        for group in groups {
            for value in group.as_mut() {
                *value = inverse_seed(value, scale).unwrap_or(PastaField::ZERO);
            }
        }
        return;
    }

    // Cursors identify (group, element) positions without summing group lengths.
    let mut end = (0, 0);
    while end.0 < groups.len() {
        let start = end;
        let mut used = 0;
        let mut products = NonzeroInversionLanes::new();
        while end.0 < groups.len() && used < scratch.len() {
            let group = groups[end.0].as_mut();
            let count = (group.len() - end.1).min(scratch.len() - used);
            // Parity follows positions within the batch, including zeros.
            for (index, value) in group[end.1..end.1 + count].iter().enumerate() {
                let index = used + index;
                if !value.is_zero()
                    && let Some(product) = products.push(index, value)
                {
                    scratch[index] = product;
                }
            }
            used += count;
            end.1 += count;
            if end.1 == group.len() {
                end = (end.0 + 1, 0);
            }
        }
        let Some(mut inverses) = products.invert_scaled(scale) else {
            continue;
        };

        // Replay only this batch in reverse; `end` remains the next batch's start.
        let mut cursor = end;
        while cursor != start {
            if cursor.1 == 0 {
                cursor.0 -= 1;
                cursor.1 = groups[cursor.0].as_mut().len();
            }
            let begin = if cursor.0 == start.0 { start.1 } else { 0 };
            for value in groups[cursor.0].as_mut()[begin..cursor.1].iter_mut().rev() {
                used -= 1;
                if !value.is_zero() {
                    *value = inverses.pop(used, value, &scratch[used]);
                }
            }
            cursor.1 = begin;
        }
    }
}

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
