use core::fmt;

use super::{PastaField, PrimeModulus};

/// Two product chains sharing one inversion.
///
/// Push nonzero factors in index order, invert the lane products, then pop in
/// reverse order with the saved prefixes. Separate even and odd lanes shorten
/// multiplication dependencies. Callers can seed the lanes with their first
/// factors to avoid multiplication by one and unused final updates.
pub(crate) struct InversionLanes<M: PrimeModulus>(pub(crate) [PastaField<M>; 2]);

impl<M: PrimeModulus> InversionLanes<M> {
    pub(crate) fn push(&mut self, index: usize, value: &PastaField<M>) -> PastaField<M> {
        let product = &mut self.0[index & 1];
        let prefix = *product;
        *product = product.mul(value);
        prefix
    }

    pub(crate) fn invert(self) -> Self {
        // For lane products a and b, multiplying (a*b)^-1 by the opposite
        // product recovers each lane's inverse with one field inversion.
        let inverse = self.0[0]
            .mul(&self.0[1])
            .invert()
            .expect("a product of nonzero field elements is nonzero");
        Self([inverse.mul(&self.0[1]), inverse.mul(&self.0[0])])
    }

    pub(crate) fn pop(
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

    pub(crate) fn invert(mut self) -> Option<Self> {
        let lane = match self.first {
            [None, None] => return None,
            [Some(_), Some(_)] => {
                self.products = self.products.invert();
                return Some(self);
            }
            [Some(_), None] => 0,
            [None, Some(_)] => 1,
        };
        self.products.0[lane] = self.products.0[lane].invert().expect("nonzero product");
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
    let mut inverses = products.invert();
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
/// assert_eq!(values[0].mul(&Fp::from_u64(7)), Fp::ONE);
/// assert_eq!(values[1], Fp::ZERO);
/// ```
pub fn batch_invert<M: PrimeModulus>(values: &mut [PastaField<M>], scratch: &mut [PastaField<M>]) {
    batch_invert_groups(&mut [values], scratch)
}

/// Inverts nonzero values across disjoint slices with shared inversions.
///
/// This is [`batch_invert`] over the concatenation of `groups`, without
/// flattening or copying their contents. Empty groups are allowed. With one
/// scratch field per input element, all groups share one inversion. Smaller
/// scratch bounds each batch within a group; empty scratch uses individual
/// inversions. Initial scratch contents do not matter and any unused tail is
/// untouched. Empty or all-zero inputs perform no inversion. No allocation is
/// performed. Arithmetic is variable-time, including the locations of zeros.
///
/// Each group's [`AsMut::as_mut`] must expose the same slice throughout the
/// call. Arrays, mutable slices, and vectors satisfy this requirement. A custom
/// implementation that changes its view can produce incorrect results or panic.
pub fn batch_invert_groups<M: PrimeModulus>(
    groups: &mut [impl AsMut<[PastaField<M>]>],
    scratch: &mut [PastaField<M>],
) {
    let required = combined_len(groups.iter_mut().map(|group| group.as_mut().len()));
    let Some(required) = required.filter(|&required| required <= scratch.len()) else {
        for group in groups {
            if scratch.is_empty() {
                for value in group.as_mut() {
                    *value = value.invert().unwrap_or(PastaField::ZERO);
                }
            } else {
                for chunk in group.as_mut().chunks_mut(scratch.len()) {
                    batch_invert(chunk, scratch);
                }
            }
        }
        return;
    };
    let scratch = &mut scratch[..required];
    // Parity follows the concatenated input, including zeros and empty groups.
    let mut products = NonzeroInversionLanes::new();
    for (index, (value, prefix)) in groups
        .iter_mut()
        .flat_map(|group| group.as_mut().iter_mut())
        .zip(scratch.iter_mut())
        .enumerate()
    {
        if !value.is_zero()
            && let Some(product) = products.push(index, value)
        {
            *prefix = product;
        }
    }
    let Some(mut inverses) = products.invert() else {
        return;
    };
    for (value, (index, prefix)) in groups
        .iter_mut()
        .flat_map(|group| group.as_mut().iter_mut())
        .rev()
        .zip(scratch.iter().enumerate().rev())
    {
        if !value.is_zero() {
            *value = inverses.pop(index, value, prefix);
        }
    }
}

/// Inverts denominators read from immutable records and visits their inverses.
///
/// `denominator` must return the same reduced field value for a record on every
/// call. It may be evaluated more than once; stability and reducedness are not
/// checked. Violations remain memory-safe but can cause wrong results or panics.
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
/// let records = [(Fp::from_u64(6), Fp::from_u64(2)),
///                (Fp::from_u64(20), Fp::from_u64(5))];
/// let mut quotients = [Fp::ZERO; 2];
/// try_batch_invert_by(&records, |record| record.1, &mut [Fp::ZERO; 2],
///     |index, record, inverse| {
///         quotients[index] = record.0.mul(&inverse);
///         Ok::<_, BatchInversionError>(())
///     }).unwrap();
/// assert_eq!(quotients, [Fp::from_u64(3), Fp::from_u64(4)]);
/// ```
pub fn try_batch_invert_by<R, M: PrimeModulus, E: From<BatchInversionError>>(
    records: &[R],
    denominator: impl Fn(&R) -> PastaField<M>,
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
                denominator(record)
                    .invert()
                    .expect("stable nonzero denominator"),
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
        let mut inverses = products.invert().expect("nonempty nonzero batch");
        for (index, record) in records.iter().enumerate().rev() {
            let inverse = inverses.pop(index, &denominator(record), &scratch[index]);
            visit(chunk * scratch.len() + index, record, inverse)?;
        }
    }
    Ok(())
}

fn combined_len(lengths: impl IntoIterator<Item = usize>) -> Option<usize> {
    lengths.into_iter().try_fold(0usize, usize::checked_add)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combined_length_overflow() {
        assert_eq!(combined_len([usize::MAX, 1]), None);
        assert_eq!(combined_len([0, usize::MAX, 0]), Some(usize::MAX));
    }
}
