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

/// Invalid input or scratch lengths for batch inversion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum BatchInversionError {
    /// The scratch slice is shorter than the combined input.
    ScratchTooSmall {
        /// Required number of field elements.
        required: usize,
        /// Supplied number of field elements.
        provided: usize,
    },
    /// The combined input length cannot be represented by `usize`.
    SizeOverflow,
}

impl fmt::Display for BatchInversionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScratchTooSmall { required, provided } => {
                write!(
                    f,
                    "batch inversion needs {required} scratch fields, got {provided}"
                )
            }
            Self::SizeOverflow => f.write_str("batch inversion input length overflow"),
        }
    }
}

impl core::error::Error for BatchInversionError {}

/// Replaces each nonzero value by its inverse, preserving zeros.
///
/// Uses one inversion if any input is nonzero, and none otherwise. The caller
/// supplies at least `values.len()` scratch fields; their initial contents do
/// not matter and the unused tail is untouched. No allocation is performed.
/// Arithmetic is variable-time, including the locations of zeros.
///
/// # Errors
///
/// Returns [`BatchInversionError::ScratchTooSmall`] before changing either
/// slice if scratch is too short.
///
/// ```
/// use zakura_udon::field::{Fp, batch_invert};
/// let mut values = [Fp::from_u64(7), Fp::ZERO, Fp::from_u64(3)];
/// batch_invert(&mut values, &mut [Fp::ZERO; 3]).unwrap();
/// assert_eq!(values[0].mul(&Fp::from_u64(7)), Fp::ONE);
/// assert_eq!(values[1], Fp::ZERO);
/// ```
pub fn batch_invert<M: PrimeModulus>(
    values: &mut [PastaField<M>],
    scratch: &mut [PastaField<M>],
) -> Result<(), BatchInversionError> {
    batch_invert_groups(&mut [values], scratch)
}

/// Inverts nonzero values across disjoint slices with a single inversion.
///
/// This is [`batch_invert`] over the concatenation of `groups`, without
/// flattening or copying their contents. Empty groups are allowed. Scratch
/// needs one field per input element, including zeros. Its initial contents
/// do not matter; any unused tail is untouched. Empty or all-zero inputs
/// perform no inversion. No allocation is performed. Arithmetic is
/// variable-time, including the locations of zeros.
///
/// Each group's [`AsMut::as_mut`] must expose the same slice throughout the
/// call. Arrays, mutable slices, and vectors satisfy this requirement. A custom
/// implementation that changes its view can produce incorrect results or panic.
///
/// # Errors
///
/// Returns [`BatchInversionError::SizeOverflow`] if the combined length
/// overflows, or [`BatchInversionError::ScratchTooSmall`] if scratch is too
/// short. All lengths are checked before changing inputs or scratch.
pub fn batch_invert_groups<M: PrimeModulus>(
    groups: &mut [impl AsMut<[PastaField<M>]>],
    scratch: &mut [PastaField<M>],
) -> Result<(), BatchInversionError> {
    let required = combined_len(groups.iter_mut().map(|group| group.as_mut().len()))?;
    if scratch.len() < required {
        return Err(BatchInversionError::ScratchTooSmall {
            required,
            provided: scratch.len(),
        });
    }
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
        return Ok(());
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
    Ok(())
}

fn combined_len(lengths: impl IntoIterator<Item = usize>) -> Result<usize, BatchInversionError> {
    lengths.into_iter().try_fold(0usize, |total, len| {
        total
            .checked_add(len)
            .ok_or(BatchInversionError::SizeOverflow)
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn combined_length_overflow() {
        assert_eq!(
            combined_len([usize::MAX, 1]),
            Err(BatchInversionError::SizeOverflow)
        );
        assert_eq!(combined_len([0, usize::MAX, 0]), Ok(usize::MAX));
    }
}
