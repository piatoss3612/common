//! Batch inversion through the field traits with caller-owned scratch.

use super::Field;

pub(crate) fn inverse_seed<F: Field>(value: &F, scale: Option<&F>) -> Option<F> {
    value
        .invert()
        .map(|inverse| scale.map_or(inverse, |scale| inverse * scale))
}

/// Two product chains sharing one inversion.
///
/// Push nonzero factors in index order, invert the lane products, then pop in
/// reverse order with the saved prefixes. Separate even and odd lanes shorten
/// multiplication dependencies. Callers can seed the lanes with their first
/// factors to avoid multiplication by one and unused final updates.
pub(in crate::field) struct InversionLanes<F: Field>(pub(in crate::field) [F; 2]);

impl<F: Field> InversionLanes<F> {
    pub(in crate::field) fn push(&mut self, index: usize, value: &F) -> F {
        let product = &mut self.0[index & 1];
        let prefix = *product;
        *product *= value;
        prefix
    }

    pub(crate) fn invert(self, scale: Option<&F>) -> Self {
        // For lane products a and b, multiplying (a*b)^-1 by the opposite
        // product recovers each lane's inverse with one field inversion.
        // Applying the scale to this seed carries it into both recovery lanes.
        let inverse = inverse_seed(&(self.0[0] * self.0[1]), scale)
            .expect("a product of nonzero field elements is nonzero");
        Self([inverse * self.0[1], inverse * self.0[0]])
    }

    pub(in crate::field) fn pop(&mut self, index: usize, value: &F, prefix: &F) -> F {
        let inverse = &mut self.0[index & 1];
        // The prefix cancels earlier factors; multiplying by this value then
        // removes it from the lane inverse for the next step.
        let result = *inverse * prefix;
        *inverse *= value;
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
pub(crate) struct NonzeroInversionLanes<F: Field> {
    products: InversionLanes<F>,
    first: [Option<usize>; 2],
}

impl<F: Field> NonzeroInversionLanes<F> {
    pub(crate) fn new() -> Self {
        Self {
            products: InversionLanes([F::ONE; 2]),
            first: [None; 2],
        }
    }

    pub(crate) fn push(&mut self, index: usize, value: &F) -> Option<F> {
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

    pub(crate) fn invert_scaled(mut self, scale: Option<&F>) -> Option<Self> {
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

    pub(crate) fn pop(&mut self, index: usize, value: &F, prefix: &F) -> F {
        if self.first[index & 1] == Some(index) {
            self.products.0[index & 1]
        } else {
            self.products.pop(index, value, prefix)
        }
    }
}

/// Replaces each nonzero value by its inverse, preserving zeros.
///
/// Uses one inversion per nonzero batch. With at least `values.len()` scratch
/// fields the entire input is one batch; smaller buffers bound the batch size,
/// and empty scratch inverts values individually. Initial scratch contents do
/// not matter and the unused tail is untouched. No allocation is performed.
/// Empty or all-zero inputs perform no inversion.
/// Arithmetic is variable-time, including the locations of zeros.
///
/// Dispatches through [`Field::batch_invert`].
///
/// ```
/// use zakura_udon::field::{Fp, batch_invert};
/// let mut values = [Fp::from_u64(7), Fp::ZERO, Fp::from_u64(3)];
/// batch_invert(&mut values, &mut [Fp::ZERO; 2]);
/// assert_eq!(values[0].mul(&<Fp>::from_u64(7)).reduce(), Fp::ONE);
/// assert!(values[1].is_zero());
/// ```
pub fn batch_invert<F: Field>(values: &mut [F], scratch: &mut [F]) {
    F::batch_invert(values, scratch)
}

/// Equivalent to [`batch_invert`], including support for small or empty scratch.
///
/// See [`batch_invert`] for the scratch and zero-preservation contract.
pub fn batch_invert_with_scratch<F: Field>(values: &mut [F], scratch: &mut [F]) {
    batch_invert(values, scratch)
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
pub fn batch_invert_groups<F: Field>(groups: &mut [impl AsMut<[F]>], scratch: &mut [F]) {
    batch_invert_groups_inner(groups, None, scratch)
}

pub(crate) fn batch_invert_groups_inner<F: Field>(
    groups: &mut [impl AsMut<[F]>],
    scale: Option<&F>,
    scratch: &mut [F],
) {
    if scratch.is_empty() {
        for group in groups {
            for value in group.as_mut() {
                if !value.is_zero() {
                    *value = inverse_seed(value, scale).expect("nonzero value");
                }
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
