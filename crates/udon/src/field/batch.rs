//! Batch inversion through the field traits with caller-owned scratch.

use super::{Field, pasta::invert_groups};

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
    invert_groups(groups, scratch, F::ONE, F::is_zero, F::invert)
}
