//! Product sums through the field traits.

use super::Field;

/// Returns the inner product of two equal-length slices.
///
/// Dispatches through [`Field::sum_of_products_slice`], preserving contiguous
/// storage for the field's slice kernels. Use [`dot_iter`] for noncontiguous
/// inputs, such as reversed or strided sequences.
///
/// # Panics
///
/// Panics if the lengths differ.
pub fn dot<F: Field>(lhs: &[F], rhs: &[F]) -> F {
    F::sum_of_products_slice(lhs, rhs)
}

/// Returns the inner product of two equal-length iterator sequences.
///
/// Dispatches through [`Field::sum_of_product_pairs`]; Pasta fields defer
/// Montgomery reduction across the sum.
///
/// # Panics
///
/// Panics if the lengths differ.
pub fn dot_iter<'a, F: Field, A, B>(lhs: A, rhs: B) -> F
where
    A: IntoIterator<Item = &'a F>,
    B: IntoIterator<Item = &'a F>,
    A::IntoIter: ExactSizeIterator,
    B::IntoIter: ExactSizeIterator,
{
    let lhs = lhs.into_iter();
    let rhs = rhs.into_iter();
    assert_eq!(
        lhs.len(),
        rhs.len(),
        "dot product operands must have equal length"
    );
    F::sum_of_product_pairs(lhs.zip(rhs))
}
