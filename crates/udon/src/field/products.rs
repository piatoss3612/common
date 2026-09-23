//! Product sums through the field traits.

use super::Field;

/// Returns the inner product of two equal-length sequences.
///
/// Dispatches through [`Field::sum_of_product_pairs`]; Pasta fields defer
/// Montgomery reduction across the sum.
///
/// # Panics
///
/// Panics if the lengths differ.
pub fn dot<'a, F: Field, A, B>(lhs: A, rhs: B) -> F
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
