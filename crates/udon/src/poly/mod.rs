//! Polynomial utilities over the field traits.
//!
//! Polynomials are coefficient slices in ascending degree. Every function is
//! generic over [`Field`], or over [`FftField`] where a product may route
//! through an evaluation [`Domain`], and none allocates: results are written
//! into caller-owned slices. The algorithms are exact reference shapes rather
//! than tuned kernels, and they are variable-time like the arithmetic they
//! use.
//!
//! ```
//! use zakura_udon::{field::Fp, poly};
//!
//! // p(X) = 1 + 2X + 3X²
//! let p = [Fp::from_u64(1), Fp::from_u64(2), Fp::from_u64(3)];
//! assert_eq!(poly::evaluate(&p, Fp::from_u64(2)), Fp::from_u64(17));
//!
//! let mut product = [Fp::ZERO; 5];
//! poly::multiply(&p, &p, &mut product, &mut []);
//! assert_eq!(poly::evaluate(&product, Fp::from_u64(2)), Fp::from_u64(289));
//! ```

use crate::{
    fft::Domain,
    field::{FftField, Field},
};

pub use crate::field::dot;

#[cfg(test)]
mod tests;

/// Evaluates the polynomial with the given ascending coefficients at `x` by
/// Horner's rule. No coefficients evaluate to zero.
pub fn evaluate<'a, F: Field, I>(coefficients: I, x: F) -> F
where
    I: IntoIterator<Item = &'a F>,
    I::IntoIter: DoubleEndedIterator,
{
    let mut result = F::ZERO;
    for coefficient in coefficients.into_iter().rev() {
        result *= x;
        result += coefficient;
    }
    result
}

/// Returns the coefficients of `p / (X - root)`, highest degree first.
///
/// The division is exact when `root` is a root of `p`; otherwise the
/// remainder `p(root)` is dropped. Synthetic division derives each quotient
/// coefficient from the current dividend coefficient and `root` times the
/// previous quotient coefficient. A constant `p` yields an empty quotient.
///
/// # Panics
///
/// Panics if `p` has no coefficients.
pub fn divide_by_root_iter<'a, F: Field, I>(p: I, root: F) -> impl Iterator<Item = F> + 'a
where
    I: IntoIterator<Item = F> + 'a,
    I::IntoIter: DoubleEndedIterator,
{
    let mut coefficients = p.into_iter().rev().peekable();
    assert!(
        coefficients.peek().is_some(),
        "cannot divide a polynomial without coefficients by a linear factor"
    );

    let mut carry = F::ZERO;
    core::iter::from_fn(move || {
        let current = coefficients.next()?;
        // The constant term only contributes to the remainder.
        coefficients.peek()?;

        let quotient_coefficient = current + carry;
        carry = quotient_coefficient * root;
        Some(quotient_coefficient)
    })
}

/// Writes the ascending coefficients of `p / (X - root)` into `quotient`.
///
/// See [`divide_by_root_iter`] for the division contract.
///
/// # Panics
///
/// Panics before mutation if `p` has no coefficients or `quotient` does not
/// have exactly `p.len() - 1` elements.
pub fn divide_by_root<F: Field>(p: &[F], root: F, quotient: &mut [F]) {
    assert!(
        !p.is_empty(),
        "cannot divide a polynomial without coefficients by a linear factor"
    );
    assert_eq!(quotient.len(), p.len() - 1, "quotient length");
    for (slot, coefficient) in quotient
        .iter_mut()
        .rev()
        .zip(divide_by_root_iter(p.iter().copied(), root))
    {
        *slot = coefficient;
    }
}

/// Returns `1 + ratio + ratio^2 + ... + ratio^(terms - 1)`.
///
/// Doubling the covered block uses `O(log terms)` multiplications. Zero terms
/// give zero.
pub fn geometric_sum<F: Field>(mut ratio: F, mut terms: usize) -> F {
    let mut block = F::ONE;
    let mut sum = F::ZERO;
    let mut step = F::ONE;
    while terms > 0 {
        if terms & 1 == 1 {
            sum += step * block;
            step *= ratio;
        }
        block += ratio * block;
        ratio = ratio.square();
        terms >>= 1;
    }
    sum
}

/// Writes the product `a * b` into `product`.
///
/// `product` must have `a.len() + b.len() - 1` elements, or none if either
/// input is empty. Small products use schoolbook convolution. Larger products
/// use an evaluation [`Domain`] when `scratch` provides at least twice the
/// domain size in elements and the domain size is supported; otherwise the
/// schoolbook path runs. The choice affects only cost: both paths are exact.
/// Scratch contents are unspecified afterwards.
///
/// # Panics
///
/// Panics before mutation if `product` has the wrong length.
pub fn multiply<F: FftField>(a: &[F], b: &[F], product: &mut [F], scratch: &mut [F]) {
    if a.is_empty() || b.is_empty() {
        assert!(product.is_empty(), "product length");
        return;
    }
    let product_len = a.len() + b.len() - 1;
    assert_eq!(product.len(), product_len, "product length");

    let size = product_len.next_power_of_two();
    let schoolbook_cost = a.len().saturating_mul(b.len());
    // Each reference FFT does a scaling and a twiddle update per butterfly.
    // Count all three transforms, the pointwise products, and normalization;
    // omitting the stage count can favor FFTs for large, thin products.
    let transform_cost = size.saturating_mul(3 * size.ilog2() as usize + 2);
    let transform_scratch = size.checked_mul(2);
    if schoolbook_cost <= transform_cost
        || transform_scratch.is_none_or(|required| scratch.len() < required)
    {
        multiply_schoolbook(a, b, product);
        return;
    }
    match Domain::<F>::for_size(size) {
        Ok(domain) => multiply_by_transform(domain, a, b, product, scratch),
        Err(_) => multiply_schoolbook(a, b, product),
    }
}

// Both inputs are nonempty and the product has the right length.
fn multiply_schoolbook<F: Field>(a: &[F], b: &[F], product: &mut [F]) {
    product.fill(F::ZERO);
    for (i, lhs) in a.iter().enumerate() {
        for (j, rhs) in b.iter().enumerate() {
            product[i + j] += *lhs * rhs;
        }
    }
}

// Both inputs fit the domain and scratch holds two domain-sized halves.
fn multiply_by_transform<F: FftField>(
    domain: Domain<F>,
    a: &[F],
    b: &[F],
    product: &mut [F],
    scratch: &mut [F],
) {
    let size = domain.size();
    let (lower, upper) = scratch[..2 * size].split_at_mut(size);
    lower[..a.len()].copy_from_slice(a);
    lower[a.len()..].fill(F::ZERO);
    upper[..b.len()].copy_from_slice(b);
    upper[b.len()..].fill(F::ZERO);

    domain.transform(lower);
    domain.transform(upper);
    for (lower, upper) in lower.iter_mut().zip(upper.iter()) {
        *lower *= upper;
    }
    domain.inverse_transform(lower);
    product.copy_from_slice(&lower[..product.len()]);
}
