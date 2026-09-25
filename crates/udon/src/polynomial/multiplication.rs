use crate::{
    exec::{ExecutionOptions, SerialExecutor},
    fft::{
        Direction, Domain, ElementOrder, Expansion, ExpansionOrder, ExpansionStorage, InputSupport,
        StorageLayout, Transform, TransformRequest,
        execution::{ExpansionPlan, FftPlan},
    },
    field::{FftField, PastaField, PrimeModulus},
};

#[cfg(test)]
#[path = "tests/multiplication.rs"]
mod tests;

/// Writes the product `a * b` into `product`.
///
/// `product` must have `a.len() + b.len() - 1` elements, or none if either
/// input is empty. Small products use schoolbook convolution. Larger products
/// use an evaluation [`Domain`] when `scratch` provides at least twice the
/// domain size in elements and the domain size is supported; otherwise the
/// schoolbook path runs. The choice affects only cost: both paths are exact.
/// Scratch contents are unspecified afterwards.
///
/// Dispatches through [`FftField::multiply_polynomials`]. Pasta expands each
/// coefficient prefix into bit-reversed evaluations, multiplies during the
/// second expansion, and interpolates directly from that order. This avoids
/// full zero-padded forward transforms and separate evaluation permutations.
///
/// # Panics
///
/// Panics before mutation if `product` has the wrong length.
pub fn multiply<F: FftField>(a: &[F], b: &[F], product: &mut [F], scratch: &mut [F]) {
    F::multiply_polynomials(a, b, product, scratch)
}

// Shape validation and the cost/scratch decision are independent of field
// arithmetic, so native and consumer implementations share them.
fn transform_log_size(a: usize, b: usize, product: usize, scratch: usize) -> Option<u32> {
    let product_len = if a == 0 || b == 0 {
        0
    } else {
        a.checked_add(b - 1).expect("product length overflow")
    };
    assert_eq!(product, product_len, "product length");
    if product_len == 0 {
        return None;
    }

    let size = product_len.checked_next_power_of_two()?;
    let schoolbook_cost = a.saturating_mul(b);
    // Estimate all three transforms, the pointwise products, and normalization
    // conservatively. The field chooses the actual FFT schedule; omitting the
    // stage count here can favor FFTs for large, thin products.
    let transform_cost = size.saturating_mul(3 * size.ilog2() as usize + 2);
    let transform_scratch = size.checked_mul(2);
    if schoolbook_cost <= transform_cost
        || transform_scratch.is_none_or(|required| scratch < required)
    {
        None
    } else {
        Some(size.ilog2())
    }
}

pub(crate) fn multiply_default<F: FftField>(
    a: &[F],
    b: &[F],
    product: &mut [F],
    scratch: &mut [F],
) {
    let domain = transform_log_size(a.len(), b.len(), product.len(), scratch.len())
        .and_then(|log_size| Domain::<F>::new(log_size).ok());
    if let Some(domain) = domain {
        multiply_by_transform(domain, a, b, product, scratch);
    } else {
        multiply_schoolbook(a, b, product, F::ZERO, F::mul_add);
    }
}

pub(crate) fn multiply_pasta<M: PrimeModulus>(
    a: &[PastaField<M>],
    b: &[PastaField<M>],
    product: &mut [PastaField<M>],
    scratch: &mut [PastaField<M>],
) {
    let domain = transform_log_size(a.len(), b.len(), product.len(), scratch.len())
        .and_then(|log_size| Domain::new(log_size).ok());
    if let Some(domain) = domain {
        multiply_by_expansion(domain, a, b, product, scratch);
    } else {
        multiply_schoolbook(a, b, product, PastaField::ZERO, PastaField::mul_add);
    }
}

// The product has the validated length, including zero for empty inputs.
fn multiply_schoolbook<F: Copy>(
    a: &[F],
    b: &[F],
    product: &mut [F],
    zero: F,
    mul_add: impl Fn(&F, &F, &F) -> F,
) {
    product.fill(zero);
    for (i, lhs) in a.iter().enumerate() {
        for (j, rhs) in b.iter().enumerate() {
            product[i + j] = mul_add(lhs, rhs, &product[i + j]);
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

// Each input fits a base domain no larger than the product domain. Residues
// retain the same global bit-reversed order even when the base sizes differ.
fn multiply_by_expansion<M: PrimeModulus>(
    domain: Domain<PastaField<M>>,
    a: &[PastaField<M>],
    b: &[PastaField<M>],
    product: &mut [PastaField<M>],
    scratch: &mut [PastaField<M>],
) {
    let options = ExecutionOptions::default().with_memory_limit(0);
    let expansion = |len: usize| {
        let base = Domain::for_size(len.next_power_of_two())
            .expect("input fits the supported product domain");
        ExpansionPlan::new(
            Expansion::new(Transform::new(base.subgroup()), domain.subgroup(), None)
                .expect("input domain fits the product domain"),
            ExpansionStorage::Coefficients,
            ExpansionOrder::BitReversed,
            InputSupport::Prefix(len),
            ElementOrder::Natural,
            StorageLayout::Contiguous,
            options,
        )
        .expect("serial coefficient expansion requires no extra scratch")
    };
    let left = expansion(a.len());
    let right = expansion(b.len());
    let inverse = FftPlan::new(
        Transform::new(domain.subgroup()),
        TransformRequest {
            input_order: ElementOrder::BitReversed,
            ..TransformRequest::new(Direction::Inverse)
        },
        StorageLayout::Contiguous,
        options,
    )
    .expect("serial interpolation requires no extra scratch");

    let (factor, values) = scratch[..2 * domain.size()].split_at_mut(domain.size());
    left.execute(a, factor, &mut [], None, &mut [], &SerialExecutor);
    right.execute(b, values, &mut [], Some(factor), &mut [], &SerialExecutor);
    inverse.execute(None, values, None, &mut [], &SerialExecutor);
    product.copy_from_slice(&values[..product.len()]);
}
