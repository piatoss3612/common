use crate::field::{PastaField, PrimeModulus, ReductionState};

#[cfg(test)]
#[path = "tests/vanishing.rs"]
mod tests;

/// Insufficient output capacity for a vanishing polynomial.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum VanishingError {
    /// The number of roots plus one cannot be represented by `usize`.
    SizeOverflow,
    /// Output cannot hold all coefficients, including the leading one.
    OutputTooShort {
        /// Required number of coefficients.
        required: usize,
        /// Supplied number of coefficients.
        actual: usize,
    },
}

impl core::fmt::Display for VanishingError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SizeOverflow => f.write_str("vanishing polynomial extent overflows usize"),
            Self::OutputTooShort { required, actual } => write!(
                f,
                "vanishing polynomial requires {required} coefficients, received {actual}"
            ),
        }
    }
}

impl core::error::Error for VanishingError {}

fn vanishing_extent(root_count: usize) -> Result<usize, VanishingError> {
    root_count
        .checked_add(1)
        .ok_or(VanishingError::SizeOverflow)
}

/// Writes the monic polynomial `product(X - roots[i])` in ascending order.
///
/// Returns the coefficient extent `roots.len() + 1`, including the leading one.
/// Repeated roots retain their multiplicities; an empty root slice produces the
/// constant one. Roots may be loose or reduced, and output uses loose elements.
/// Only the returned prefix is written; any output tail is unchanged. Extent
/// overflow and insufficient output are rejected before any write.
///
/// Rust's borrows keep roots disjoint from writable output. Preparation is
/// serial, uses quadratic work in the root count and constant auxiliary space,
/// and requires no allocation, scratch, or inversion. The result can be reused
/// as the divisor of [`super::divide_monic_in_place`].
///
/// ```
/// use zakura_udon::{field::Fp, polynomial::vanishing_polynomial};
///
/// let roots = [2, 3].map(<Fp>::from_u64);
/// let mut coefficients = [<Fp>::ZERO; 3];
/// assert_eq!(vanishing_polynomial(&roots, &mut coefficients).unwrap(), 3);
/// assert_eq!(coefficients.map(Fp::reduce), [6, -5, 1].map(|n| <Fp>::from_i64(n).reduce()));
/// ```
pub fn vanishing_polynomial<M: PrimeModulus, S: ReductionState>(
    roots: &[PastaField<M, S>],
    output: &mut [PastaField<M>],
) -> Result<usize, VanishingError> {
    let extent = vanishing_extent(roots.len())?;
    if output.len() < extent {
        return Err(VanishingError::OutputTooShort {
            required: extent,
            actual: output.len(),
        });
    }
    output[0] = PastaField::ONE;
    for (degree, root) in roots.iter().enumerate() {
        let negative_root = root.neg();
        output[degree + 1] = PastaField::ONE;
        for j in (1..=degree).rev() {
            output[j] = output[j].mul_add(&negative_root, &output[j - 1]);
        }
        output[0] = output[0].mul(&negative_root);
    }
    Ok(extent)
}
