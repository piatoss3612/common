//! Batch inversion of the nonzero Jacobian scales.

use super::{CurveError, PastaCurve, Point, ProjectivePoint, check_length, check_scratch};
use crate::field::PastaField;

/// Normalizes points in order, preserving identity positions.
///
/// `output` must have exactly `points.len()` elements; `scratch` needs at least
/// that many field elements. Initial scratch contents do not matter, and its
/// unused tail is untouched. Returns [`CurveError::LengthMismatch`] for an
/// incorrect output length or [`CurveError::ScratchTooSmall`] for short scratch.
/// Every returned error leaves both buffers unchanged.
///
/// A batch containing nonidentity points uses one inversion; an empty or
/// all-identity batch uses none.
///
/// ```
/// use zakura_udon::{
///     curve::{PallasPoint, PallasProjective, batch_normalize},
///     field::Fp,
/// };
///
/// let g = PallasProjective::GENERATOR;
/// let input = [g.double(), PallasProjective::IDENTITY, g];
/// let mut output = [PallasPoint::IDENTITY; 3];
/// let mut scratch = [Fp::ZERO; 3];
/// batch_normalize(&input, &mut output, &mut scratch).unwrap();
/// assert_eq!(output, input.map(|point| point.to_point()));
/// ```
pub fn batch_normalize<C: PastaCurve>(
    points: &[ProjectivePoint<C>],
    output: &mut [Point<C>],
    scratch: &mut [PastaField<C::Base>],
) -> Result<(), CurveError> {
    check_length("output", points.len(), output.len())?;
    check_scratch("field", points.len(), scratch.len())?;
    normalize(points, &mut scratch[..points.len()], |index, point| {
        output[index] = point
    });
    Ok(())
}

// Callers check lengths before any mutation. A sink lets fixed-base preparation
// write nonidentity points directly into its affine table without a Point buffer.
pub(super) fn normalize<C: PastaCurve>(
    points: &[ProjectivePoint<C>],
    scratch: &mut [PastaField<C::Base>],
    mut write: impl FnMut(usize, Point<C>),
) {
    // Separate even and odd prefix products shorten the multiplication
    // dependency chain. scratch[i] is the product of earlier nonzero z values
    // in i's lane. Skipping identities keeps both products invertible.
    let mut products = [PastaField::ONE; 2];
    let mut any_nonidentity = false;
    for (index, point) in points.iter().enumerate() {
        let lane = index & 1;
        scratch[index] = products[lane];
        if !point.is_identity() {
            any_nonidentity = true;
            products[lane] = products[lane].mul(&point.z);
        }
    }
    if !any_nonidentity {
        for index in 0..points.len() {
            write(index, Point::IDENTITY);
        }
        return;
    }
    // For lane products a and b, (a*b)^-1 gives a^-1 after multiplying by b,
    // and b^-1 after multiplying by a, sharing one field inversion.
    let combined_inverse = products[0]
        .mul(&products[1])
        .invert()
        .expect("a product of nonzero field elements is nonzero");
    let mut inverses = [
        combined_inverse.mul(&products[1]),
        combined_inverse.mul(&products[0]),
    ];
    for (index, point) in points.iter().enumerate().rev() {
        if point.is_identity() {
            write(index, Point::IDENTITY);
            continue;
        }
        let lane = index & 1;
        // The lane inverse still includes this z; its prefix cancels every
        // earlier factor. Multiplying by z then removes it for the next step.
        let inverse = inverses[lane].mul(&scratch[index]);
        inverses[lane] = inverses[lane].mul(&point.z);
        write(index, point.normalize_with_inverse(&inverse).to_point());
    }
}
