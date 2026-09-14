//! Batch normalization with a shared field inversion.

use super::{CurveError, PastaCurve, Point, ProjectivePoint, check_length, check_scratch};
use crate::field::{NonzeroInversionLanes, PastaField};

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

// Callers check lengths before any mutation. A sink lets table preparation
// write directly into the selected entry representation without a Point buffer.
pub(super) fn normalize<C: PastaCurve>(
    points: &[ProjectivePoint<C>],
    scratch: &mut [PastaField<C::Base>],
    mut write: impl FnMut(usize, Point<C>),
) {
    // Skipping identities keeps both products invertible; their scratch slots
    // need no prefix because the reverse pass also skips them.
    let mut products = NonzeroInversionLanes::new();
    for (index, (point, prefix)) in points.iter().zip(scratch.iter_mut()).enumerate() {
        if !point.is_identity()
            && let Some(product) = products.push(index, &point.z)
        {
            *prefix = product;
        }
    }
    let Some(mut inverses) = products.invert() else {
        for index in 0..points.len() {
            write(index, Point::IDENTITY);
        }
        return;
    };
    for (index, (point, prefix)) in points.iter().zip(scratch.iter()).enumerate().rev() {
        if point.is_identity() {
            write(index, Point::IDENTITY);
            continue;
        }
        let inverse = inverses.pop(index, &point.z, prefix);
        write(index, point.normalize_with_inverse(&inverse).to_point());
    }
}
