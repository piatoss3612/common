//! Batch normalization with a shared field inversion.

use super::{PastaCurve, Point, ProjectivePoint, assert_length};
use crate::field::{NonzeroInversionLanes, PastaField};

/// Normalizes points in order, preserving identity positions.
///
/// `output` must have exactly `points.len()` elements. Scratch bounds the batch size:
/// one field per point shares one inversion across the entire input; smaller buffers
/// process bounded batches and empty scratch normalizes individually. Initial scratch
/// contents do not matter, and its unused tail is untouched. A mismatched output length
/// panics before either buffer is changed.
///
/// Each batch containing points whose `z` is neither zero nor one uses one
/// inversion. Identity and already-affine points are excluded from the product;
/// batches containing only those points use no inversion.
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
/// batch_normalize(&input, &mut output, &mut scratch);
/// assert_eq!(output, input.map(|point| point.to_point()));
/// ```
pub fn batch_normalize<C: PastaCurve>(
    points: &[ProjectivePoint<C>],
    output: &mut [Point<C>],
    scratch: &mut [PastaField<C::Base>],
) {
    assert_length("output", points.len(), output.len());
    normalize(points, scratch, |index, point| output[index] = point);
}

// Callers check lengths before any mutation. A sink lets table preparation
// write directly into the selected entry representation without a Point buffer.
pub(super) fn normalize<C: PastaCurve>(
    points: &[ProjectivePoint<C>],
    scratch: &mut [PastaField<C::Base>],
    mut write: impl FnMut(usize, Point<C>),
) {
    if scratch.len() < points.len() {
        if scratch.is_empty() {
            for (index, point) in points.iter().enumerate() {
                write(index, point.to_point());
            }
        } else {
            for (chunk, points) in points.chunks(scratch.len()).enumerate() {
                let offset = chunk * scratch.len();
                normalize_full(points, scratch, |index, point| write(offset + index, point));
            }
        }
        return;
    }
    normalize_full(points, scratch, write);
}

fn normalize_full<C: PastaCurve>(
    points: &[ProjectivePoint<C>],
    scratch: &mut [PastaField<C::Base>],
    mut write: impl FnMut(usize, Point<C>),
) {
    // Only nontrivial denominators enter the products. Skipped scratch slots
    // need no prefix because the reverse pass also skips them.
    let mut products = NonzeroInversionLanes::new();
    for (index, (point, prefix)) in points.iter().zip(scratch.iter_mut()).enumerate() {
        if !point.is_identity()
            && !point.z.is_one()
            && let Some(product) = products.push(index, &point.z)
        {
            *prefix = product;
        }
    }
    let Some(mut inverses) = products.invert() else {
        for (index, point) in points.iter().enumerate() {
            write(index, point.to_point());
        }
        return;
    };
    for (index, (point, prefix)) in points.iter().zip(scratch.iter()).enumerate().rev() {
        if point.is_identity() {
            write(index, Point::IDENTITY);
            continue;
        }
        if point.z.is_one() {
            write(index, point.to_point());
            continue;
        }
        let inverse = inverses.pop(index, &point.z, prefix);
        write(index, point.normalize_with_inverse(&inverse).to_point());
    }
}
