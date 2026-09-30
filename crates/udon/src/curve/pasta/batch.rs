//! Batch normalization and affine steps with shared field inversions.

use super::{PastaCurve, Point, ProjectivePoint, assert_length};
use crate::field::{NonzeroInversionLanes, PastaField, invert_nonzero};

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
pub(crate) fn normalize<C: PastaCurve>(
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
    let mut products = NonzeroInversionLanes::new(PastaField::ONE, PastaField::mul);
    for (index, (point, prefix)) in points.iter().zip(scratch.iter_mut()).enumerate() {
        if !point.is_identity()
            && !point.z.is_one()
            && let Some(product) = products.push(index, &point.z)
        {
            *prefix = product;
        }
    }
    let Some(mut inverses) = products.invert(PastaField::invert) else {
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

// Complete affine steps across independent lanes. Accumulators have z=0 or 1;
// exceptional chords are excluded from inversion and handled explicitly.
pub(crate) fn affine_step<C: PastaCurve>(
    points: &mut [ProjectivePoint<C>],
    scratch: &mut [PastaField<C::Base>],
    add: Option<impl Fn(usize) -> super::AffinePoint<C>>,
) {
    let (denom, prefix) = scratch.split_at_mut(points.len());
    // Pasta has no nonidentity two-torsion, so doubling denominators are nonzero.
    let mut active = points.len();
    for (i, (p, d)) in points.iter().zip(denom.iter_mut()).enumerate() {
        *d = if p.is_identity() {
            active -= 1;
            PastaField::ZERO
        } else if let Some(ref add) = add {
            let q = add(i);
            let h = q.x.sub(&p.x);
            if h.is_zero() {
                if q.y.sub(&p.y).is_zero() {
                    p.y.double()
                } else {
                    active -= 1;
                    PastaField::ZERO
                }
            } else {
                h
            }
        } else {
            p.y.double()
        };
    }
    // Most ladder steps have no exceptional lanes. Keep their inversion loop
    // free of sparse-lane checks; zero-only steps need no inversion at all.
    let all_active = active == points.len();
    if all_active {
        invert_nonzero(denom, prefix);
    } else if active != 0 {
        let mut products = NonzeroInversionLanes::new(PastaField::ONE, PastaField::mul);
        for (i, d) in denom.iter().enumerate() {
            if !d.is_zero()
                && let Some(product) = products.push(i, d)
            {
                prefix[i] = product;
            }
        }
        let mut inverses = products
            .invert(PastaField::invert)
            .expect("active affine lanes");
        for (i, (d, prefix)) in denom.iter_mut().zip(prefix.iter()).enumerate().rev() {
            if !d.is_zero() {
                *d = inverses.pop(i, d, prefix);
            }
        }
    }
    for (i, (p, inverse)) in points.iter_mut().zip(denom.iter()).enumerate() {
        let q = add.as_ref().map(|add| add(i));
        if p.is_identity() {
            if let Some(q) = q {
                *p = q.to_projective();
            }
            continue;
        }
        if !all_active && inverse.is_zero() {
            *p = ProjectivePoint::IDENTITY;
            continue;
        }
        let (numerator, x2) = match q {
            Some(q) if !q.x.sub(&p.x).is_zero() => (q.y.sub(&p.y), q.x.into_loose()),
            _ => (super::AffinePoint::<C>::tangent_numerator_at(&p.x), p.x),
        };
        let slope = numerator.mul(inverse);
        (p.x, p.y) = super::AffinePoint::<C>::slope_coordinates(&p.x, &p.y, &x2, &slope);
    }
}
