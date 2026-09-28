//! Affine bucket reduction with one shared inversion per pair-tree level.

use crate::field::invert_nonzero;

use super::{AffinePoint, PastaCurve, ProjectivePoint};
use crate::field::PastaField;

/// Sums one bounded batch, sharing inversions until the final small layer.
///
/// The caller supplies at least one field per input point. Cancelled pairs
/// disappear; no identity is ever stored as a nonidentity affine point.
pub(super) fn sum<C: PastaCurve>(
    points: &mut [AffinePoint<C>],
    fields: &mut [PastaField<C::Base>],
) -> ProjectivePoint<C> {
    let mut lens = [points.len()];
    // Small layers cannot amortize another inversion. Finish projectively
    // without imposing a normalization inversion on the returned sum.
    while lens[0] > 16 {
        let count = lens[0];
        reduce_fused::<C, true>(&mut points[..count], &[0], &mut lens, fields);
    }
    let mut sum = ProjectivePoint::IDENTITY;
    for point in &points[..lens[0]] {
        sum = sum.add_mixed(point);
    }
    sum
}

/// Reduces disjoint buckets to zero or one affine point, updating their lengths.
///
/// Bucket `i` occupies `points[starts[i]..starts[i] + lens[i]]`. The caller
/// reserves disjoint ranges, one start per length, and two field elements per
/// possible pair (`points.len() / 2`). Survivors end up at their bucket's start.
pub(crate) fn reduce<C: PastaCurve>(
    points: &mut [AffinePoint<C>],
    starts: &[usize],
    lens: &mut [usize],
    fields: &mut [PastaField<C::Base>],
) {
    while lens.iter().any(|&n| n > 1) {
        reduce_fused::<C, true>(points, starts, lens, fields);
    }
}

/// Reduces one pair-tree level while recovering inverses during point addition.
///
/// Storage follows [`reduce`]. Returns the number of noncancelling pairs added.
/// With `INCOMPLETE`, first try chord denominators for every pair. A zero product
/// retries [`reduce_level`] with complete formulas before changing points or
/// lengths, so callers need not exclude doubling or cancellation.
#[inline(always)]
fn reduce_fused<C: PastaCurve, const INCOMPLETE: bool>(
    points: &mut [AffinePoint<C>],
    starts: &[usize],
    lens: &mut [usize],
    fields: &mut [PastaField<C::Base>],
) -> usize {
    let (denom, suffix) = fields.split_at_mut(points.len() / 2);
    let mut staged = 0;
    for (&start, &len) in starts.iter().zip(lens.iter()) {
        for pair in points[start..start + len].chunks_exact(2) {
            let (p, q) = (pair[0], pair[1]);
            if !INCOMPLETE && p.x == q.x && p.y != q.y {
                continue;
            }
            denom[staged] = if !INCOMPLETE && p.x == q.x {
                p.y.double()
            } else {
                q.x.sub(&p.x)
            };
            staged += 1;
        }
    }
    let mut inverses = [PastaField::ONE; 2];
    if staged != 0 {
        // Even and odd denominators form independent multiplication chains.
        // suffix[i] excludes denom[i] and contains later factors in that lane.
        // Seeding each lane with its last denominator avoids multiplying by one;
        // its endpoint needs no suffix entry during inverse recovery.
        let mut products = [PastaField::ONE; 2];
        products[(staged - 1) & 1] = denom[staged - 1];
        if staged > 1 {
            products[(staged - 2) & 1] = denom[staged - 2];
        }
        for i in (0..staged.saturating_sub(2)).rev() {
            suffix[i] = products[i & 1];
            products[i & 1] = products[i & 1].mul(&denom[i]);
        }
        let product = if staged == 1 {
            products[0]
        } else {
            products[0].mul(&products[1])
        };
        let Some(inverse) = product.invert() else {
            // Neither destinations nor lengths have been touched. Recompute
            // the denominators, handling doubles and cancelling pairs.
            return reduce_level::<C, false>(points, starts, lens, fields);
        };
        inverses = if staged == 1 {
            [inverse, PastaField::ONE]
        } else {
            [inverse.mul(&products[1]), inverse.mul(&products[0])]
        };
    }
    // Forward inverse recovery lets each pair be read before writing its sum.
    // Within a bucket, pair j reads positions 2j and 2j+1 before writing at most
    // position j. Odd survivors and later pairs therefore remain intact until
    // needed, even when cancellations reduce the number of outputs.
    let mut read = 0;
    for (&start, len) in starts.iter().zip(lens.iter_mut()) {
        let old = *len;
        let mut written = 0;
        for i in (0..old.saturating_sub(1)).step_by(2) {
            let (p, q) = (points[start + i], points[start + i + 1]);
            if !INCOMPLETE && p.x == q.x && p.y != q.y {
                continue;
            }
            let inverse = if read < staged.saturating_sub(2) {
                let result = inverses[read & 1].mul(&suffix[read]);
                inverses[read & 1] = inverses[read & 1].mul(&denom[read]);
                result
            } else {
                inverses[read & 1]
            };
            read += 1;
            let numerator = if !INCOMPLETE && p.x == q.x {
                p.tangent_numerator()
            } else {
                q.y.sub(&p.y)
            };
            let slope = numerator.mul(&inverse);
            points[start + written] = AffinePoint::from_slope(&p.x, &p.y, &q.x, &slope);
            written += 1;
        }
        if old & 1 != 0 {
            points[start + written] = points[start + old - 1];
            written += 1;
        }
        *len = written;
    }
    debug_assert_eq!(read, staged);
    staged
}

#[cfg(test)]
pub(crate) fn reduce_with<C: PastaCurve, const FUSED: bool>(
    points: &mut [AffinePoint<C>],
    starts: &[usize],
    lens: &mut [usize],
    fields: &mut [PastaField<C::Base>],
    mut level: impl FnMut(usize, usize),
) {
    while lens.iter().any(|&n| n > 1) {
        let terms = lens.iter().sum();
        let staged = reduce_level::<C, FUSED>(points, starts, lens, fields);
        level(staged, terms);
    }
}

/// Reduces one pair-tree level and returns the number of staged denominators.
///
/// Storage follows [`reduce`]. This complete path handles the exceptional pairs
/// detected by [`reduce_fused`]. `FUSED` selects a `mul_sub` expression for the
/// output's y-coordinate in timing controls; it does not fuse inverse recovery.
#[inline(always)]
pub(crate) fn reduce_level<C: PastaCurve, const FUSED: bool>(
    points: &mut [AffinePoint<C>],
    starts: &[usize],
    lens: &mut [usize],
    fields: &mut [PastaField<C::Base>],
) -> usize {
    let (denom, prefix) = fields.split_at_mut(points.len() / 2);
    let mut staged = 0;
    for (&start, &len) in starts.iter().zip(lens.iter()) {
        for pair in points[start..start + len].chunks_exact(2) {
            let (p, q) = (pair[0], pair[1]);
            if p.x == q.x && p.y != q.y {
                continue;
            }
            denom[staged] = if p.x == q.x {
                p.y.double()
            } else {
                q.x.sub(&p.x)
            };
            staged += 1;
        }
    }
    invert_nonzero(&mut denom[..staged], prefix);
    // Even when every pair cancels, this pass must compact odd survivors
    // and publish the new lengths. Pair j reads positions 2j and 2j+1 before
    // writing at most position j; it cannot overwrite a later unread pair.
    let mut read = 0;
    for (&start, len) in starts.iter().zip(lens.iter_mut()) {
        let old = *len;
        let mut written = 0;
        for i in (0..old.saturating_sub(1)).step_by(2) {
            let (p, q) = (points[start + i], points[start + i + 1]);
            if p.x == q.x && p.y != q.y {
                continue;
            }
            let numerator = if p.x == q.x {
                p.tangent_numerator()
            } else {
                q.y.sub(&p.y)
            };
            let slope = numerator.mul(&denom[read]);
            read += 1;
            points[start + written] = if FUSED {
                AffinePoint::from_slope_fused(&p.x, &p.y, &q.x, &slope)
            } else {
                AffinePoint::from_slope(&p.x, &p.y, &q.x, &slope)
            };
            written += 1;
        }
        if old & 1 != 0 {
            points[start + written] = points[start + old - 1];
            written += 1;
        }
        *len = written;
    }
    debug_assert_eq!(read, staged);
    staged
}

/// Retained native benchmark control with the original six-field staging layout.
#[cfg(test)]
pub(crate) fn reduce_original<C: PastaCurve>(
    points: &mut [AffinePoint<C>],
    starts: &[usize],
    lens: &mut [usize],
    fields: &mut [PastaField<C::Base>],
    writes: &mut [usize],
) {
    let capacity = points.len() / 2;
    let (x1, fields) = fields.split_at_mut(capacity);
    let (y1, fields) = fields.split_at_mut(capacity);
    let (x2, fields) = fields.split_at_mut(capacity);
    let (numerator, fields) = fields.split_at_mut(capacity);
    let (denom, prefix) = fields.split_at_mut(capacity);
    loop {
        // Read every pair's operands before completing additions in place.
        // Compacting odd survivors cannot overwrite an unread pair; cancelled
        // pairs disappear without needing an inverse or an identity encoding.
        let mut staged = 0;
        for (&start, len) in starts.iter().zip(lens.iter_mut()) {
            let old = *len;
            *len = 0;
            for i in (0..old.saturating_sub(1)).step_by(2) {
                let p = points[start + i];
                let q = points[start + i + 1];
                if p.x == q.x && p.y != q.y {
                    continue;
                }
                x1[staged] = p.x.into_loose();
                y1[staged] = p.y.into_loose();
                x2[staged] = q.x.into_loose();
                if p.x == q.x {
                    numerator[staged] = p.tangent_numerator();
                    denom[staged] = p.y.double();
                } else {
                    numerator[staged] = q.y.sub(&p.y);
                    denom[staged] = q.x.sub(&p.x);
                }
                writes[staged] = start + *len;
                *len += 1;
                staged += 1;
            }
            if old & 1 != 0 {
                points[start + *len] = points[start + old - 1];
                *len += 1;
            }
        }
        if staged == 0 {
            break;
        }
        invert_nonzero(&mut denom[..staged], prefix);
        for i in 0..staged {
            let slope = numerator[i].mul(&denom[i]);
            points[writes[i]] = AffinePoint::from_slope(&x1[i], &y1[i], &x2[i], &slope);
        }
        if lens.iter().all(|&n| n <= 1) {
            break;
        }
    }
}

/// Weights survivor `i` by `i + 1`, skipping points whose occupancy is zero.
pub(crate) fn collapse<C: PastaCurve>(
    points: &[AffinePoint<C>],
    occupied: &[usize],
) -> ProjectivePoint<C> {
    let Some(last) = occupied.iter().rposition(|&n| n != 0) else {
        return ProjectivePoint::IDENTITY;
    };
    let mut running = ProjectivePoint::IDENTITY;
    let mut sum = ProjectivePoint::IDENTITY;
    for (point, &len) in points[..=last].iter().zip(&occupied[..=last]).rev() {
        if len != 0 {
            running = running.add_mixed(point);
        }
        sum = sum.add(&running);
    }
    sum
}

#[cfg(test)]
#[path = "tests/reduce.rs"]
mod tests;
