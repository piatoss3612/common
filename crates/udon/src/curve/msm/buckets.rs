//! Affine bucket reduction with one shared inversion per pair-tree level.

use core::marker::PhantomData;

use crate::{
    curve::{AffinePoint, PastaCurve, ProjectivePoint, batch::invert_nonzero},
    field::PastaField,
};

/// Reduces disjoint buckets to zero or one affine point, updating their lengths.
///
/// Bucket `i` occupies `points[starts[i]..starts[i] + lens[i]]`. The caller must
/// reserve disjoint ranges within `points`, one start per length, six field
/// elements per possible pair, and one write index per possible pair. Pair
/// capacity is `points.len() / 2`; survivors end up at their bucket's start.
pub(super) fn reduce<C: PastaCurve>(
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
                x1[staged] = p.x;
                y1[staged] = p.y;
                x2[staged] = q.x;
                if p.x == q.x {
                    let xx = p.x.square();
                    numerator[staged] = xx.double().add(&xx);
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
            let x = slope.square().sub(&x1[i]).sub(&x2[i]);
            let y = slope.mul(&x1[i].sub(&x)).sub(&y1[i]);
            points[writes[i]] = AffinePoint {
                x,
                y,
                marker: PhantomData,
            };
        }
        if lens.iter().all(|&n| n <= 1) {
            break;
        }
    }
}

/// Weights survivor `i` by `i + 1`, skipping points whose occupancy is zero.
pub(super) fn collapse<C: PastaCurve>(
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
