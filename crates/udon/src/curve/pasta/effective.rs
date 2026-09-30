//! Temporary Eisenstein tables and Jacobian ladders on isomorphic curves.

use super::{PastaCurve, ProjectivePoint, eisenstein::EisensteinScalar};
use crate::field::{CanonicalUint, PastaField};
use core::marker::PhantomData;

/// Multiplies a nonidentity base by a canonical scalar without inversion.
///
/// Table entries and the ladder stay on y²=x³+5*d⁶. Restoring d in the final
/// Jacobian denominator yields an ordinary point. The private scratch types
/// cannot be confused with affine points or retained POD tables.
pub(super) fn multiply<C: PastaCurve>(
    base: &ProjectivePoint<C>,
    scalar: CanonicalUint,
) -> ProjectivePoint<C> {
    let prepared = EisensteinScalar::<C>::from_canonical(scalar);
    let Some((&top, digits)) = prepared.digits().split_last() else {
        return ProjectivePoint::IDENTITY;
    };
    let mut storage = [PastaField::ZERO; 24];
    let table = EffectiveTable::prepare(base, &mut storage);
    let mut result = Jacobian {
        xy: table.digit(top),
        z: PastaField::ONE,
    };
    for &code in digits.iter().rev() {
        result = result.double();
        if code != 0 {
            result = result.add(table.digit(code));
        }
    }
    ProjectivePoint {
        x: result.xy.x,
        y: result.xy.y,
        z: result.z.mul(&table.denominator),
        marker: PhantomData,
    }
}

// Scratch on E_d; it never implements an ordinary point or POD interface.
#[derive(Clone, Copy)]
struct Jacobian<C: PastaCurve> {
    xy: Coordinates<C>,
    z: PastaField<C::Base>,
}

impl<C: PastaCurve> Jacobian<C> {
    fn double(self) -> Self {
        // The a=0 formulas do not use the curve's constant term. A zero z
        // remains zero, including for the private identity representation.
        let b = self.xy.y.square();
        let c = b.square();
        let d = self.xy.x.mul(&b);
        let e = self.xy.x.square().triple().half();
        let x = e.square().sub(&d.double());
        Self {
            xy: Coordinates {
                x,
                y: e.mul_sub(&d.sub(&x), &c),
            },
            z: self.z.mul(&self.xy.y),
        }
    }

    fn add(self, rhs: Coordinates<C>) -> Self {
        if self.z.is_zero() {
            return Self {
                xy: rhs,
                z: PastaField::ONE,
            };
        }
        let zz = self.z.square();
        let u = rhs.x.mul(&zz);
        let s = rhs.y.mul(&zz).mul(&self.z);
        let h = u.sub(&self.xy.x);
        let r = s.sub(&self.xy.y);
        if h.is_zero() {
            // Scalar schedules may encounter equal or inverse points. Keep
            // the ladder complete even though table construction is incomplete.
            return if r.is_zero() {
                self.double()
            } else {
                Self {
                    xy: rhs,
                    z: PastaField::ZERO,
                }
            };
        }
        let hh = h.square();
        let hhh = h.mul(&hh);
        let v = self.xy.x.mul(&hh);
        let x = r.square().sub(&hhh).sub(&v.double());
        Self {
            xy: Coordinates {
                x,
                y: r.mul_sub_product(&v.sub(&x), &self.xy.y, &hhh),
            },
            z: self.z.mul(&h),
        }
    }
}

// These coordinates satisfy y² = x³ + 5*d^6 for a separately retained d.
// They are deliberately neither AffinePoint nor CurveTableEntry: treating them
// as ordinary affine/POD values would lose the omitted denominator.
#[derive(Clone, Copy)]
struct Coordinates<C: PastaCurve> {
    x: PastaField<C::Base>,
    y: PastaField<C::Base>,
}

impl<C: PastaCurve> Coordinates<C> {
    fn transform(mut self, rotation: usize, negative: bool) -> Self {
        self.x = match rotation {
            0 => self.x,
            1 => self.x.mul(&PastaField::<C::Base>::ZETA),
            2 => self.x.mul(&PastaField::<C::Base>::ZETA_INVERSE),
            _ => unreachable!("three rotations"),
        };
        if negative {
            self.y = self.y.neg();
        }
        self
    }

    fn rescale(self, ratio: &PastaField<C::Base>) -> Self {
        let square = ratio.square();
        Self {
            x: self.x.mul(&square),
            y: self.y.mul(&square).mul(ratio),
        }
    }
}

struct EffectiveTable<'a, C: PastaCurve> {
    coordinates: &'a [PastaField<C::Base>],
    endomorphism_x: &'a [PastaField<C::Base>],
    denominator: PastaField<C::Base>,
}

impl<'a, C: PastaCurve> EffectiveTable<'a, C> {
    fn prepare(base: &ProjectivePoint<C>, storage: &'a mut [PastaField<C::Base>]) -> Self {
        debug_assert!(!base.is_identity());
        debug_assert_eq!(storage.len(), 24);
        let (storage, rotations) = storage.split_at_mut(16);
        // D=2P has denominator d=P.z*P.y. Its raw x/y lie on E_d, with
        // equation y²=x³+5*d^6. Map P onto E_d by scaling its raw x/y by
        // P.y²/P.y³. This uses no inverse, even for projectively scaled input.
        let yy = base.y.square();
        let yyyy = yy.square();
        let xyy = base.x.mul(&yy);
        let e = base.x.square().triple().half();
        let dx = e.square().sub(&xyy.double());
        let double = ProjectivePoint::<C> {
            x: dx,
            y: e.mul_sub(&xyy.sub(&dx), &yyyy),
            z: base.z.mul(&base.y),
            marker: PhantomData,
        };
        let operand = Coordinates::<C> {
            x: double.x,
            y: double.y,
        };
        let mut current = Coordinates::<C> { x: xyy, y: yyyy };
        let mut z = PastaField::ONE;
        storage[0] = current.x;
        storage[1] = current.y;
        // Each tuple transforms the previous representative, adds 2P, then
        // transforms the result into the next representative. Coefficients use
        // phi(a,b)=(-b,a-b), with REPRESENTATIVES' order for storage.
        const CHAIN: [(usize, usize, bool, usize, bool); 7] = [
            (4, 0, false, 0, false),
            (7, 1, true, 0, false),
            (2, 2, false, 1, false),
            (6, 1, true, 0, false),
            (5, 1, true, 2, true),
            (1, 0, true, 0, true),
            (3, 1, true, 0, false),
        ];
        let mut ratios = [PastaField::ONE; 7];
        for (step, &(index, before, negate_before, after, negate_after)) in CHAIN.iter().enumerate()
        {
            current = current.transform(before, negate_before);
            let zz = z.square();
            let h = operand.x.mul(&zz).sub(&current.x);
            let r = operand.y.mul(&zz).mul(&z).sub(&current.y);
            // No transformed representative equals +/-2P: the corresponding
            // nonzero Eisenstein differences have norm below the prime order.
            // Hence all h and z stay nonzero for every nonidentity base.
            debug_assert!(!h.is_zero());
            let hh = h.square();
            let hhh = hh.mul(&h);
            let v = current.x.mul(&hh);
            let x = r.square().sub(&hhh).sub(&v.double());
            let y = r.mul_sub_product(&v.sub(&x), &current.y, &hhh);
            current = Coordinates { x, y }.transform(after, negate_after);
            z = z.mul(&h);
            ratios[step] = h;
            storage[2 * index] = current.x;
            storage[2 * index + 1] = current.y;
        }
        // z_i divides z_final through the recorded ratios. Multiplying each
        // entry by the suffix ratio squared/cubed puts all entries on E_(d*z).
        // The reverse products recover these ratios without division.
        let mut suffix = PastaField::ONE;
        for step in (0..7).rev() {
            suffix = suffix.mul(&ratios[step]);
            let index = if step == 0 { 0 } else { CHAIN[step - 1].0 };
            let entry = Coordinates::<C> {
                x: storage[2 * index],
                y: storage[2 * index + 1],
            }
            .rescale(&suffix);
            storage[2 * index] = entry.x;
            storage[2 * index + 1] = entry.y;
        }
        for (rotation, xy) in rotations.iter_mut().zip(storage.chunks_exact(2)) {
            *rotation = xy[0].mul(&PastaField::<C::Base>::ZETA);
        }
        Self {
            coordinates: storage,
            endomorphism_x: rotations,
            denominator: double.z.mul(&z),
        }
    }

    fn digit(&self, code: u8) -> Coordinates<C> {
        let value = usize::from(code - 1);
        let index = value / 6;
        let x = self.coordinates[2 * index];
        let rotated = self.endomorphism_x[index];
        let y = self.coordinates[2 * index + 1];
        Coordinates {
            x: match (value % 6) / 2 {
                0 => x,
                1 => rotated,
                _ => x.add(&rotated).neg(),
            },
            y: if value & 1 == 0 { y } else { y.neg() },
        }
    }
}

#[cfg(test)]
#[path = "tests/effective.rs"]
mod tests;
