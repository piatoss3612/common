//! Affine `BigUint` formulas with ordinary residues, independent of runtime
//! Montgomery and Jacobian kernels.

use num_bigint::BigUint;

use super::*;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct Reference {
    pub coordinates: Option<(BigUint, BigUint)>,
}

impl Reference {
    pub fn identity() -> Self {
        Self { coordinates: None }
    }

    pub fn generator(p: &BigUint) -> Self {
        Self {
            coordinates: Some((p - 1_u32, BigUint::from(2_u32))),
        }
    }

    pub fn from_point<C: PastaCurve>(point: &Point<C>) -> Self {
        Self {
            coordinates: point.coordinates().map(|(x, y)| {
                (
                    BigUint::from_bytes_le(&x.to_bytes()),
                    BigUint::from_bytes_le(&y.to_bytes()),
                )
            }),
        }
    }

    pub fn add(&self, rhs: &Self, p: &BigUint) -> Self {
        let (Some((x1, y1)), Some((x2, y2))) = (&self.coordinates, &rhs.coordinates) else {
            return if self.coordinates.is_none() {
                rhs.clone()
            } else {
                self.clone()
            };
        };
        let (numerator, denominator) = if x1 == x2 {
            if (y1 + y2) % p == BigUint::from(0_u32) {
                return Self::identity();
            }
            (x1 * x1 * 3_u32 % p, y1 * 2_u32 % p)
        } else {
            ((y2 + p - y1) % p, (x2 + p - x1) % p)
        };
        let slope = numerator * denominator.modpow(&(p - 2_u32), p) % p;
        let x = (&slope * &slope + p + p - x1 - x2) % p;
        let y = (&slope * (x1 + p - &x) + p - y1) % p;
        Self {
            coordinates: Some((x, y)),
        }
    }

    pub fn mul(&self, scalar: &BigUint, p: &BigUint) -> Self {
        let mut result = Self::identity();
        for bit in (0..scalar.bits()).rev() {
            result = result.add(&result, p);
            if scalar.bit(bit) {
                result = result.add(self, p);
            }
        }
        result
    }

    pub fn assert_point<C: PastaCurve>(&self, point: &Point<C>) {
        assert_eq!(self, &Self::from_point(point));
    }
}
