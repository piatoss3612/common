use super::*;
use crate::field::CanonicalUint;
use crate::test_support::field_samples;
use std::{vec, vec::Vec};

mod arithmetic;
mod contracts;
mod eisenstein;
mod eisenstein_batch;
mod fixed_base;
mod glv;
mod reference;

fn scalar_corpus<C: PastaCurve>() -> Vec<PastaField<C::Scalar>> {
    let mut values = vec![PastaField::ZERO, PastaField::ONE, PastaField::ONE.neg()];
    for integer in [2, 3, 7, 8, 15, 16, 17, 127, 128, 255, 256, u64::MAX] {
        values.push(PastaField::from_u64(integer));
    }
    for bit in [63, 64, 65, 127, 128, 129, 191, 192, 193, 253, 254] {
        let power =
            PastaField::from_canonical_uint(CanonicalUint::power_of_two(bit).unwrap()).unwrap();
        values.extend([
            power.sub(&PastaField::ONE),
            power,
            power.add(&PastaField::ONE),
        ]);
    }
    values.extend(field_samples::<C::Scalar>().take(8));
    values
}

fn scaled<C: PastaCurve>(point: &Point<C>, scale: u64) -> ProjectivePoint<C> {
    let Some(point) = point.as_affine() else {
        // All z = 0 representations must compare equal, even with nonzero x/y.
        return ProjectivePoint {
            x: PastaField::ONE,
            y: PastaField::ONE,
            z: PastaField::ZERO,
            marker: PhantomData,
        };
    };
    let z = PastaField::from_u64(scale);
    ProjectivePoint {
        x: point.x.mul(&z.square()),
        y: point.y.mul(&z.square()).mul(&z),
        z,
        marker: PhantomData,
    }
}

fn invalid_field<M: PrimeModulus>() -> PastaField<M> {
    *bento::AlignedBytes([0xff; 32]).as_value()
}
