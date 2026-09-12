//! Ordinary scalar multiplication shared by affine and projective bases.

use super::{PastaCurve, ProjectivePoint};
use crate::field::PastaField;

pub(super) fn multiply<C: PastaCurve>(
    scalar: &PastaField<C::Scalar>,
    add_base: impl Fn(&ProjectivePoint<C>) -> ProjectivePoint<C>,
) -> ProjectivePoint<C> {
    let scalar = scalar.to_canonical_uint();
    let mut result = ProjectivePoint::IDENTITY;
    if let Some(high) = scalar.highest_set_bit() {
        for bit in (0..=high).rev() {
            result = result.double();
            if scalar.bit(bit).unwrap() {
                result = add_base(&result);
            }
        }
    }
    result
}
