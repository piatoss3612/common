//! Reference MSM using the scalar ladder.

use super::*;

pub(super) fn check<C: PastaCurve>(
    input: &Input<'_, C>,
    actual: ProjectivePoint<C>,
) -> Result<(), &'static str> {
    if actual == reference(input) {
        Ok(())
    } else {
        Err("MSM differs from the binary-ladder reference")
    }
}

pub(super) fn reference<C: PastaCurve>(input: &Input<'_, C>) -> ProjectivePoint<C> {
    let mut sum = ProjectivePoint::IDENTITY;
    let Scalars::Raw(scalars) = input.scalars else {
        panic!("reference needs original scalars")
    };
    for (i, k) in scalars.iter().enumerate() {
        let j = input.indices.map_or(i, |indices| indices[i] as usize);
        let base = match input.bases {
            Bases::Affine(b) => b[j].to_projective(),
            Bases::Prepared(b) => b[j].to_affine().to_projective(),
            Bases::Points(b) => b[j].to_projective(),
            Bases::Compact(b) => b.get(j).unwrap().base().to_projective(),
            Bases::CompactPrepared(b) => b.get(j).unwrap().base().to_projective(),
            Bases::Odd(b) => b.originals()[j].to_projective(),
            Bases::OddPrepared(b) => b.originals()[j].to_affine().to_projective(),
            Bases::Alpha(b) => b.originals()[j].to_projective(),
            Bases::AlphaPrepared(b) => b.originals()[j].to_affine().to_projective(),
        };
        sum = sum.add(&test_reference::multiply(k, |sum| sum.add(&base)));
    }
    sum
}
