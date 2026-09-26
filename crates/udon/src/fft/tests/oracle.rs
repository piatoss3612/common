//! Direct polynomial evaluation and reference transforms.

use super::*;

pub(super) fn evaluate<M: PrimeModulus>(
    coefficients: &[PastaField<M>],
    point: PastaField<M>,
) -> PastaField<M> {
    coefficients
        .iter()
        .rev()
        .fold(PastaField::ZERO, |acc, coefficient| {
            acc.mul(&point).add(coefficient)
        })
}

pub(in crate::fft) fn direct<M: PrimeModulus>(
    coefficients: &[PastaField<M>],
    domain: CosetDomain<M>,
) -> Vec<PastaField<M>> {
    let mut point = domain.shift();
    (0..domain.size())
        .map(|_| {
            let value = evaluate(coefficients, point);
            point = point.mul(&domain.domain().root());
            value
        })
        .collect()
}

pub(super) fn check_forward<M: PrimeModulus>(
    coefficients: &[PastaField<M>],
    domain: CosetDomain<M>,
    actual: &[PastaField<M>],
) -> Result<(), &'static str> {
    if actual == direct(coefficients, domain) {
        Ok(())
    } else {
        Err("FFT differs from direct polynomial evaluation")
    }
}

pub(super) fn reference_coset<M: PrimeModulus>(
    coefficients: &[PastaField<M>],
    domain: CosetDomain<M>,
) -> Vec<PastaField<M>> {
    let mut values = vec![PastaField::ZERO; domain.size()];
    let mut scale = PastaField::ONE;
    for (out, coefficient) in values.iter_mut().zip(coefficients) {
        *out = coefficient.mul(&scale);
        scale = scale.mul(&domain.shift());
    }
    reference::transform(&mut values, &domain.domain().root());
    values
}

pub(in crate::fft) fn ordered<M: PrimeModulus>(
    values: &[PastaField<M>],
    order: ElementOrder,
) -> Vec<PastaField<M>> {
    (0..values.len())
        .map(|i| {
            values[if order == ElementOrder::Natural {
                i
            } else {
                bit_reverse(i, values.len().ilog2())
            }]
        })
        .collect()
}

pub(in crate::fft) fn inverse_direct<M: PrimeModulus>(
    evaluations: &[PastaField<M>],
    domain: CosetDomain<M>,
    normalized: bool,
) -> Vec<PastaField<M>> {
    (0..domain.size())
        .map(|degree| {
            let step = domain.domain().inverse_root().pow_u64(degree as u64);
            let mut power = PastaField::ONE;
            let mut sum = PastaField::ZERO;
            for value in evaluations {
                sum = sum.add(&value.mul(&power));
                power = power.mul(&step);
            }
            sum = sum.mul(&domain.inverse_shift().pow_u64(degree as u64));
            if normalized {
                sum.mul(&domain.domain().size_inverse())
            } else {
                sum
            }
        })
        .collect()
}
