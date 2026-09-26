//! Targeted comparisons for arithmetic specialization points.

use super::*;
use crate::field::word::{adc, mac};
use std::{hint::black_box, time::Instant};

// Separate the product from low-half Montgomery cancellation, allowing the
// addend to enter the high half before its final carry propagation.
fn multiply_add<M: PrimeModulus>(
    a: PastaField<M>,
    b: PastaField<M>,
    c: PastaField<M>,
) -> PastaField<M> {
    let mut wide = crate::field::word::multiply_wide(&a.limbs, &b.limbs);
    let mut carry = 0;
    for (limb, term) in wide[4..].iter_mut().zip(c.limbs) {
        (*limb, carry) = adc(*limb, term, carry);
    }
    // ab + cR < 4p^2 + 2pR < R^2. REDC can nevertheless reach 4p,
    // so preserve the final carry before reducing modulo 2p.
    debug_assert_eq!(carry, 0);
    let [mut r0, mut r1, mut r2, mut r3, t4, t5, t6, t7] = wide;
    for _ in 0..4 {
        let k = r0.wrapping_mul(M::MONTGOMERY_INV);
        let (_, carry) = mac(r0, k, M::MODULUS[0], 0);
        let (s0, carry) = mac(r1, k, M::MODULUS[1], carry);
        let (s1, carry) = adc(r2, 0, carry);
        let (s2, carry) = adc(r3, k << 62, carry);
        (r0, r1, r2, r3) = (s0, s1, s2, (k >> 2) + carry);
    }
    let (r0, carry) = adc(r0, t4, 0);
    let (r1, carry) = adc(r1, t5, carry);
    let (r2, carry) = adc(r2, t6, carry);
    let (r3, carry) = adc(r3, t7, carry);
    PastaField::from_montgomery(crate::field::pasta::montgomery::reduce_twice_modulus::<M>(
        [r0, r1, r2, r3],
        carry,
    ))
}

fn check<M: PrimeModulus>() {
    // Cross every quotient boundary of the 254-bit fold, including borrow
    // repair and the largest loose representatives.
    let p = modulus::<M>();
    let inverse_r = (BigUint::from(1u8) << 256usize).modpow(&(&p - 2u8), &p);
    for k in [3u8, 4, 8] {
        for q in 1u8..=16 {
            let boundary = (BigUint::from(q) << 254usize) / k;
            for raw in [&boundary - 1u8, boundary.clone(), &boundary + 1u8] {
                if raw >= &p * 2u8 {
                    continue;
                }
                let value = PastaField::<M>::from_montgomery_limbs(limbs(&raw));
                let result = match k {
                    3 => value.triple(),
                    4 => value.mul_by_4(),
                    _ => value.mul_by_8(),
                };
                assert_value(result, &(&raw * &inverse_r * k));
            }
        }
    }
    let values = samples::<M>(32);
    for (a, x) in &values {
        assert_value(a.triple(), &(x * 3u8));
        assert_value(a.mul_by_4(), &(x * 4u8));
        assert_value(a.mul_by_8(), &(x * 8u8));
        assert_eq!(a.triple().half().reduce(), a.add(&a.half()).reduce());
        for (b, y) in &values {
            for (c, z) in values.iter().step_by(9) {
                assert_value(multiply_add(*a, *b, *c), &(x * y + z));
                assert_value(multiply_add(*a, *b, c.neg()), &(x * y + modulus::<M>() - z));
            }
        }
    }
}

#[test]
fn specialization_candidates_match_integers() {
    check::<PallasBase>();
    check::<PallasScalar>();
}

fn timed<M: PrimeModulus>(
    values: &[PastaField<M>],
    operation: impl Fn(PastaField<M>) -> PastaField<M>,
) -> f64 {
    let mut samples = [0.0f64; 7];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..4096 {
            for value in black_box(values) {
                black_box(operation(*value));
            }
        }
        *sample = start.elapsed().as_nanos() as f64 / (4096 * values.len()) as f64;
    }
    samples.sort_by(f64::total_cmp);
    samples[3]
}

fn compare<M: PrimeModulus>(name: &str) {
    let values: Vec<_> = samples::<M>(256)
        .into_iter()
        .map(|(value, _)| value)
        .collect();
    let b = black_box(values[19]);
    let c = black_box(values[27]);
    for (label, baseline, candidate) in [
        (
            "mul_add",
            timed(&values, |a| a.mul_add(&b, &c)),
            timed(&values, |a| multiply_add(a, b, c)),
        ),
        (
            "mul_sub",
            timed(&values, |a| a.mul_sub(&b, &c)),
            timed(&values, |a| multiply_add(a, b, c.neg())),
        ),
        (
            "triple",
            timed(&values, |a| a.double().add(&a)),
            timed(&values, |a| a.triple()),
        ),
        (
            "mul_by_4",
            timed(&values, |a| a.double().double()),
            timed(&values, |a| a.mul_by_4()),
        ),
        (
            "mul_by_8",
            timed(&values, |a| a.double().double().double()),
            timed(&values, |a| a.mul_by_8()),
        ),
        (
            "three_halves",
            timed(&values, |a| a.triple().half()),
            timed(&values, |a| a.add(&a.half())),
        ),
    ] {
        std::println!("{name}/{label}: baseline {baseline:.3}, candidate {candidate:.3} ns/value");
    }
}

#[test]
#[ignore = "targeted timing experiment"]
fn compare_arithmetic_specializations() {
    compare::<PallasBase>("Fp");
    compare::<PallasScalar>("Fq");
}
