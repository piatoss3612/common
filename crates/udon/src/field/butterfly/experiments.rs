//! Isolated target experiments; none of these candidates select runtime policy.

use super::*;
use crate::field::{PallasBase, PallasScalar};
use crate::test_support::{CORPUS_SEED, xorshift64};
use std::{hint::black_box, time::Instant};

// These experiments exercise the private [0, 2p) representation. The ordinary
// field constructor requires canonical limbs and must not receive this data.
fn loose<M: PrimeModulus>(limbs: [u64; 4]) -> PastaField<M> {
    PastaField {
        limbs,
        marker: core::marker::PhantomData,
    }
}

#[inline]
fn correct<M: PrimeModulus, const MASKED: bool>(
    left: &mut PastaField<M>,
    product: &mut PastaField<M>,
) {
    if !MASKED {
        butterfly(left, product, None);
        return;
    }
    let modulus = M::TWICE_MODULUS;
    let (sum, carry) = super::super::word::add_limbs(&left.limbs, &product.limbs);
    let (reduced, borrow) = subtract_limbs(&sum, &modulus);
    let mask = 0u64.wrapping_sub(carry | (borrow ^ 1));
    let (mut difference, borrow) = subtract_limbs(&left.limbs, &product.limbs);
    let correction = 0u64.wrapping_sub(borrow);
    let mut carry = 0;
    for index in 0..4 {
        left.limbs[index] = (sum[index] & !mask) | (reduced[index] & mask);
        (difference[index], carry) = adc(difference[index], modulus[index] & correction, carry);
    }
    product.limbs = difference;
}

#[inline]
fn pair<M: PrimeModulus, const MASKED: bool, const INTERLEAVED: bool>(
    values: &mut [PastaField<M>; 4],
    twiddle: &PastaField<M>,
) {
    if INTERLEAVED {
        let mut first = loose(multiply::<M>(&values[1].limbs, &twiddle.limbs));
        let mut second = loose(multiply::<M>(&values[3].limbs, &twiddle.limbs));
        correct::<M, MASKED>(&mut values[0], &mut first);
        correct::<M, MASKED>(&mut values[2], &mut second);
        values[1] = first;
        values[3] = second;
    } else {
        for values in values.chunks_exact_mut(2) {
            let mut product = loose(multiply::<M>(&values[1].limbs, &twiddle.limbs));
            correct::<M, MASKED>(&mut values[0], &mut product);
            values[1] = product;
        }
    }
}

fn check<M: PrimeModulus>() {
    let minus_one = subtract_limbs(&M::MODULUS, &[1, 0, 0, 0]).0;
    let max = subtract_limbs(&M::TWICE_MODULUS, &[1, 0, 0, 0]).0;
    let values = [[0; 4], [1, 0, 0, 0], minus_one, M::MODULUS, max];
    for left in values {
        for right in values {
            for twiddle in [
                PastaField::<M>::ONE,
                PastaField::<_>::ONE.neg(),
                PastaField::from_u64(7),
            ] {
                let original = [left, right, right, left].map(loose::<M>);
                let mut expected = original;
                for pair in expected.chunks_exact_mut(2) {
                    let (left, right) = pair.split_at_mut(1);
                    butterfly(&mut left[0], &mut right[0], Some(&twiddle));
                }
                let mut masked = original;
                pair::<M, true, false>(&mut masked, &twiddle);
                assert_eq!(
                    masked.map(|value| value.reduce()),
                    expected.map(|value| value.reduce())
                );
                let mut interleaved = original;
                pair::<M, false, true>(&mut interleaved, &twiddle);
                assert_eq!(
                    interleaved.map(|value| value.reduce()),
                    expected.map(|value| value.reduce())
                );
                pair::<M, true, true>(&mut masked, &twiddle);
                pair::<M, false, false>(&mut expected, &twiddle);
                assert_eq!(
                    masked.map(|value| value.reduce()),
                    expected.map(|value| value.reduce())
                );
            }
        }
    }
}

#[test]
fn correction_and_interleaving_candidates_preserve_the_loose_bound() {
    check::<PallasBase>();
    check::<PallasScalar>();
}

// Keep four separately inspectable bodies so a code-generation comparison can
// establish whether source-level branches survive on a particular target.
#[inline(never)]
fn timed<M: PrimeModulus, const MASKED: bool, const INTERLEAVED: bool>(
    input: &[[PastaField<M>; 4]],
    twiddle: &PastaField<M>,
) -> f64 {
    let mut samples = [0.0f64; 5];
    for sample in &mut samples {
        let start = Instant::now();
        for _ in 0..256 {
            for values in input {
                let mut values = black_box(*values);
                pair::<M, MASKED, INTERLEAVED>(&mut values, black_box(twiddle));
                black_box(values);
            }
        }
        *sample = start.elapsed().as_nanos() as f64 / (256 * input.len()) as f64;
    }
    samples.sort_by(f64::total_cmp);
    samples[2]
}

fn measure<M: PrimeModulus>(field: &str) {
    let mut seed = CORPUS_SEED;
    let random: std::vec::Vec<[PastaField<M>; 4]> = (0..1024)
        .map(|_| {
            core::array::from_fn(|_| {
                let mut limbs = core::array::from_fn(|_| xorshift64(&mut seed));
                limbs[3] &= (1 << 63) - 1;
                loose(limbs)
            })
        })
        .collect();
    let boundary = subtract_limbs(&M::TWICE_MODULUS, &[1, 0, 0, 0]).0;
    for (distribution, input) in [
        ("zero", std::vec![[PastaField::ZERO; 4]; 1024]),
        ("loose_random", random),
        ("boundary", std::vec![[loose(boundary); 4]; 1024]),
    ] {
        let twiddle = PastaField::from_u64(7);
        let serial = timed::<M, false, false>(&input, &twiddle);
        let interleaved = timed::<M, false, true>(&input, &twiddle);
        let masked = timed::<M, true, false>(&input, &twiddle);
        let combined = timed::<M, true, true>(&input, &twiddle);
        std::println!(
            "{field}/{distribution}: ns per pair, branching={serial:.2}, interleaved={interleaved:.2}, masked={masked:.2}, masked_interleaved={combined:.2}"
        );
    }
}

#[test]
#[ignore = "target-specific timing experiment; run in release with --nocapture"]
fn compare_fft_butterfly_candidates() {
    measure::<PallasBase>("Fp");
    measure::<PallasScalar>("Fq");
}
