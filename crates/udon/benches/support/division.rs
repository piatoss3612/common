use std::hint::black_box;

use criterion::{BatchSize, Criterion};
use zakura_udon::{
    field::{PastaField, PrimeModulus},
    polynomial::{divide_linear_in_place, evaluate},
};

use super::{SEED_A, SEED_B, values};

fn quotient_only<M: PrimeModulus>(coefficients: &mut [PastaField<M>], point: &PastaField<M>) {
    let Some(last) = coefficients.last() else {
        return;
    };
    let mut carry = *last;
    for index in (1..coefficients.len().saturating_sub(1)).rev() {
        let next = carry.mul_add(point, &coefficients[index]);
        coefficients[index + 1] = carry;
        carry = next;
    }
    if coefficients.len() > 1 {
        coefficients[1] = carry;
    }
}

fn separate<M: PrimeModulus>(coefficients: &mut [PastaField<M>], point: &PastaField<M>) -> usize {
    let remainder = evaluate(coefficients, point);
    quotient_only(coefficients, point);
    if let Some(constant) = coefficients.first_mut() {
        *constant = remainder;
    }
    coefficients.len().min(1)
}

pub fn benchmarks<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let corpus = values::<M, 8192>(SEED_A);
    let [random_point] = values::<M, 1>(SEED_B);
    for length in [0usize, 1, 2, 3, 8, 32, 1024, 8192] {
        for (name, point) in [
            ("dense", random_point),
            ("zero", PastaField::ZERO),
            ("one", PastaField::ONE),
        ] {
            if name != "dense" && ![32, 1024].contains(&length) {
                continue;
            }
            let input = &corpus[..length];
            let mut fused = input.to_vec();
            let mut reference = input.to_vec();
            let len = divide_linear_in_place(&mut fused, &point);
            let reference_len = separate(&mut reference, &point);
            assert_eq!(len, reference_len);
            assert!(
                fused
                    .iter()
                    .zip(&reference)
                    .all(|(a, b)| a.reduce() == b.reduce())
            );
            let mut group =
                criterion.benchmark_group(format!("{field}/linear_division/{length}/{name}"));
            group.bench_function("with_remainder", |b| {
                b.iter_batched_ref(
                    || input.to_vec(),
                    |coefficients| {
                        let result =
                            divide_linear_in_place(black_box(coefficients), black_box(&point));
                        black_box(coefficients);
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            });
            group.bench_function("separate_evaluation", |b| {
                b.iter_batched_ref(
                    || input.to_vec(),
                    |coefficients| {
                        let result = separate(black_box(coefficients), black_box(&point));
                        black_box(coefficients);
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            });
            group.finish();
        }
    }
}
