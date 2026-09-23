use std::hint::black_box;

use criterion::{BatchSize, Criterion};
use zakura_udon::{
    field::{PastaField, PrimeModulus},
    polynomial::{
        MonicDivisionError, divide_linear_in_place, divide_monic_in_place, vanishing_polynomial,
    },
};

use super::{SEED_A, SEED_B, values};

fn general<M: PrimeModulus>(
    coefficients: &mut [PastaField<M>],
    divisor: &[PastaField<M>],
) -> Result<usize, MonicDivisionError> {
    let Some((leading, lower)) = divisor.split_last() else {
        return Err(MonicDivisionError::EmptyDivisor);
    };
    if !leading.is_one() {
        return Err(MonicDivisionError::NonMonicDivisor);
    }
    let degree = lower.len();
    if degree > 0 && coefficients.len() > degree {
        for index in (0..coefficients.len() - degree).rev() {
            let negative_quotient = coefficients[index + degree].neg();
            for (value, divisor) in coefficients[index..index + degree].iter_mut().zip(lower) {
                *value = negative_quotient.mul_add(divisor, value);
            }
        }
    }
    Ok(coefficients.len().min(degree))
}

fn successive<M: PrimeModulus>(
    coefficients: &mut [PastaField<M>],
    roots: &[PastaField<M>],
    scratch: &mut [PastaField<M>],
) -> usize {
    let degree = roots.len();
    let n = coefficients.len();
    if degree == 0 || n <= degree {
        return n.min(degree);
    }
    let (remainders, remainder) = scratch[..2 * degree].split_at_mut(degree);
    let mut offset = 0;
    for (root, out) in roots.iter().zip(remainders.iter_mut()) {
        let split = divide_linear_in_place(&mut coefficients[offset..], root);
        *out = coefficients[offset];
        offset += split;
    }
    // Successive scalar remainders are Newton coefficients. Expand them to
    // retain the full polynomial remainder in the same layout as monic division.
    remainder[0] = remainders[degree - 1];
    for (extent, index) in (0..degree - 1).rev().enumerate() {
        let extent = extent + 1;
        let negative_root = roots[index].neg();
        remainder[extent] = remainder[extent - 1];
        for j in (1..extent).rev() {
            remainder[j] = remainder[j].mul_add(&negative_root, &remainder[j - 1]);
        }
        remainder[0] = remainder[0].mul_add(&negative_root, &remainders[index]);
    }
    coefficients[..degree].copy_from_slice(remainder);
    degree
}

pub fn benchmarks<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let corpus = values::<M, 8192>(SEED_A);
    let all_roots = values::<M, 8>(SEED_B);
    for degree in [2usize, 3, 8] {
        let roots = &all_roots[..degree];
        let mut divisor = vec![PastaField::ZERO; degree + 1];
        vanishing_polynomial(roots, &mut divisor).unwrap();
        let mut scratch = vec![PastaField::ZERO; 2 * degree];
        for length in [0, degree, degree + 1, 32, 1024, 8192] {
            let input = &corpus[..length];
            let mut checked = input.to_vec();
            let split = divide_monic_in_place(&mut checked, &divisor).unwrap();
            for method in [0, 1] {
                let mut reference = input.to_vec();
                let reference_split = if method == 0 {
                    general(&mut reference, &divisor).unwrap()
                } else {
                    successive(&mut reference, roots, &mut scratch)
                };
                assert_eq!(split, reference_split);
                assert!(
                    checked
                        .iter()
                        .zip(&reference)
                        .all(|(a, b)| a.reduce() == b.reduce())
                );
            }
            let mut group =
                criterion.benchmark_group(format!("{field}/monic_division/{degree}/{length}"));
            group.bench_function("monic", |b| {
                b.iter_batched_ref(
                    || input.to_vec(),
                    |coefficients| {
                        let result =
                            divide_monic_in_place(black_box(coefficients), black_box(&divisor));
                        black_box(coefficients);
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            });
            group.bench_function("general", |b| {
                b.iter_batched_ref(
                    || input.to_vec(),
                    |coefficients| {
                        let result = general(black_box(coefficients), black_box(&divisor));
                        black_box(coefficients);
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            });
            group.bench_function("successive_linear", |b| {
                b.iter_batched_ref(
                    || input.to_vec(),
                    |coefficients| {
                        let result = successive(
                            black_box(coefficients),
                            black_box(roots),
                            black_box(&mut scratch),
                        );
                        black_box(coefficients);
                        black_box(result)
                    },
                    BatchSize::SmallInput,
                );
            });
            if length == 1024 {
                group.bench_function("prepare_and_divide", |b| {
                    b.iter_batched_ref(
                        || input.to_vec(),
                        |coefficients| {
                            vanishing_polynomial(black_box(roots), black_box(&mut divisor))
                                .unwrap();
                            let result =
                                divide_monic_in_place(black_box(coefficients), black_box(&divisor));
                            black_box(coefficients);
                            black_box(result)
                        },
                        BatchSize::SmallInput,
                    );
                });
            }
            group.finish();
        }
        let mut group = criterion.benchmark_group(format!("{field}/vanishing/{degree}"));
        group.bench_function("prepare", |b| {
            b.iter(|| {
                let result = vanishing_polynomial(black_box(roots), black_box(&mut divisor));
                black_box(&divisor);
                black_box(result)
            });
        });
        group.finish();
    }
}
