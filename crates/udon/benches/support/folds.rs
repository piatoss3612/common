use std::hint::black_box;

use criterion::Criterion;
use zakura_udon::{
    field::{PastaField, PrimeModulus},
    polynomial::fold_weighted,
};

use super::{SEED_A, SEED_B, values};

fn row_sums<M: PrimeModulus>(
    inputs: &[&[PastaField<M>]],
    weights: &[PastaField<M>],
    output: &mut [PastaField<M>],
) {
    assert_eq!(inputs.len(), weights.len());
    let extent = inputs.iter().map(|row| row.len()).max().unwrap_or(0);
    let output = &mut output[..extent];
    let Some((first, rest)) = inputs.split_first() else {
        return;
    };
    let weight = &weights[0];
    if weight.is_zero() {
        output[..first.len()].fill(PastaField::ZERO);
    } else if weight.is_one() {
        output[..first.len()].copy_from_slice(first);
    } else {
        for (out, value) in output.iter_mut().zip(*first) {
            *out = value.mul(weight);
        }
    }
    output[first.len()..].fill(PastaField::ZERO);
    for (input, weight) in rest.iter().zip(&weights[1..]) {
        if weight.is_zero() {
            continue;
        }
        if weight.is_one() {
            for (out, value) in output.iter_mut().zip(*input) {
                *out = out.add(value);
            }
        } else {
            for (out, value) in output.iter_mut().zip(*input) {
                *out = value.mul_add(weight, out);
            }
        }
    }
}

fn powers<M: PrimeModulus>(challenge: &PastaField<M>, weights: &mut [PastaField<M>]) {
    if let Some((first, rest)) = weights.split_first_mut() {
        *first = PastaField::ONE;
        let mut previous = *first;
        for weight in rest {
            *weight = previous.mul(challenge);
            previous = *weight;
        }
    }
}

pub fn benchmarks<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let corpus = values::<M, 1024>(SEED_A);
    let all_weights = values::<M, 32>(SEED_B);
    for shape in ["dense", "unit_first", "ragged"] {
        for (count, length) in [
            (0, 0),
            (1, 8),
            (2, 8),
            (8, 8),
            (32, 8),
            (1, 1024),
            (2, 1024),
            (8, 1024),
            (32, 1024),
        ] {
            if shape == "ragged" && (count < 8 || length < 1024) {
                continue;
            }
            let rows: Vec<Vec<_>> = (0..count)
                .map(|i| {
                    let len = if shape == "ragged" {
                        length - (i * 137 % (length + 1))
                    } else {
                        length
                    };
                    (0..len)
                        .map(|j| corpus[(j + i * 31) % corpus.len()])
                        .collect()
                })
                .collect();
            let inputs: Vec<_> = rows.iter().map(Vec::as_slice).collect();
            let mut weights = all_weights[..count].to_vec();
            if shape == "unit_first" && count != 0 {
                weights[0] = PastaField::ONE;
            }
            let mut output = vec![PastaField::ZERO; length];
            let mut expected = output.clone();
            row_sums(&inputs, &weights, &mut expected);
            fold_weighted(&inputs, &weights, &mut output).unwrap();
            assert!(
                output
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.reduce() == b.reduce())
            );

            let mut group = criterion
                .benchmark_group(format!("{field}/fold_weighted/{shape}/{count}x{length}"));
            group.bench_function("fold", |b| {
                b.iter(|| {
                    fold_weighted(
                        black_box(&inputs),
                        black_box(&weights),
                        black_box(&mut output),
                    )
                    .unwrap();
                    black_box(&output);
                });
            });
            group.bench_function("row_sums", |b| {
                b.iter(|| {
                    row_sums(
                        black_box(&inputs),
                        black_box(&weights),
                        black_box(&mut output),
                    );
                    black_box(&output);
                });
            });
            if shape == "unit_first" {
                let challenge = all_weights[0];
                powers(&challenge, &mut weights);
                row_sums(&inputs, &weights, &mut expected);
                fold_weighted(&inputs, &weights, &mut output).unwrap();
                assert!(
                    output
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.reduce() == b.reduce())
                );
                group.bench_function("powers_and_fold", |b| {
                    b.iter(|| {
                        powers(black_box(&challenge), black_box(&mut weights));
                        fold_weighted(
                            black_box(&inputs),
                            black_box(&weights),
                            black_box(&mut output),
                        )
                        .unwrap();
                        black_box(&output);
                    });
                });
                group.bench_function("powers_and_row_sums", |b| {
                    b.iter(|| {
                        powers(black_box(&challenge), black_box(&mut weights));
                        row_sums(
                            black_box(&inputs),
                            black_box(&weights),
                            black_box(&mut output),
                        );
                        black_box(&output);
                    });
                });
            }
            group.finish();
        }
    }
}
