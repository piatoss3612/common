use std::hint::black_box;

use criterion::Criterion;
use zakura_udon::{
    field::{PastaField, PrimeModulus},
    polynomial::{EvaluationPlan, evaluate},
};

use super::{SEED_A, SEED_B, values};

fn horner_many<M: PrimeModulus>(
    inputs: &[&[PastaField<M>]],
    point: &PastaField<M>,
    output: &mut [PastaField<M>],
) {
    assert!(output.len() >= inputs.len());
    for (input, out) in inputs.iter().zip(output) {
        *out = evaluate(input, point);
    }
}

pub fn benchmarks<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let corpus = values::<M, 1024>(SEED_A);
    let [point] = values::<M, 1>(SEED_B);
    for length in [0usize, 1, 2, 3, 8, 32, 64, 1024] {
        let required = EvaluationPlan::<M>::power_count(length);
        let mut storage = vec![PastaField::ZERO; required];
        let plan = EvaluationPlan::prepare(&point, &mut storage);
        let mut preparation = vec![PastaField::ZERO; required];
        let mut group = criterion.benchmark_group(format!("{field}/evaluation/{length}"));
        group.bench_function("prepare", |b| {
            b.iter(|| {
                black_box(EvaluationPlan::prepare(
                    black_box(&point),
                    black_box(&mut preparation),
                ));
            });
        });
        group.bench_function("bind", |b| {
            b.iter(|| EvaluationPlan::bind(black_box(&point), black_box(plan.powers())));
        });
        group.finish();
        for count in [1, 2, 4, 8] {
            if length < 3 && count != 1 {
                continue;
            }
            for ragged in [false, true] {
                if ragged && (length < 32 || count != 8) {
                    continue;
                }
                let rows: Vec<Vec<_>> = (0..count)
                    .map(|i| {
                        let len = if ragged {
                            length * (count - i) / count
                        } else {
                            length
                        };
                        (0..len)
                            .map(|j| corpus[(j + i * 37) % corpus.len()])
                            .collect()
                    })
                    .collect();
                let inputs: Vec<_> = rows.iter().map(Vec::as_slice).collect();
                let mut output = vec![PastaField::ZERO; count];
                let mut expected = output.clone();
                horner_many(&inputs, &point, &mut expected);
                plan.evaluate_many(&inputs, &mut output).unwrap();
                assert!(
                    output
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.reduce() == b.reduce())
                );
                let shape = if ragged { "ragged" } else { "dense" };
                let mut group = criterion
                    .benchmark_group(format!("{field}/evaluation/{length}/{shape}/reuse_{count}"));
                group.bench_function("horner", |b| {
                    b.iter(|| {
                        horner_many(
                            black_box(&inputs),
                            black_box(&point),
                            black_box(&mut output),
                        );
                        black_box(&output);
                    });
                });
                group.bench_function("retained", |b| {
                    b.iter(|| {
                        black_box(&plan)
                            .evaluate_many(black_box(&inputs), black_box(&mut output))
                            .unwrap();
                        black_box(&output);
                    });
                });
                group.bench_function("prepare_and_evaluate", |b| {
                    b.iter(|| {
                        let plan =
                            EvaluationPlan::prepare(black_box(&point), black_box(&mut preparation));
                        plan.evaluate_many(black_box(&inputs), black_box(&mut output))
                            .unwrap();
                        black_box(&output);
                    });
                });
                group.finish();
            }
        }
    }
}
