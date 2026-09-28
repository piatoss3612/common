use std::hint::black_box;

use criterion::Criterion;
use zakura_udon::{
    field::{PastaField, PrimeModulus, batch_invert_groups},
    polynomial::{InterpolationPlan, evaluate},
};

use super::{SEED_A, SEED_B, values};

pub fn benchmarks<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let nodes = values::<M, 64>(SEED_A);
    let coefficients = values::<M, 64>(SEED_B);
    let queries = values::<M, 16>(SEED_A ^ SEED_B);
    for n in [0, 1, 4, 16, 64] {
        let points = &nodes[..n];
        let ordinates: Vec<_> = points
            .iter()
            .map(|x| evaluate(&coefficients[..n], x))
            .collect();
        let mut weights = vec![PastaField::ZERO; n];
        let mut work = vec![PastaField::ZERO; n];
        let plan = InterpolationPlan::prepare(points, &mut weights, &mut work).unwrap();
        let mut output = vec![PastaField::ZERO; n];
        plan.interpolate(&ordinates, &mut output, &mut work)
            .unwrap();
        assert!(
            output
                .iter()
                .zip(&coefficients)
                .all(|(a, b)| a.reduce() == b.reduce())
        );
        for query in &queries {
            assert_eq!(
                plan.evaluate(&ordinates, query, &mut work)
                    .unwrap()
                    .reduce(),
                evaluate(&coefficients[..n], query).reduce()
            );
        }
        let mut prepared = vec![PastaField::ZERO; n];
        let mut inversion = vec![PastaField::ZERO; n];
        let mut group = criterion.benchmark_group(format!("{field}/interpolation/{n}"));
        group.bench_function("prepare", |b| {
            b.iter(|| {
                black_box(
                    InterpolationPlan::prepare(
                        black_box(points),
                        black_box(&mut prepared),
                        black_box(&mut inversion),
                    )
                    .unwrap(),
                );
            })
        });
        group.bench_function("prepare_bounded", |b| {
            b.iter(|| {
                black_box(
                    InterpolationPlan::prepare(
                        black_box(points),
                        black_box(&mut prepared),
                        black_box(&mut inversion[..n.min(2)]),
                    )
                    .unwrap(),
                );
            })
        });
        group.bench_function("bind", |b| {
            b.iter(|| {
                black_box(
                    InterpolationPlan::bind(black_box(points), black_box(plan.weights())).unwrap(),
                );
            })
        });
        group.bench_function("interpolate_retained", |b| {
            b.iter(|| {
                black_box(&plan)
                    .interpolate(
                        black_box(&ordinates),
                        black_box(&mut output),
                        black_box(&mut work),
                    )
                    .unwrap();
                black_box(&output);
            })
        });
        group.bench_function("prepare_and_interpolate", |b| {
            b.iter(|| {
                let plan = InterpolationPlan::prepare(
                    black_box(points),
                    black_box(&mut prepared),
                    black_box(&mut inversion),
                )
                .unwrap();
                plan.interpolate(
                    black_box(&ordinates),
                    black_box(&mut output),
                    black_box(&mut work),
                )
                .unwrap();
                black_box(&output);
            })
        });
        if n != 0 {
            group.bench_function("node", |b| {
                b.iter(|| {
                    black_box(&plan)
                        .evaluate(
                            black_box(&ordinates),
                            black_box(&points[n / 2]),
                            black_box(&mut work),
                        )
                        .unwrap()
                })
            });
        }
        group.finish();
        for count in [1, 16] {
            if n < 4 && count != 1 {
                continue;
            }
            let queries = &queries[..count];
            let mut results = vec![PastaField::ZERO; count];
            let mut group =
                criterion.benchmark_group(format!("{field}/interpolation/{n}/queries_{count}"));
            group.bench_function("retained", |b| {
                b.iter(|| {
                    for (query, result) in black_box(queries).iter().zip(&mut results) {
                        *result = black_box(&plan)
                            .evaluate(black_box(&ordinates), query, black_box(&mut work))
                            .unwrap();
                    }
                    black_box(&results);
                })
            });
            group.bench_function("retained_coefficients", |b| {
                b.iter(|| {
                    for (query, result) in black_box(queries).iter().zip(&mut results) {
                        *result = evaluate(black_box(&coefficients[..n]), query);
                    }
                    black_box(&results);
                })
            });
            group.bench_function("interpolate_and_horner", |b| {
                b.iter(|| {
                    black_box(&plan)
                        .interpolate(
                            black_box(&ordinates),
                            black_box(&mut output),
                            black_box(&mut work),
                        )
                        .unwrap();
                    for (query, result) in black_box(queries).iter().zip(&mut results) {
                        *result = evaluate(black_box(&output), query);
                    }
                    black_box(&results);
                })
            });
            group.bench_function("prepare_and_evaluate", |b| {
                b.iter(|| {
                    let plan = InterpolationPlan::prepare(
                        black_box(points),
                        black_box(&mut prepared),
                        black_box(&mut inversion),
                    )
                    .unwrap();
                    for (query, result) in black_box(queries).iter().zip(&mut results) {
                        *result = plan
                            .evaluate(black_box(&ordinates), query, black_box(&mut work))
                            .unwrap();
                    }
                    black_box(&results);
                })
            });
            group.bench_function("prepare_interpolate_and_horner", |b| {
                b.iter(|| {
                    let plan = InterpolationPlan::prepare(
                        black_box(points),
                        black_box(&mut prepared),
                        black_box(&mut inversion),
                    )
                    .unwrap();
                    plan.interpolate(
                        black_box(&ordinates),
                        black_box(&mut output),
                        black_box(&mut work),
                    )
                    .unwrap();
                    for (query, result) in black_box(queries).iter().zip(&mut results) {
                        *result = evaluate(black_box(&output), query);
                    }
                    black_box(&results);
                })
            });
            group.finish();
        }
        if n < 4 {
            continue;
        }
        let mut groups = vec![vec![PastaField::ZERO; n]; 4];
        let mut scratch = vec![PastaField::ZERO; 4 * n];
        for weights in &mut groups {
            InterpolationPlan::prepare_denominators(points, weights).unwrap();
        }
        batch_invert_groups(&mut groups, &mut scratch);
        assert!(groups.iter().all(|weights| {
            weights
                .iter()
                .zip(plan.weights())
                .all(|(a, b)| a.reduce() == b.reduce())
        }));
        let mut group = criterion.benchmark_group(format!("{field}/interpolation/{n}/grouped"));
        group.bench_function("separate", |b| {
            b.iter(|| {
                for weights in &mut groups {
                    black_box(
                        InterpolationPlan::prepare(
                            black_box(points),
                            black_box(weights),
                            black_box(&mut scratch[..n]),
                        )
                        .unwrap(),
                    );
                }
            })
        });
        group.bench_function("shared", |b| {
            b.iter(|| {
                let descriptors: [_; 4] = core::array::from_fn(|i| {
                    InterpolationPlan::prepare_denominators(
                        black_box(points),
                        black_box(&mut groups[i]),
                    )
                    .unwrap()
                });
                batch_invert_groups(black_box(&mut groups), black_box(&mut scratch));
                for (descriptor, weights) in descriptors.into_iter().zip(&groups) {
                    black_box(descriptor.complete(weights).unwrap());
                }
            })
        });
        group.finish();
    }
}
