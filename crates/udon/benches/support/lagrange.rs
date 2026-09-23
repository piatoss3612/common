use std::hint::black_box;

use criterion::Criterion;
use zakura_udon::{
    fft::Domain,
    field::{PastaField, PrimeModulus, batch_invert, batch_invert_groups},
};

pub fn benchmarks<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let point = super::inputs::<M>(1)[0];
    for (shift, coset) in [("subgroup", false), ("zeta", true)] {
        for (log, start, count) in [
            (0, 0, 1),
            (10, 512, 0),
            (10, 511, 1),
            (10, 508, 8),
            (10, 480, 32),
            (10, 0, 1024),
        ] {
            let subgroup = Domain::<M>::new(log).unwrap();
            let domain = if coset {
                subgroup.coset()
            } else {
                subgroup.subgroup()
            };
            let range = start..start + count;
            let mut output = vec![PastaField::ZERO; count];
            let mut scratch = vec![PastaField::ZERO; count];
            domain
                .evaluate_lagrange(&point, range.clone(), &mut output, &mut scratch)
                .unwrap();
            let expected = output.clone();
            let completion = domain
                .prepare_lagrange(&point, range.clone(), &mut output)
                .unwrap();
            batch_invert(&mut output, &mut scratch);
            completion.complete(&mut output).unwrap();
            assert!(
                output
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.reduce() == b.reduce())
            );
            let mut group = criterion.benchmark_group(format!(
                "{field}/lagrange/{}/{shift}/{start}_{count}",
                domain.size()
            ));
            group.bench_function("scaled", |b| {
                b.iter(|| {
                    black_box(domain)
                        .evaluate_lagrange(
                            black_box(&point),
                            range.clone(),
                            black_box(&mut output),
                            black_box(&mut scratch),
                        )
                        .unwrap();
                    black_box(&output);
                });
            });
            group.bench_function("separate_scale", |b| {
                b.iter(|| {
                    let completion = black_box(domain)
                        .prepare_lagrange(black_box(&point), range.clone(), black_box(&mut output))
                        .unwrap();
                    batch_invert(black_box(&mut output), black_box(&mut scratch));
                    completion.complete(black_box(&mut output)).unwrap();
                    black_box(&output);
                });
            });
            group.bench_function("prepare", |b| {
                b.iter(|| {
                    black_box(
                        black_box(domain)
                            .prepare_lagrange(
                                black_box(&point),
                                range.clone(),
                                black_box(&mut output),
                            )
                            .unwrap(),
                    );
                    black_box(&output);
                });
            });
            if count == 1024 {
                group.bench_function("bounded_32", |b| {
                    b.iter(|| {
                        black_box(domain)
                            .evaluate_lagrange(
                                black_box(&point),
                                range.clone(),
                                black_box(&mut output),
                                black_box(&mut scratch[..32]),
                            )
                            .unwrap();
                        black_box(&output);
                    });
                });
            }
            if count == 8 {
                let mut full_output = vec![PastaField::ZERO; domain.size()];
                let mut full_scratch = vec![PastaField::ZERO; domain.size()];
                domain
                    .evaluate_lagrange(
                        &point,
                        0..domain.size(),
                        &mut full_output,
                        &mut full_scratch,
                    )
                    .unwrap();
                assert!(
                    expected
                        .iter()
                        .zip(&full_output[range.clone()])
                        .all(|(a, b)| a.reduce() == b.reduce())
                );
                group.bench_function("full_domain", |b| {
                    b.iter(|| {
                        black_box(domain)
                            .evaluate_lagrange(
                                black_box(&point),
                                0..domain.size(),
                                black_box(&mut full_output),
                                black_box(&mut full_scratch),
                            )
                            .unwrap();
                        black_box(&full_output[range.clone()]);
                    });
                });
            }
            if count == 32 {
                for (name, index) in [("node_inside", start + 7), ("node_outside", start - 1)] {
                    let node = domain.shift().mul(&subgroup.root().pow_u64(index as u64));
                    domain
                        .evaluate_lagrange(&node, range.clone(), &mut output, &mut scratch)
                        .unwrap();
                    assert!(
                        output
                            .iter()
                            .enumerate()
                            .all(|(i, value)| if start + i == index {
                                value.is_one()
                            } else {
                                value.is_zero()
                            })
                    );
                    group.bench_function(name, |b| {
                        b.iter(|| {
                            black_box(domain)
                                .evaluate_lagrange(
                                    black_box(&node),
                                    range.clone(),
                                    black_box(&mut output),
                                    black_box(&mut scratch),
                                )
                                .unwrap();
                            black_box(&output);
                        });
                    });
                }
            }
            group.finish();
        }
    }

    let domain = Domain::<M>::new(10).unwrap().coset();
    let points: [_; 4] = super::inputs::<M>(4).try_into().unwrap();
    for count in [8, 256] {
        let range = 256..256 + count;
        let mut values = core::array::from_fn::<_, 4, _>(|_| vec![PastaField::ZERO; count]);
        let mut scratch = vec![PastaField::ZERO; 4 * count];
        let completions = core::array::from_fn::<_, 4, _>(|i| {
            domain
                .prepare_lagrange(&points[i], range.clone(), &mut values[i])
                .unwrap()
        });
        batch_invert_groups(&mut values, &mut scratch);
        for (completion, values) in completions.into_iter().zip(&mut values) {
            completion.complete(values).unwrap();
        }
        let expected = values.clone();
        for (point, values) in points.iter().zip(&mut values) {
            domain
                .evaluate_lagrange(point, range.clone(), values, &mut scratch[..count])
                .unwrap();
        }
        assert!(
            values
                .iter()
                .flatten()
                .zip(expected.iter().flatten())
                .all(|(a, b)| a.reduce() == b.reduce())
        );
        let mut group = criterion.benchmark_group(format!("{field}/lagrange/grouped/4_{count}"));
        group.bench_function("shared", |b| {
            b.iter(|| {
                let completions = core::array::from_fn::<_, 4, _>(|i| {
                    black_box(domain)
                        .prepare_lagrange(
                            black_box(&points[i]),
                            range.clone(),
                            black_box(&mut values[i]),
                        )
                        .unwrap()
                });
                batch_invert_groups(black_box(&mut values), black_box(&mut scratch));
                for (completion, values) in completions.into_iter().zip(&mut values) {
                    completion.complete(black_box(values)).unwrap();
                }
                black_box(&values);
            });
        });
        group.bench_function("separate", |b| {
            b.iter(|| {
                for (point, values) in points.iter().zip(&mut values) {
                    black_box(domain)
                        .evaluate_lagrange(
                            black_box(point),
                            range.clone(),
                            black_box(values),
                            black_box(&mut scratch[..count]),
                        )
                        .unwrap();
                }
                black_box(&values);
            });
        });
        group.finish();
    }
}
