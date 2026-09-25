use super::Pool;
use criterion::{Bencher, BenchmarkId, Criterion, Throughput};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};
use zakura_udon::{
    curve::{
        AffinePoint, EisensteinScalar, EisensteinTableBatch, PastaCurve, Point, ProjectivePoint,
        batch_normalize,
    },
    exec::{Executor, SerialExecutor, TaskBudget},
    field::PastaField,
};

// Keep pool entry, allocation, and input restoration outside arithmetic timing.
fn measure<C: PastaCurve>(
    b: &mut Bencher,
    inputs: &[ProjectivePoint<C>],
    pool: Option<&rayon::ThreadPool>,
    mut operation: impl FnMut(&mut [ProjectivePoint<C>]) + Send,
) {
    b.iter_custom(|iterations| {
        let mut run = || {
            let mut points = inputs.to_vec();
            let mut elapsed = Duration::ZERO;
            for _ in 0..iterations {
                points.copy_from_slice(inputs);
                let start = Instant::now();
                operation(&mut points);
                elapsed += start.elapsed();
            }
            elapsed
        };
        if let Some(pool) = pool {
            pool.install(run)
        } else {
            run()
        }
    });
}

fn run<C: PastaCurve, X: Executor>(
    c: &mut Criterion,
    curve: &str,
    bases: &[AffinePoint<C>],
    scalar: &PastaField<C::Scalar>,
    workers: usize,
    executor: &X,
    pool: Option<&rayon::ThreadPool>,
) {
    let budget = TaskBudget::new(workers).unwrap();
    let prepared = EisensteinScalar::new(scalar);
    let mut group = c.benchmark_group(format!("{curve}/projective_batch/workers_{workers}"));
    for n in [32, 64, 128, 512, 2048] {
        let inputs: Vec<_> = bases[..n]
            .iter()
            .map(|p| p.to_projective().double())
            .collect();
        let expected: Vec<_> = inputs.iter().map(|p| p.mul(scalar)).collect();
        let r = EisensteinTableBatch::<C>::requirements(n).unwrap();
        let mut affine = vec![Point::IDENTITY; n];
        let mut nonidentity = vec![AffinePoint::GENERATOR; n];
        let mut entries = vec![AffinePoint::GENERATOR; r.table_entries];
        let mut projective = vec![ProjectivePoint::IDENTITY; r.projective_scratch];
        let mut fields = vec![
            PastaField::ZERO;
            r.field_scratch
                .max(EisensteinTableBatch::<C>::multiplication_scratch(n).unwrap())
                .max(n)
        ];
        batch_normalize(&inputs, &mut affine, &mut fields);
        for (a, p) in nonidentity.iter_mut().zip(&affine) {
            *a = *p.as_affine().unwrap();
        }
        let tables = EisensteinTableBatch::prepare(
            &nonidentity,
            &mut entries,
            &mut projective,
            &mut fields,
            budget,
            executor,
        );
        let mut checked = inputs.clone();
        tables.mul_prepared(&prepared, &mut checked, &mut fields, budget, executor);
        assert_eq!(checked, expected);
        batch_normalize(&checked, &mut affine, &mut fields);
        for (actual, expected) in affine.iter().zip(&expected) {
            assert_eq!(actual.to_projective(), *expected);
        }
        group.throughput(Throughput::Elements(n as u64));
        group.bench_with_input(BenchmarkId::new("mul_prepared", n), &n, |b, _| {
            measure(b, &inputs, pool, |points| {
                tables.mul_prepared(black_box(&prepared), points, &mut fields, budget, executor);
                black_box(points);
            });
        });
        group.bench_with_input(BenchmarkId::new("normalize_prepare", n), &n, |b, _| {
            measure(b, &inputs, pool, |points| {
                batch_normalize(black_box(points), &mut affine, &mut fields);
                for (a, p) in nonidentity.iter_mut().zip(&affine) {
                    *a = *p.as_affine().unwrap();
                }
                black_box(EisensteinTableBatch::prepare(
                    &nonidentity,
                    &mut entries,
                    &mut projective,
                    &mut fields,
                    budget,
                    executor,
                ));
            });
        });
        for normalized in [false, true] {
            let output = if normalized { "affine" } else { "projective" };
            group.bench_with_input(
                BenchmarkId::new(format!("normalized_tables_{output}"), n),
                &n,
                |b, _| {
                    measure(b, &inputs, pool, |points| {
                        batch_normalize(black_box(points), &mut affine, &mut fields);
                        for (a, p) in nonidentity.iter_mut().zip(&affine) {
                            *a = *p.as_affine().unwrap();
                        }
                        let tables = EisensteinTableBatch::prepare(
                            &nonidentity,
                            &mut entries,
                            &mut projective,
                            &mut fields,
                            budget,
                            executor,
                        );
                        tables.mul(black_box(scalar), points, &mut fields, budget, executor);
                        if normalized {
                            batch_normalize(points, &mut affine, &mut fields);
                        }
                        black_box((&*points, &affine));
                    });
                },
            );
        }
        if workers == 1 {
            group.bench_with_input(BenchmarkId::new("individual_projective", n), &n, |b, _| {
                measure(b, &inputs, pool, |points| {
                    for p in points.iter_mut() {
                        *p = black_box(&*p).mul(black_box(scalar));
                    }
                    black_box(points);
                });
            });
        }
    }
    group.finish();
}

pub(super) fn bench<C: PastaCurve>(
    c: &mut Criterion,
    curve: &str,
    bases: &[AffinePoint<C>],
    scalar: &PastaField<C::Scalar>,
) {
    run(c, curve, bases, scalar, 1, &SerialExecutor, None);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    run(c, curve, bases, scalar, 4, &Pool, Some(&pool));
}
