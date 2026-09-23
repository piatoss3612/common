use super::{Buffers, Pool};
use criterion::{Bencher, BenchmarkId, Criterion, Throughput};
use std::{hint::black_box, time::Instant};
use zakura_udon::{
    curve::{
        AffinePoint, PastaCurve, ProjectivePoint,
        msm::{Bases, Input, PreparedScalars, ScalarStorage, SharedScalarInput},
    },
    exec::{ExecutionOptions, Executor, SerialExecutor, TaskBudget},
    field::PastaField,
};

pub(super) fn measure(
    b: &mut Bencher,
    pool: Option<&rayon::ThreadPool>,
    mut run: impl FnMut() + Send,
) {
    b.iter_custom(|iterations| {
        let work = || {
            let start = Instant::now();
            for _ in 0..iterations {
                run();
            }
            start.elapsed()
        };
        if let Some(pool) = pool {
            pool.install(work)
        } else {
            let mut work = work;
            work()
        }
    });
}

pub(super) fn bench<C: PastaCurve>(
    c: &mut Criterion,
    curve: &str,
    bases: &[AffinePoint<C>],
    full: &[PastaField<C::Scalar>],
) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    run(c, curve, bases, full, 1, &SerialExecutor, None);
    run(c, curve, bases, full, 4, &Pool, Some(&pool));
}

fn run<C: PastaCurve, X: Executor>(
    c: &mut Criterion,
    curve: &str,
    bases: &[AffinePoint<C>],
    full: &[PastaField<C::Scalar>],
    workers: usize,
    executor: &X,
    pool: Option<&rayon::ThreadPool>,
) {
    let mut group = c.benchmark_group(format!("{curve}/shared_scalars/workers_{workers}"));
    for n in [8, 32, 128, 512, 2048] {
        for outputs in [2, 8] {
            let row_major: Vec<_> = (0..n * outputs).map(|i| bases[i % bases.len()]).collect();
            let term_major: Vec<_> = (0..n * outputs)
                .map(|i| row_major[(i % outputs) * n + i / outputs])
                .collect();
            let short: Vec<_> = (0..n)
                .map(|i| PastaField::from_u64((i % 17) as u64))
                .collect();
            for (corpus, scalars) in [("full", &full[..n]), ("short", &short)] {
                if workers == 1 && outputs == 2 {
                    let mut storage = vec![ScalarStorage::<C>::ZERO; n];
                    group.throughput(Throughput::Elements(n as u64));
                    group.bench_function(BenchmarkId::new(format!("prepare/{corpus}"), n), |b| {
                        b.iter(|| {
                            black_box(PreparedScalars::prepare(
                                black_box(scalars),
                                &mut storage,
                                TaskBudget::SERIAL,
                                &SerialExecutor,
                            ));
                        });
                    });
                }
                let mut records = vec![ScalarStorage::ZERO; n];
                let prepared = PreparedScalars::prepare(
                    scalars,
                    &mut records,
                    TaskBudget::SERIAL,
                    &SerialExecutor,
                );
                let rows: Vec<_> = row_major
                    .chunks(n)
                    .map(|row| Input::new_prepared(Bases::Affine(row), prepared))
                    .collect();
                let expected: Vec<_> = (0..outputs)
                    .map(|j| {
                        let scalar = (0..n).fold(PastaField::ZERO, |sum, i| {
                            sum.add(&scalars[i].mul(&full[(j * n + i) % bases.len()]))
                        });
                        AffinePoint::<C>::GENERATOR.mul_projective(&scalar)
                    })
                    .collect();
                for (capacity, limit) in [("all", usize::MAX), ("64k", 65536), ("96k", 98304)] {
                    let options = ExecutionOptions::DEFAULT
                        .with_task_budget(TaskBudget::new(workers).unwrap())
                        .with_memory_limit(limit);
                    let mut independent = Buffers::new(rows[0].requirements(options).unwrap());
                    let mut output = vec![ProjectivePoint::IDENTITY; outputs];
                    for (input, output) in rows.iter().zip(&mut output) {
                        *output = input
                            .execute(options, &SerialExecutor, independent.borrow())
                            .unwrap();
                    }
                    assert_eq!(output, expected);
                    group.throughput(Throughput::Elements((n * outputs) as u64));
                    let case = format!("{corpus}/cap_{capacity}/outputs_{outputs}/{n}");
                    group.bench_function(BenchmarkId::new("separate", &case), |b| {
                        measure(b, pool, || {
                            for (input, output) in rows.iter().zip(&mut output) {
                                *output = black_box(input)
                                    .execute(options, executor, independent.borrow())
                                    .unwrap();
                            }
                            black_box(&output);
                        })
                    });
                    for (layout, storage, output_stride, term_stride) in [
                        ("rows", &row_major, n, 1),
                        ("terms", &term_major, 1, outputs),
                    ] {
                        let matrix = SharedScalarInput::new(
                            Bases::Affine(storage),
                            prepared,
                            outputs,
                            output_stride,
                            term_stride,
                        )
                        .unwrap();
                        let mut buffers = Buffers::new(matrix.requirements(options).unwrap());
                        matrix
                            .execute(&mut output, options, &SerialExecutor, buffers.borrow())
                            .unwrap();
                        assert_eq!(output, expected);
                        group.bench_function(
                            BenchmarkId::new(format!("matrix_{layout}"), &case),
                            |b| {
                                measure(b, pool, || {
                                    black_box(matrix)
                                        .execute(&mut output, options, executor, buffers.borrow())
                                        .unwrap();
                                    black_box(&output);
                                })
                            },
                        );
                    }
                }
            }
        }
    }
    group.finish();
}
