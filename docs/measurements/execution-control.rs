//! Public synchronous API control, usable before and after the run migration.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::{hint::black_box, time::Duration};
use zakura_udon::{
    curve::{
        AffinePoint, Pallas, ProjectivePoint,
        msm::{Bases, ExecutionOptions, Input, Requirements, ScalarStorage, Scratch},
    },
    exec::{Executor, TaskBudget},
    fft::{Direction, Domain, ExecutionOptions as FftOptions, Plan},
    field::{Fp, Fq},
};

struct Pool;
impl Executor for Pool {
    fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
    where
        L: FnOnce() -> A + Send,
        R: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        rayon::join(left, right)
    }
}

struct Buffers {
    scalars: Vec<ScalarStorage<Pallas>>,
    digits: Vec<u8>,
    affine: Vec<AffinePoint<Pallas>>,
    projective: Vec<ProjectivePoint<Pallas>>,
    field: Vec<Fp>,
    indices: Vec<usize>,
}

impl Buffers {
    fn new(requirements: impl Iterator<Item = Requirements>) -> Self {
        let mut counts = [0; 6];
        for r in requirements {
            for (max, count) in counts.iter_mut().zip([
                r.scalars(),
                r.digits(),
                r.affine(),
                r.projective(),
                r.field(),
                r.indices(),
            ]) {
                *max = (*max).max(count);
            }
        }
        Self {
            scalars: vec![ScalarStorage::ZERO; counts[0]],
            digits: vec![0; counts[1]],
            affine: vec![AffinePoint::GENERATOR; counts[2]],
            projective: vec![ProjectivePoint::IDENTITY; counts[3]],
            field: vec![Fp::ZERO; counts[4]],
            indices: vec![0; counts[5]],
        }
    }

    fn borrow(&mut self) -> Scratch<'_, Pallas> {
        Scratch::new(
            &mut self.scalars,
            &mut self.digits,
            &mut self.affine,
            &mut self.projective,
            &mut self.field,
            &mut self.indices,
        )
    }
}

fn isolated(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution/control");
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));
    for threads in [1, 4, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for terms in [32, 1024, 8192] {
            let bases = vec![AffinePoint::<Pallas>::GENERATOR; terms];
            let scalars: Vec<_> = (0..terms)
                .map(|i| Fq::from_u64(i as u64 + 2).invert().unwrap())
                .collect();
            let input = Input::new(Bases::Affine(&bases), &scalars).unwrap();
            let options =
                ExecutionOptions::SERIAL.with_task_budget(TaskBudget::new(threads).unwrap());
            let mut buffers = Buffers::new(core::iter::once(input.requirements(options).unwrap()));
            group.bench_function(BenchmarkId::new(format!("msm/{terms}"), threads), |b| {
                b.iter_custom(|iterations| {
                    pool.install(|| {
                        let start = std::time::Instant::now();
                        for _ in 0..iterations {
                            black_box(input.execute(options, &Pool, buffers.borrow()).unwrap());
                        }
                        start.elapsed()
                    })
                })
            });
        }
        for size in [64, 2048, 16384] {
            let domain = Domain::for_size(size).unwrap();
            for coset in [false, true] {
                let plan = Plan::without_tables(
                    domain
                        .coset(if coset { Fp::from_u64(7) } else { Fp::ONE })
                        .unwrap(),
                );
                let options = FftOptions {
                    max_tasks: threads,
                    ..FftOptions::default()
                };
                let fields = plan.scratch_requirements(options).unwrap().field_elements;
                let mut scratch = vec![Fp::ZERO; fields];
                for direction in [Direction::Forward, Direction::Inverse] {
                    let input: Vec<_> = (0..size).map(|i| Fp::from_u64(i as u64 + 1)).collect();
                    let mut values = input.clone();
                    group.bench_function(
                        BenchmarkId::new(format!("fft/{size}/{coset}/{direction:?}"), threads),
                        |b| {
                            b.iter_custom(|iterations| {
                                pool.install(|| {
                                    let start = std::time::Instant::now();
                                    for _ in 0..iterations {
                                        values.copy_from_slice(&input);
                                        if direction == Direction::Forward {
                                            plan.forward(&mut values, options, &Pool, &mut scratch)
                                                .unwrap();
                                        } else {
                                            plan.inverse(&mut values, options, &Pool, &mut scratch)
                                                .unwrap();
                                        }
                                        black_box(&values);
                                    }
                                    start.elapsed()
                                })
                            })
                        },
                    );
                }
            }
        }
    }
    group.finish();
}

criterion_group!(benches, isolated);
criterion_main!(benches);
