//! Fixed-arena mixed workload. Run with `cargo bench --bench execution`.

#[path = "support/msm.rs"]
mod bench_msm;

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use zakura_udon::{
    curve::{
        AffinePoint, Pallas, ProjectivePoint,
        msm::{self, Bases, Input, Requirements, ScalarStorage, Scratch},
    },
    exec::{ExecutionOptions, Executor, TaskBudget},
    fft::{Direction, Domain, StorageLayout, Transform, TransformRequest, run::FftPlan},
    field::{CanonicalUint, Fp, Fq},
};

#[path = "../tests/support/msm_run.rs"]
#[allow(dead_code)]
mod msm_run;

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

    fn bytes(&self) -> usize {
        self.scalars.capacity() * size_of::<ScalarStorage<Pallas>>()
            + self.digits.capacity()
            + self.affine.capacity() * size_of::<AffinePoint<Pallas>>()
            + self.projective.capacity() * size_of::<ProjectivePoint<Pallas>>()
            + self.field.capacity() * size_of::<Fp>()
            + self.indices.capacity() * size_of::<usize>()
            + size_of::<Self>()
    }
}

#[derive(Clone, Copy, Debug)]
enum Mode {
    Partition,
    Independent,
    Batched,
    UnlimitedCap,
    Capped,
}

fn mixed(c: &mut Criterion) {
    const TERMS: usize = 8192;
    const CEILING: usize = 16 * 1024 * 1024;
    let bases: Vec<_> = (0..TERMS)
        .scan(ProjectivePoint::<Pallas>::GENERATOR, |point, _| {
            let result = *point.to_point().as_affine().unwrap();
            *point = point.add(&ProjectivePoint::GENERATOR);
            Some(result)
        })
        .collect();
    let mut seed = 0x243f_6a88_85a3_08d3_u64;
    let scalars: Vec<_> = (0..TERMS)
        .map(|_| {
            let mut limbs = core::array::from_fn(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            });
            limbs[3] &= (1 << 62) - 1;
            Fq::from_canonical_uint(CanonicalUint::from_limbs(limbs)).unwrap()
        })
        .collect();
    let rounds: Vec<_> = (0..11)
        .map(|round| {
            let n = TERMS >> round;
            [
                Input::new(Bases::Affine(&bases[..n]), &scalars[..n]),
                Input::new(
                    Bases::Affine(&bases[..(n / 8).max(1)]),
                    &scalars[..(n / 8).max(1)],
                ),
            ]
        })
        .collect();
    let coefficients: Vec<_> = (0..16384).map(|i| Fp::from_u64(i * i + 1)).collect();
    let plans: Vec<_> = (11..=14)
        .map(|log| Transform::new(Domain::new(log).unwrap().subgroup()))
        .collect();
    let mut group = c.benchmark_group("execution/synchronous/shrinking");
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(300));
    group.measurement_time(Duration::from_secs(1));
    for threads in [1, 3, 4, 16] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for mode in [
            Mode::Partition,
            Mode::Independent,
            Mode::Batched,
            Mode::UnlimitedCap,
            Mode::Capped,
        ] {
            let tasks = if matches!(mode, Mode::Partition) {
                (threads / 4).max(1)
            } else {
                threads
            };
            let mut options =
                ExecutionOptions::default().with_task_budget(TaskBudget::new(tasks).unwrap());
            if matches!(mode, Mode::UnlimitedCap) {
                options = options.with_memory_limit(usize::MAX);
            } else if matches!(mode, Mode::Capped) {
                options = options.with_memory_limit(1024 * 1024);
            }
            let independent = matches!(mode, Mode::Partition | Mode::Independent);
            let mut first = Buffers::new(rounds.iter().map(|inputs| {
                bench_msm::batch_requirements(
                    if independent { &inputs[..1] } else { inputs },
                    options,
                )
                .unwrap()
            }));
            let mut second = Buffers::new(
                rounds
                    .iter()
                    .filter(|_| independent)
                    .map(|inputs| inputs[1].requirements(options).unwrap()),
            );
            let fft_options =
                ExecutionOptions::default().with_task_budget(TaskBudget::new(tasks).unwrap());
            let fields = plans[3].scratch_requirements(fft_options).unwrap();
            let mut fft_scratch = [vec![Fp::ZERO; fields], vec![Fp::ZERO; fields]];
            let mut values = [vec![Fp::ZERO; 16384], vec![Fp::ZERO; 2048]];
            let mut app = vec![0_u64; 8192];
            let bytes = first.bytes()
                + second.bytes()
                + fft_scratch
                    .iter()
                    .chain(values.iter())
                    .map(|v| v.capacity() * size_of::<Fp>())
                    .sum::<usize>()
                + app.capacity() * size_of::<u64>();
            if bytes > CEILING {
                eprintln!(
                    "execution baseline {mode:?}/{threads}: inadmissible, {bytes} exceeds {CEILING} bytes"
                );
                continue;
            }
            eprintln!(
                "execution baseline {mode:?}/{threads}: provisioned arena {bytes}/{CEILING} bytes"
            );
            group.bench_function(BenchmarkId::new(format!("{mode:?}"), threads), |b| {
                b.iter(|| {
                    pool.install(|| {
                        for _ in 0..2 {
                            for (round, inputs) in rounds.iter().enumerate() {
                                let index = 3 - round.min(3);
                                let size = plans[index].domain().size();
                                let mut outputs = [ProjectivePoint::IDENTITY; 2];
                                Pool.join(
                                    || {
                                        if independent {
                                            let (left, right) = outputs.split_at_mut(1);
                                            Pool.join(
                                                || {
                                                    left[0] = inputs[0]
                                                        .execute(options, &Pool, first.borrow())
                                                        .unwrap()
                                                },
                                                || {
                                                    right[0] = inputs[1]
                                                        .execute(options, &Pool, second.borrow())
                                                        .unwrap()
                                                },
                                            );
                                        } else {
                                            bench_msm::execute_batch(
                                                inputs,
                                                &mut outputs,
                                                options,
                                                &Pool,
                                                first.borrow(),
                                            )
                                            .unwrap();
                                        }
                                    },
                                    || {
                                        let [large, small] = &mut values;
                                        let [large_scratch, small_scratch] = &mut fft_scratch;
                                        Pool.join(
                                            || {
                                                large[..size]
                                                    .copy_from_slice(&coefficients[..size]);
                                                plans[index]
                                                    .forward(
                                                        &mut large[..size],
                                                        fft_options,
                                                        &Pool,
                                                        large_scratch,
                                                    )
                                                    .unwrap();
                                            },
                                            || {
                                                Pool.join(
                                                    || {
                                                        small
                                                            .copy_from_slice(&coefficients[..2048]);
                                                        plans[0]
                                                            .forward(
                                                                small,
                                                                fft_options,
                                                                &Pool,
                                                                small_scratch,
                                                            )
                                                            .unwrap();
                                                    },
                                                    || {
                                                        for (i, value) in app.iter_mut().enumerate()
                                                        {
                                                            *value = value
                                                                .wrapping_add(
                                                                    i as u64 + round as u64,
                                                                )
                                                                .rotate_left(17);
                                                        }
                                                    },
                                                );
                                            },
                                        );
                                    },
                                );
                                // This consumer and challenge fence are part of each round.
                                black_box((&outputs, &values, &app));
                            }
                        }
                    });
                })
            });
        }
    }
    group.finish();
}

fn isolated(c: &mut Criterion) {
    let mut group = c.benchmark_group("execution/isolated");
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
                .map(|i| <Fq>::from_u64(i as u64 + 2).invert().unwrap())
                .collect();
            let input = Input::new(Bases::Affine(&bases), &scalars);
            let options =
                ExecutionOptions::default().with_task_budget(TaskBudget::new(threads).unwrap());
            let plan = msm::run::MsmPlan::new(terms, options).unwrap();
            let mut synchronous =
                Buffers::new(core::iter::once(input.requirements(options).unwrap()));
            let mut bounded = Buffers::new(core::iter::once(plan.requirements()));
            for mode in ["synchronous", "runs"] {
                group.bench_function(
                    BenchmarkId::new(format!("msm/{terms}/{mode}"), threads),
                    |b| {
                        b.iter_custom(|iterations| {
                            pool.install(|| {
                                let start = std::time::Instant::now();
                                for _ in 0..iterations {
                                    black_box(match mode {
                                        "synchronous" => input
                                            .execute(options, &Pool, synchronous.borrow())
                                            .unwrap(),
                                        "runs" => plan.execute(input, &Pool, bounded.borrow()),
                                        _ => unreachable!(),
                                    });
                                }
                                start.elapsed()
                            })
                        })
                    },
                );
            }
        }
        for size in [64, 2048, 16384] {
            let domain = Domain::for_size(size).unwrap();
            for coset in [false, true] {
                let plan = Transform::new(if coset {
                    domain.coset()
                } else {
                    domain.subgroup()
                });
                let options =
                    ExecutionOptions::default().with_task_budget(TaskBudget::new(threads).unwrap());
                let fields = plan.scratch_requirements(options).unwrap();
                let mut synchronous_scratch = vec![Fp::ZERO; fields];
                for direction in [Direction::Forward, Direction::Inverse] {
                    let planned = FftPlan::new(
                        plan,
                        TransformRequest::new(direction),
                        StorageLayout::Contiguous,
                        options,
                    )
                    .unwrap();
                    let mut bounded_scratch = vec![Fp::ZERO; planned.retained_fields()];
                    let input: Vec<_> = (0..size).map(|i| Fp::from_u64(i as u64 + 1)).collect();
                    let mut values = input.clone();
                    for mode in ["synchronous", "planned"] {
                        group.bench_function(
                            BenchmarkId::new(
                                format!("fft/{size}/{coset}/{direction:?}/{mode}"),
                                threads,
                            ),
                            |b| {
                                b.iter_custom(|iterations| {
                                    pool.install(|| {
                                        let start = std::time::Instant::now();
                                        for _ in 0..iterations {
                                            values.copy_from_slice(&input);
                                            match mode {
                                                "synchronous" => {
                                                    if direction == Direction::Forward {
                                                        plan.forward(
                                                            &mut values,
                                                            options,
                                                            &Pool,
                                                            &mut synchronous_scratch,
                                                        )
                                                        .unwrap();
                                                    } else {
                                                        plan.inverse(
                                                            &mut values,
                                                            options,
                                                            &Pool,
                                                            &mut synchronous_scratch,
                                                        )
                                                        .unwrap();
                                                    }
                                                }
                                                _ => planned.execute(
                                                    None,
                                                    &mut values,
                                                    None,
                                                    &mut bounded_scratch,
                                                    &Pool,
                                                ),
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
    }
    group.finish();
}

criterion_group!(benches, mixed, isolated);
criterion_main!(benches);
