use std::{
    hint::black_box,
    num::NonZeroUsize,
    time::{Duration, Instant},
};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use zakura_udon::{
    curve::{
        AffinePoint, CurveTableEntry, EisensteinScalar, EisensteinTableBatch, Pallas, PastaCurve,
        Point, PreparedAffinePoint, ProjectivePoint, Vesta,
        msm::{self, Bases, ExecutionOptions, Input, PreparedScalars, Requirements, Scratch},
    },
    exec::{Executor, SerialExecutor, TaskBudget},
    field::{CanonicalUint, PastaField, PrimeModulus},
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

struct Buffers<C: PastaCurve> {
    digits: Vec<u8>,
    affine: Vec<AffinePoint<C>>,
    projective: Vec<ProjectivePoint<C>>,
    field: Vec<PastaField<C::Base>>,
    indices: Vec<usize>,
}
impl<C: PastaCurve> Buffers<C> {
    fn new(r: Requirements) -> Self {
        Self {
            digits: vec![0; r.digits],
            affine: vec![AffinePoint::GENERATOR; r.affine],
            projective: vec![ProjectivePoint::IDENTITY; r.projective],
            field: vec![PastaField::ZERO; r.field],
            indices: vec![0; r.indices],
        }
    }
    fn borrow(&mut self) -> Scratch<'_, C> {
        Scratch {
            digits: &mut self.digits,
            affine: &mut self.affine,
            projective: &mut self.projective,
            field: &mut self.field,
            indices: &mut self.indices,
        }
    }
}

fn values<M: PrimeModulus>(n: usize) -> Vec<PastaField<M>> {
    let mut seed = 0x243f_6a88_85a3_08d3_u64;
    (0..n)
        .map(|_| {
            let mut limbs = std::array::from_fn(|_| {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            });
            limbs[3] &= (1 << 62) - 1;
            PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs)).unwrap()
        })
        .collect()
}

fn compact<C: PastaCurve, E: CurveTableEntry<C>>(
    c: &mut Criterion,
    curve: &str,
    layout: &str,
    bases: &[AffinePoint<C>],
    scalar: &PastaField<C::Scalar>,
) {
    let mut group = c.benchmark_group(format!("{curve}/{layout}"));
    let prepared = EisensteinScalar::new(scalar);
    for n in [1, 8, 32, 64, 128, 512] {
        let r = EisensteinTableBatch::<C, E>::requirements(n).unwrap();
        let mut entries = vec![E::from_affine(&bases[0]); r.table_entries];
        let mut projective = vec![ProjectivePoint::IDENTITY; r.projective_scratch];
        let mut field = vec![PastaField::ZERO; r.field_scratch];
        group.throughput(Throughput::Elements(n as u64));
        let case = format!("full/serial/cap_all/{n}");
        group.bench_with_input(BenchmarkId::new("prepare_batch", &case), &n, |b, &n| {
            b.iter(|| {
                let tables = EisensteinTableBatch::prepare(
                    black_box(&bases[..n]),
                    &mut entries,
                    &mut projective,
                    &mut field,
                    TaskBudget::SERIAL,
                    &SerialExecutor,
                )
                .unwrap();
                black_box(tables);
            })
        });
        let tables = EisensteinTableBatch::prepare(
            &bases[..n],
            &mut entries,
            &mut projective,
            &mut field,
            TaskBudget::SERIAL,
            &SerialExecutor,
        )
        .unwrap();
        let mut output = vec![ProjectivePoint::IDENTITY; n];
        let mut scratch = vec![
            PastaField::ZERO;
            EisensteinTableBatch::<C, E>::multiplication_scratch(n).unwrap()
        ];
        tables
            .mul_prepared(
                &prepared,
                &mut output,
                &mut scratch,
                TaskBudget::SERIAL,
                &SerialExecutor,
            )
            .unwrap();
        for (i, result) in output.iter().enumerate() {
            assert_eq!(*result, bases[i].mul_projective(scalar));
            assert_eq!(*result, tables.get(i).unwrap().mul_prepared(&prepared));
        }
        group.bench_with_input(BenchmarkId::new("mul", &case), &n, |b, &n| {
            b.iter(|| {
                for (i, result) in output.iter_mut().enumerate().take(n) {
                    *result = black_box(tables.get(i).unwrap()).mul(black_box(scalar));
                }
                black_box(&output);
            })
        });
        group.bench_with_input(BenchmarkId::new("mul_prepared", &case), &n, |b, &n| {
            b.iter(|| {
                for (i, result) in output.iter_mut().enumerate().take(n) {
                    *result = black_box(tables.get(i).unwrap()).mul_prepared(black_box(&prepared));
                }
                black_box(&output);
            })
        });
        group.bench_with_input(BenchmarkId::new("mul_same_scalar", &case), &n, |b, _| {
            b.iter(|| {
                tables
                    .mul_prepared(
                        black_box(&prepared),
                        &mut output,
                        &mut scratch,
                        TaskBudget::SERIAL,
                        &SerialExecutor,
                    )
                    .unwrap();
                black_box(&output);
            })
        });
    }
    group.finish();
}

fn curve<C: PastaCurve>(c: &mut Criterion, curve: &str) {
    let full = values::<C::Scalar>(4096);
    let short: Vec<_> = (0..4096)
        .map(|i| PastaField::from_u64((i * 137) as u64))
        .collect();
    let affine: Vec<_> = full
        .iter()
        .map(|k| {
            *AffinePoint::<C>::GENERATOR
                .mul_projective(k)
                .to_point()
                .as_affine()
                .unwrap()
        })
        .collect();
    let prepared: Vec<_> = affine
        .iter()
        .map(PreparedAffinePoint::from_affine)
        .collect();
    let points: Vec<_> = affine
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i % 11 == 0 {
                Point::IDENTITY
            } else {
                p.to_point()
            }
        })
        .collect();
    let indices: Vec<_> = (0..4096).map(|i| ((i * 13) % 4096) as u32).collect();
    corpus(c, curve, &full, &affine);
    compact::<C, AffinePoint<C>>(c, curve, "eisenstein", &affine, &full[0]);
    compact::<C, PreparedAffinePoint<C>>(c, curve, "eisenstein_cached", &affine, &full[0]);

    for (access, indexed) in [("dense", false), ("indexed", true)] {
        let mut group = c.benchmark_group(format!("{curve}/msm/{access}"));
        for n in [8, 31, 32, 64, 128, 255, 256, 512, 1024, 4096] {
            group.throughput(Throughput::Elements(n as u64));
            for (layout, bases) in [
                ("affine", Bases::Affine(&affine[..n])),
                ("cached", Bases::Prepared(&prepared[..n])),
                ("points", Bases::Points(&points[..n])),
            ] {
                let ix: Vec<_> = indices[..n].iter().map(|i| i % n as u32).collect();
                for (corpus, scalars) in [("full", &full[..n]), ("short", &short[..n])] {
                    if corpus == "short" && layout != "affine" {
                        continue;
                    }
                    let input = if indexed {
                        Input::indexed(bases, &ix, scalars)
                    } else {
                        Input::new(bases, scalars)
                    }
                    .unwrap();
                    let mut buffers =
                        Buffers::new(input.requirements(ExecutionOptions::SERIAL).unwrap());
                    // The fixture bases are known generator multiples. Check
                    // their scalar inner product independently of MSM recoding.
                    let expected =
                        scalars
                            .iter()
                            .enumerate()
                            .fold(PastaField::ZERO, |sum, (i, scalar)| {
                                let base = if indexed { ix[i] as usize } else { i };
                                if layout == "points" && base % 11 == 0 {
                                    sum
                                } else {
                                    sum.add(&full[base].mul(scalar))
                                }
                            });
                    assert_eq!(
                        input
                            .execute(ExecutionOptions::SERIAL, &SerialExecutor, buffers.borrow())
                            .unwrap(),
                        AffinePoint::<C>::GENERATOR.mul_projective(&expected)
                    );
                    let case = format!("{layout}/{corpus}/serial/cap_all");
                    group.bench_with_input(BenchmarkId::new(case, n), &input, |b, input| {
                        b.iter(|| {
                            black_box(input)
                                .execute(
                                    ExecutionOptions::SERIAL,
                                    &SerialExecutor,
                                    buffers.borrow(),
                                )
                                .unwrap()
                        })
                    });
                }
            }
        }
        group.finish();
    }

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for family in ["ipa", "commitments"] {
        let mut group = c.benchmark_group(format!("{curve}/msm_batch/{family}"));
        for n in [128, 1024] {
            let sizes = if family == "ipa" {
                vec![n, n]
            } else {
                vec![n, n / 2, n / 4, 17]
            };
            let jobs: Vec<_> = sizes
                .iter()
                .map(|&n| {
                    if family == "ipa" {
                        Input::indexed(Bases::Prepared(&prepared), &indices[..n], &full[..n])
                            .unwrap()
                    } else {
                        Input::new(Bases::Prepared(&prepared[..n]), &full[..n]).unwrap()
                    }
                })
                .collect();
            let expected: Vec<_> = sizes
                .iter()
                .map(|&n| {
                    let scalar = (0..n).fold(PastaField::ZERO, |sum, i| {
                        let base = if family == "ipa" {
                            indices[i] as usize
                        } else {
                            i
                        };
                        sum.add(&full[base].mul(&full[i]))
                    });
                    AffinePoint::<C>::GENERATOR.mul_projective(&scalar)
                })
                .collect();
            for (execution, tasks) in [("serial", 1), ("rayon4", 4)] {
                for (cap, maximum) in [("all", None), ("512", NonZeroUsize::new(512))] {
                    let options = ExecutionOptions {
                        task_budget: TaskBudget::new(tasks).unwrap(),
                        max_terms_per_pass: maximum,
                    };
                    let mut buffers =
                        Buffers::new(msm::batch_requirements(&jobs, options).unwrap());
                    let mut output = vec![ProjectivePoint::IDENTITY; jobs.len()];
                    msm::execute_batch(
                        &jobs,
                        &mut output,
                        options,
                        &SerialExecutor,
                        buffers.borrow(),
                    )
                    .unwrap();
                    assert_eq!(output, expected);
                    group.throughput(Throughput::Elements(sizes.iter().sum::<usize>() as u64));
                    let case = format!("cached/full/{execution}/cap_{cap}");
                    group.bench_with_input(BenchmarkId::new(case, n), &jobs, |b, jobs| {
                        if tasks == 1 {
                            b.iter(|| {
                                msm::execute_batch(
                                    black_box(jobs),
                                    &mut output,
                                    options,
                                    &SerialExecutor,
                                    buffers.borrow(),
                                )
                                .unwrap();
                                black_box(&output);
                            });
                        } else {
                            pool.install(|| {
                                b.iter(|| {
                                    msm::execute_batch(
                                        black_box(jobs),
                                        &mut output,
                                        options,
                                        &Pool,
                                        buffers.borrow(),
                                    )
                                    .unwrap();
                                    black_box(&output);
                                })
                            });
                        }
                    });
                }
            }
        }
        group.finish();
    }
}

// These fixtures exercise scalar-dependent dispatch and exceptional bucket
// pairs independently of the ordinary layout/access benchmark matrix.
fn corpus<C: PastaCurve>(
    c: &mut Criterion,
    curve: &str,
    full: &[PastaField<C::Scalar>],
    affine: &[AffinePoint<C>],
) {
    let random128: Vec<_> = full
        .iter()
        .map(|s| {
            let mut limbs = s.to_canonical_uint().limbs();
            limbs[2] = 0;
            limbs[3] = 0;
            PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs)).unwrap()
        })
        .collect();
    let sparse: Vec<_> = (0..full.len())
        .map(|i| {
            let mut limbs = [0; 4];
            let bit = 129 + i % 125;
            limbs[bit / 64] = 1 << (bit % 64);
            limbs[0] = 1 << (i % 64);
            PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs)).unwrap()
        })
        .collect();
    let random96: Vec<_> = random128
        .iter()
        .map(|s| {
            let mut limbs = s.to_canonical_uint().limbs();
            limbs[1] &= u32::MAX as u64;
            PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs)).unwrap()
        })
        .collect();
    let sparse128: Vec<_> = (0..full.len())
        .map(|i| {
            let limbs = [1 << (i % 64), 1 << (i % 64), 0, 0];
            PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs)).unwrap()
        })
        .collect();
    let ones = vec![PastaField::ONE; full.len()];
    let equal = vec![AffinePoint::<C>::GENERATOR; full.len()];
    let inverse: Vec<_> = (0..full.len())
        .map(|i| {
            if i % 2 == 0 {
                AffinePoint::<C>::GENERATOR
            } else {
                AffinePoint::GENERATOR.neg()
            }
        })
        .collect();
    let mut group = c.benchmark_group(format!("{curve}/msm_corpus"));
    // Eviction is outside timing. This is a repeatable cache-pressure scenario,
    // not a guarantee about any particular processor's cache hierarchy.
    let mut eviction = vec![0u64; 64 * 1024 * 1024 / 8];
    for n in [1, 8, 16, 31, 32, 47, 48, 64, 127, 128, 256, 1024] {
        group.throughput(Throughput::Elements(n as u64));
        for (name, bases, scalars) in [
            ("full", &affine[..n], &full[..n]),
            ("random128", &affine[..n], &random128[..n]),
            ("random96", &affine[..n], &random96[..n]),
            ("sparse_high", &affine[..n], &sparse[..n]),
            ("sparse128", &affine[..n], &sparse128[..n]),
            ("equal", &equal[..n], &full[..n]),
            ("inverse", &inverse[..n], &full[..n]),
            ("cancellation", &inverse[..n], &ones[..n]),
        ] {
            let input = Input::new(Bases::Affine(bases), scalars).unwrap();
            let mut buffers = Buffers::new(input.requirements(ExecutionOptions::SERIAL).unwrap());
            let expected = scalars
                .iter()
                .enumerate()
                .fold(PastaField::ZERO, |sum, (i, s)| match name {
                    "equal" => sum.add(s),
                    "inverse" | "cancellation" => {
                        if i % 2 == 0 {
                            sum.add(s)
                        } else {
                            sum.sub(s)
                        }
                    }
                    _ => sum.add(&full[i].mul(s)),
                });
            assert_eq!(
                input
                    .execute(ExecutionOptions::SERIAL, &SerialExecutor, buffers.borrow())
                    .unwrap(),
                AffinePoint::<C>::GENERATOR.mul_projective(&expected)
            );
            group.bench_function(BenchmarkId::new(format!("{name}/warm"), n), |b| {
                b.iter(|| {
                    black_box(input)
                        .execute(ExecutionOptions::SERIAL, &SerialExecutor, buffers.borrow())
                        .unwrap()
                })
            });
            if matches!(name, "full" | "random128" | "cancellation") {
                let mut storage = vec![0; PreparedScalars::<C>::storage_len(n).unwrap()];
                group.bench_function(
                    BenchmarkId::new(format!("{name}/prepare_scalars"), n),
                    |b| {
                        b.iter(|| {
                            black_box(
                                PreparedScalars::<C>::prepare(
                                    black_box(scalars),
                                    &mut storage,
                                    TaskBudget::SERIAL,
                                    &SerialExecutor,
                                )
                                .unwrap(),
                            );
                        })
                    },
                );
                let retained = PreparedScalars::<C>::prepare(
                    scalars,
                    &mut storage,
                    TaskBudget::SERIAL,
                    &SerialExecutor,
                )
                .unwrap();
                let reused = Input::new_prepared(Bases::Affine(bases), retained).unwrap();
                let mut buffers =
                    Buffers::new(reused.requirements(ExecutionOptions::SERIAL).unwrap());
                assert_eq!(
                    reused
                        .execute(ExecutionOptions::SERIAL, &SerialExecutor, buffers.borrow())
                        .unwrap(),
                    AffinePoint::<C>::GENERATOR.mul_projective(&expected)
                );
                group.bench_function(BenchmarkId::new(format!("{name}/reused"), n), |b| {
                    b.iter(|| {
                        black_box(reused)
                            .execute(ExecutionOptions::SERIAL, &SerialExecutor, buffers.borrow())
                            .unwrap()
                    })
                });
            }
            if n == 128 || n == 1024 {
                group.bench_function(BenchmarkId::new(format!("{name}/cold"), n), |b| {
                    b.iter_custom(|iterations| {
                        let mut elapsed = Duration::ZERO;
                        for _ in 0..iterations {
                            for word in eviction.iter_mut().step_by(8) {
                                *word = word.wrapping_add(1);
                            }
                            black_box(&eviction);
                            let start = Instant::now();
                            black_box(input)
                                .execute(
                                    ExecutionOptions::SERIAL,
                                    &SerialExecutor,
                                    buffers.borrow(),
                                )
                                .map(black_box)
                                .unwrap();
                            elapsed += start.elapsed();
                        }
                        elapsed
                    })
                });
            }
        }
    }
    group.finish();
}

fn benchmarks(c: &mut Criterion) {
    curve::<Pallas>(c, "pallas");
    curve::<Vesta>(c, "vesta");
}
criterion_group!(benches, benchmarks);
criterion_main!(benches);
