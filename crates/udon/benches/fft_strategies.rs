//! Opt-in FFT strategies under explicit memory, order, and concurrency choices.
//!
//! Reused-output cases include initialization, scaling, and required ordering.
//! Setup, allocation, and disposable-input restoration are measured separately
//! or excluded explicitly. No results from this suite tune the runtime defaults.

use std::{hint::black_box, num::NonZeroUsize, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use zakura_udon::{
    exec::{Executor, SerialExecutor},
    fft::{
        run::{ExpansionPlan, FftPlan, InterpolationPlan},
        *,
    },
    field::{CanonicalUint, PallasBase, PallasScalar, PastaField, PrimeModulus},
};

struct Runner {
    tasks: usize,
    pool: Option<rayon::ThreadPool>,
}

impl Executor for Runner {
    fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
    where
        L: FnOnce() -> A + Send,
        R: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        if self.pool.is_some() {
            rayon::join(left, right)
        } else {
            SerialExecutor.join(left, right)
        }
    }
}

impl Runner {
    fn install(&self, work: impl FnOnce() + Send) {
        if let Some(pool) = &self.pool {
            pool.install(work);
        } else {
            work();
        }
    }

    fn transform<'t, M: PrimeModulus>(
        &self,
        plan: Plan<'t, M>,
        request: TransformRequest,
        separate: bool,
        columns: bool,
        codelet: Codelet,
    ) -> FftPlan<'t, M> {
        let mut operation = FftPlan::new(
            plan,
            zakura_udon::fft::TransformRequest {
                input_storage: if separate {
                    zakura_udon::fft::InputStorage::Preserve
                } else {
                    zakura_udon::fft::InputStorage::InPlace
                },
                ..request
            },
            nz(1024.min(plan.domain().size() / 2).max(1)),
            codelet,
        )
        .unwrap()
        .with_contiguous_permutation();
        if columns {
            operation = operation.with_columns(nz(32), nz(self.tasks)).unwrap();
        }
        operation
    }
}
fn nz(n: usize) -> NonZeroUsize {
    NonZeroUsize::new(n).unwrap()
}

fn inputs<M: PrimeModulus>(size: usize) -> Vec<PastaField<M>> {
    let mut seed = 0x243f_6a88_85a3_08d3u64;
    (0..size)
        .map(|_| {
            let mut limbs = core::array::from_fn(|_| {
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

fn transforms<M: PrimeModulus>(criterion: &mut Criterion, field: &str, runner: &Runner) {
    for size in [2048, 16384, 1 << 20] {
        for (shift_name, shift) in [
            ("subgroup", PastaField::ONE),
            ("zeta", PastaField::zeta()),
            ("generic_7", PastaField::from_u64(7)),
        ] {
            let domain = Domain::<M>::for_size(size).unwrap().coset(shift).unwrap();
            let plan = Plan::without_tables(domain);
            let input = inputs(size);
            let mut group = criterion.benchmark_group(format!(
                "{field}/strategies/{size}/{shift_name}/tasks_{}",
                runner.tasks
            ));
            group.throughput(Throughput::Elements(size as u64));
            let mut bench = |name: &str, operation: FftPlan<'_, M>| {
                let mut output = vec![PastaField::ZERO; size];
                let mut scratch = vec![PastaField::ZERO; operation.retained_fields()];
                let id = BenchmarkId::new(name, format!("scratch_{}", scratch.len() * 32));
                group.bench_function(id, |b| {
                    runner.install(|| {
                        b.iter(|| {
                            operation
                                .execute(
                                    Some(black_box(&input)),
                                    black_box(&mut output),
                                    None,
                                    black_box(&mut scratch),
                                    nz(runner.tasks),
                                    runner,
                                )
                                .unwrap();
                            black_box(&output);
                        })
                    })
                });
            };
            for columns in [false, true] {
                for direction in [Direction::Forward, Direction::Inverse] {
                    bench(
                        &format!("columns_{columns}/{direction:?}"),
                        runner.transform(
                            plan,
                            TransformRequest::new(direction),
                            true,
                            columns,
                            Codelet::Radix2,
                        ),
                    );
                }
            }
            for scatter in [false, true] {
                let mut operation = runner.transform(
                    plan,
                    TransformRequest::new(Direction::Forward),
                    true,
                    true,
                    Codelet::Radix2,
                );
                if scatter {
                    operation = operation.with_scatter_initialization();
                }
                bench(&format!("initialization/scatter_{scatter}"), operation);
            }
            for codelet in [Codelet::Radix2, Codelet::Radix4, Codelet::Radix8] {
                bench(
                    &format!("DIF/{codelet:?}"),
                    runner.transform(
                        plan,
                        TransformRequest {
                            output_order: ElementOrder::BitReversed,
                            ..TransformRequest::new(Direction::Forward)
                        },
                        true,
                        true,
                        codelet,
                    ),
                );
            }
            for (name, table_size, storage) in [
                ("dense", size, TwiddleStorage::Dense),
                ("local_packed", 256, TwiddleStorage::StagePacked),
                ("packed", size, TwiddleStorage::StagePacked),
                ("strided", size * 2, TwiddleStorage::Dense),
            ] {
                let description = TwiddleDescription {
                    size: table_size,
                    inverse: false,
                    storage,
                };
                let mut values = vec![PastaField::ZERO; description.requirements().unwrap()];
                let table = TwiddleTable::prepare(description, &mut values).unwrap();
                for direction in [Direction::Forward, Direction::Inverse] {
                    bench(
                        &format!("twiddles/{name}/{direction:?}"),
                        runner
                            .transform(
                                plan,
                                TransformRequest::new(direction),
                                true,
                                true,
                                Codelet::Radix2,
                            )
                            .with_twiddles(table),
                    );
                }
            }
            for inverse in [false, true] {
                let mut values = vec![PastaField::ZERO; size / 2];
                let tables = if inverse {
                    TablesMut {
                        inverse: Some(&mut values),
                        ..TablesMut::default()
                    }
                } else {
                    TablesMut {
                        forward: Some(&mut values),
                        ..TablesMut::default()
                    }
                }
                .prepare(domain)
                .unwrap();
                for direction in [Direction::Forward, Direction::Inverse] {
                    bench(
                        &format!("plan_twiddles/inverse_{inverse}/{direction:?}"),
                        runner.transform(
                            Plan::new(tables),
                            TransformRequest::new(direction),
                            true,
                            true,
                            Codelet::Radix2,
                        ),
                    );
                }
            }
            let mut powers = vec![PastaField::ZERO; size];
            let scales = PowerTable::prepare(PastaField::ONE, shift, &mut powers).unwrap();
            bench(
                "coefficient_scales",
                runner
                    .transform(
                        plan,
                        TransformRequest::new(Direction::Forward),
                        true,
                        true,
                        Codelet::Radix2,
                    )
                    .with_forward_scales(scales)
                    .unwrap(),
            );
            group.finish();
        }
    }
}

fn pipelines<M: PrimeModulus>(criterion: &mut Criterion, field: &str, runner: &Runner) {
    let base = Plan::without_tables(Domain::<M>::new(11).unwrap().subgroup());
    let extended = Domain::new(14)
        .unwrap()
        .coset(PastaField::from_u64(7))
        .unwrap();
    let input = inputs(base.domain().size());
    let mut group = criterion.benchmark_group(format!(
        "{field}/expansion_strategies/tasks_{}",
        runner.tasks
    ));
    group.throughput(Throughput::Elements(extended.size() as u64));
    for normalization in [
        None,
        Some(ExpansionScaleNormalization::Coefficients),
        Some(ExpansionScaleNormalization::UnscaledInverse),
    ] {
        let mut scales = vec![PastaField::ZERO; extended.size()];
        let expansion = Expansion::new(base, extended, None).unwrap();
        let expansion = if let Some(normalization) = normalization {
            expansion
                .with_scales(
                    ExpansionScales::prepare(
                        base.domain().size(),
                        extended,
                        normalization,
                        &mut scales,
                    )
                    .unwrap(),
                )
                .unwrap()
        } else {
            expansion
        };
        for order in [ExpansionOrder::Residues, ExpansionOrder::BitReversed] {
            let scale = if normalization == Some(ExpansionScaleNormalization::Coefficients) {
                InverseScale::Normalized
            } else {
                InverseScale::Unscaled
            };
            for (storage_name, storage) in [
                ("ReuseOutput", ExpansionStorage::ReuseOutput),
                (
                    "CoefficientWorkspace",
                    ExpansionStorage::CoefficientWorkspace { scale },
                ),
                (
                    "DisposableInput",
                    ExpansionStorage::DisposableInput { scale },
                ),
            ] {
                let operation = ExpansionPlan::new(
                    expansion,
                    storage,
                    order,
                    InputSupport::Full,
                    ElementOrder::Natural,
                    nz(1024),
                    Codelet::Radix2,
                )
                .unwrap();
                let mut output = vec![PastaField::ZERO; extended.size()];
                let mut workspace = vec![PastaField::ZERO; operation.coefficient_fields()];
                let mut scratch =
                    vec![PastaField::ZERO; operation.scratch_fields(nz(runner.tasks)).unwrap()];
                let id = format!(
                    "{normalization:?}/{order:?}/{storage_name}/scratch_{}_coefficients_{}",
                    scratch.len() * 32,
                    workspace.len() * 32
                );
                group.bench_function(id, |b| {
                    runner.install(|| {
                        b.iter_batched_ref(
                            || input.clone(),
                            |working| {
                                if matches!(storage, ExpansionStorage::DisposableInput { .. }) {
                                    black_box(
                                        operation
                                            .execute_disposable(
                                                black_box(working),
                                                &mut output,
                                                None,
                                                &mut scratch,
                                                nz(runner.tasks),
                                                runner,
                                            )
                                            .unwrap(),
                                    );
                                } else {
                                    black_box(
                                        operation
                                            .execute(
                                                black_box(&input),
                                                &mut output,
                                                &mut workspace,
                                                None,
                                                &mut scratch,
                                                nz(runner.tasks),
                                                runner,
                                            )
                                            .unwrap(),
                                    );
                                }
                                black_box(&output);
                            },
                            criterion::BatchSize::SmallInput,
                        )
                    })
                });
            }
        }
    }
    group.finish();

    let mut group =
        criterion.benchmark_group(format!("{field}/batch_strategies/tasks_{}", runner.tasks));
    for columns in [false, true] {
        let operation = runner.transform(
            base,
            TransformRequest::new(Direction::Forward),
            false,
            columns,
            Codelet::Radix2,
        );
        for count in [1, 4, 8] {
            let values = inputs::<M>(base.domain().size() * count);
            let fields = operation.batch_fields(count, nz(runner.tasks)).unwrap();
            let mut scratch = vec![PastaField::ZERO; fields];
            group.bench_function(
                format!("columns_{columns}/{count}/scratch_{}", fields * 32),
                |b| {
                    runner.install(|| {
                        b.iter_batched_ref(
                            || values.clone(),
                            |output| {
                                operation
                                    .execute_batch(
                                        black_box(output),
                                        &mut scratch,
                                        nz(runner.tasks),
                                        runner,
                                    )
                                    .unwrap();
                                black_box(output);
                            },
                            criterion::BatchSize::SmallInput,
                        )
                    })
                },
            );
        }
    }
    group.finish();

    let mut group =
        criterion.benchmark_group(format!("{field}/class_strategies/tasks_{}", runner.tasks));
    for mode in ["fused", "parallel", "sum"] {
        let tasks = if mode == "fused" { 1 } else { runner.tasks };
        let transform = runner.transform(
            base,
            TransformRequest::new(Direction::Inverse),
            false,
            false,
            Codelet::Radix2,
        );
        let operation = InterpolationPlan::new([transform; 4], mode == "sum").unwrap();
        group.bench_function(mode, |b| {
            runner.install(|| {
                b.iter_batched_ref(
                    || [input.clone(), input.clone(), input.clone(), input.clone()],
                    |values| {
                        operation
                            .execute(
                                values.each_mut().map(Vec::as_mut_slice),
                                [&mut [], &mut [], &mut [], &mut []],
                                nz(tasks),
                                runner,
                            )
                            .unwrap();
                        black_box(values);
                    },
                    criterion::BatchSize::SmallInput,
                )
            });
        });
    }
    group.finish();
}

fn expansion_prefixes<M: PrimeModulus>(criterion: &mut Criterion, field: &str, runner: &Runner) {
    let domain = Domain::<M>::new(11).unwrap().subgroup();
    let extended = Domain::new(14).unwrap().coset(PastaField::zeta()).unwrap();
    let mut twiddles = vec![PastaField::ZERO; domain.size() / 2];
    let tables = TablesMut {
        forward: Some(&mut twiddles),
        ..TablesMut::default()
    }
    .prepare(domain)
    .unwrap();
    let input = inputs(domain.size());
    let factor = inputs(extended.size());
    let mut group =
        criterion.benchmark_group(format!("{field}/expansion_prefixes/tasks_{}", runner.tasks));
    group.throughput(Throughput::Elements(extended.size() as u64));
    for (name, base) in [
        ("computed", Plan::without_tables(domain)),
        ("table", Plan::new(tables)),
    ] {
        let expansion = Expansion::new(base, extended, None).unwrap();
        for order in [ExpansionOrder::Residues, ExpansionOrder::BitReversed] {
            let mut output = vec![PastaField::ZERO; extended.size()];
            let layout = if order == ExpansionOrder::Residues {
                EvaluationLayout::Residues(expansion.layout())
            } else {
                EvaluationLayout::BitReversed
            };
            let mut ordered_factor = vec![PastaField::ZERO; extended.size()];
            for (row, value) in factor.iter().enumerate() {
                ordered_factor[layout.index(row, extended.size()).unwrap()] = *value;
            }
            for prefix in [1, 10, 128, 256, domain.size()] {
                let operation = ExpansionPlan::new(
                    expansion,
                    ExpansionStorage::Coefficients,
                    order,
                    InputSupport::Prefix(prefix),
                    ElementOrder::Natural,
                    nz(1024),
                    Codelet::Radix2,
                )
                .unwrap();
                let mut scratch =
                    vec![PastaField::ZERO; operation.scratch_fields(nz(runner.tasks)).unwrap()];
                for product in [false, true] {
                    group.bench_function(
                        format!("{name}/{order:?}/prefix_{prefix}/product_{product}"),
                        |b| {
                            runner.install(|| {
                                b.iter(|| {
                                    let input = black_box(&input[..prefix]);
                                    operation
                                        .execute(
                                            input,
                                            &mut output,
                                            &mut [],
                                            product.then_some(ordered_factor.as_slice()),
                                            &mut scratch,
                                            nz(runner.tasks),
                                            runner,
                                        )
                                        .unwrap();
                                    black_box(&output);
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

fn preparation<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let mut group = criterion.benchmark_group(format!("{field}/strategy_preparation"));
    for storage in [TwiddleStorage::Dense, TwiddleStorage::StagePacked] {
        let description = TwiddleDescription {
            size: 1 << 20,
            inverse: false,
            storage,
        };
        let mut values = vec![PastaField::<M>::ZERO; description.requirements().unwrap()];
        group.bench_function(format!("{storage:?}/bytes_{}", values.len() * 32), |b| {
            b.iter(|| {
                black_box(TwiddleTable::prepare(description, black_box(&mut values)).unwrap());
            })
        });
    }
    // Import validation is setup work; keep it outside execution measurements.
    for size in [1 << 14, 1 << 20] {
        for storage in [TwiddleStorage::Dense, TwiddleStorage::StagePacked] {
            let description = TwiddleDescription {
                size,
                inverse: false,
                storage,
            };
            let mut values = vec![PastaField::<M>::ZERO; description.requirements().unwrap()];
            TwiddleTable::prepare(description, &mut values).unwrap();
            group.bench_function(format!("checked_import/{storage:?}/{size}"), |b| {
                b.iter(|| black_box(TwiddleTable::bind(description, black_box(&values)).unwrap()))
            });
        }
    }
    let base_size = 2048;
    let extended = Domain::<M>::for_size(16384)
        .unwrap()
        .coset(PastaField::from_u64(7))
        .unwrap();
    let mut scales = vec![PastaField::ZERO; extended.size()];
    for normalization in [
        ExpansionScaleNormalization::Coefficients,
        ExpansionScaleNormalization::UnscaledInverse,
    ] {
        group.bench_function(
            format!("expansion/{normalization:?}/bytes_{}", scales.len() * 32),
            |b| {
                b.iter(|| {
                    black_box(
                        ExpansionScales::prepare(
                            base_size,
                            extended,
                            normalization,
                            black_box(&mut scales),
                        )
                        .unwrap(),
                    );
                })
            },
        );
    }
    group.bench_function(
        format!("coefficient_powers/bytes_{}", scales.len() * 32),
        |b| {
            b.iter(|| {
                black_box(
                    PowerTable::prepare(PastaField::ONE, extended.shift(), black_box(&mut scales))
                        .unwrap(),
                );
            })
        },
    );
    group.finish();
}

fn subgroup_expansion<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let base = Plan::without_tables(Domain::<M>::for_size(2048).unwrap().subgroup());
    let input = inputs(base.domain().size());
    let mut group = criterion.benchmark_group(format!("{field}/subgroup_expansion"));
    for residues in [1, 8] {
        let extended = Domain::for_size(2048 * residues).unwrap().subgroup();
        let expansion = Expansion::new(base, extended, None).unwrap();
        for order in [ExpansionOrder::Residues, ExpansionOrder::BitReversed] {
            let operation = ExpansionPlan::new(
                expansion,
                ExpansionStorage::ReuseOutput,
                order,
                InputSupport::Full,
                ElementOrder::Natural,
                nz(base.domain().size()),
                Codelet::Radix2,
            )
            .unwrap();
            let mut output = vec![PastaField::ZERO; extended.size()];
            let mut scratch = vec![PastaField::ZERO; operation.scratch_fields(nz(1)).unwrap()];
            group.bench_function(BenchmarkId::new(format!("{order:?}"), residues), |b| {
                b.iter(|| {
                    operation
                        .execute(
                            black_box(&input),
                            &mut output,
                            &mut [],
                            None,
                            &mut scratch,
                            nz(1),
                            &SerialExecutor,
                        )
                        .unwrap();
                    black_box(&output);
                })
            });
        }
    }
    group.finish();
}

fn benchmarks(criterion: &mut Criterion) {
    subgroup_expansion::<PallasBase>(criterion, "Fp");
    subgroup_expansion::<PallasScalar>(criterion, "Fq");
    for tasks in [1, 4] {
        let runner = Runner {
            tasks,
            pool: (tasks > 1).then(|| {
                rayon::ThreadPoolBuilder::new()
                    .num_threads(tasks)
                    .build()
                    .unwrap()
            }),
        };
        transforms::<PallasBase>(criterion, "Fp", &runner);
        transforms::<PallasScalar>(criterion, "Fq", &runner);
        pipelines::<PallasBase>(criterion, "Fp", &runner);
        pipelines::<PallasScalar>(criterion, "Fq", &runner);
        expansion_prefixes::<PallasBase>(criterion, "Fp", &runner);
        expansion_prefixes::<PallasScalar>(criterion, "Fq", &runner);
    }
    preparation::<PallasBase>(criterion, "Fp");
    preparation::<PallasScalar>(criterion, "Fq");
}

criterion_group! {
    name = benches;
    config = Criterion::default().sample_size(10).warm_up_time(Duration::from_millis(300)).measurement_time(Duration::from_secs(1));
    targets = benchmarks
}
criterion_main!(benches);
