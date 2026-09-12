//! Opt-in FFT strategies under explicit memory, order, and concurrency choices.
//!
//! Reused-output cases include initialization, scaling, and required ordering.
//! Setup, allocation, and disposable-input restoration are measured separately
//! or excluded explicitly. No results from this suite tune the runtime defaults.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use zakura_udon::{
    exec::{Executor, SerialExecutor},
    fft::*,
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

    fn strategy(&self, size: usize) -> Strategy {
        Strategy {
            execution: ExecutionOptions {
                tile_len: 1024.min(size / 2),
                columns_per_task: 32,
                max_tasks: self.tasks,
            },
            budget: ResourceBudget::for_tasks(self.tasks),
            ..Strategy::serial()
        }
    }
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
            let strategy = runner.strategy(size);
            let mut group = criterion.benchmark_group(format!(
                "{field}/strategies/{size}/{shift_name}/tasks_{}",
                runner.tasks
            ));
            group.throughput(Throughput::Elements(size as u64));
            let mut bench = |name: &str, operation: PreparedOperation<'_, M>| {
                let required = operation.requirements();
                let mut output = vec![PastaField::ZERO; size];
                let mut scratch = vec![PastaField::ZERO; required.scratch_fields];
                let id = BenchmarkId::new(
                    name,
                    format!(
                        "tables_{}_scratch_{}",
                        required.retained_table_bytes,
                        scratch.len() * 32
                    ),
                );
                group.bench_function(id, |b| {
                    runner.install(|| {
                        b.iter(|| {
                            operation
                                .execute_into(
                                    black_box(&input),
                                    black_box(&mut output),
                                    runner,
                                    black_box(&mut scratch),
                                )
                                .unwrap();
                            black_box(&output);
                        })
                    })
                });
            };
            for backend in [Backend::InPlace, Backend::Blocked] {
                for direction in [Direction::Forward, Direction::Inverse] {
                    bench(
                        &format!("{backend:?}/{direction:?}"),
                        plan.configure(
                            TransformRequest::new(direction),
                            Strategy {
                                backend,
                                ..strategy
                            },
                        )
                        .unwrap(),
                    );
                }
            }
            for initialization in [
                Initialization::Scatter,
                Initialization::Gather,
                Initialization::Blocked,
            ] {
                bench(
                    &format!("initialization/{initialization:?}"),
                    plan.configure(
                        TransformRequest::new(Direction::Forward),
                        Strategy {
                            initialization,
                            ..strategy
                        },
                    )
                    .unwrap(),
                );
            }
            for codelet in [Codelet::Radix2, Codelet::Radix4, Codelet::Radix8] {
                bench(
                    &format!("DIF/{codelet:?}"),
                    plan.configure(
                        TransformRequest {
                            output_order: InputOrder::BitReversed,
                            ..TransformRequest::new(Direction::Forward)
                        },
                        Strategy {
                            codelet,
                            ..strategy
                        },
                    )
                    .unwrap(),
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
                        plan.configure(TransformRequest::new(direction), strategy)
                            .unwrap()
                            .with_twiddles(table)
                            .unwrap(),
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
                        Plan::new(tables)
                            .configure(TransformRequest::new(direction), strategy)
                            .unwrap(),
                    );
                }
            }
            let mut powers = vec![PastaField::ZERO; size];
            let scales = PowerTable::prepare(PastaField::ONE, shift, &mut powers).unwrap();
            bench(
                "coefficient_scales",
                plan.configure(TransformRequest::new(Direction::Forward), strategy)
                    .unwrap()
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
                let strategy = runner.strategy(base.domain().size());
                let operation = expansion
                    .configure(
                        order,
                        storage,
                        ExpansionStrategy {
                            transform: strategy.execution,
                            budget: strategy.budget,
                        },
                    )
                    .unwrap();
                let required = operation.requirements();
                let mut output = vec![PastaField::ZERO; extended.size()];
                let mut workspace = vec![PastaField::ZERO; required.coefficient_fields];
                let mut scratch = vec![PastaField::ZERO; required.scratch_fields];
                let id = format!(
                    "{normalization:?}/{order:?}/{storage_name}/tables_{}_scratch_{}_coefficients_{}",
                    required.retained_table_bytes,
                    scratch.len() * 32,
                    workspace.len() * 32
                );
                group.bench_function(id, |b| {
                    runner.install(|| {
                        b.iter_batched_ref(
                            || input.clone(),
                            |working| {
                                match storage {
                                    ExpansionStorage::ReuseOutput => operation.execute_into(
                                        black_box(&input),
                                        &mut output,
                                        runner,
                                        &mut scratch,
                                    ),
                                    ExpansionStorage::CoefficientWorkspace { .. } => operation
                                        .execute_with_workspace(
                                            black_box(&input),
                                            &mut output,
                                            &mut workspace,
                                            runner,
                                            &mut scratch,
                                        )
                                        .map(|view| {
                                            black_box(view);
                                        }),
                                    ExpansionStorage::DisposableInput { .. } => operation
                                        .execute_disposable(
                                            black_box(working),
                                            &mut output,
                                            runner,
                                            &mut scratch,
                                        )
                                        .map(|view| {
                                            black_box(view);
                                        }),
                                    ExpansionStorage::Coefficients => unreachable!(),
                                }
                                .unwrap();
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
    for backend in [Backend::InPlace, Backend::Blocked] {
        let operation = base
            .configure(
                TransformRequest::new(Direction::Forward),
                Strategy {
                    backend,
                    ..runner.strategy(base.domain().size())
                },
            )
            .unwrap();
        for count in [1, 4, 8] {
            let values = inputs::<M>(base.domain().size() * count);
            let fields = operation.batch_requirements(count).unwrap().field_elements;
            let mut scratch = vec![PastaField::ZERO; fields];
            group.bench_function(
                format!("{backend:?}/{count}/scratch_{}", fields * 32),
                |b| {
                    runner.install(|| {
                        b.iter_batched_ref(
                            || values.clone(),
                            |output| {
                                operation
                                    .execute_batch(black_box(output), runner, &mut scratch)
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
    let options = runner.strategy(base.domain().size()).execution;
    let parallel = InterpolationOptions {
        transform: options,
        max_class_tasks: 4,
        max_tasks: runner.tasks,
    };
    let parallel_fields = parallel
        .requirements(base.domain().size(), &[base.domain().size(); 3])
        .unwrap()
        .scratch_fields;
    let serial_fields = options
        .interpolation_requirements(base.domain().size(), &[base.domain().size(); 3])
        .unwrap()
        .field_elements;
    for mode in ["fused", "parallel", "sum"] {
        let fields = if mode == "parallel" {
            parallel_fields
        } else {
            serial_fields
        };
        let mut scratch = vec![PastaField::ZERO; fields];
        group.bench_function(format!("{mode}/scratch_{}", fields * 32), |b| {
            runner.install(|| {
                b.iter_batched_ref(
                    || [input.clone(), input.clone(), input.clone(), input.clone()],
                    |values| {
                        let [output, a, b, c] = values;
                        let mut output = Class::new(base, output, InputOrder::Natural).unwrap();
                        let mut lifts = [
                            Class::new(base, a, InputOrder::Natural).unwrap(),
                            Class::new(base, b, InputOrder::Natural).unwrap(),
                            Class::new(base, c, InputOrder::Natural).unwrap(),
                        ];
                        match mode {
                            "fused" => interpolate_classes(
                                &mut output,
                                &mut lifts,
                                options,
                                runner,
                                &mut scratch,
                            ),
                            "parallel" => interpolate_classes_parallel(
                                &mut output,
                                &mut lifts,
                                parallel,
                                runner,
                                &mut scratch,
                            ),
                            "sum" => interpolate_sum(
                                &mut output,
                                &mut lifts,
                                options,
                                runner,
                                &mut scratch,
                            ),
                            _ => unreachable!(),
                        }
                        .unwrap();
                        black_box(output.values());
                    },
                    criterion::BatchSize::SmallInput,
                )
            })
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
            let strategy = runner.strategy(domain.size());
            let operation = expansion
                .configure(
                    order,
                    ExpansionStorage::Coefficients,
                    ExpansionStrategy {
                        transform: strategy.execution,
                        budget: strategy.budget,
                    },
                )
                .unwrap();
            let mut scratch = vec![PastaField::ZERO; operation.requirements().scratch_fields];
            let mut output = vec![PastaField::ZERO; extended.size()];
            // Give each order the same factor by natural evaluation row.
            let mut ordered_factor = vec![PastaField::ZERO; extended.size()];
            for (row, value) in factor.iter().enumerate() {
                ordered_factor[operation.layout().index(row, extended.size()).unwrap()] = *value;
            }
            let factor = operation.view(&ordered_factor).unwrap();
            for prefix in [1, 10, 128, 256, domain.size()] {
                for product in [false, true] {
                    group.bench_function(
                        format!("{name}/{order:?}/prefix_{prefix}/product_{product}"),
                        |b| {
                            runner.install(|| {
                                b.iter(|| {
                                    let input = black_box(&input[..prefix]);
                                    if product {
                                        operation.execute_product_into(
                                            input,
                                            factor,
                                            &mut output,
                                            runner,
                                            &mut scratch,
                                        )
                                    } else {
                                        operation.execute_into(
                                            input,
                                            &mut output,
                                            runner,
                                            &mut scratch,
                                        )
                                    }
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

fn benchmarks(criterion: &mut Criterion) {
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
