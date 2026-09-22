//! FFT execution under caller-owned memory, order, and concurrency constraints.
//!
//! Reused-output cases include initialization, scaling, and required ordering.
//! Setup, allocation, and disposable-input restoration are measured separately
//! or excluded explicitly. Udon selects the arithmetic for each case.

use std::{hint::black_box, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use zakura_udon::{
    exec::{ExecutionOptions, Executor, SerialExecutor, TaskBudget},
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

    fn options(&self) -> ExecutionOptions {
        ExecutionOptions::default().with_task_budget(TaskBudget::new(self.tasks).unwrap())
    }

    fn transform<'t, M: PrimeModulus>(
        &self,
        plan: Transform<'t, M>,
        request: TransformRequest,
        separate: bool,
        memory_limit: Option<usize>,
    ) -> FftPlan<'t, M> {
        let options = memory_limit.map_or(self.options(), |limit| {
            self.options().with_memory_limit(limit)
        });
        FftPlan::new(
            plan,
            TransformRequest {
                input_storage: if separate {
                    InputStorage::Preserve
                } else {
                    InputStorage::InPlace
                },
                ..request
            },
            StorageLayout::Contiguous,
            options,
        )
        .unwrap()
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
            ("zeta", PastaField::ZETA),
            ("generic_7", PastaField::from_u64(7)),
        ] {
            let domain = Domain::<M>::for_size(size).unwrap().coset(shift).unwrap();
            let plan = Transform::new(domain);
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
                            operation.execute(
                                Some(black_box(&input)),
                                black_box(&mut output),
                                None,
                                black_box(&mut scratch),
                                runner,
                            );
                            black_box(&output);
                        })
                    })
                });
            };
            for memory_limit in [Some(0), None] {
                for direction in [Direction::Forward, Direction::Inverse] {
                    bench(
                        &format!("memory_{memory_limit:?}/{direction:?}"),
                        runner.transform(
                            plan,
                            TransformRequest::new(direction),
                            true,
                            memory_limit,
                        ),
                    );
                }
            }
            for output_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                bench(
                    &format!("order/{output_order:?}"),
                    runner.transform(
                        plan,
                        TransformRequest {
                            output_order,
                            ..TransformRequest::new(Direction::Forward)
                        },
                        true,
                        None,
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
                            .transform(plan, TransformRequest::new(direction), true, None)
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
                .prepare(domain);
                for direction in [Direction::Forward, Direction::Inverse] {
                    bench(
                        &format!("plan_twiddles/inverse_{inverse}/{direction:?}"),
                        runner.transform(tables, TransformRequest::new(direction), true, None),
                    );
                }
            }
            let mut powers = vec![PastaField::ZERO; size];
            let scales = PowerTable::prepare(PastaField::ONE, shift, &mut powers);
            bench(
                "coefficient_scales",
                runner
                    .transform(plan, TransformRequest::new(Direction::Forward), true, None)
                    .with_forward_scales(scales),
            );
            group.finish();
        }
    }
}

fn pipelines<M: PrimeModulus>(criterion: &mut Criterion, field: &str, runner: &Runner) {
    let base = Transform::new(Domain::<M>::new(11).unwrap().subgroup());
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
            expansion.with_scales(
                ExpansionScales::prepare(
                    base.domain().size(),
                    extended,
                    normalization,
                    &mut scales,
                )
                .unwrap(),
            )
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
                    StorageLayout::Contiguous,
                    runner.options(),
                )
                .unwrap();
                let mut output = vec![PastaField::ZERO; extended.size()];
                let mut workspace = vec![PastaField::ZERO; operation.coefficient_fields()];
                let mut scratch = vec![PastaField::ZERO; operation.scratch_fields()];
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
                                    black_box(operation.execute_disposable(
                                        black_box(working),
                                        &mut output,
                                        None,
                                        &mut scratch,
                                        runner,
                                    ));
                                } else {
                                    black_box(operation.execute(
                                        black_box(&input),
                                        &mut output,
                                        &mut workspace,
                                        None,
                                        &mut scratch,
                                        runner,
                                    ));
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
    for memory_limit in [Some(0), None] {
        let operation = runner.transform(
            base,
            TransformRequest::new(Direction::Forward),
            false,
            memory_limit,
        );
        for count in [1, 4, 8] {
            let values = inputs::<M>(base.domain().size() * count);
            let fields = operation.batch_fields(count).unwrap();
            let mut scratch = vec![PastaField::ZERO; fields];
            group.bench_function(
                format!("memory_{memory_limit:?}/{count}/scratch_{}", fields * 32),
                |b| {
                    runner.install(|| {
                        b.iter_batched_ref(
                            || values.clone(),
                            |output| {
                                operation.execute_batch(black_box(output), &mut scratch, runner);
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
        let operation = InterpolationPlan::new(
            [(base, ElementOrder::Natural); 4],
            mode == "sum",
            StorageLayout::Contiguous,
            ExecutionOptions::default().with_task_budget(TaskBudget::new(tasks).unwrap()),
        )
        .unwrap();
        let mut scratch: [_; 4] =
            core::array::from_fn(|i| vec![PastaField::ZERO; operation.snapshot_fields(i).unwrap()]);
        group.bench_function(mode, |b| {
            runner.install(|| {
                b.iter_batched_ref(
                    || [input.clone(), input.clone(), input.clone(), input.clone()],
                    |values| {
                        operation.execute(
                            values.each_mut().map(Vec::as_mut_slice),
                            scratch.each_mut().map(Vec::as_mut_slice),
                            runner,
                        );
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
    let extended = Domain::new(14).unwrap().coset(PastaField::ZETA).unwrap();
    let mut twiddles = vec![PastaField::ZERO; domain.size() / 2];
    let tables = TablesMut {
        forward: Some(&mut twiddles),
        ..TablesMut::default()
    }
    .prepare(domain);
    let input = inputs(domain.size());
    let factor = inputs(extended.size());
    let mut group =
        criterion.benchmark_group(format!("{field}/expansion_prefixes/tasks_{}", runner.tasks));
    group.throughput(Throughput::Elements(extended.size() as u64));
    for (name, base) in [("computed", Transform::new(domain)), ("table", tables)] {
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
                    StorageLayout::Contiguous,
                    runner.options(),
                )
                .unwrap();
                let mut scratch = vec![PastaField::ZERO; operation.scratch_fields()];
                for product in [false, true] {
                    group.bench_function(
                        format!("{name}/{order:?}/prefix_{prefix}/product_{product}"),
                        |b| {
                            runner.install(|| {
                                b.iter(|| {
                                    let input = black_box(&input[..prefix]);
                                    operation.execute(
                                        input,
                                        &mut output,
                                        &mut [],
                                        product.then_some(ordered_factor.as_slice()),
                                        &mut scratch,
                                        runner,
                                    );
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
                black_box(PowerTable::prepare(
                    PastaField::ONE,
                    extended.shift(),
                    black_box(&mut scales),
                ));
            })
        },
    );
    group.finish();
}

fn subgroup_expansion<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let base = Transform::new(Domain::<M>::for_size(2048).unwrap().subgroup());
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
                StorageLayout::Contiguous,
                ExecutionOptions::default(),
            )
            .unwrap();
            let mut output = vec![PastaField::ZERO; extended.size()];
            let mut scratch = vec![PastaField::ZERO; operation.scratch_fields()];
            group.bench_function(BenchmarkId::new(format!("{order:?}"), residues), |b| {
                b.iter(|| {
                    operation.execute(
                        black_box(&input),
                        &mut output,
                        &mut [],
                        None,
                        &mut scratch,
                        &SerialExecutor,
                    );
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
