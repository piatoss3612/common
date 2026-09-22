//! Measures table preparation separately from execution that reuses buffers.
//!
//! Serial controls and persistent Rayon pools separate scheduling overhead from
//! parallel scaling. Layout conversion is included only in natural-output cases.

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use zakura_udon::{
    exec::{ExecutionOptions, Executor, SerialExecutor, TaskBudget},
    fft::{
        CoefficientView, CosetDomain, Direction, Domain, ElementOrder, Expansion, InputStorage,
        InputSupport, StorageLayout, TableRequirements, Tables, TablesMut, TransformRequest,
        reference, run::InterpolationPlan,
    },
    field::{CanonicalUint, PallasBase, PallasScalar, PastaField, PrimeModulus},
};

struct Prepared<M: PrimeModulus> {
    mask: u8,
    forward: Vec<PastaField<M>>,
    inverse: Vec<PastaField<M>>,
    finish: Vec<PastaField<M>>,
    scales: Vec<PastaField<M>>,
}

impl<M: PrimeModulus> Prepared<M> {
    fn new(domain: CosetDomain<M>) -> Self {
        Self::selected(domain, 15)
    }

    fn selected(domain: CosetDomain<M>, mask: u8) -> Self {
        let sizes = TableRequirements::for_domain(domain);
        let mut result = Self {
            mask,
            forward: vec![PastaField::ZERO; if mask & 1 != 0 { sizes.twiddles } else { 0 }],
            inverse: vec![PastaField::ZERO; if mask & 2 != 0 { sizes.twiddles } else { 0 }],
            finish: vec![PastaField::ZERO; if mask & 4 != 0 { sizes.twiddles } else { 0 }],
            scales: vec![
                PastaField::ZERO;
                if mask & 8 != 0 {
                    sizes.inverse_scales
                } else {
                    0
                }
            ],
        };
        result.prepare(domain);
        result
    }

    fn prepare(&mut self, domain: CosetDomain<M>) {
        TablesMut {
            forward: (self.mask & 1 != 0).then_some(&mut self.forward),
            inverse: (self.mask & 2 != 0).then_some(&mut self.inverse),
            inverse_finish: (self.mask & 4 != 0).then_some(&mut self.finish),
            inverse_scales: (self.mask & 8 != 0).then_some(&mut self.scales),
        }
        .prepare(domain);
    }

    fn bytes(&self) -> usize {
        (self.forward.len() + self.inverse.len() + self.finish.len() + self.scales.len()) * 32
    }

    fn tables(&self) -> Tables<'_, M> {
        Tables {
            forward: (self.mask & 1 != 0).then_some(&self.forward),
            inverse: (self.mask & 2 != 0).then_some(&self.inverse),
            inverse_finish: (self.mask & 4 != 0).then_some(&self.finish),
            inverse_scales: (self.mask & 8 != 0).then_some(&self.scales),
        }
    }
}

// Pools are constructed once for the entire suite. The timed loop runs on a
// worker, so pool entry and worker creation are outside each measurement.
struct Runner {
    name: &'static str,
    pool: Option<rayon::ThreadPool>,
    tasks: usize,
}

impl Runner {
    fn install(&self, f: impl FnOnce() + Send) {
        if let Some(pool) = &self.pool {
            pool.install(f);
        } else {
            f();
        }
    }
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

const PARALLEL: ExecutionOptions =
    ExecutionOptions::DEFAULT.with_task_budget(TaskBudget::new(4).unwrap());

fn reference_coset<M: PrimeModulus>(
    values: &mut [PastaField<M>],
    domain: CosetDomain<M>,
    inverse: bool,
) {
    let (shift, mut scale) = if inverse {
        reference::transform(values, &domain.domain().inverse_root());
        (domain.inverse_shift(), domain.domain().size_inverse())
    } else {
        (domain.shift(), PastaField::ONE)
    };
    for value in values.iter_mut() {
        *value = value.mul(&scale);
        scale = scale.mul(&shift);
    }
    if !inverse {
        reference::transform(values, &domain.domain().root());
    }
}

fn transforms<M: PrimeModulus>(
    criterion: &mut Criterion,
    field: &str,
    shift_name: &str,
    shift: PastaField<M>,
    runners: &[Runner],
) {
    for log_size in [11, 14, 20] {
        let domain = Domain::new(log_size).unwrap().coset(shift).unwrap();
        let input = inputs::<M>(domain.size());
        for (direction, inverse, profiles) in [
            ("forward", false, &[("none", 0), ("twiddles", 1)][..]),
            (
                "inverse",
                true,
                &[("none", 0), ("twiddles", 2), ("finish", 14)][..],
            ),
        ] {
            let mut group = criterion.benchmark_group(format!(
                "{field}/fft/{}/{shift_name}/{direction}",
                domain.size()
            ));
            group.throughput(Throughput::Elements(domain.size() as u64));
            group.bench_function("reference", |b| {
                b.iter_batched_ref(
                    || input.clone(),
                    |values| {
                        reference_coset(black_box(values), domain, inverse);
                        black_box(values);
                    },
                    BatchSize::PerIteration,
                );
            });
            group.finish();
            for &(profile, mask) in profiles {
                let mut tables = Prepared::selected(domain, mask);
                if mask != 0 {
                    let mut setup = criterion.benchmark_group(format!(
                        "{field}/fft_setup/{}/{shift_name}/{direction}",
                        domain.size()
                    ));
                    setup.bench_function(profile, |b| {
                        b.iter(|| black_box(&mut tables).prepare(black_box(domain)))
                    });
                    setup.finish();
                }
                let plan = tables.tables().bind(domain).unwrap();
                let mut group = criterion.benchmark_group(format!(
                    "{field}/fft/{}/{shift_name}/{direction}/{profile}",
                    domain.size()
                ));
                for runner in runners {
                    let options = ExecutionOptions::default()
                        .with_task_budget(TaskBudget::new(runner.tasks).unwrap());
                    let mut scratch =
                        vec![
                            PastaField::ZERO;
                            plan.scratch_requirements(options).unwrap().field_elements
                        ];
                    eprintln!(
                        "{field}/fft/{}/{shift_name}/{direction}/{profile}/{}: tables {} bytes; scratch {} bytes",
                        domain.size(),
                        runner.name,
                        tables.bytes(),
                        scratch.len() * 32
                    );
                    group.bench_function(runner.name, |b| {
                        runner.install(|| {
                            b.iter_batched_ref(
                                || input.clone(),
                                |values| {
                                    if inverse {
                                        plan.inverse(
                                            black_box(values),
                                            options,
                                            runner,
                                            &mut scratch,
                                        )
                                        .unwrap();
                                    } else {
                                        plan.forward(
                                            black_box(values),
                                            options,
                                            runner,
                                            &mut scratch,
                                        )
                                        .unwrap();
                                    }
                                    black_box(values);
                                },
                                BatchSize::PerIteration,
                            );
                        })
                    });
                    // Direct-output comparisons include initialization in both cases.
                    // Keep this matrix to serial controls; pool scaling is above.
                    if runner.name == "serial" || runner.name == "tiled_serial" {
                        let mut output = vec![PastaField::ZERO; domain.size()];
                        for into in [false, true] {
                            group.bench_function(
                                format!(
                                    "{}/{}",
                                    runner.name,
                                    if into { "into" } else { "copy_in_place" }
                                ),
                                |b| {
                                    b.iter(|| {
                                        if into {
                                            if inverse {
                                                plan.execute(
                                                    TransformRequest {
                                                        input_storage: InputStorage::Preserve,
                                                        ..TransformRequest::new(Direction::Inverse)
                                                    },
                                                    Some(CoefficientView::normalized(black_box(
                                                        &input,
                                                    ))),
                                                    &mut output,
                                                    options,
                                                    runner,
                                                    &mut scratch,
                                                )
                                                .unwrap();
                                            } else {
                                                plan.execute(
                                                    TransformRequest {
                                                        input_storage: InputStorage::Preserve,
                                                        ..TransformRequest::new(Direction::Forward)
                                                    },
                                                    Some(CoefficientView::normalized(black_box(
                                                        &input,
                                                    ))),
                                                    &mut output,
                                                    options,
                                                    runner,
                                                    &mut scratch,
                                                )
                                                .unwrap();
                                            }
                                        } else {
                                            output.copy_from_slice(black_box(&input));
                                            if inverse {
                                                plan.inverse(
                                                    &mut output,
                                                    options,
                                                    runner,
                                                    &mut scratch,
                                                )
                                                .unwrap();
                                            } else {
                                                plan.forward(
                                                    &mut output,
                                                    options,
                                                    runner,
                                                    &mut scratch,
                                                )
                                                .unwrap();
                                            }
                                        }
                                        black_box(&output);
                                    })
                                },
                            );
                        }
                        if !inverse {
                            for prefix_len in [0, 1, 5, domain.size() / 8] {
                                group.bench_function(
                                    format!("{}/prefix_{prefix_len}", runner.name),
                                    |b| {
                                        b.iter(|| {
                                            plan.execute(
                                                TransformRequest {
                                                    input_storage: InputStorage::Preserve,
                                                    support: InputSupport::Prefix(prefix_len),
                                                    ..TransformRequest::new(Direction::Forward)
                                                },
                                                Some(CoefficientView::normalized(black_box(
                                                    &input[..prefix_len],
                                                ))),
                                                &mut output,
                                                options,
                                                runner,
                                                &mut scratch,
                                            )
                                            .unwrap();
                                            black_box(&output);
                                        })
                                    },
                                );
                            }
                        }
                    }
                }
                group.finish();
            }
        }
    }
}

fn expansions<M: PrimeModulus>(
    criterion: &mut Criterion,
    field: &str,
    shift_name: &str,
    shift: PastaField<M>,
    runners: &[Runner],
) {
    let base = Domain::new(11).unwrap().subgroup();
    let tables = Prepared::selected(base, 15);
    let plan = tables.tables().bind(base).unwrap();
    let coefficient_plan = (Tables {
        forward: Some(&tables.forward),
        ..Tables::default()
    })
    .bind(base)
    .unwrap();
    let coefficients = inputs::<M>(base.size());
    let mut evaluations = coefficients.clone();
    plan.forward(
        &mut evaluations,
        ExecutionOptions::default(),
        &SerialExecutor,
        &mut [],
    )
    .unwrap();
    for log_size in [11, 12, 14] {
        let extended = Domain::new(log_size).unwrap().coset(shift).unwrap();
        let expansion = Expansion::new(plan, extended, None).unwrap();
        let mut scales = vec![PastaField::ZERO; expansion.scale_count()];
        let mut setup = criterion.benchmark_group(format!("{field}/fft_setup/{shift_name}"));
        setup.bench_function(BenchmarkId::new("residue_scales", extended.size()), |b| {
            b.iter(|| {
                black_box(expansion.prepare_scales(black_box(&mut scales)));
            });
        });
        setup.finish();
        let scales = expansion.prepare_scales(&mut scales);
        let dense_tables = Prepared::selected(extended, 1);
        let dense_plan = dense_tables.tables().bind(extended).unwrap();
        let factor_values = inputs::<M>(extended.size());
        let factor = expansion.view(&factor_values);
        let mut natural_factor = vec![PastaField::ZERO; extended.size()];
        expansion
            .layout()
            .copy_to_natural(&factor_values, &mut natural_factor);
        let mut output = vec![PastaField::ZERO; extended.size()];
        let mut natural = output.clone();
        let mut group = criterion.benchmark_group(format!(
            "{field}/expansion/{}/{shift_name}",
            extended.size()
        ));
        group.throughput(Throughput::Elements(extended.size() as u64));
        // Equal polynomial inputs, table families, serial execution, and consumers.
        // Native cases expose kernel costs; natural cases include residue conversion.
        for prefix in [0, 1, 5, base.size()] {
            for product in [false, true] {
                if product && prefix != 1 && prefix != 5 {
                    continue;
                }
                for method in ["dense", "prefix", "residues"] {
                    for natural_output in [false, true] {
                        if natural_output && method != "residues" {
                            continue;
                        }
                        let name = format!(
                            "compare/{prefix}/{}/{method}/{}",
                            if product { "product" } else { "evaluate" },
                            if natural_output { "natural" } else { "native" }
                        );
                        let expansion =
                            Expansion::new(coefficient_plan, extended, Some(scales)).unwrap();
                        group.bench_function(name, |b| {
                            b.iter(|| {
                                let input = black_box(&coefficients[..prefix]);
                                match method {
                                    "dense" => {
                                        output[..prefix].copy_from_slice(input);
                                        output[prefix..].fill(PastaField::ZERO);
                                        dense_plan
                                            .forward(
                                                &mut output,
                                                ExecutionOptions::default(),
                                                &SerialExecutor,
                                                &mut [],
                                            )
                                            .unwrap();
                                    }
                                    "prefix" => dense_plan
                                        .execute(
                                            TransformRequest {
                                                input_storage: InputStorage::Preserve,
                                                support: InputSupport::Prefix(input.len()),
                                                ..TransformRequest::new(Direction::Forward)
                                            },
                                            Some(CoefficientView::normalized(input)),
                                            &mut output,
                                            ExecutionOptions::default(),
                                            &SerialExecutor,
                                            &mut [],
                                        )
                                        .unwrap(),
                                    _ if product => expansion
                                        .short_product(
                                            input,
                                            factor,
                                            &mut output,
                                            ExecutionOptions::default(),
                                            &SerialExecutor,
                                            &mut [],
                                        )
                                        .unwrap(),
                                    _ => expansion
                                        .coefficients(
                                            input,
                                            &mut output,
                                            ExecutionOptions::default(),
                                            &SerialExecutor,
                                            &mut [],
                                        )
                                        .unwrap(),
                                }
                                if product && method != "residues" {
                                    for (value, factor) in output.iter_mut().zip(&natural_factor) {
                                        *value = value.mul(factor);
                                    }
                                }
                                if natural_output {
                                    expansion.layout().copy_to_natural(&output, &mut natural);
                                    black_box(&natural);
                                } else {
                                    black_box(&output);
                                }
                            })
                        });
                    }
                }
            }
        }
        for prepared in [false, true] {
            let scales = prepared.then_some(scales);
            for runner in runners {
                let options = ExecutionOptions::default()
                    .with_task_budget(TaskBudget::new(runner.tasks).unwrap());
                let profile = if prepared { "scales" } else { "no_scales" };
                for from_evaluations in [false, true] {
                    let expansion = Expansion::new(
                        if from_evaluations {
                            plan
                        } else {
                            coefficient_plan
                        },
                        extended,
                        scales,
                    )
                    .unwrap();
                    let required = if from_evaluations {
                        expansion.evaluation_scratch(options)
                    } else {
                        expansion.coefficient_scratch(options)
                    }
                    .unwrap();
                    let mut scratch = vec![PastaField::ZERO; required.field_elements];
                    let table_bytes = if from_evaluations {
                        tables.bytes()
                    } else {
                        tables.forward.len() * 32
                    } + scales.map_or(0, |scales| scales.as_slice().len() * 32);
                    eprintln!(
                        "{field}/expansion/{}/{shift_name}/{profile}/{}/automatic/{}: tables {table_bytes} bytes; scratch {} bytes; dense tables {} bytes",
                        extended.size(),
                        runner.name,
                        if from_evaluations {
                            "evaluations"
                        } else {
                            "coefficients"
                        },
                        scratch.len() * 32,
                        dense_tables.bytes()
                    );
                    group.bench_function(
                        format!(
                            "{profile}/{}/automatic/{}",
                            runner.name,
                            if from_evaluations {
                                "evaluations"
                            } else {
                                "coefficients"
                            }
                        ),
                        |b| {
                            runner.install(|| {
                                b.iter(|| {
                                    if from_evaluations {
                                        expansion
                                            .evaluations(
                                                black_box(&evaluations),
                                                &mut output,
                                                options,
                                                runner,
                                                &mut scratch,
                                            )
                                            .unwrap();
                                    } else {
                                        expansion
                                            .coefficients(
                                                black_box(&coefficients),
                                                &mut output,
                                                options,
                                                runner,
                                                &mut scratch,
                                            )
                                            .unwrap();
                                    }
                                    black_box(&output);
                                })
                            })
                        },
                    );
                }
            }
        }
        group.finish();
    }
}

fn interpolation<M: PrimeModulus>(
    criterion: &mut Criterion,
    field: &str,
    shift_name: &str,
    shift: PastaField<M>,
) {
    let domains = [14, 13, 12].map(|log| Domain::new(log).unwrap().coset(shift).unwrap());
    let tables = domains.map(Prepared::new);
    let plans = core::array::from_fn::<_, 3, _>(|i| tables[i].tables().bind(domains[i]).unwrap());
    let input = domains.map(|domain| inputs::<M>(domain.size()));
    let mut scratch = vec![
        PastaField::ZERO;
        plans[0]
            .scratch_requirements(PARALLEL)
            .unwrap()
            .field_elements
    ];
    let mut group = criterion.benchmark_group(format!("{field}/class_interpolation/{shift_name}"));
    group.throughput(Throughput::Elements(domains[0].size() as u64));
    group.bench_function("fused", |b| {
        b.iter_batched_ref(
            || input.clone(),
            |[output, a, b]| {
                InterpolationPlan::new(
                    plans.map(|plan| (plan, ElementOrder::Natural)),
                    false,
                    StorageLayout::Contiguous,
                    ExecutionOptions::default(),
                )
                .unwrap()
                .execute(
                    [black_box(output), black_box(a), black_box(b)],
                    [&mut [], &mut [], &mut []],
                    &SerialExecutor,
                );
                black_box(output);
            },
            BatchSize::PerIteration,
        );
    });
    group.bench_function("separate", |b| {
        b.iter_batched_ref(
            || input.clone(),
            |values| {
                for (plan, values) in plans.iter().zip(values.iter_mut()) {
                    plan.inverse(black_box(values), PARALLEL, &SerialExecutor, &mut scratch)
                        .unwrap();
                }
                let [output, a, b] = values;
                for lift in [a, b] {
                    for (output, coefficient) in output.iter_mut().zip(lift) {
                        *output = output.add(coefficient);
                    }
                }
                black_box(output);
            },
            BatchSize::PerIteration,
        );
    });
    group.finish();
}

fn field_benchmarks<M: PrimeModulus>(criterion: &mut Criterion, field: &str, runners: &[Runner]) {
    for (name, shift) in [
        ("subgroup", PastaField::ONE),
        ("zeta", PastaField::ZETA),
        ("generic_7", PastaField::from_u64(7)),
    ] {
        transforms::<M>(criterion, field, name, shift, runners);
        expansions::<M>(criterion, field, name, shift, runners);
        interpolation::<M>(criterion, field, name, shift);
    }
}

fn benchmarks(criterion: &mut Criterion) {
    let runners: Vec<_> = [
        ("serial", 1, false),
        ("tiled_serial", 4, false),
        ("rayon_1", 1, true),
        ("rayon_2", 2, true),
        ("rayon_4", 4, true),
    ]
    .into_iter()
    .map(|(name, tasks, threaded)| Runner {
        name,
        tasks,
        pool: threaded.then(|| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(tasks)
                .build()
                .unwrap()
        }),
    })
    .collect();
    field_benchmarks::<PallasBase>(criterion, "Fp", &runners);
    field_benchmarks::<PallasScalar>(criterion, "Fq", &runners);
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
