//! Measures table preparation separately from execution that reuses buffers.
//!
//! Every case uses a serial executor, including tiled transforms. Comparisons
//! therefore measure arithmetic and scheduling overhead without pool effects.

use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use zakura_udon::{
    fft::{
        Class, CosetDomain, Domain, ExecutionOptions, Expansion, ExpansionOptions, InputOrder,
        Plan, ResidueView, SerialExecutor, TableRequirements, Tables, TablesMut,
        interpolate_classes, reference,
    },
    field::{CanonicalUint, PallasBase, PallasScalar, PastaField, PrimeModulus},
};

struct Prepared<M: PrimeModulus> {
    permutation: Vec<u32>,
    forward: Vec<PastaField<M>>,
    inverse: Vec<PastaField<M>>,
    finish: Vec<PastaField<M>>,
    scales: Vec<PastaField<M>>,
}

impl<M: PrimeModulus> Prepared<M> {
    fn new(domain: CosetDomain<M>) -> Self {
        let sizes = TableRequirements::for_domain(domain);
        let mut result = Self {
            permutation: vec![0; sizes.permutation],
            forward: vec![PastaField::ZERO; sizes.twiddles],
            inverse: vec![PastaField::ZERO; sizes.twiddles],
            finish: vec![PastaField::ZERO; sizes.twiddles],
            scales: vec![PastaField::ZERO; sizes.inverse_scales],
        };
        result.prepare(domain);
        result
    }

    fn prepare(&mut self, domain: CosetDomain<M>) {
        TablesMut {
            bit_reversed: Some(&mut self.permutation),
            forward: Some(&mut self.forward),
            inverse: Some(&mut self.inverse),
            inverse_finish: Some(&mut self.finish),
            inverse_scales: Some(&mut self.scales),
        }
        .prepare(domain)
        .unwrap();
    }

    fn tables(&self) -> Tables<'_, M> {
        Tables {
            bit_reversed: Some(&self.permutation),
            forward: Some(&self.forward),
            inverse: Some(&self.inverse),
            inverse_finish: Some(&self.finish),
            inverse_scales: Some(&self.scales),
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

const TILED: ExecutionOptions = ExecutionOptions {
    tile_len: 2048,
    columns_per_task: 128,
    max_tasks: 4,
};

// Keep at least two tiles even in the smallest benchmark domain.
fn tiled(size: usize) -> ExecutionOptions {
    ExecutionOptions {
        tile_len: TILED.tile_len.min(size / 2),
        ..TILED
    }
}

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
) {
    for log_size in [11, 14, 20] {
        let domain = Domain::new(log_size).unwrap().coset(shift).unwrap();
        let mut tables = Prepared::new(domain);
        let input = inputs::<M>(domain.size());
        let mut setup = criterion.benchmark_group(format!("{field}/fft_setup/{shift_name}"));
        setup.throughput(Throughput::Elements(domain.size() as u64));
        setup.bench_function(BenchmarkId::new("tables", domain.size()), |b| {
            b.iter(|| {
                black_box(&mut tables).prepare(black_box(domain));
                black_box(tables.tables());
            });
        });
        setup.finish();
        let tiled = tiled(domain.size());
        let prepared = Plan::new(domain, tables.tables()).unwrap();
        let unprepared = Plan::without_tables(domain);
        let mut scratch =
            vec![PastaField::ZERO; prepared.scratch_requirements(tiled).unwrap().field_elements];
        eprintln!(
            "{field}/fft/{}/{shift_name}: serial scratch 0 bytes; tiled scratch {} bytes",
            domain.size(),
            core::mem::size_of_val(scratch.as_slice())
        );
        let mut group =
            criterion.benchmark_group(format!("{field}/fft/{}/{shift_name}", domain.size()));
        group.throughput(Throughput::Elements(domain.size() as u64));
        for inverse in [false, true] {
            let direction = if inverse { "inverse" } else { "forward" };
            group.bench_function(format!("reference_{direction}"), |b| {
                b.iter_batched_ref(
                    || input.clone(),
                    |values| {
                        reference_coset(black_box(values), black_box(domain), inverse);
                        black_box(values);
                    },
                    BatchSize::PerIteration,
                );
            });
            for (name, plan, options) in [
                ("unprepared", unprepared, ExecutionOptions::serial()),
                ("prepared", prepared, ExecutionOptions::serial()),
                ("tiled_unprepared", unprepared, tiled),
                ("tiled_prepared", prepared, tiled),
            ] {
                group.bench_function(format!("{name}_{direction}"), |b| {
                    b.iter_batched_ref(
                        || input.clone(),
                        |values| {
                            if inverse {
                                black_box(plan)
                                    .inverse(
                                        black_box(values),
                                        options,
                                        &SerialExecutor,
                                        &mut scratch,
                                    )
                                    .unwrap();
                            } else {
                                black_box(plan)
                                    .forward(
                                        black_box(values),
                                        options,
                                        &SerialExecutor,
                                        &mut scratch,
                                    )
                                    .unwrap();
                            }
                            black_box(values);
                        },
                        BatchSize::PerIteration,
                    );
                });
            }
        }
        let mut output = vec![PastaField::ZERO; domain.size()];
        for prefix_len in [5, domain.size() / 8] {
            for (name, plan, options) in [
                ("unprepared", unprepared, ExecutionOptions::serial()),
                ("prepared", prepared, ExecutionOptions::serial()),
                ("tiled_unprepared", unprepared, tiled),
                ("tiled_prepared", prepared, tiled),
            ] {
                group.bench_function(format!("forward_prefix_{prefix_len}/{name}"), |b| {
                    b.iter(|| {
                        black_box(plan)
                            .forward_prefix(
                                black_box(&input[..prefix_len]),
                                black_box(&mut output),
                                options,
                                &SerialExecutor,
                                &mut scratch,
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

fn expansions<M: PrimeModulus>(
    criterion: &mut Criterion,
    field: &str,
    shift_name: &str,
    shift: PastaField<M>,
) {
    let base = Domain::new(11).unwrap().subgroup();
    let tables = Prepared::new(base);
    let plan = Plan::new(base, tables.tables()).unwrap();
    let coefficients = inputs::<M>(base.size());
    let mut evaluations = coefficients.clone();
    plan.forward(
        &mut evaluations,
        ExecutionOptions::serial(),
        &SerialExecutor,
        &mut [],
    )
    .unwrap();
    for log_size in [12, 14] {
        let extended = Domain::new(log_size).unwrap().coset(shift).unwrap();
        let expansion = Expansion::new(plan, extended, None).unwrap();
        let mut scales = vec![PastaField::ZERO; expansion.scale_count()];
        let mut setup = criterion.benchmark_group(format!("{field}/fft_setup/{shift_name}"));
        setup.bench_function(BenchmarkId::new("residue_scales", extended.size()), |b| {
            b.iter(|| expansion.prepare_scales(black_box(&mut scales)).unwrap());
        });
        setup.finish();
        expansion.prepare_scales(&mut scales).unwrap();
        let expansion = Expansion::new(plan, extended, Some(&scales)).unwrap();
        let factor_values = inputs::<M>(extended.size());
        let factor = ResidueView::new(&factor_values, expansion.layout()).unwrap();
        let mut output = vec![PastaField::ZERO; extended.size()];
        let options = ExpansionOptions {
            max_residue_tasks: 4,
            transform: tiled(base.size()),
        };
        let mut scratch = vec![
            PastaField::ZERO;
            expansion
                .coefficient_scratch(options)
                .unwrap()
                .field_elements
        ];
        let mut group = criterion.benchmark_group(format!(
            "{field}/expansion/{}/{shift_name}",
            extended.size()
        ));
        group.throughput(Throughput::Elements(extended.size() as u64));
        group.bench_function("coefficients", |b| {
            b.iter(|| {
                expansion
                    .coefficients(
                        black_box(&coefficients),
                        black_box(&mut output),
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                black_box(&output);
            })
        });
        group.bench_function("evaluations", |b| {
            b.iter(|| {
                expansion
                    .evaluations(
                        black_box(&evaluations),
                        black_box(&mut output),
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                black_box(&output);
            })
        });
        group.bench_function("short_product_5", |b| {
            b.iter(|| {
                expansion
                    .short_product(
                        black_box(&coefficients[..5]),
                        black_box(factor),
                        black_box(&mut output),
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                black_box(&output);
            })
        });
        let dense_tables = Prepared::new(extended);
        let dense_plan = Plan::new(extended, dense_tables.tables()).unwrap();
        // The dense baseline includes zero padding but keeps its natural-order
        // output; conversion to the expansion's residue layout is not timed.
        group.bench_function("zero_padded_fft", |b| {
            b.iter(|| {
                output[..base.size()].copy_from_slice(black_box(&coefficients));
                output[base.size()..].fill(PastaField::ZERO);
                dense_plan
                    .forward(
                        black_box(&mut output),
                        ExecutionOptions::serial(),
                        &SerialExecutor,
                        &mut [],
                    )
                    .unwrap();
                black_box(&output);
            })
        });
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
    let plans =
        core::array::from_fn::<_, 3, _>(|i| Plan::new(domains[i], tables[i].tables()).unwrap());
    let input = domains.map(|domain| inputs::<M>(domain.size()));
    let mut scratch =
        vec![PastaField::ZERO; plans[0].scratch_requirements(TILED).unwrap().field_elements];
    let mut group = criterion.benchmark_group(format!("{field}/class_interpolation/{shift_name}"));
    group.throughput(Throughput::Elements(domains[0].size() as u64));
    group.bench_function("fused", |b| {
        b.iter_batched_ref(
            || input.clone(),
            |[output, a, b]| {
                let mut output =
                    Class::new(plans[0], black_box(output), InputOrder::Natural).unwrap();
                let mut lifts = [
                    Class::new(plans[1], black_box(a), InputOrder::Natural).unwrap(),
                    Class::new(plans[2], black_box(b), InputOrder::Natural).unwrap(),
                ];
                interpolate_classes(
                    &mut output,
                    &mut lifts,
                    TILED,
                    &SerialExecutor,
                    &mut scratch,
                )
                .unwrap();
                black_box(output.values());
            },
            BatchSize::PerIteration,
        );
    });
    group.bench_function("separate", |b| {
        b.iter_batched_ref(
            || input.clone(),
            |values| {
                for (plan, values) in plans.iter().zip(values.iter_mut()) {
                    plan.inverse(black_box(values), TILED, &SerialExecutor, &mut scratch)
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

fn field_benchmarks<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    for (name, shift) in [
        ("zeta", PastaField::zeta()),
        ("generic_7", PastaField::from_u64(7)),
    ] {
        transforms::<M>(criterion, field, name, shift);
        expansions::<M>(criterion, field, name, shift);
        interpolation::<M>(criterion, field, name, shift);
    }
}

fn benchmarks(criterion: &mut Criterion) {
    field_benchmarks::<PallasBase>(criterion, "Fp");
    field_benchmarks::<PallasScalar>(criterion, "Fq");
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
