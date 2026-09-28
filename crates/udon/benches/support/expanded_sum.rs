// Constant-size `chunks_exact` predates `as_chunks`; migration is upstream
// work, and the pinned toolchain's Clippy predates the lint itself.
#![allow(unknown_lints)]
#![allow(clippy::chunks_exact_to_as_chunks)]

use super::{Buffers, Pool, shared_scalars::measure, values};
use criterion::{BenchmarkId, Criterion};
use rayon::prelude::*;
use std::hint::black_box;
use zakura_udon::{
    curve::{
        AffinePoint, CurveTableEntry, FixedBaseDescription, FixedBaseTable, PastaCurve,
        PreparedAffinePoint, ProjectivePoint,
    },
    exec::{ExecutionOptions, SerialExecutor, TaskBudget},
    field::PastaField,
    msm::{Bases, Input},
};

fn separate<C: PastaCurve, E: CurveTableEntry<C>>(
    tables: &[FixedBaseTable<'_, C, E>],
    scalars: &[PastaField<C::Scalar>],
) -> ProjectivePoint<C> {
    tables
        .iter()
        .zip(scalars)
        .fold(ProjectivePoint::IDENTITY, |sum, (table, scalar)| {
            sum.add(&table.mul(scalar))
        })
}

fn binary<C: PastaCurve>(
    base: &AffinePoint<C>,
    scalar: &PastaField<C::Scalar>,
) -> ProjectivePoint<C> {
    let mut sum = ProjectivePoint::IDENTITY;
    for byte in scalar.to_bytes().iter().rev() {
        for bit in (0..8).rev() {
            sum = sum.double();
            if byte & (1 << bit) != 0 {
                sum = sum.add_mixed(base);
            }
        }
    }
    sum
}

fn layout<C: PastaCurve, E: CurveTableEntry<C>>(
    c: &mut Criterion,
    name: &str,
    layout: &str,
    bases: &[AffinePoint<C>],
) {
    let full = values::<C::Scalar>(bases.len());
    let short: Vec<_> = (0..bases.len())
        .map(|i| PastaField::from_u64(0x1234_5678_9abc_def0_u64.wrapping_mul(i as u64 + 1)))
        .collect();
    let tiny: Vec<_> = (0..bases.len())
        .map(|i| match i % 4 {
            0 => PastaField::ZERO,
            1 => PastaField::ONE,
            2 => PastaField::<C::Scalar>::ONE.neg(),
            _ => PastaField::from_u64(17),
        })
        .collect();
    for width in [2, 4, 8] {
        let description = FixedBaseDescription { window_bits: width };
        let r = description.requirements().unwrap();
        for n in [1, 8, 32, 128] {
            if width != 4 && (n == 1 || n == 128) {
                continue;
            }
            let bases = &bases[..n];
            let mut entries = vec![E::from_affine(&bases[0]); n * r.table_entries];
            let mut projective = vec![ProjectivePoint::IDENTITY; r.table_entries];
            let mut preparation = vec![PastaField::ZERO; r.table_entries];
            let mut group = c.benchmark_group(format!("{name}/expanded_sum/{layout}/w{width}/{n}"));
            group.bench_function("prepare", |b| {
                b.iter(|| {
                    for (base, entries) in black_box(bases)
                        .iter()
                        .zip(entries.chunks_exact_mut(r.table_entries))
                    {
                        black_box(
                            FixedBaseTable::prepare_with(
                                description,
                                base,
                                entries,
                                &mut projective,
                                &mut preparation,
                            )
                            .unwrap(),
                        );
                    }
                })
            });
            for (base, entries) in bases.iter().zip(entries.chunks_exact_mut(r.table_entries)) {
                FixedBaseTable::prepare_with(
                    description,
                    base,
                    entries,
                    &mut projective,
                    &mut preparation,
                )
                .unwrap();
            }
            let tables: Vec<_> = bases
                .iter()
                .zip(entries.chunks_exact(r.table_entries))
                .map(|(base, entries)| FixedBaseTable::bind(description, base, entries).unwrap())
                .collect();
            let capacity = FixedBaseTable::sum_scratch_len(&tables).unwrap();
            let mut affine = vec![AffinePoint::GENERATOR; capacity];
            let mut fields = vec![PastaField::ZERO; capacity];
            for (shape, scalars) in [("full", &full), ("short", &short), ("tiny", &tiny)] {
                let scalars = &scalars[..n];
                let expected = bases
                    .iter()
                    .zip(scalars)
                    .fold(ProjectivePoint::IDENTITY, |sum, (base, scalar)| {
                        sum.add(&binary(base, scalar))
                    });
                let input = Input::new(Bases::Affine(bases), scalars);
                let options = ExecutionOptions::default();
                let mut scratch = Buffers::new(input.requirements(options).unwrap());
                assert_eq!(separate(&tables, scalars), expected);
                assert_eq!(
                    input
                        .execute(options, &SerialExecutor, scratch.borrow())
                        .unwrap(),
                    expected
                );
                assert_eq!(
                    FixedBaseTable::sum(&tables, scalars, &mut affine, &mut fields),
                    expected
                );
                group.bench_function(BenchmarkId::new("separate", shape), |b| {
                    b.iter(|| black_box(separate(black_box(&tables), black_box(scalars))))
                });
                group.bench_function(BenchmarkId::new("ordinary", shape), |b| {
                    b.iter(|| {
                        black_box(
                            black_box(input)
                                .execute(options, &SerialExecutor, scratch.borrow())
                                .unwrap(),
                        )
                    })
                });
                group.bench_function(BenchmarkId::new("joint", shape), |b| {
                    b.iter(|| {
                        black_box(FixedBaseTable::sum(
                            black_box(&tables),
                            black_box(scalars),
                            &mut affine,
                            &mut fields,
                        ))
                    })
                });
                if shape == "full" && capacity > 682 {
                    let options = options.with_memory_limit(65536);
                    let mut scratch = Buffers::new(input.requirements(options).unwrap());
                    assert_eq!(
                        input
                            .execute(options, &SerialExecutor, scratch.borrow())
                            .unwrap(),
                        expected
                    );
                    assert_eq!(
                        FixedBaseTable::sum(
                            &tables,
                            scalars,
                            &mut affine[..682],
                            &mut fields[..682]
                        ),
                        expected
                    );
                    group.bench_function("joint_64k", |b| {
                        b.iter(|| {
                            black_box(FixedBaseTable::sum(
                                black_box(&tables),
                                black_box(scalars),
                                &mut affine[..682],
                                &mut fields[..682],
                            ))
                        })
                    });
                    group.bench_function("ordinary_64k", |b| {
                        b.iter(|| {
                            black_box(
                                black_box(input)
                                    .execute(options, &SerialExecutor, scratch.borrow())
                                    .unwrap(),
                            )
                        })
                    });
                }
            }
            drop(tables);
            // Binding uses stack handles; table and scratch allocation stay
            // outside the preparation-inclusive timing boundary.
            group.bench_function("prepare_and_joint", |b| {
                b.iter(|| {
                    for (base, entries) in black_box(bases)
                        .iter()
                        .zip(entries.chunks_exact_mut(r.table_entries))
                    {
                        FixedBaseTable::prepare_with(
                            description,
                            base,
                            entries,
                            &mut projective,
                            &mut preparation,
                        )
                        .unwrap();
                    }
                    let first =
                        FixedBaseTable::bind(description, &bases[0], &entries[..r.table_entries])
                            .unwrap();
                    let mut handles = [first; 128];
                    for (handle, (base, entries)) in handles
                        .iter_mut()
                        .zip(bases.iter().zip(entries.chunks_exact(r.table_entries)))
                    {
                        *handle = FixedBaseTable::bind(description, base, entries).unwrap();
                    }
                    black_box(FixedBaseTable::sum(
                        &handles[..n],
                        black_box(&full[..n]),
                        &mut affine,
                        &mut fields,
                    ));
                })
            });
            group.finish();
        }
    }
}

pub(super) fn bench<C: PastaCurve>(c: &mut Criterion, name: &str, bases: &[AffinePoint<C>]) {
    layout::<C, AffinePoint<C>>(c, name, "affine", bases);
    layout::<C, PreparedAffinePoint<C>>(c, name, "cached", bases);
    let bases = &bases[..128];
    let scalars = values::<C::Scalar>(128);
    let mut entries = vec![AffinePoint::GENERATOR; 128 * 256];
    for (base, entries) in bases.iter().zip(entries.chunks_exact_mut(256)) {
        FixedBaseTable::prepare_with(
            FixedBaseDescription::default(),
            base,
            entries,
            &mut [ProjectivePoint::IDENTITY; 256],
            &mut [PastaField::ZERO; 256],
        )
        .unwrap();
    }
    let tables: Vec<_> = bases
        .iter()
        .zip(entries.chunks_exact(256))
        .map(|(base, entries)| {
            FixedBaseTable::bind(FixedBaseDescription::default(), base, entries).unwrap()
        })
        .collect();
    let expected = bases
        .iter()
        .zip(&scalars)
        .fold(ProjectivePoint::IDENTITY, |sum, (base, scalar)| {
            sum.add(&binary(base, scalar))
        });
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        let mut group = c.benchmark_group(format!("{name}/expanded_sum/workers_{workers}/128"));
        for capacity in [2048, 170] {
            let limit = if capacity == 170 { "64k" } else { "all" };
            let mut affine = vec![AffinePoint::GENERATOR; capacity * 4];
            let mut fields = vec![PastaField::ZERO; capacity * 4];
            let mut outputs = [ProjectivePoint::IDENTITY; 4];
            let mut run = || {
                outputs
                    .par_iter_mut()
                    .zip(affine.par_chunks_mut(capacity))
                    .zip(fields.par_chunks_mut(capacity))
                    .enumerate()
                    .for_each(|(i, ((out, affine), field))| {
                        *out = FixedBaseTable::sum(
                            black_box(&tables[i * 32..(i + 1) * 32]),
                            black_box(&scalars[i * 32..(i + 1) * 32]),
                            affine,
                            field,
                        );
                    });
                outputs
                    .iter()
                    .fold(ProjectivePoint::IDENTITY, |sum, p| sum.add(p))
            };
            assert_eq!(pool.install(&mut run), expected);
            group.bench_function(BenchmarkId::new("joint", limit), |b| {
                measure(b, Some(&pool), || {
                    black_box(run());
                })
            });
            let mut options =
                ExecutionOptions::default().with_task_budget(TaskBudget::new(4).unwrap());
            if capacity == 170 {
                options = options.with_memory_limit(65536);
            }
            let input = Input::new(Bases::Affine(bases), &scalars);
            let mut scratch = Buffers::new(input.requirements(options).unwrap());
            assert_eq!(
                pool.install(|| input.execute(options, &Pool, scratch.borrow()).unwrap()),
                expected
            );
            group.bench_function(BenchmarkId::new("ordinary", limit), |b| {
                measure(b, Some(&pool), || {
                    black_box(
                        black_box(input)
                            .execute(options, &Pool, scratch.borrow())
                            .unwrap(),
                    );
                })
            });
        }
        group.finish();
    }
}
