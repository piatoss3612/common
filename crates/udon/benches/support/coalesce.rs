use super::{Buffers, Pool, shared_scalars::measure, values};
use criterion::Criterion;
use std::hint::black_box;
use zakura_udon::{
    curve::{
        PastaCurve, Point, ProjectivePoint,
        msm::{Bases, CoalescingKey, CoalescingPlan, IndexedCoalescingPlan, Input},
    },
    exec::{ExecutionOptions, SerialExecutor, TaskBudget},
    field::PastaField,
};

fn binary<C: PastaCurve>(base: Point<C>, scalar: PastaField<C::Scalar>) -> ProjectivePoint<C> {
    let mut sum = ProjectivePoint::IDENTITY;
    for byte in scalar.to_bytes().iter().rev() {
        for bit in (0..8).rev() {
            sum = sum.double();
            if byte & (1 << bit) != 0 {
                sum = sum.add(&base.to_projective());
            }
        }
    }
    sum
}

pub(super) fn bench<C: PastaCurve>(c: &mut Criterion, name: &str) {
    let full = values::<C::Scalar>(2048);
    let basis: Vec<_> = full
        .iter()
        .map(|s| Point::<C>::GENERATOR.mul_projective(s).to_point())
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for n in [8, 128, 2048] {
        for indexed in [false, true] {
            for shape in ["unique", "repeated", "opposite", "cancelled"] {
                if indexed && shape == "opposite" {
                    continue;
                }
                let k = if shape == "unique" {
                    n
                } else {
                    (n / 16).max(1)
                };
                let indices: Vec<_> = (0..n)
                    .map(|i| {
                        let position = if shape == "cancelled" { i / 2 } else { i };
                        ((position * 17 + 3) % k) as u32
                    })
                    .collect();
                let points: Vec<_> = indices
                    .iter()
                    .enumerate()
                    .map(|(i, &index)| {
                        let base = basis[index as usize];
                        let negative = (shape == "opposite" && (i / k) % 2 != 0)
                            || (shape == "cancelled" && i % 2 != 0);
                        if !indexed && negative {
                            base.neg()
                        } else {
                            base
                        }
                    })
                    .collect();
                let row: Vec<_> = (0..n)
                    .map(|i| {
                        if shape == "cancelled" {
                            let s = full[i / 2];
                            if indexed && i % 2 != 0 { s.neg() } else { s }
                        } else {
                            full[i]
                        }
                    })
                    .collect();
                let expected = points
                    .iter()
                    .zip(&row)
                    .fold(ProjectivePoint::IDENTITY, |sum, (&base, &s)| {
                        sum.add(&binary(base, s))
                    });
                let mut keys = vec![CoalescingKey::EMPTY; n];
                let mut order = vec![0; n];
                let point_plan = CoalescingPlan::prepare(&points, &mut keys);
                let index_plan =
                    IndexedCoalescingPlan::prepare(Bases::Points(&basis), &indices, &mut order)
                        .unwrap();
                let mut prep_keys = vec![CoalescingKey::EMPTY; n];
                let mut prep_order = vec![0; n];
                let groups = if indexed {
                    index_plan.groups()
                } else {
                    point_plan.groups()
                };
                let mut output_points = vec![Point::IDENTITY; groups];
                let mut output_indices = vec![0; groups];
                let mut sums = vec![PastaField::ZERO; groups];
                let layout = if indexed { "indices" } else { "points" };
                let mut preparation =
                    c.benchmark_group(format!("{name}/coalesce_prepare/{layout}/{shape}/{n}"));
                preparation.bench_function("prepare", |b| {
                    b.iter(|| {
                        if indexed {
                            black_box(
                                IndexedCoalescingPlan::prepare(
                                    Bases::Points(black_box(&basis)),
                                    black_box(&indices),
                                    &mut prep_order,
                                )
                                .unwrap(),
                            );
                        } else {
                            black_box(CoalescingPlan::prepare(black_box(&points), &mut prep_keys));
                        }
                    })
                });
                preparation.bench_function("aggregate", |b| {
                    b.iter(|| {
                        if indexed {
                            black_box(index_plan.with_scalars(
                                black_box(&row),
                                &mut output_indices,
                                &mut sums,
                            ));
                        } else {
                            black_box(point_plan.with_scalars(
                                black_box(&row),
                                &mut output_points,
                                &mut sums,
                            ));
                        }
                    })
                });
                preparation.finish();
                let configurations = if n == 2048 && (shape == "repeated" || shape == "unique") {
                    &[(1, None), (4, None), (1, Some(65536)), (4, Some(65536))][..]
                } else {
                    &[(1, None)][..]
                };
                for &(tasks, limit) in configurations {
                    let mut options = ExecutionOptions::default()
                        .with_task_budget(TaskBudget::new(tasks).unwrap());
                    if let Some(limit) = limit {
                        options = options.with_memory_limit(limit);
                    }
                    let direct = if indexed {
                        Input::indexed(Bases::Points(&basis), &indices, &row).unwrap()
                    } else {
                        Input::new(Bases::Points(&points), &row)
                    };
                    let merged = if indexed {
                        index_plan.with_scalars(&row, &mut output_indices, &mut sums)
                    } else {
                        point_plan.with_scalars(&row, &mut output_points, &mut sums)
                    };
                    let mut direct_scratch = Buffers::new(direct.requirements(options).unwrap());
                    let mut merged_scratch = Buffers::new(merged.requirements(options).unwrap());
                    for parallel in [false, true] {
                        let check = || {
                            assert_eq!(
                                direct
                                    .execute(options, &Pool, direct_scratch.borrow())
                                    .unwrap(),
                                expected
                            );
                            assert_eq!(
                                merged
                                    .execute(options, &Pool, merged_scratch.borrow())
                                    .unwrap(),
                                expected
                            );
                        };
                        if parallel {
                            pool.install(check);
                        } else {
                            assert_eq!(
                                direct
                                    .execute(options, &SerialExecutor, direct_scratch.borrow())
                                    .unwrap(),
                                expected
                            );
                            assert_eq!(
                                merged
                                    .execute(options, &SerialExecutor, merged_scratch.borrow())
                                    .unwrap(),
                                expected
                            );
                        }
                    }
                    let cap = limit.map_or_else(|| "all".to_owned(), |l| l.to_string());
                    let mut group = c.benchmark_group(format!(
                        "{name}/coalesce/{layout}/{shape}/tasks_{tasks}/cap_{cap}/{n}"
                    ));
                    for method in ["direct", "retained", "prepare_and_execute"] {
                        group.bench_function(method, |b| {
                            measure(b, (tasks > 1).then_some(&pool), || {
                                let result = if method == "direct" {
                                    let input = if indexed {
                                        Input::indexed(
                                            Bases::Points(black_box(&basis)),
                                            black_box(&indices),
                                            black_box(&row),
                                        )
                                        .unwrap()
                                    } else {
                                        Input::new(
                                            Bases::Points(black_box(&points)),
                                            black_box(&row),
                                        )
                                    };
                                    input
                                        .execute(options, &Pool, direct_scratch.borrow())
                                        .unwrap()
                                } else {
                                    let input = if indexed {
                                        let plan = if method == "retained" {
                                            black_box(index_plan)
                                        } else {
                                            IndexedCoalescingPlan::prepare(
                                                Bases::Points(black_box(&basis)),
                                                black_box(&indices),
                                                &mut prep_order,
                                            )
                                            .unwrap()
                                        };
                                        plan.with_scalars(
                                            black_box(&row),
                                            &mut output_indices,
                                            &mut sums,
                                        )
                                    } else {
                                        let plan = if method == "retained" {
                                            black_box(point_plan)
                                        } else {
                                            CoalescingPlan::prepare(
                                                black_box(&points),
                                                &mut prep_keys,
                                            )
                                        };
                                        plan.with_scalars(
                                            black_box(&row),
                                            &mut output_points,
                                            &mut sums,
                                        )
                                    };
                                    input
                                        .execute(options, &Pool, merged_scratch.borrow())
                                        .unwrap()
                                };
                                black_box(result);
                            })
                        });
                    }
                    group.finish();
                }
            }
        }
    }
}
