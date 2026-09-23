use super::{Buffers, Pool, shared_scalars::measure};
use criterion::Criterion;
use std::hint::black_box;
use zakura_udon::{
    curve::{
        AffinePoint, PastaCurve, ProjectivePoint,
        msm::{Bases, Selection},
    },
    exec::{ExecutionOptions, SerialExecutor, TaskBudget},
    field::PastaField,
};

fn binary<C: PastaCurve>(
    base: AffinePoint<C>,
    scalar: PastaField<C::Scalar>,
) -> ProjectivePoint<C> {
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

pub(super) fn bench<C: PastaCurve>(
    c: &mut Criterion,
    name: &str,
    bases: &[AffinePoint<C>],
    full: &[PastaField<C::Scalar>],
) {
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let extra_base = bases[2048];
    let extra_scalar = full[2048];
    let extra = binary(extra_base, extra_scalar);
    for n in [8, 128, 2048] {
        let bases = Bases::Affine(&bases[..n]);
        let selection = Selection::new(bases);
        let mut counts = vec![0, 1, n / 16, n / 2, n];
        counts.sort_unstable();
        counts.dedup();
        for k in counts {
            let row: Vec<_> = (0..n)
                .map(|i| {
                    if (i * 17 + 3) % n < k {
                        full[i]
                    } else {
                        PastaField::ZERO
                    }
                })
                .collect();
            let known: Vec<_> = (0..n)
                .filter(|&i| (i * 17 + 3) % n < k)
                .map(|i| i as u32)
                .collect();
            let compact: Vec<_> = known.iter().map(|&i| row[i as usize]).collect();
            let retained = Selection::indexed(bases, &known).unwrap();
            let mut indices = vec![0; k];
            let mut scalars = vec![PastaField::ZERO; k];
            let expected = known.iter().zip(&compact).fold(extra, |sum, (&i, &s)| {
                let Bases::Affine(bases) = bases else {
                    unreachable!()
                };
                sum.add(&binary(bases[i as usize], s))
            });
            let mut preparation = c.benchmark_group(format!("{name}/nonzero_prepare/{n}/live_{k}"));
            preparation.bench_function("scan_compact", |b| {
                b.iter(|| {
                    black_box(
                        selection
                            .with_nonzero_scalars(black_box(&row), &mut indices, &mut scalars)
                            .unwrap(),
                    );
                })
            });
            preparation.bench_function("validate_known", |b| {
                b.iter(|| {
                    black_box(Selection::indexed(black_box(bases), black_box(&known)).unwrap());
                })
            });
            preparation.finish();
            let configurations = if n == 2048 && (k == n / 16 || k == n) {
                &[(1, None), (4, None), (1, Some(65536)), (4, Some(65536))][..]
            } else {
                &[(1, None)][..]
            };
            for &(tasks, limit) in configurations {
                let mut options =
                    ExecutionOptions::default().with_task_budget(TaskBudget::new(tasks).unwrap());
                if let Some(limit) = limit {
                    options = options.with_memory_limit(limit);
                }
                let direct = selection.with_scalars(&row);
                let input = selection
                    .with_nonzero_scalars(&row, &mut indices, &mut scalars)
                    .unwrap();
                assert_eq!(input.len(), k);
                let mut dense_scratch = Buffers::new(direct.requirements(options).unwrap());
                let mut compact_scratch = Buffers::new(input.requirements(options).unwrap());
                assert_eq!(
                    direct
                        .execute(options, &SerialExecutor, dense_scratch.borrow())
                        .unwrap()
                        .add(&extra),
                    expected
                );
                assert_eq!(
                    input
                        .execute(options, &SerialExecutor, compact_scratch.borrow())
                        .unwrap()
                        .add(&extra),
                    expected
                );
                assert_eq!(
                    retained
                        .with_scalars(&compact)
                        .execute(options, &SerialExecutor, compact_scratch.borrow())
                        .unwrap()
                        .add(&extra),
                    expected
                );
                pool.install(|| {
                    assert_eq!(
                        input
                            .execute(options, &Pool, compact_scratch.borrow())
                            .unwrap()
                            .add(&extra),
                        expected
                    );
                });
                let cap = limit.map_or_else(|| "all".to_owned(), |l| l.to_string());
                let mut group = c.benchmark_group(format!(
                    "{name}/nonzero/{n}/live_{k}/tasks_{tasks}/cap_{cap}"
                ));
                for method in ["dense", "scan_execute", "known_execute", "retained_execute"] {
                    group.bench_function(method, |b| {
                        measure(b, (tasks > 1).then_some(&pool), || {
                            let result = if method == "dense" {
                                selection
                                    .with_scalars(black_box(&row))
                                    .execute(options, &Pool, dense_scratch.borrow())
                                    .unwrap()
                            } else {
                                let input = if method == "scan_execute" {
                                    selection
                                        .with_nonzero_scalars(
                                            black_box(&row),
                                            &mut indices,
                                            &mut scalars,
                                        )
                                        .unwrap()
                                } else {
                                    let selection = if method == "known_execute" {
                                        Selection::indexed(black_box(bases), black_box(&known))
                                            .unwrap()
                                    } else {
                                        black_box(retained)
                                    };
                                    selection.with_scalars(black_box(&compact))
                                };
                                input
                                    .execute(options, &Pool, compact_scratch.borrow())
                                    .unwrap()
                            };
                            black_box(result.add(
                                &black_box(extra_base).mul_projective(black_box(&extra_scalar)),
                            ));
                        })
                    });
                }
                group.finish();
            }
        }
    }
}
