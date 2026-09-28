use super::{Buffers, Pool, shared_scalars::measure, values};
use criterion::{BenchmarkId, Criterion};
use std::hint::black_box;
use zakura_udon::{
    curve::{PastaCurve, Point, ProjectivePoint},
    exec::{ExecutionOptions, SerialExecutor, TaskBudget},
    field::{ConstantPrefix, PastaField},
    msm::{Bases, BasisSum, Input},
};

fn binary<C: PastaCurve>(point: Point<C>, scalar: PastaField<C::Scalar>) -> ProjectivePoint<C> {
    let mut sum = ProjectivePoint::IDENTITY;
    for byte in scalar.to_bytes().iter().rev() {
        for bit in (0..8).rev() {
            sum = sum.double();
            if byte & (1 << bit) != 0 {
                sum = sum.add(&point.to_projective());
            }
        }
    }
    sum
}

fn correction_input<'a, C: PastaCurve>(
    basis: BasisSum<'a, C>,
    tail: bool,
    constant: PastaField<C::Scalar>,
    indices: &'a [u32],
    values: &'a [PastaField<C::Scalar>],
    differences: &'a mut [PastaField<C::Scalar>],
) -> Input<'a, C> {
    if tail {
        basis.tail_corrections(
            ConstantPrefix::new(basis.original().len(), &constant, values).unwrap(),
            differences,
        )
    } else {
        basis.corrections(indices, values).unwrap()
    }
}

pub(super) fn bench<C: PastaCurve>(c: &mut Criterion, name: &str) {
    let scalars = values::<C::Scalar>(2048);
    let points: Vec<_> = scalars
        .iter()
        .map(|s| Point::<C>::GENERATOR.mul_projective(s).to_point())
        .collect();
    let constant = scalars[7];
    let extra_base = points[11];
    let extra_scalar = scalars[13];
    let extra_reference = binary(extra_base, extra_scalar);
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    for n in [8, 32, 256, 2048] {
        let points = &points[..n];
        let retained = BasisSum::prepare(points);
        let mut group = c.benchmark_group(format!("{name}/region_prepare"));
        group.bench_function(BenchmarkId::from_parameter(n), |b| {
            b.iter(|| black_box(BasisSum::prepare(black_box(points))));
        });
        group.finish();
        for k in [0, 1, n / 16, n / 2, n]
            .into_iter()
            .collect::<std::collections::BTreeSet<_>>()
        {
            for tail in [false, true] {
                let indices: Vec<_> = (0..k).map(|i| ((i * 17 + 1) % n) as u32).collect();
                let values = &scalars[..k];
                let mut dense = vec![constant; n];
                if tail {
                    dense[n - k..].copy_from_slice(values);
                } else {
                    for (&index, value) in indices.iter().zip(values) {
                        dense[index as usize] = dense[index as usize].add(value);
                    }
                }
                let expected = points
                    .iter()
                    .zip(&dense)
                    .fold(extra_reference, |sum, (base, scalar)| {
                        sum.add(&binary(*base, *scalar))
                    });
                let configurations = if n == 2048 && (k == n / 16 || k == n) {
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
                    let direct = Input::new(Bases::Points(points), &dense);
                    let mut direct_scratch = Buffers::new(direct.requirements(options).unwrap());
                    let mut differences = vec![PastaField::ZERO; k];
                    let corrections = correction_input(
                        retained,
                        tail,
                        constant,
                        &indices,
                        values,
                        &mut differences,
                    );
                    let mut correction_scratch =
                        Buffers::new(corrections.requirements(options).unwrap());
                    let actual = retained
                        .sum()
                        .mul_projective(&constant)
                        .add(
                            &corrections
                                .execute(options, &SerialExecutor, correction_scratch.borrow())
                                .unwrap(),
                        )
                        .add(&extra_base.mul_projective(&extra_scalar));
                    assert_eq!(actual, expected);
                    assert_eq!(
                        direct
                            .execute(options, &SerialExecutor, direct_scratch.borrow())
                            .unwrap()
                            .add(&extra_base.mul_projective(&extra_scalar)),
                        expected
                    );
                    pool.install(|| {
                        let actual = retained
                            .sum()
                            .mul_projective(&constant)
                            .add(
                                &corrections
                                    .execute(options, &Pool, correction_scratch.borrow())
                                    .unwrap(),
                            )
                            .add(&extra_base.mul_projective(&extra_scalar));
                        assert_eq!(actual, expected);
                        assert_eq!(
                            direct
                                .execute(options, &Pool, direct_scratch.borrow())
                                .unwrap()
                                .add(&extra_base.mul_projective(&extra_scalar)),
                            expected
                        );
                    });
                    let shape = if tail { "tail" } else { "sparse" };
                    let cap = limit.map_or_else(|| "all".to_owned(), |l| l.to_string());
                    let mut group = c.benchmark_group(format!(
                        "{name}/constant_regions/{shape}/tasks_{tasks}/cap_{cap}/{n}/{k}"
                    ));
                    for method in ["direct", "retained", "prepare_and_execute"] {
                        group.bench_function(method, |b| {
                            measure(b, (tasks > 1).then_some(&pool), || {
                                let extra =
                                    black_box(extra_base).mul_projective(black_box(&extra_scalar));
                                let result = if method == "direct" {
                                    Input::new(Bases::Points(black_box(points)), black_box(&dense))
                                        .execute(options, &Pool, direct_scratch.borrow())
                                        .unwrap()
                                } else {
                                    let basis = if method == "retained" {
                                        black_box(retained)
                                    } else {
                                        BasisSum::prepare(black_box(points))
                                    };
                                    let input = correction_input(
                                        basis,
                                        tail,
                                        black_box(constant),
                                        black_box(&indices),
                                        black_box(values),
                                        &mut differences,
                                    );
                                    basis.sum().mul_projective(black_box(&constant)).add(
                                        &input
                                            .execute(options, &Pool, correction_scratch.borrow())
                                            .unwrap(),
                                    )
                                };
                                black_box(result.add(&extra));
                            });
                        });
                    }
                    group.finish();
                }
            }
        }
    }
}
