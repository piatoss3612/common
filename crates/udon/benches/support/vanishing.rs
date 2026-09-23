use std::hint::black_box;

use criterion::{BatchSize, Criterion};
use zakura_udon::{
    exec::{ExecutionOptions, SerialExecutor},
    fft::{
        Direction, Domain, ElementOrder, StorageLayout, Transform, TransformRequest,
        VanishingDivision, execution::FftPlan,
    },
    field::{PastaField, PrimeModulus},
};

pub fn benchmarks<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    for (shift_name, shift) in [
        ("zeta", PastaField::<M>::ZETA),
        ("general", PastaField::from_u64(7)),
    ] {
        for (size, count, order, full) in [
            (64, 4, ElementOrder::Natural, true),
            (1024, 4, ElementOrder::Natural, false),
            (1024, 4, ElementOrder::BitReversed, false),
            (1024, 4, ElementOrder::BitReversed, true),
            (16384, 16, ElementOrder::BitReversed, false),
        ] {
            let domain = Domain::<PastaField<M>>::for_size(size).unwrap();
            let n = size / count;
            let division = VanishingDivision::new(domain, &shift, n).unwrap();
            let output_count = if full { count } else { count - 1 };
            let mut factor_storage = vec![PastaField::ZERO; count];
            let factors =
                division.prepare_factors(&mut factor_storage, &mut vec![PastaField::ZERO; count]);
            let tables = super::Prepared::new(domain.subgroup());
            let forward = FftPlan::new(
                tables.tables().bind(domain.subgroup()),
                TransformRequest {
                    output_order: order,
                    ..TransformRequest::new(Direction::Forward)
                },
                StorageLayout::Contiguous,
                ExecutionOptions::default(),
            )
            .unwrap();
            // The periodic control uses the existing optimized coset finish.
            // A general shift uses subgroup interpolation then one untwist pass.
            let inverse_domain = if shift_name == "zeta" {
                domain.coset()
            } else {
                domain.subgroup()
            };
            let inverse_tables = super::Prepared::new(inverse_domain);
            let inverse = FftPlan::new(
                inverse_tables.tables().bind(inverse_domain),
                TransformRequest::new(Direction::Inverse),
                StorageLayout::Contiguous,
                ExecutionOptions::default(),
            )
            .unwrap();
            let coefficients = super::inputs::<M>(size);
            let mut evaluations = coefficients.clone();
            let mut power = PastaField::ONE;
            for value in &mut evaluations {
                *value = value.mul(&power);
                power = power.mul(&shift);
            }
            Transform::new(domain.subgroup())
                .forward(
                    &mut evaluations,
                    ExecutionOptions::default(),
                    &SerialExecutor,
                    &mut [],
                )
                .unwrap();
            let mut scratch = vec![
                PastaField::ZERO;
                forward
                    .retained_fields()
                    .max(inverse.retained_fields())
                    .max(count)
            ];
            let mut transformed = evaluations.clone();
            forward.execute(None, &mut transformed, None, &mut scratch, &SerialExecutor);
            let mut pieces = vec![vec![PastaField::ZERO; n]; output_count];
            let mut destinations: Vec<_> = pieces.iter_mut().map(Vec::as_mut_slice).collect();
            division.write_pieces(&transformed, order, &mut destinations, &mut scratch);
            let mut expected = evaluations.clone();
            factors.divide_in_place(&mut expected, ElementOrder::Natural);
            inverse.execute(None, &mut expected, None, &mut scratch, &SerialExecutor);
            let inverse_shift = shift.invert().unwrap();
            let untwist = |values: &mut [PastaField<M>]| {
                if shift_name == "general" {
                    let mut power = PastaField::ONE;
                    for value in values {
                        *value = value.mul(&power);
                        power = power.mul(&inverse_shift);
                    }
                }
            };
            untwist(&mut expected);
            assert!(
                destinations
                    .iter()
                    .flat_map(|piece| piece.iter())
                    .zip(&expected)
                    .all(|(a, b)| a.reduce() == b.reduce())
            );
            let order_name = if order == ElementOrder::Natural {
                "natural"
            } else {
                "reversed"
            };
            let mut group = criterion.benchmark_group(format!(
                "{field}/vanishing/{size}/{count}/{shift_name}/{order_name}/{output_count}"
            ));
            group.bench_function("finish", |b| {
                b.iter(|| {
                    black_box(division).write_pieces(
                        black_box(&transformed),
                        order,
                        black_box(&mut destinations),
                        black_box(&mut scratch),
                    );
                    black_box(&destinations);
                })
            });
            group.bench_function("forward_and_finish", |b| {
                b.iter_batched_ref(
                    || evaluations.clone(),
                    |values| {
                        forward.execute(
                            None,
                            black_box(values),
                            None,
                            &mut scratch,
                            &SerialExecutor,
                        );
                        black_box(division).write_pieces(
                            values,
                            order,
                            black_box(&mut destinations),
                            black_box(&mut scratch),
                        );
                        black_box(&destinations);
                    },
                    BatchSize::PerIteration,
                )
            });
            group.bench_function("pointwise_and_inverse", |b| {
                b.iter_batched_ref(
                    || evaluations.clone(),
                    |values| {
                        black_box(factors)
                            .divide_in_place(black_box(values), ElementOrder::Natural);
                        inverse.execute(None, values, None, &mut scratch, &SerialExecutor);
                        untwist(values);
                        for (piece, coefficients) in
                            destinations.iter_mut().zip(values.chunks_exact(n))
                        {
                            piece.copy_from_slice(coefficients);
                        }
                        black_box(&destinations);
                    },
                    BatchSize::PerIteration,
                )
            });
            if order == ElementOrder::Natural && full {
                group.bench_function("prepare_plan", |b| {
                    b.iter(|| {
                        black_box(
                            VanishingDivision::new(black_box(domain), black_box(&shift), n)
                                .unwrap(),
                        );
                    })
                });
                let mut storage = vec![PastaField::ZERO; count];
                group.bench_function("prepare_factors", |b| {
                    b.iter(|| {
                        black_box(division.prepare_factors(
                            black_box(&mut storage),
                            black_box(&mut scratch[..count]),
                        ));
                    })
                });
            }
            group.finish();
        }
    }
}
