//! Compares recoding and shared-scalar folds across prepared point banks.
//!
//! Run `compare_prepared_recoding` alone in release mode with `--ignored` and
//! `--nocapture`. Preparation timings are reported separately; correctness
//! checks are outside the execution timing loops. The point-bank comparison
//! forces α kernels to measure their crossover even below the planner's size
//! threshold. Task budgets share one eight-thread pool, so they describe the
//! allowed work partitions rather than a separate pool size for each sample.
use super::experiments::timing;
use super::*;
use crate::curve::{EisensteinTableBatch, FixedBaseDescription, FixedBaseTable};
use std::{hint::black_box, time::Instant};

#[test]
#[ignore = "isolated recoding and shared-fold timings; run release, alone, with --nocapture"]
fn compare_prepared_recoding() {
    type C = Vesta;
    let g = AffinePoint::<C>::GENERATOR;
    let affine: Vec<_> = (1..=2048)
        .map(|i| {
            *g.mul_projective(&PastaField::from_u64(i))
                .to_point()
                .as_affine()
                .unwrap()
        })
        .collect();
    let originals: Vec<_> = affine
        .iter()
        .map(PreparedAffinePoint::from_affine)
        .collect();
    let scalars: Vec<_> = field_samples::<<C as PastaCurve>::Scalar>()
        .take(2048)
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(8)
        .build()
        .unwrap();
    for width in [0, 4, 5, 6, 7] {
        let started = Instant::now();
        let mut codes = Vec::new();
        let mut coefficients = Vec::new();
        let book = (width >= 5).then(|| {
            let d = AlphaDescription::new(width).unwrap();
            codes.resize(d.codes(), 0);
            coefficients.resize(d.layers(), AlphaCoefficient::ZERO);
            AlphaCodebook::prepare(
                d,
                &mut codes,
                &mut coefficients,
                &mut vec![0; d.codebook_scratch()],
            )
        });
        let count = if width == 0 {
            0
        } else if width == 4 {
            8 * 2048
        } else {
            book.unwrap().description().layers() * 2048
        };
        let mut entries = vec![PreparedAffinePoint::from_affine(&g); count];
        let bases = match width {
            0 => Bases::Prepared(&originals),
            4 => {
                let r =
                    EisensteinTableBatch::<C, PreparedAffinePoint<C>>::requirements(2048).unwrap();
                Bases::CompactPrepared(EisensteinTableBatch::prepare(
                    &affine,
                    &mut entries,
                    &mut vec![ProjectivePoint::IDENTITY; r.projective_scratch],
                    &mut vec![PastaField::ZERO; r.field_scratch],
                    TaskBudget::SERIAL,
                    &SerialExecutor,
                ))
            }
            _ => Bases::AlphaPrepared(AlphaTable::prepare(
                book.unwrap(),
                &affine,
                &mut entries,
                &mut vec![ProjectivePoint::IDENTITY; 2048],
                &mut vec![PastaField::ZERO; 2048],
                TaskBudget::SERIAL,
                &SerialExecutor,
            )),
        };
        let retained = count * 96
            + book.map_or(0, |book| {
                let d = book.description();
                d.codes() * 4 + d.layers() * 12
            });
        std::println!(
            "prepare,width={width},bytes={retained},us={:.3}",
            started.elapsed().as_secs_f64() * 1e6
        );
        for n in [1, 10, 16, 24, 32, 44, 64, 128, 512, 2048] {
            let input = Input::new(bases.range(0..n), &scalars[..n]);
            let expected = reference(&input);
            for workers in [1, 2, 3, 8] {
                // Force the table candidate so this remains a comparison even
                // when automatic selection chooses the retained originals.
                let arithmetic = if width >= 5 {
                    ArithmeticOptions::DEFAULT
                        .with_algorithm(Algorithm::Alpha {
                            accumulation: Accumulation::Auto,
                        })
                        .unwrap()
                } else {
                    ArithmeticOptions::DEFAULT
                };
                let options = BatchOptions::new(arithmetic)
                    .with_task_budget(TaskBudget::new(workers).unwrap());
                let r = input.requirements_with(options).unwrap();
                let mut buffers = Buffers::new(r);
                assert_eq!(
                    pool.install(|| input
                        .execute_with(options, &Pool, buffers.borrow())
                        .unwrap()),
                    expected
                );
                timing(
                    &std::format!("msm/{width}/workers_{workers}"),
                    n,
                    r.bytes::<C>().unwrap(),
                    || {
                        black_box(pool.install(|| {
                            input
                                .execute_with(options, &Pool, buffers.borrow())
                                .unwrap()
                        }));
                    },
                );
            }
        }
        // Rows of 128 shared terms cross every width's serial crossover, so
        // automatic selection folds over the table; outputs partition the bank.
        let mut weights = scalars[..128].to_vec();
        weights[0] = PastaField::ONE;
        let mut records = [ScalarStorage::ZERO; 128];
        let weights =
            PreparedScalars::prepare(&weights, &mut records, TaskBudget::SERIAL, &SerialExecutor);
        let matrix = SharedScalarInput::new(bases, weights, 16, 1, 16).unwrap();
        let options = ExecutionOptions::default();
        let r = matrix.requirements(options).unwrap();
        let mut buffers = Buffers::new(r);
        let mut expected = vec![ProjectivePoint::IDENTITY; 16];
        matrix
            .execute(&mut expected, options, &SerialExecutor, buffers.borrow())
            .unwrap();
        let mut output = expected.clone();
        timing(
            &std::format!("fold/msm/{width}"),
            2048,
            r.bytes::<C>().unwrap(),
            || {
                matrix
                    .execute(
                        black_box(&mut output),
                        options,
                        &SerialExecutor,
                        buffers.borrow(),
                    )
                    .unwrap();
            },
        );
        if let Bases::AlphaPrepared(table) = bases {
            let matrix = SharedScalarInput::new(
                Bases::OddPrepared(table.odd_multiples(width).unwrap()),
                weights,
                16,
                1,
                16,
            )
            .unwrap();
            let r = matrix.requirements(options).unwrap();
            let mut buffers = Buffers::new(r);
            matrix
                .execute(&mut output, options, &SerialExecutor, buffers.borrow())
                .unwrap();
            assert_eq!(output, expected);
            timing(
                &std::format!("fold/wnaf/{width}"),
                2048,
                r.bytes::<C>().unwrap(),
                || {
                    matrix
                        .execute(
                            black_box(&mut output),
                            options,
                            &SerialExecutor,
                            buffers.borrow(),
                        )
                        .unwrap();
                },
            );
        }
    }
    for width in [4, 8] {
        let description = FixedBaseDescription { window_bits: width };
        let count = description.requirements().unwrap().table_entries;
        let mut entries = vec![PreparedAffinePoint::from_affine(&g); count];
        let table = FixedBaseTable::prepare_with(
            description,
            &g,
            &mut entries,
            &mut vec![ProjectivePoint::IDENTITY; count],
            &mut vec![PastaField::ZERO; count],
        )
        .unwrap();
        let tables = [table; 2];
        let mut points = [g; 256];
        let mut field = [PastaField::ZERO; 256];
        let expected = table.mul(&scalars[2]).add(&table.mul(&scalars[3]));
        assert_eq!(
            FixedBaseTable::sum(&tables, &scalars[2..4], &mut points, &mut field),
            expected
        );
        timing(
            &std::format!("fixed/separate/{width}"),
            2,
            count * 96,
            || {
                black_box(
                    table
                        .mul(black_box(&scalars[2]))
                        .add(&table.mul(black_box(&scalars[3]))),
                );
            },
        );
        timing(&std::format!("fixed/joint/{width}"), 2, count * 96, || {
            black_box(FixedBaseTable::sum(
                &tables,
                black_box(&scalars[2..4]),
                &mut points,
                &mut field,
            ));
        });
    }
}
