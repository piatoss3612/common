use std::hint::black_box;

use criterion::Criterion;
use zakura_udon::{
    exec::{ExecutionOptions, SerialExecutor},
    fft::{
        ConstantPrefixExpansion, Direction, Domain, ElementOrder, Expansion, ExpansionOrder,
        ExpansionScaleNormalization, ExpansionScales, ExpansionStorage, InputSupport,
        StorageLayout, TransformRequest,
        run::{ExpansionPlan, FftPlan},
    },
    field::{ConstantPrefix, PastaField, PrimeModulus},
};

pub fn benchmarks<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    for n in [64, 1024, 16384] {
        let base = Domain::<M>::for_size(n).unwrap().subgroup();
        let target = Domain::<M>::for_size(n * 4).unwrap().coset();
        let size = target.size();
        let tables = super::Prepared::new(base);
        let transform = tables.tables().bind(base);
        let inverse = FftPlan::new(
            transform,
            TransformRequest::new(Direction::Inverse),
            StorageLayout::Contiguous,
            ExecutionOptions::default(),
        )
        .unwrap();
        let expansion = Expansion::new(transform, target, None).unwrap();
        let dense_plan = ExpansionPlan::new(
            expansion,
            ExpansionStorage::ReuseOutput,
            ExpansionOrder::Residues,
            InputSupport::Full,
            ElementOrder::Natural,
            StorageLayout::Contiguous,
            ExecutionOptions::default(),
        )
        .unwrap();
        let mut scales = vec![PastaField::ZERO; size];
        let scales = ExpansionScales::prepare(
            n,
            target,
            ExpansionScaleNormalization::UnscaledInverse,
            &mut scales,
        )
        .unwrap();
        let scaled_expansion = expansion.with_scales(scales);
        let scaled_plan = ExpansionPlan::new(
            scaled_expansion,
            ExpansionStorage::ReuseOutput,
            ExpansionOrder::Residues,
            InputSupport::Full,
            ElementOrder::Natural,
            StorageLayout::Contiguous,
            ExecutionOptions::default(),
        )
        .unwrap();
        let mut fft_scratch = vec![
            PastaField::ZERO;
            inverse
                .retained_fields()
                .max(dense_plan.scratch_fields())
                .max(scaled_plan.scratch_fields())
        ];
        let mut samples = vec![PastaField::ZERO; size];
        let mut inversion_scratch = vec![PastaField::ZERO; size];
        let plan =
            ConstantPrefixExpansion::prepare(base, target, &mut samples, &mut inversion_scratch)
                .unwrap();
        let mut fresh_samples = vec![PastaField::ZERO; size];
        let mut prep =
            criterion.benchmark_group(format!("{field}/constant_prefix/{n}/preparation"));
        prep.bench_function("fft_scales", |b| {
            b.iter(|| {
                black_box(
                    ExpansionScales::prepare(
                        black_box(n),
                        black_box(target),
                        ExpansionScaleNormalization::UnscaledInverse,
                        black_box(&mut fresh_samples),
                    )
                    .unwrap(),
                );
            });
        });
        prep.bench_function("samples", |b| {
            b.iter(|| {
                black_box(
                    ConstantPrefixExpansion::prepare(
                        black_box(base),
                        black_box(target),
                        black_box(&mut fresh_samples),
                        black_box(&mut inversion_scratch),
                    )
                    .unwrap(),
                );
            });
        });
        prep.bench_function("samples_bounded_32", |b| {
            b.iter(|| {
                black_box(
                    ConstantPrefixExpansion::prepare(
                        black_box(base),
                        black_box(target),
                        black_box(&mut fresh_samples),
                        black_box(&mut inversion_scratch[..32]),
                    )
                    .unwrap(),
                );
            });
        });
        prep.bench_function("bind", |b| {
            b.iter(|| {
                black_box(
                    ConstantPrefixExpansion::bind(
                        black_box(base),
                        black_box(target),
                        black_box(plan.samples()),
                    )
                    .unwrap(),
                );
            });
        });
        prep.finish();
        let tails: &[usize] = if n == 64 {
            &[0, 1, 4, 8, 16, 32, 64]
        } else {
            &[0, 1, 4, 8, 16, 32]
        };
        let tail_values = super::inputs::<M>(*tails.last().unwrap());
        let constant = PastaField::<M>::from_u64(7);
        let mut dense = vec![PastaField::ZERO; n];
        let mut coefficients = vec![PastaField::ZERO; n];
        let mut residues = vec![PastaField::ZERO; size];
        let mut output = vec![PastaField::ZERO; size];
        for &t in tails {
            let input = ConstantPrefix::new(n, &constant, &tail_values[..t]).unwrap();
            let mut delta = vec![PastaField::ZERO; t];
            input.write_values(&mut dense).unwrap();
            input.write_values(&mut coefficients).unwrap();
            inverse.execute(
                None,
                &mut coefficients,
                None,
                &mut fft_scratch,
                &SerialExecutor,
            );
            base.interpolate_constant_prefix(input, &mut dense, &mut delta)
                .unwrap();
            assert!(
                dense
                    .iter()
                    .zip(&coefficients)
                    .all(|(a, b)| a.reduce() == b.reduce())
            );
            input.write_values(&mut dense).unwrap();
            dense_plan.execute(
                &dense,
                &mut residues,
                &mut [],
                None,
                &mut fft_scratch,
                &SerialExecutor,
            );
            expansion.layout().copy_to_natural(&residues, &mut output);
            let expected = output.clone();
            plan.evaluate(input, &mut output).unwrap();
            assert!(
                output
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.reduce() == b.reduce())
            );
            scaled_plan.execute(
                &dense,
                &mut residues,
                &mut [],
                None,
                &mut fft_scratch,
                &SerialExecutor,
            );
            scaled_expansion
                .layout()
                .copy_to_natural(&residues, &mut output);
            assert!(
                output
                    .iter()
                    .zip(&expected)
                    .all(|(a, b)| a.reduce() == b.reduce())
            );

            let mut group =
                criterion.benchmark_group(format!("{field}/constant_prefix/{n}/tail_{t}"));
            group.bench_function("coefficients_direct", |b| {
                b.iter(|| {
                    black_box(base)
                        .interpolate_constant_prefix(
                            black_box(input),
                            black_box(&mut coefficients),
                            black_box(&mut delta),
                        )
                        .unwrap();
                    black_box(&coefficients);
                });
            });
            group.bench_function("coefficients_fft", |b| {
                b.iter(|| {
                    black_box(input)
                        .write_values(black_box(&mut coefficients))
                        .unwrap();
                    inverse.execute(
                        None,
                        black_box(&mut coefficients),
                        None,
                        black_box(&mut fft_scratch),
                        &SerialExecutor,
                    );
                    black_box(&coefficients);
                });
            });
            group.bench_function("extend_retained", |b| {
                b.iter(|| {
                    black_box(plan)
                        .evaluate(black_box(input), black_box(&mut output))
                        .unwrap();
                    black_box(&output);
                });
            });
            group.bench_function("extend_fft", |b| {
                b.iter(|| {
                    black_box(input)
                        .write_values(black_box(&mut dense))
                        .unwrap();
                    dense_plan.execute(
                        black_box(&dense),
                        black_box(&mut residues),
                        &mut [],
                        None,
                        black_box(&mut fft_scratch),
                        &SerialExecutor,
                    );
                    expansion
                        .layout()
                        .copy_to_natural(black_box(&residues), black_box(&mut output));
                    black_box(&output);
                });
            });
            group.bench_function("extend_fft_scales", |b| {
                b.iter(|| {
                    black_box(input)
                        .write_values(black_box(&mut dense))
                        .unwrap();
                    scaled_plan.execute(
                        black_box(&dense),
                        black_box(&mut residues),
                        &mut [],
                        None,
                        black_box(&mut fft_scratch),
                        &SerialExecutor,
                    );
                    scaled_expansion
                        .layout()
                        .copy_to_natural(black_box(&residues), black_box(&mut output));
                    black_box(&output);
                });
            });
            if t == 1 || t == 8 {
                group.bench_function("prepare_and_extend", |b| {
                    b.iter(|| {
                        let plan = ConstantPrefixExpansion::prepare(
                            black_box(base),
                            black_box(target),
                            black_box(&mut fresh_samples),
                            black_box(&mut inversion_scratch),
                        )
                        .unwrap();
                        plan.evaluate(black_box(input), black_box(&mut output))
                            .unwrap();
                        black_box(&output);
                    });
                });
            }
            group.finish();
        }
    }
}
