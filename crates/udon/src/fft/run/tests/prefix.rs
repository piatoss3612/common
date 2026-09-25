//! Isolated timing of prefix power generation and task coset scaling.
use super::*;
use crate::{
    exec::run::ReadView,
    fft::{factors::ForwardShift, reverse, run::FftKernel},
};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};

impl<M: PrimeModulus> FftKernel<'_, M> {
    fn initialize_prefix_baseline(
        &self,
        values: &mut [PastaField<M>],
        source: &dyn ReadView<PastaField<M>>,
    ) {
        let plan = self.plan;
        let repeat = plan.first() / 2;
        let width = plan.size() / repeat;
        let mut offset = 0;
        while offset < values.len() {
            let index = self.start + offset;
            let degree = reverse(index / repeat, width.ilog2());
            let mut value = source.get(degree).copied().unwrap_or(PastaField::ZERO);
            if degree < source.len() && plan.twist() {
                let power = plan
                    .residue_scales
                    .map_or_else(|| plan.shift.at(degree), |scales| scales[degree]);
                value = value.mul(&power).mul(&plan.input_scale);
            }
            let len = (repeat - index % repeat).min(values.len() - offset);
            values[offset..offset + len].fill(value);
            offset += len;
        }
    }

    fn twist_baseline(&self, values: &mut [PastaField<M>], order: ElementOrder) {
        let plan = self.plan;
        if let Some(scales) = plan.residue_scales {
            for (offset, value) in values.iter_mut().enumerate() {
                let index = if order == ElementOrder::Natural {
                    self.start + offset
                } else {
                    reverse(self.start + offset, plan.size().ilog2())
                };
                *value = value.mul(&scales[index]);
                if plan.input_scale.reduce() != PastaField::<M>::ONE.reduce() {
                    *value = value.mul(&plan.input_scale);
                }
            }
        } else if let Some(cycle) = plan.shift.cycle() {
            let cycle = cycle.scaled(plan.input_scale);
            for (offset, value) in values.iter_mut().enumerate() {
                let degree = if order == ElementOrder::Natural {
                    self.start + offset
                } else {
                    reverse(self.start + offset, plan.size().ilog2())
                };
                *value = value.mul(&cycle.at(degree));
            }
        } else {
            let ForwardShift::Residue { shift, inverse } = plan.shift else {
                unreachable!()
            };
            if order == ElementOrder::Natural {
                let mut power = shift.pow_u64(self.start as u64).mul(&plan.input_scale);
                let len = values.len();
                for (index, value) in values.iter_mut().enumerate() {
                    *value = value.mul(&power);
                    if index + 1 < len {
                        power = power.mul(&shift);
                    }
                }
            } else {
                let powers = crate::fft::factors::BitReversedPowers::new(
                    shift,
                    inverse,
                    plan.size().ilog2(),
                );
                let mut power = powers.at(self.start).mul(&plan.input_scale);
                let len = values.len();
                for (offset, value) in values.iter_mut().enumerate() {
                    *value = value.mul(&power);
                    if offset + 1 < len {
                        power = powers.next(self.start + offset, power);
                    }
                }
            }
        }
    }
}

fn median(mut run: impl FnMut()) -> f64 {
    let mut samples = [0.0f64; 7];
    for sample in &mut samples {
        let start = Instant::now();
        let mut count = 0;
        while start.elapsed() < Duration::from_millis(15) {
            for _ in 0..16 {
                run();
            }
            count += 16;
        }
        *sample = start.elapsed().as_nanos() as f64 / count as f64;
    }
    samples.sort_by(f64::total_cmp);
    samples[3]
}

fn compare<M: PrimeModulus>(name: &str) {
    let size = 4096;
    let domain = Domain::<M>::for_size(size).unwrap();
    let shift = PastaField::<M>::from_u64(7);
    for prefix in [10, 128, 256, 1024] {
        let source: Vec<_> = crate::test_support::field_samples::<M>()
            .take(prefix)
            .collect();
        for residue in [false, true] {
            let mut plan = FftPlan::with_strategy(
                Transform::new(domain.coset()),
                TransformRequest {
                    input_storage: InputStorage::Preserve,
                    support: InputSupport::Prefix(prefix),
                    ..TransformRequest::new(Direction::Forward)
                },
                NonZeroUsize::new(size).unwrap(),
                Codelet::Radix2,
            )
            .unwrap();
            if residue {
                plan.shift = ForwardShift::Residue {
                    shift,
                    inverse: shift.invert().unwrap(),
                };
            }
            for length in [256, size] {
                let kernel = FftKernel {
                    plan: black_box(plan),
                    kind: WorkKind::Local,
                    start: 0,
                    block: 0,
                    product: false,
                    column: 0,
                    band: 0,
                };
                let mut values = vec![PastaField::ZERO; length];
                kernel.initialize_prefix_baseline(&mut values, &source.as_slice());
                let expected = values.clone();
                kernel.initialize_prefix(&mut values, &source.as_slice());
                assert!(
                    values
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.reduce() == b.reduce())
                );
                let baseline = median(|| {
                    kernel.initialize_prefix_baseline(
                        black_box(&mut values),
                        black_box(&source.as_slice()),
                    );
                    black_box(&values);
                });
                let candidate = median(|| {
                    kernel.initialize_prefix(black_box(&mut values), black_box(&source.as_slice()));
                    black_box(&values);
                });
                std::println!(
                    "{name}/prefix/{prefix}/residue_{residue}/length_{length}: {baseline:.1} -> {candidate:.1} ns"
                );
            }
        }
    }
    let plan = FftPlan::with_strategy(
        Transform::new(domain.coset()),
        TransformRequest::new(Direction::Forward),
        NonZeroUsize::new(size).unwrap(),
        Codelet::Radix2,
    )
    .unwrap();
    let kernel = FftKernel {
        plan: black_box(plan),
        kind: WorkKind::Local,
        start: 0,
        block: 0,
        product: false,
        column: 0,
        band: 0,
    };
    for order in [ElementOrder::Natural, ElementOrder::BitReversed] {
        let mut values: Vec<_> = crate::test_support::field_samples::<M>()
            .take(size)
            .collect();
        let mut expected = values.clone();
        kernel.twist_baseline(&mut expected, order);
        kernel.twist(&mut values, order);
        assert!(
            values
                .iter()
                .zip(&expected)
                .all(|(a, b)| a.reduce() == b.reduce())
        );
        let baseline = median(|| {
            kernel.twist_baseline(black_box(&mut values), order);
            black_box(&values);
        });
        let candidate = median(|| {
            kernel.twist(black_box(&mut values), order);
            black_box(&values);
        });
        std::println!("{name}/twist/{order:?}: {baseline:.1} -> {candidate:.1} ns");
    }
}

#[test]
#[ignore = "targeted timing experiment"]
fn compare_prefix_initialization() {
    compare::<PallasBase>("Fp");
    compare::<PallasScalar>("Fq");
}
