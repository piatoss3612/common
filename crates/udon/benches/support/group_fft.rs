//! Complete serial group transforms and coefficient-to-Lagrange conversion.

use criterion::{Bencher, BenchmarkId, Criterion, Throughput};
use std::{
    hint::black_box,
    time::{Duration, Instant},
};
use zakura_udon::{
    curve::{PastaCurve, Point, ProjectivePoint, batch_normalize},
    fft::{Domain, reference},
    field::PastaField,
};

fn generator_multiple<C: PastaCurve>(scalar: &PastaField<C::Scalar>) -> ProjectivePoint<C> {
    let scalar = scalar.to_canonical_uint();
    let mut point = ProjectivePoint::IDENTITY;
    for bit in (0..255).rev() {
        point = point.double();
        if scalar.bit(bit).unwrap() {
            point = point.add(&ProjectivePoint::GENERATOR);
        }
    }
    point
}

// Known generator multiples let Horner evaluation check each DFT row without
// using any FFT schedule or the production scalar-multiplication routine.
fn direct<C: PastaCurve>(
    coefficients: &[PastaField<C::Scalar>],
    root: PastaField<C::Scalar>,
    scale: PastaField<C::Scalar>,
) -> Vec<ProjectivePoint<C>> {
    let mut row = PastaField::ONE;
    (0..coefficients.len())
        .map(|_| {
            let value = coefficients
                .iter()
                .rev()
                .fold(PastaField::ZERO, |sum, coefficient| {
                    sum.mul(&row).add(coefficient)
                });
            row = row.mul(&root);
            generator_multiple::<C>(&value.mul(&scale))
        })
        .collect()
}

// Input restoration and allocation are excluded. The operation includes all
// root powers, scalar preparation, permutations, and requested output passes.
fn measure<C: PastaCurve>(
    b: &mut Bencher,
    input: &[ProjectivePoint<C>],
    mut operation: impl FnMut(&mut [ProjectivePoint<C>]),
) {
    let mut working = input.to_vec();
    b.iter_custom(|iterations| {
        let mut elapsed = Duration::ZERO;
        for _ in 0..iterations {
            working.copy_from_slice(input);
            let start = Instant::now();
            operation(black_box(&mut working));
            elapsed += start.elapsed();
        }
        elapsed
    });
}

pub(super) fn benchmarks<C: PastaCurve>(criterion: &mut Criterion, curve: &str) {
    for shape in ["dense", "sparse", "cancellation"] {
        let mut group = criterion.benchmark_group(format!("{curve}/group_fft/{shape}"));
        for size in [1, 2, 4, 8, 16, 64, 256, 1024] {
            let domain = Domain::<C::Scalar>::for_size(size).unwrap();
            let mut coefficients = super::inputs::<C::Scalar>(size + 3)[3..].to_vec();
            let repeated = PastaField::from_u64(7);
            for (index, coefficient) in coefficients.iter_mut().enumerate() {
                match shape {
                    "sparse" if index % (size / 4).max(1) != 0 => *coefficient = PastaField::ZERO,
                    "cancellation" => {
                        *coefficient = if index % 2 == 0 {
                            repeated
                        } else {
                            repeated.neg()
                        };
                    }
                    _ => {}
                }
            }
            let input: Vec<_> = coefficients
                .iter()
                .map(|scalar| generator_multiple::<C>(scalar).to_point().to_projective())
                .collect();
            let forward = direct::<C>(&coefficients, domain.root(), PastaField::ONE);
            let inverse = direct::<C>(&coefficients, domain.inverse_root(), domain.size_inverse());
            let mut checked = input.clone();
            reference::transform(&mut checked, &domain.root());
            assert_eq!(checked, forward);
            checked.copy_from_slice(&input);
            reference::inverse_transform(
                &mut checked,
                &domain.inverse_root(),
                &domain.size_inverse(),
            );
            assert_eq!(checked, inverse);
            let mut affine = vec![Point::IDENTITY; size];
            let mut scratch = vec![PastaField::ZERO; size];
            batch_normalize(&checked, &mut affine, &mut scratch);
            for (actual, expected) in affine.iter().zip(&inverse) {
                assert_eq!(actual.to_projective(), *expected);
            }
            group.throughput(Throughput::Elements(size as u64));
            group.bench_with_input(BenchmarkId::new("forward", size), &size, |b, _| {
                measure(b, &input, |values| {
                    reference::transform(values, black_box(&domain.root()));
                    black_box(values);
                });
            });
            group.bench_with_input(BenchmarkId::new("inverse", size), &size, |b, _| {
                measure(b, &input, |values| {
                    reference::inverse_transform(
                        values,
                        black_box(&domain.inverse_root()),
                        black_box(&domain.size_inverse()),
                    );
                    black_box(values);
                });
            });
            group.bench_with_input(BenchmarkId::new("basis_conversion", size), &size, |b, _| {
                measure(b, &input, |values| {
                    reference::inverse_transform(
                        values,
                        black_box(&domain.inverse_root()),
                        black_box(&domain.size_inverse()),
                    );
                    batch_normalize(values, &mut affine, &mut scratch);
                    black_box((&*values, &affine));
                });
            });
        }
        group.finish();
    }
}
