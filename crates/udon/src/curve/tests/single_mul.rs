use super::*;
use crate::curve::eisenstein;
use std::hint::black_box;

// A normalization-based control with the same recoding and cached rotations.
fn normalized<C: PastaCurve>(
    base: &ProjectivePoint<C>,
    affine: Option<&AffinePoint<C>>,
    scalar: &PastaField<C::Scalar>,
) -> ProjectivePoint<C> {
    let points = match affine {
        Some(p) => eisenstein::representatives_affine(p),
        None => representatives(base),
    };
    let mut entries = [PreparedAffinePoint::from_affine(&AffinePoint::GENERATOR); 8];
    let mut field = [PastaField::ZERO; 8];
    eisenstein::normalize(&points, &mut field, &mut entries);
    eisenstein::multiply(&entries, EisensteinScalar::<C>::new(scalar).digits())
}

fn compare<C: PastaCurve>(c: &mut criterion::Criterion, name: &str) {
    let scalars: Vec<_> = field_samples::<C::Scalar>().take(8192).collect();
    let mut p = ProjectivePoint::<C>::GENERATOR;
    let bases: Vec<_> = (0..8192)
        .map(|_| {
            p = p.add_mixed(&AffinePoint::GENERATOR);
            p
        })
        .collect();
    let affine: Vec<_> = bases
        .iter()
        .map(|p| *p.to_point().as_affine().unwrap())
        .collect();
    let mut output = vec![ProjectivePoint::IDENTITY; bases.len()];
    let mut group = c.benchmark_group(std::format!("{name}/single_mul"));
    for n in [1, 64, 512, 8192] {
        for i in 0..n {
            let expected = multiply(&scalars[i], |sum| sum.add(&bases[i]));
            assert_eq!(bases[i].mul(&scalars[i]), expected);
            assert_eq!(affine[i].mul_projective(&scalars[i]), expected);
            assert_eq!(normalized(&bases[i], None, &scalars[i]), expected);
            assert_eq!(
                normalized(&bases[i], Some(&affine[i]), &scalars[i]),
                expected
            );
        }
        group.throughput(criterion::Throughput::Elements(n as u64));
        for affine_input in [false, true] {
            let input = if affine_input { "affine" } else { "projective" };
            for effective in [false, true] {
                let method = if effective { "effective" } else { "normalized" };
                group.bench_function(
                    criterion::BenchmarkId::new(std::format!("{input}_{method}"), n),
                    |b| {
                        b.iter(|| {
                            for i in 0..n {
                                let s = black_box(&scalars[i]);
                                output[i] = match (effective, affine_input) {
                                    (true, true) => black_box(&affine[i]).mul_projective(s),
                                    (true, false) => black_box(&bases[i]).mul(s),
                                    (false, true) => {
                                        normalized(&bases[i], Some(black_box(&affine[i])), s)
                                    }
                                    (false, false) => normalized(black_box(&bases[i]), None, s),
                                };
                            }
                            black_box(&output[..n]);
                        })
                    },
                );
            }
        }
    }
    group.finish();
}

#[test]
#[ignore = "Criterion timing experiment; run without concurrent builds or tests"]
fn compare_single_multiplication_tables() {
    let mut c = criterion::Criterion::default()
        .sample_size(20)
        .warm_up_time(std::time::Duration::from_millis(100))
        .measurement_time(std::time::Duration::from_millis(300))
        .without_plots();
    compare::<Pallas>(&mut c, "pallas");
    compare::<Vesta>(&mut c, "vesta");
    c.final_summary();
}
