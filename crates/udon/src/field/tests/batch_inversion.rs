//! Retained control for measuring the inversion endpoint optimization.

use super::*;
use crate::test_support::field_samples;

// The former two-lane schedule, including its multiplications by one and
// unused final updates. Keeping it here allows a same-process comparison.
fn legacy<M: PrimeModulus>(values: &mut [PastaField<M>], prefix: &mut [PastaField<M>]) {
    let mut products = [PastaField::ONE; 2];
    for (i, value) in values.iter().enumerate() {
        prefix[i] = products[i & 1];
        products[i & 1] = products[i & 1].mul(value);
    }
    let inverse = products[0].mul(&products[1]).invert().unwrap();
    let mut inverses = [inverse.mul(&products[1]), inverse.mul(&products[0])];
    for (i, value) in values.iter_mut().enumerate().rev() {
        let result = inverses[i & 1].mul(&prefix[i]);
        inverses[i & 1] = inverses[i & 1].mul(value);
        *value = result;
    }
}

fn compare<M: PrimeModulus>(c: &mut criterion::Criterion, name: &str) {
    use std::hint::black_box;
    let mut group = c.benchmark_group(std::format!("{name}/inversion_endpoints"));
    for n in [1, 2, 3, 8, 32, 128, 1024] {
        let values: Vec<_> = field_samples::<M>()
            .filter(|v| !v.is_zero())
            .take(n)
            .collect();
        let mut prefix = vec![PastaField::ZERO; n];
        let mut old = values.clone();
        let mut new = values.clone();
        legacy(&mut old, &mut prefix);
        crate::field::invert_nonzero(&mut new, &mut prefix);
        assert_eq!(
            (old).iter().map(|value| value.reduce()).collect::<Vec<_>>(),
            (new).iter().map(|value| value.reduce()).collect::<Vec<_>>()
        );
        for (v, inverse) in values.iter().zip(&new) {
            assert_eq!((v.mul(inverse)).reduce(), (PastaField::<_>::ONE).reduce());
        }
        // Consecutive iterations alternate values and their inverses; both
        // schedules see the same operands, with no resetting inside timing.
        group.bench_function(criterion::BenchmarkId::new("legacy", n), |b| {
            b.iter(|| {
                legacy(black_box(&mut old), black_box(&mut prefix));
                black_box(&old);
            })
        });
        group.bench_function(criterion::BenchmarkId::new("endpoints", n), |b| {
            b.iter(|| {
                crate::field::invert_nonzero(black_box(&mut new), black_box(&mut prefix));
                black_box(&new);
            })
        });
    }
    group.finish();
}

#[test]
#[ignore = "Criterion timing experiment; run without concurrent builds or tests"]
fn compare_batch_inversion_endpoints() {
    let mut c = criterion::Criterion::default()
        .sample_size(30)
        .warm_up_time(std::time::Duration::from_millis(300))
        .measurement_time(std::time::Duration::from_secs(1))
        .without_plots();
    compare::<crate::field::PallasBase>(&mut c, "Fp");
    compare::<crate::field::PallasScalar>(&mut c, "Fq");
    c.final_summary();
}
