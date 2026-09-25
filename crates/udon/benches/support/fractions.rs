use std::hint::black_box;

use criterion::{BatchSize, Criterion};
use zakura_udon::field::{
    PastaField, PrimeModulus, batch_invert, fraction_prefixes, fraction_prefixes_in_place,
};

use super::{SEED_A, SEED_B, values};

fn inverted_denominators<M: PrimeModulus>(
    numerators: &[PastaField<M>],
    denominators: &[PastaField<M>],
    initial: &PastaField<M>,
    output: &mut [PastaField<M>],
    scratch: &mut [PastaField<M>],
) {
    let n = denominators.len();
    assert_eq!(numerators.len(), n);
    assert!(output.len() > n && scratch.len() >= n);
    if initial.is_zero() {
        output[..=n].fill(PastaField::ZERO);
        return;
    }
    output[..n].copy_from_slice(denominators);
    batch_invert(&mut output[..n], scratch);
    let mut product = *initial;
    for i in 0..n {
        let inverse = output[i];
        output[i] = product;
        product = product.mul(&numerators[i]).mul(&inverse);
    }
    output[n] = product;
}

fn inverted_denominators_in_place<M: PrimeModulus>(
    values: &mut [PastaField<M>],
    inverses: &mut [PastaField<M>],
    initial: &PastaField<M>,
    scratch: &mut [PastaField<M>],
) {
    let n = inverses.len();
    assert!(values.len() > n && scratch.len() >= n);
    if initial.is_zero() {
        values[..=n].fill(PastaField::ZERO);
        return;
    }
    batch_invert(inverses, scratch);
    let mut product = *initial;
    for i in 0..n {
        let numerator = values[i];
        values[i] = product;
        product = product.mul(&numerator).mul(&inverses[i]);
    }
    values[n] = product;
}

pub fn benchmarks<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let numerators = values::<M, 1024>(SEED_A);
    let all_denominators = values::<M, 1024>(SEED_B);
    for shape in [
        "dense",
        "zero_first",
        "zero_middle",
        "zero_last",
        "zero_initial",
    ] {
        for n in [0, 1, 2, 3, 8, 64, 1023, 1024] {
            if shape != "dense" && n != 8 && n != 1024 {
                continue;
            }
            let numerators = &numerators[..n];
            let mut denominators = all_denominators[..n].to_vec();
            match shape {
                "zero_first" => denominators[0] = PastaField::ZERO,
                "zero_middle" => denominators[n / 2] = PastaField::ZERO,
                "zero_last" => denominators[n - 1] = PastaField::ZERO,
                _ => {}
            }
            let initial = if shape == "zero_initial" {
                PastaField::ZERO
            } else {
                PastaField::from_u64(7)
            };
            let mut output = vec![PastaField::ZERO; n + 1];
            let mut scratch = vec![PastaField::ZERO; n];
            let mut expected = vec![initial];
            for (num, den) in numerators.iter().zip(&denominators) {
                expected.push(
                    expected
                        .last()
                        .unwrap()
                        .mul(num)
                        .mul(&den.invert().unwrap_or(PastaField::ZERO)),
                );
            }
            let check = |output: &[PastaField<M>]| {
                assert!(
                    output
                        .iter()
                        .zip(&expected)
                        .all(|(a, b)| a.reduce() == b.reduce())
                );
            };
            fraction_prefixes(
                numerators,
                &denominators,
                &initial,
                &mut output,
                &mut scratch,
            )
            .unwrap();
            check(&output);
            inverted_denominators(
                numerators,
                &denominators,
                &initial,
                &mut output,
                &mut scratch,
            );
            check(&output);
            let mut input = numerators.to_vec();
            input.push(PastaField::ZERO);
            output.copy_from_slice(&input);
            fraction_prefixes_in_place(&mut output, &denominators, &initial, &mut scratch).unwrap();
            check(&output);
            output.copy_from_slice(&input);
            inverted_denominators_in_place(
                &mut output,
                &mut denominators.clone(),
                &initial,
                &mut scratch,
            );
            check(&output);

            let mut group =
                criterion.benchmark_group(format!("{field}/fraction_prefixes/{shape}/{n}"));
            group.bench_function("into", |b| {
                b.iter(|| {
                    fraction_prefixes(
                        black_box(numerators),
                        black_box(&denominators),
                        black_box(&initial),
                        black_box(&mut output),
                        black_box(&mut scratch),
                    )
                    .unwrap();
                    black_box(&output);
                });
            });
            group.bench_function("inverted_into", |b| {
                b.iter(|| {
                    inverted_denominators(
                        black_box(numerators),
                        black_box(&denominators),
                        black_box(&initial),
                        black_box(&mut output),
                        black_box(&mut scratch),
                    );
                    black_box(&output);
                });
            });
            group.bench_function("in_place", |b| {
                b.iter_batched_ref(
                    || input.clone(),
                    |output| {
                        fraction_prefixes_in_place(
                            black_box(output),
                            black_box(&denominators),
                            black_box(&initial),
                            black_box(&mut scratch),
                        )
                        .unwrap();
                        black_box(output);
                    },
                    BatchSize::SmallInput,
                );
            });
            group.bench_function("inverted_in_place", |b| {
                b.iter_batched_ref(
                    || (input.clone(), denominators.clone()),
                    |(output, inverses)| {
                        inverted_denominators_in_place(
                            black_box(output),
                            black_box(inverses),
                            black_box(&initial),
                            black_box(&mut scratch),
                        );
                        black_box(output);
                    },
                    BatchSize::SmallInput,
                );
            });
            group.finish();
        }
    }
}
