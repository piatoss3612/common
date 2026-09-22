use std::{fmt::Write, hint::black_box};

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use zakura_udon::field::{
    CanonicalUint, PallasBase, PallasScalar, PastaField, PrimeModulus, ProductSum,
};

const SEED_A: u64 = 0x243f_6a88_85a3_08d3;
const SEED_B: u64 = 0x1319_8a2e_0370_7344;

// Deterministic operands fill all four limbs. Masking to 254 bits keeps the
// canonical integers below both moduli; preparation is outside timed loops.
fn values<M: PrimeModulus, const N: usize>(mut seed: u64) -> [PastaField<M>; N] {
    std::array::from_fn(|_| {
        let mut limbs = std::array::from_fn(|_| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        });
        limbs[3] &= (1 << 62) - 1;
        PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs)).unwrap()
    })
}

// Keep each operation monomorphized and hide its inputs on every iteration.
// Criterion consumes the returned output so the operation cannot be removed.
fn bench<I: ?Sized, O>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    input: &I,
    operation: impl Fn(&I) -> O,
) {
    group.bench_function(name, |b| b.iter(|| operation(black_box(input))));
}

fn arithmetic<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let [a, b, c, d] = values::<M, 4>(SEED_A);
    let mut group = criterion.benchmark_group(format!("{field}/arithmetic"));
    bench(&mut group, "add", &(a, b), |(a, b)| a.add(b));
    bench(&mut group, "sub", &(a, b), |(a, b)| a.sub(b));
    bench(&mut group, "neg", &a, |a| a.neg());
    bench(&mut group, "mul", &(a, b), |(a, b)| a.mul(b));
    bench(&mut group, "square", &a, |a| a.square());
    bench(&mut group, "double", &a, |a| a.double());
    bench(&mut group, "triple", &a, |a| a.triple());
    bench(&mut group, "mul_by_4", &a, |a| a.mul_by_4());
    bench(&mut group, "mul_by_8", &a, |a| a.mul_by_8());
    bench(&mut group, "mul_add", &(a, b, c), |(a, b, c)| {
        a.mul_add(b, c)
    });
    bench(&mut group, "mul_sub", &(a, b, c), |(a, b, c)| {
        a.mul_sub(b, c)
    });
    bench(
        &mut group,
        "mul_sub_product",
        &(a, b, c, d),
        |(a, b, c, d)| a.mul_sub_product(b, c, d),
    );
    bench(
        &mut group,
        "mul_sub_double_product",
        &(a, b, c, d),
        |(a, b, c, d)| a.mul_sub_double_product(b, c, d),
    );
    bench(&mut group, "cmp", &(a, b), |(a, b)| a.cmp(b));
    group.finish();

    let operands = values::<M, 128>(SEED_B);
    let mut group = criterion.benchmark_group(format!("{field}/corpus"));
    group.throughput(Throughput::Elements(128));
    bench(&mut group, "mul_dependent", &operands, |values| {
        values.iter().fold(a, |acc, value| acc.mul(value))
    });
    bench(&mut group, "square_dependent", &a, |value| {
        let mut acc = *value;
        for _ in 0..128 {
            acc = acc.square();
        }
        acc
    });
    bench(&mut group, "mul_independent", &operands, |values| {
        values.map(|value| value.mul(&a))
    });
    bench(&mut group, "square_independent", &operands, |values| {
        values.map(|value| value.square())
    });
    bench(&mut group, "invert", &operands, |values| {
        values.map(|value| value.invert())
    });
    let squares = operands.map(|value| value.square());
    bench(&mut group, "sqrt_square", &squares, |values| {
        values.map(|value| value.sqrt())
    });
    let nonsquares = squares.map(|value| value.mul(&PastaField::from_u64(5)));
    bench(&mut group, "sqrt_nonsquare", &nonsquares, |values| {
        values.map(|value| value.sqrt())
    });
    group.finish();

    let mut group = criterion.benchmark_group(format!("{field}/pow_u64"));
    for (name, exponent) in [("small", 17), ("sparse", 1 << 63), ("dense", u64::MAX)] {
        bench(&mut group, name, &(a, exponent), |(a, exponent)| {
            a.pow_u64(*exponent)
        });
    }
    group.finish();

    let mut group = criterion.benchmark_group(format!("{field}/invert"));
    for (name, value) in [
        ("dense_a", a),
        ("dense_b", b),
        ("one", PastaField::ONE),
        ("zero", PastaField::ZERO),
    ] {
        bench(&mut group, name, &value, |value| value.invert());
    }
    group.finish();

    let mut group = criterion.benchmark_group(format!("{field}/sqrt"));
    for (name, value) in [("dense_a", a), ("dense_b", b)] {
        assert!(!value.is_zero());
        let square = value.square();
        // Five generates both multiplicative groups, so multiplying a nonzero
        // square by five gives a nonsquare. Validate the fixture before timing.
        let nonsquare = square.mul(&PastaField::from_u64(5));
        assert_eq!(square.sqrt().unwrap().square(), square);
        assert!(nonsquare.sqrt().is_none());
        bench(&mut group, &format!("square/{name}"), &square, |value| {
            value.sqrt()
        });
        bench(
            &mut group,
            &format!("nonsquare/{name}"),
            &nonsquare,
            |value| value.sqrt(),
        );
    }
    bench(&mut group, "one", &PastaField::<M>::ONE, |value| {
        value.sqrt()
    });
    bench(&mut group, "zero", &PastaField::<M>::ZERO, |value| {
        value.sqrt()
    });
    group.finish();
}

fn encoding<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let [value] = values::<M, 1>(SEED_A);
    let integer = value.to_canonical_uint();
    let bytes = value.to_bytes();
    let modulus = CanonicalUint::from_limbs(M::MODULUS);
    let unreduced = CanonicalUint::from_limbs([u64::MAX; 4]);
    let wide = [0xa7; 64];
    let mut group = criterion.benchmark_group(format!("{field}/encoding"));
    bench(&mut group, "from_u64", &u64::MAX, |value| {
        PastaField::<M>::from_u64(*value)
    });
    for (name, value) in [("positive", i64::MAX), ("negative", i64::MIN)] {
        bench(&mut group, &format!("from_i64/{name}"), &value, |value| {
            PastaField::<M>::from_i64(*value)
        });
    }
    for (name, value) in [("canonical", integer), ("rejected", modulus)] {
        bench(
            &mut group,
            &format!("from_canonical_uint/{name}"),
            &value,
            |value| PastaField::<M>::from_canonical_uint(*value),
        );
    }
    bench(&mut group, "from_uint_reduced", &unreduced, |value| {
        PastaField::<M>::from_uint_reduced(*value)
    });
    for (name, bytes) in [("canonical", bytes), ("rejected", modulus.to_le_bytes())] {
        bench(&mut group, &format!("from_bytes/{name}"), &bytes, |bytes| {
            PastaField::<M>::from_bytes(*bytes)
        });
    }
    bench(&mut group, "from_wide_bytes_reduced", &wide, |bytes| {
        PastaField::<M>::from_wide_bytes_reduced(bytes)
    });
    bench(&mut group, "to_canonical_uint", &value, |value| {
        value.to_canonical_uint()
    });
    bench(&mut group, "to_bytes", &value, |value| value.to_bytes());
    bench(
        &mut group,
        "from_montgomery_limbs",
        &value.montgomery_limbs(),
        |limbs| PastaField::<M>::from_montgomery_limbs(*limbs),
    );
    bench(&mut group, "is_odd", &value, |value| value.is_odd());
    let mut formatted = String::with_capacity(66);
    group.bench_function("debug", |b| {
        b.iter(|| {
            formatted.clear();
            write!(&mut formatted, "{:?}", black_box(&value)).unwrap();
            black_box(&formatted);
        });
    });
    group.finish();

    let mut group = criterion.benchmark_group(format!("{field}/from_bytes_reduced"));
    // Include short encodings, the 32- and 64-byte paths, and generic reduction
    // with both complete and partial final limbs.
    for size in [8, 31, 32, 33, 63, 64, 65, 128, 1024] {
        let bytes = vec![0xa7; size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &bytes, |b, bytes| {
            b.iter(|| PastaField::<M>::from_bytes_reduced(black_box(bytes.as_slice())));
        });
    }
    group.finish();
}

fn parameters<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let mut group = criterion.benchmark_group(format!("{field}/parameters"));
    for log_size in [1, 16, 31, 32] {
        group.bench_with_input(
            BenchmarkId::new("root_of_unity", log_size),
            &log_size,
            |b, log_size| b.iter(|| PastaField::<M>::root_of_unity(black_box(*log_size))),
        );
        group.bench_with_input(
            BenchmarkId::new("root_of_unity_inverse", log_size),
            &log_size,
            |b, log_size| b.iter(|| PastaField::<M>::root_of_unity_inverse(black_box(*log_size))),
        );
    }
    // The inverse domain-size table ends at 32; larger exponents use pow_u64.
    for log_size in [16, 32, 33, u32::MAX] {
        group.bench_with_input(
            BenchmarkId::new("power_of_two_inverse", log_size),
            &log_size,
            |b, log_size| b.iter(|| PastaField::<M>::power_of_two_inverse(black_box(*log_size))),
        );
    }
    group.finish();
}

fn accumulator<M: PrimeModulus>(lhs: &[PastaField<M>], rhs: &[PastaField<M>]) -> ProductSum<M> {
    let mut sum = ProductSum::new();
    for (lhs, rhs) in lhs.iter().zip(rhs) {
        sum.add_product(lhs, rhs);
    }
    sum
}

fn product_sum<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let lhs = values::<M, 16>(SEED_A);
    let rhs = values::<M, 16>(SEED_B);
    let other = accumulator(&rhs, &lhs);
    let mut group = criterion.benchmark_group(format!("{field}/ProductSum"));
    // Rebuild a populated accumulator outside each timed iteration. Returning
    // the mutated state through black_box keeps writes observable without
    // including finish in the add/merge measurements.
    group.bench_function("add_product", |b| {
        b.iter_batched_ref(
            || accumulator(&lhs, &rhs),
            |sum| {
                sum.add_product(black_box(&lhs[0]), black_box(&rhs[0]));
                black_box(sum);
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("add_term", |b| {
        b.iter_batched_ref(
            || accumulator(&lhs, &rhs),
            |sum| {
                sum.add_term(black_box(&lhs[0]));
                black_box(sum);
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("merge", |b| {
        b.iter_batched_ref(
            || accumulator(&lhs, &rhs),
            |sum| {
                sum.merge(black_box(&other));
                black_box(sum);
            },
            BatchSize::SmallInput,
        );
    });
    group.bench_function("finish", |b| {
        b.iter_batched(
            || accumulator(&lhs, &rhs),
            |sum| sum.finish(),
            BatchSize::SmallInput,
        );
    });
    group.finish();
}

fn inner_products<M: PrimeModulus, const N: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let lhs = values::<M, N>(SEED_A);
    let rhs = values::<M, N>(SEED_B);
    if N != 0 {
        group.throughput(Throughput::Elements(N as u64));
    }
    group.bench_function(BenchmarkId::new("sum_of_products", N), |b| {
        b.iter(|| PastaField::sum_of_products(black_box(&lhs), black_box(&rhs)));
    });
    group.bench_function(BenchmarkId::new("sum_of_products_slice", N), |b| {
        b.iter(|| {
            PastaField::sum_of_products_slice(black_box(lhs.as_slice()), black_box(rhs.as_slice()))
        });
    });
    group.bench_function(BenchmarkId::new("sum_of_product_pairs", N), |b| {
        b.iter(|| {
            PastaField::sum_of_product_pairs(
                black_box(lhs.as_slice())
                    .iter()
                    .zip(black_box(rhs.as_slice())),
            )
        });
    });
}

fn field<M: PrimeModulus>(criterion: &mut Criterion, name: &str) {
    arithmetic::<M>(criterion, name);
    encoding::<M>(criterion, name);
    parameters::<M>(criterion, name);
    product_sum::<M>(criterion, name);
    let mut group = criterion.benchmark_group(format!("{name}/inner_product"));
    inner_products::<M, 0>(&mut group);
    inner_products::<M, 1>(&mut group);
    inner_products::<M, 2>(&mut group);
    inner_products::<M, 3>(&mut group);
    inner_products::<M, 4>(&mut group);
    inner_products::<M, 16>(&mut group);
    // Array and slice sums use 32-term blocks on AArch64 and switch to four
    // lanes at 64 terms elsewhere. Include boundaries and remainders for both.
    inner_products::<M, 31>(&mut group);
    inner_products::<M, 32>(&mut group);
    inner_products::<M, 33>(&mut group);
    inner_products::<M, 63>(&mut group);
    inner_products::<M, 64>(&mut group);
    inner_products::<M, 65>(&mut group);
    inner_products::<M, 256>(&mut group);
    inner_products::<M, 1024>(&mut group);
    group.finish();
}

fn canonical_uint(criterion: &mut Criterion) {
    let value = CanonicalUint::from_limbs([
        u64::MAX,
        u64::MAX,
        0x1319_8a2e_0370_7344,
        0x243f_6a88_85a3_08d3,
    ]);
    let mut group = criterion.benchmark_group("CanonicalUint");
    bench(&mut group, "from_le_bytes", &value.to_le_bytes(), |bytes| {
        CanonicalUint::from_le_bytes(*bytes)
    });
    bench(&mut group, "to_le_bytes", &value, |value| {
        value.to_le_bytes()
    });
    bench(&mut group, "bit", &(value, 137), |(value, index)| {
        value.bit(*index)
    });
    bench(&mut group, "highest_set_bit", &value, |value| {
        value.highest_set_bit()
    });
    for (name, offset, width) in [("within_limb", 8, 16), ("cross_limb", 60, 64)] {
        bench(
            &mut group,
            &format!("window/{name}"),
            &(value, offset, width),
            |(value, offset, width)| value.window(*offset, *width),
        );
    }
    for shift in [64, 67] {
        bench(
            &mut group,
            &format!("shr/{shift}"),
            &(value, shift),
            |(value, shift)| value.shr(*shift),
        );
    }
    bench(
        &mut group,
        "bit_slice",
        &(value, 60, 130),
        |(value, offset, width)| value.bit_slice(*offset, *width),
    );
    bench(
        &mut group,
        "checked_add_u128",
        &(value, u128::MAX),
        |(value, rhs)| value.checked_add_u128(*rhs),
    );
    bench(
        &mut group,
        "fits_in_bits",
        &(value, 253),
        |(value, bits)| value.fits_in_bits(*bits),
    );
    bench(&mut group, "power_of_two", &137, |exponent| {
        CanonicalUint::power_of_two(*exponent)
    });
    let other = value.checked_add_u128(1).unwrap();
    bench(&mut group, "cmp", &(value, other), |(a, b)| a.cmp(b));
    group.finish();
}

fn benchmarks(criterion: &mut Criterion) {
    field::<PallasBase>(criterion, "Fp");
    field::<PallasScalar>(criterion, "Fq");
    batch_inversion::<PallasBase>(criterion, "Fp");
    batch_inversion::<PallasScalar>(criterion, "Fq");
    canonical_uint(criterion);
}

fn batch_inversion<M: PrimeModulus>(criterion: &mut Criterion, field: &str) {
    let mut group = criterion.benchmark_group(format!("{field}/batch_invert"));
    for size in [1, 2, 8, 64] {
        for sparse in [false, true] {
            let input: Vec<_> = (0..size)
                .map(|i| {
                    if sparse && i % 2 == 1 {
                        PastaField::ZERO
                    } else {
                        PastaField::<M>::from_u64(i as u64 + 3)
                    }
                })
                .collect();
            let mut scratch = vec![PastaField::ZERO; size];
            let shape = if sparse { "one_lane" } else { "dense" };
            group.bench_function(BenchmarkId::new(shape, size), |b| {
                b.iter_batched_ref(
                    || input.clone(),
                    |values| {
                        zakura_udon::field::batch_invert(
                            black_box(values),
                            black_box(&mut scratch),
                        );
                        black_box(values);
                    },
                    BatchSize::SmallInput,
                )
            });
        }
    }
    group.finish();
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
