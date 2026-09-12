use std::hint::black_box;

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use zakura_udon::{
    curve::{
        AffinePoint, FixedBaseDescription, FixedBaseTable, Pallas, PastaCurve, Point,
        ProjectivePoint, Vesta, batch_normalize, curve_rhs,
    },
    field::{CanonicalUint, PastaField, PrimeModulus},
};

const CORPUS_SIZE: usize = 32;

// Full-limb deterministic inputs, kept below both moduli and prepared before timing.
fn values<M: PrimeModulus>() -> [PastaField<M>; CORPUS_SIZE] {
    let mut seed = 0x243f_6a88_85a3_08d3_u64;
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

// Criterion consumes returned results; hide every input on each iteration too.
fn bench<I: ?Sized, O>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    name: &str,
    input: &I,
    operation: impl Fn(&I) -> O,
) {
    group.bench_function(name, |b| b.iter(|| operation(black_box(input))));
}

fn curve<C: PastaCurve>(criterion: &mut Criterion, name: &str) {
    let scalar = PastaField::<C::Scalar>::from_canonical_uint(CanonicalUint::from_limbs([
        u64::MAX,
        0x1319_8a2e_0370_7344,
        0xa409_3822_299f_31d0,
        0x243f_6a88_85a3_08d3,
    ]))
    .unwrap();
    let generator = AffinePoint::<C>::GENERATOR;
    let lhs = generator.mul_projective(&scalar);
    let rhs = lhs.double();
    let affine = rhs.to_point().as_affine().copied().unwrap();
    assert_ne!(*lhs.coordinates().2, PastaField::ONE);
    assert_ne!(*rhs.coordinates().2, PastaField::ONE);
    let encoding = affine.to_bytes();
    assert_eq!(Point::<C>::from_bytes(encoding), Some(affine.to_point()));
    let mut group = criterion.benchmark_group(format!("{name}/operations"));
    group.bench_function("add", |b| b.iter(|| black_box(&lhs).add(black_box(&rhs))));
    group.bench_function("add_mixed", |b| {
        b.iter(|| black_box(&lhs).add_mixed(black_box(&affine)))
    });
    group.bench_function("double", |b| b.iter(|| black_box(&lhs).double()));
    group.bench_function("normalize", |b| b.iter(|| black_box(&lhs).to_point()));
    group.bench_function("decode", |b| {
        b.iter(|| AffinePoint::<C>::from_bytes(black_box(encoding)))
    });
    group.bench_function("mul_affine", |b| {
        b.iter(|| black_box(&affine).mul_projective(black_box(&scalar)))
    });
    group.bench_function("mul_projective", |b| {
        b.iter(|| black_box(&lhs).mul(black_box(&scalar)))
    });
    bench(&mut group, "sub", &(lhs, rhs), |(lhs, rhs)| lhs.sub(rhs));
    bench(&mut group, "neg", &lhs, |point| point.neg());

    // Equal group elements with different Jacobian scales exercise the full
    // comparison and exceptional addition paths, rather than identical copies.
    let lhs_affine = lhs.to_point().as_affine().copied().unwrap();
    let lifted = lhs_affine.to_projective();
    assert_eq!(lhs, lifted);
    for (case, other, expected_equal) in [
        ("equal_scaled", lifted, true),
        ("inverse", lifted.neg(), false),
        ("distinct", rhs, false),
        ("identity", ProjectivePoint::IDENTITY, false),
    ] {
        assert_eq!(lhs == other, expected_equal);
        bench(
            &mut group,
            &format!("eq/{case}"),
            &(lhs, other),
            |(lhs, rhs)| lhs == rhs,
        );
    }
    for (case, left, right) in [
        ("equal_scaled", lhs, lifted),
        ("inverse", lhs, lifted.neg()),
        ("identity_lhs", ProjectivePoint::IDENTITY, rhs),
        ("identity_rhs", lhs, ProjectivePoint::IDENTITY),
    ] {
        bench(
            &mut group,
            &format!("add/{case}"),
            &(left, right),
            |(lhs, rhs)| lhs.add(rhs),
        );
    }
    for (case, left, right) in [
        ("equal", lhs, lhs_affine),
        ("inverse", lhs, lhs_affine.neg()),
        ("identity", ProjectivePoint::IDENTITY, affine),
    ] {
        bench(
            &mut group,
            &format!("add_mixed/{case}"),
            &(left, right),
            |(lhs, rhs)| lhs.add_mixed(rhs),
        );
    }
    bench(
        &mut group,
        "double/identity",
        &ProjectivePoint::<C>::IDENTITY,
        |point| point.double(),
    );
    bench(
        &mut group,
        "normalize/identity",
        &ProjectivePoint::<C>::IDENTITY,
        |point| point.to_point(),
    );
    group.finish();

    let mut group = criterion.benchmark_group(format!("{name}/point"));
    let left = lhs_affine.to_point();
    let right = affine.to_point();
    bench(&mut group, "add", &(left, right), |(lhs, rhs)| lhs.add(rhs));
    bench(&mut group, "sub", &(left, right), |(lhs, rhs)| lhs.sub(rhs));
    bench(&mut group, "double", &left, |point| point.double());
    bench(&mut group, "neg", &left, |point| point.neg());
    bench(&mut group, "affine_neg", &affine, |point| point.neg());
    group.finish();

    coordinates_and_encoding(criterion, name, &affine);
    batch_normalization(criterion, name, lhs, rhs);
    multiplication(criterion, name, &affine, scalar);
}

fn coordinates_and_encoding<C: PastaCurve>(
    criterion: &mut Criterion,
    name: &str,
    affine: &AffinePoint<C>,
) {
    let (&x, &y) = affine.coordinates();
    let unreduced: PastaField<C::Base> = *bento::AlignedBytes([0xff; 32]).as_value();
    let mut group = criterion.benchmark_group(format!("{name}/coordinates"));
    bench(&mut group, "curve_rhs", &x, curve_rhs);
    // Invalid stored residues must be rejected before curve arithmetic.
    for (case, coordinates, expected) in [
        ("valid", (x, y), Some(affine.to_point())),
        ("off_curve", (x, PastaField::ZERO), None),
        ("unreduced_x", (unreduced, y), None),
        ("unreduced_y", (x, unreduced), None),
        (
            "identity",
            (PastaField::ZERO, PastaField::ZERO),
            Some(Point::IDENTITY),
        ),
    ] {
        let (x, y) = coordinates;
        assert_eq!(Point::<C>::from_xy(x, y), expected);
        assert_eq!(
            AffinePoint::<C>::from_xy(x, y),
            expected.and_then(|point| point.as_affine().copied())
        );
        bench(
            &mut group,
            &format!("point_from_xy/{case}"),
            &coordinates,
            |&(x, y)| Point::<C>::from_xy(x, y),
        );
        bench(
            &mut group,
            &format!("affine_from_xy/{case}"),
            &coordinates,
            |&(x, y)| AffinePoint::<C>::from_xy(x, y),
        );
    }
    group.finish();

    let even = if y.is_odd() { affine.neg() } else { *affine };
    let odd = even.neg();
    let mut signed_identity = [0; 32];
    signed_identity[31] = 0x80;
    let noncanonical = CanonicalUint::from_limbs(C::Base::MODULUS).to_le_bytes();
    let nonsquare_x = values::<C::Base>()
        .into_iter()
        .find(|x| !x.is_zero() && curve_rhs(x).sqrt().is_none())
        .unwrap();
    let mut group = criterion.benchmark_group(format!("{name}/encoding"));
    bench(&mut group, "affine_to_bytes", affine, |point| {
        point.to_bytes()
    });
    for (case, point) in [
        ("nonidentity", affine.to_point()),
        ("identity", Point::IDENTITY),
    ] {
        bench(
            &mut group,
            &format!("point_to_bytes/{case}"),
            &point,
            |point| point.to_bytes(),
        );
    }
    for (case, bytes, expected) in [
        ("valid_even", even.to_bytes(), Some(even.to_point())),
        ("valid_odd", odd.to_bytes(), Some(odd.to_point())),
        ("identity", [0; 32], Some(Point::IDENTITY)),
        ("signed_identity", signed_identity, None),
        ("noncanonical_x", noncanonical, None),
        ("nonsquare_rhs", nonsquare_x.to_bytes(), None),
    ] {
        assert_eq!(Point::<C>::from_bytes(bytes), expected);
        assert_eq!(
            AffinePoint::<C>::from_bytes(bytes),
            expected.and_then(|point| point.as_affine().copied())
        );
        bench(
            &mut group,
            &format!("point_from_bytes/{case}"),
            &bytes,
            |bytes| Point::<C>::from_bytes(*bytes),
        );
        bench(
            &mut group,
            &format!("affine_from_bytes/{case}"),
            &bytes,
            |bytes| AffinePoint::<C>::from_bytes(*bytes),
        );
    }
    group.finish();

    // Vary square-root inputs: a single compressed point cannot characterize decoding.
    let points = values::<C::Scalar>().map(|scalar| affine.mul_projective(&scalar).to_point());
    let encodings = points.map(|point| point.to_bytes());
    for (&bytes, &point) in encodings.iter().zip(&points) {
        assert_eq!(Point::<C>::from_bytes(bytes), Some(point));
    }
    let mut group = criterion.benchmark_group(format!("{name}/encoding/corpus"));
    group.throughput(Throughput::Elements(CORPUS_SIZE as u64));
    bench(&mut group, "encode", &points, |points| {
        points.map(|point| point.to_bytes())
    });
    bench(&mut group, "decode", &encodings, |bytes| {
        bytes.map(Point::<C>::from_bytes)
    });
    group.finish();
}

fn batch_normalization<C: PastaCurve>(
    criterion: &mut Criterion,
    name: &str,
    lhs: ProjectivePoint<C>,
    rhs: ProjectivePoint<C>,
) {
    let mut group = criterion.benchmark_group(format!("{name}/batch_normalize"));
    // Include both prefix-product lanes, odd lengths, and larger amortized batches.
    for size in [1, 2, 3, 8, 64, 1024] {
        let mut current = lhs;
        let points: Vec<_> = (0..size)
            .map(|_| {
                current = current.add(&rhs);
                current
            })
            .collect();
        group.throughput(Throughput::Elements(size as u64));
        for shape in ["nonidentity", "mixed", "identity"] {
            if shape == "mixed" && size == 1 {
                continue;
            }
            let points: Vec<_> = points
                .iter()
                .enumerate()
                .map(|(index, point)| {
                    if shape == "identity" || (shape == "mixed" && index % 3 == 1) {
                        ProjectivePoint::IDENTITY
                    } else {
                        *point
                    }
                })
                .collect();
            let expected: Vec<_> = points.iter().map(ProjectivePoint::to_point).collect();
            let mut output = vec![Point::IDENTITY; size];
            let mut scratch = vec![PastaField::ZERO; size];
            batch_normalize(&points, &mut output, &mut scratch).unwrap();
            assert_eq!(output, expected);
            // Keep the original dense-batch IDs for existing Criterion baselines.
            let id = if shape == "nonidentity" {
                BenchmarkId::from_parameter(size)
            } else {
                BenchmarkId::new(shape, size)
            };
            group.bench_function(id, |b| {
                b.iter(|| {
                    batch_normalize(
                        black_box(&points),
                        black_box(&mut output),
                        black_box(&mut scratch),
                    )
                    .unwrap();
                    black_box(&output);
                })
            });
            // Same input and output layout exposes the benefit of sharing an inversion.
            group.bench_function(BenchmarkId::new(format!("individual/{shape}"), size), |b| {
                b.iter(|| {
                    for (point, output) in black_box(&points).iter().zip(black_box(&mut output)) {
                        *output = point.to_point();
                    }
                    black_box(&output);
                })
            });
        }
    }
    group.finish();
}

fn multiplication<C: PastaCurve>(
    criterion: &mut Criterion,
    name: &str,
    affine: &AffinePoint<C>,
    dense: PastaField<C::Scalar>,
) {
    let scalars = [
        ("zero", PastaField::ZERO),
        ("one", PastaField::ONE),
        ("small", PastaField::from_u64(17)),
        (
            "sparse_high",
            PastaField::from_canonical_uint(CanonicalUint::power_of_two(254).unwrap()).unwrap(),
        ),
        ("dense_low", PastaField::from_u64(u64::MAX)),
        ("minus_one", PastaField::ONE.neg()),
    ];
    // Compare scalar methods on the same base, retaining nontrivial Jacobian z.
    let projective = affine.to_projective().double().add_mixed(&affine.neg());
    assert_eq!(projective, affine.to_projective());
    assert_ne!(*projective.coordinates().2, PastaField::ONE);
    let point = affine.to_point();
    let mut group = criterion.benchmark_group(format!("{name}/scalar_mul"));
    for (case, scalar) in scalars {
        let expected = affine.mul_projective(&scalar);
        assert_eq!(point.mul_projective(&scalar), expected);
        assert_eq!(projective.mul(&scalar), expected);
        bench(
            &mut group,
            &format!("affine/{case}"),
            &(*affine, scalar),
            |(point, scalar)| point.mul_projective(scalar),
        );
        bench(
            &mut group,
            &format!("projective/{case}"),
            &(projective, scalar),
            |(point, scalar)| point.mul(scalar),
        );
        bench(
            &mut group,
            &format!("point/{case}"),
            &(point, scalar),
            |(point, scalar)| point.mul_projective(scalar),
        );
    }
    bench(
        &mut group,
        "point/dense",
        &(point, dense),
        |(point, scalar)| point.mul_projective(scalar),
    );
    bench(
        &mut group,
        "point/identity",
        &(Point::<C>::IDENTITY, dense),
        |(point, scalar)| point.mul_projective(scalar),
    );
    bench(
        &mut group,
        "projective/identity",
        &(ProjectivePoint::<C>::IDENTITY, dense),
        |(point, scalar)| point.mul(scalar),
    );
    group.finish();

    let corpus = values::<C::Scalar>();
    let expected = corpus.map(|scalar| affine.mul_projective(&scalar));
    assert_eq!(corpus.map(|scalar| projective.mul(&scalar)), expected);
    let mut group = criterion.benchmark_group(format!("{name}/scalar_mul/corpus"));
    group.throughput(Throughput::Elements(CORPUS_SIZE as u64));
    bench(
        &mut group,
        "affine",
        &(*affine, corpus),
        |(point, scalars)| scalars.map(|scalar| point.mul_projective(&scalar)),
    );
    bench(
        &mut group,
        "projective",
        &(projective, corpus),
        |(point, scalars)| scalars.map(|scalar| point.mul(&scalar)),
    );
    group.finish();

    for window_bits in 2..=8 {
        let description = FixedBaseDescription { window_bits };
        let required = description.requirements().unwrap();
        let mut entries = vec![AffinePoint::<C>::GENERATOR; required.affine_points];
        let mut projective = vec![ProjectivePoint::IDENTITY; required.projective_scratch];
        let mut field = vec![PastaField::ZERO; required.field_scratch];
        // Allocation and buffer initialization are outside timed preparation.
        // Execution then borrows this expanded storage without setup per scalar.
        let mut group = criterion.benchmark_group(format!("{name}/fixed_base/w{window_bits}"));
        group.bench_function("prepare", |b| {
            b.iter(|| {
                let table = FixedBaseTable::prepare(
                    black_box(description),
                    black_box(affine),
                    black_box(&mut entries),
                    black_box(&mut projective),
                    black_box(&mut field),
                )
                .unwrap();
                black_box(table.as_slice());
            })
        });
        let table = FixedBaseTable::prepare(
            description,
            affine,
            &mut entries,
            &mut projective,
            &mut field,
        )
        .unwrap();
        assert_eq!(table.mul(&dense), affine.mul_projective(&dense));
        assert_eq!(
            table.mul(&PastaField::ONE.neg()),
            affine.neg().to_projective()
        );
        assert_eq!(corpus.map(|scalar| table.mul(&scalar)), expected);
        group.bench_function("bind", |b| {
            b.iter(|| {
                FixedBaseTable::bind(
                    black_box(description),
                    black_box(affine),
                    black_box(table.as_slice()),
                )
                .unwrap()
            })
        });
        bench(
            &mut group,
            "bind_trusted",
            &(description, *affine, table.as_slice()),
            |(description, base, entries)| {
                FixedBaseTable::bind_trusted(*description, base, entries).unwrap()
            },
        );
        bench(&mut group, "validate", &table, |table| {
            table.validate().unwrap()
        });
        group.bench_function("mul", |b| {
            b.iter(|| black_box(&table).mul(black_box(&dense)))
        });
        // High-bit scalars also exercise the final carry in widths 3 and 5;
        // the original dense input has only 254 bits and never takes that path.
        for (case, scalar) in scalars {
            assert_eq!(table.mul(&scalar), affine.mul_projective(&scalar));
            bench(
                &mut group,
                &format!("mul/{case}"),
                &(table, scalar),
                |(table, scalar)| table.mul(scalar),
            );
        }
        group.throughput(Throughput::Elements(CORPUS_SIZE as u64));
        bench(
            &mut group,
            "mul/corpus",
            &(table, corpus),
            |(table, scalars)| scalars.map(|scalar| table.mul(&scalar)),
        );
        group.finish();
    }
}

fn benchmarks(criterion: &mut Criterion) {
    curve::<Pallas>(criterion, "Pallas");
    curve::<Vesta>(criterion, "Vesta");
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
