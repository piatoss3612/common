use std::hint::black_box;

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use zakura_udon::{
    curve::{
        AffinePoint, CurveError, CurveTableEntry, EisensteinTable, FixedBaseDescription,
        FixedBaseTable, Pallas, PastaCurve, Point, PreparedAffinePoint, ProjectivePoint, Vesta,
        batch_normalize, glv_decompose,
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
    endomorphisms(criterion, name, &affine, rhs);
    batch_normalization(criterion, name, lhs, rhs);
    multiplication(criterion, name, &affine, scalar);
}

fn endomorphisms<C: PastaCurve>(
    criterion: &mut Criterion,
    name: &str,
    affine: &AffinePoint<C>,
    projective: ProjectivePoint<C>,
) {
    assert_eq!(projective.to_point(), affine.to_point());
    assert_ne!(*projective.coordinates().2, PastaField::ONE);
    let mut group = criterion.benchmark_group(format!("{name}/endomorphism"));
    assert_eq!(affine.endomorphism().endomorphism().endomorphism(), *affine);
    bench(&mut group, "affine", affine, AffinePoint::endomorphism);
    for (case, point) in [
        ("nonidentity", affine.to_point()),
        ("identity", Point::IDENTITY),
    ] {
        assert_eq!(point.endomorphism().endomorphism().endomorphism(), point);
        bench(
            &mut group,
            &format!("point/{case}"),
            &point,
            Point::endomorphism,
        );
    }
    for (case, point) in [
        ("scaled", projective),
        ("identity", ProjectivePoint::IDENTITY),
    ] {
        assert_eq!(
            point.endomorphism().to_point(),
            point.to_point().endomorphism()
        );
        assert_eq!(point.endomorphism().coordinates().2, point.coordinates().2);
        bench(
            &mut group,
            &format!("projective/{case}"),
            &point,
            ProjectivePoint::endomorphism,
        );
    }
    group.finish();

    let prepared = PreparedAffinePoint::from_affine(affine);
    assert_eq!(prepared.to_affine(), *affine);
    let mut group = criterion.benchmark_group(format!("{name}/prepared_affine"));
    bench(
        &mut group,
        "from_affine",
        affine,
        PreparedAffinePoint::from_affine,
    );
    assert!(prepared.valid_cache());
    bench(&mut group, "valid_cache/valid", &prepared, |entry| {
        entry.valid_cache()
    });
    for (case, bytes) in [("inconsistent", [0; 32]), ("unreduced", [0xff; 32])] {
        let mut raw = bento::AlignedBytes([0; 96]);
        raw.0.copy_from_slice(bento::bytes_of(&prepared));
        raw.0[32..64].copy_from_slice(&bytes);
        // Typed POD views need static storage; create the fixture before timing.
        let damaged = *Box::leak(Box::new(raw)).as_value::<PreparedAffinePoint<C>>();
        assert_eq!(damaged.affine(), *affine);
        assert!(!damaged.valid_cache());
        bench(
            &mut group,
            &format!("valid_cache/{case}"),
            &damaged,
            |entry| entry.valid_cache(),
        );
    }
    group.finish();

    entry_rotations(criterion, name, "affine", affine);
    entry_rotations(criterion, name, "cached", &prepared);
}

fn entry_rotations<C: PastaCurve, E: CurveTableEntry<C>>(
    criterion: &mut Criterion,
    name: &str,
    layout: &str,
    entry: &E,
) {
    let mut group = criterion.benchmark_group(format!("{name}/table_entry/{layout}"));
    let mut expected = entry.affine();
    for rotation in 1..3 {
        expected = expected.endomorphism();
        assert_eq!(entry.rotated(rotation), expected);
        bench(
            &mut group,
            &format!("rotated/{rotation}"),
            &(*entry, rotation),
            |(entry, rotation)| entry.rotated(*rotation),
        );
    }
    group.finish();
}

fn signed_scalar<M: PrimeModulus>(value: i128) -> PastaField<M> {
    let magnitude = PastaField::from_bytes_reduced(&value.unsigned_abs().to_le_bytes());
    if value < 0 {
        magnitude.neg()
    } else {
        magnitude
    }
}

fn decomposition<C: PastaCurve>(
    criterion: &mut Criterion,
    name: &str,
    scalars: &[(&str, PastaField<C::Scalar>)],
    corpus: &[PastaField<C::Scalar>; CORPUS_SIZE],
) {
    // Reconstruction checks the public contract without pinning a particular
    // valid decomposition. Both half signs affect subsequent table lookups.
    let mut signs = [false; 4];
    for scalar in scalars.iter().map(|(_, scalar)| scalar).chain(corpus) {
        let (a, b) = glv_decompose::<C>(scalar);
        assert!(a.unsigned_abs() < 1_u128 << 127);
        assert!(b.unsigned_abs() < 1_u128 << 127);
        assert_eq!(
            signed_scalar(a).add(&PastaField::ZETA.mul(&signed_scalar(b))),
            *scalar
        );
        if a != 0 && b != 0 {
            signs[usize::from(a < 0) * 2 + usize::from(b < 0)] = true;
        }
    }
    assert!(signs.into_iter().all(|seen| seen));
    let mut group = criterion.benchmark_group(format!("{name}/glv_decompose"));
    for &(case, scalar) in scalars {
        bench(&mut group, case, &scalar, glv_decompose::<C>);
    }
    group.throughput(Throughput::Elements(CORPUS_SIZE as u64));
    bench(&mut group, "corpus", corpus, |scalars| {
        scalars.map(|scalar| glv_decompose::<C>(&scalar))
    });
    group.finish();
}

fn coordinates_and_encoding<C: PastaCurve>(
    criterion: &mut Criterion,
    name: &str,
    affine: &AffinePoint<C>,
) {
    let (&x, &y) = affine.coordinates();
    let unreduced: PastaField<C::Base> = *bento::AlignedBytes([0xff; 32]).as_value();
    let mut group = criterion.benchmark_group(format!("{name}/coordinates"));

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
        .find(|x| {
            !x.is_zero()
                && x.square()
                    .mul(x)
                    .add(&PastaField::from_u64(5))
                    .sqrt()
                    .is_none()
        })
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
        (
            "sparse_64",
            PastaField::from_canonical_uint(CanonicalUint::power_of_two(63).unwrap()).unwrap(),
        ),
        ("dense_low", PastaField::from_u64(u64::MAX)),
        (
            "sparse_65",
            PastaField::from_canonical_uint(CanonicalUint::power_of_two(64).unwrap()).unwrap(),
        ),
        (
            "dense_65",
            PastaField::from_canonical_uint(CanonicalUint::from_limbs([u64::MAX, 1, 0, 0]))
                .unwrap(),
        ),
        ("dense", dense),
        ("minus_one", PastaField::ONE.neg()),
        ("lambda", PastaField::ZETA),
        ("minus_lambda", PastaField::ZETA.neg()),
        ("one_plus_lambda", PastaField::ONE.add(&PastaField::ZETA)),
        ("one_minus_lambda", PastaField::ONE.sub(&PastaField::ZETA)),
    ];
    let corpus = values::<C::Scalar>();
    decomposition::<C>(criterion, name, &scalars, &corpus);
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

    let expected = corpus.map(|scalar| affine.mul_projective(&scalar));
    assert_eq!(corpus.map(|scalar| projective.mul(&scalar)), expected);
    assert_eq!(corpus.map(|scalar| point.mul_projective(&scalar)), expected);
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
    bench(&mut group, "point", &(point, corpus), |(point, scalars)| {
        scalars.map(|scalar| point.mul_projective(&scalar))
    });
    group.finish();

    expanded::<C, AffinePoint<C>>(criterion, name, "fixed_base", affine, dense, &scalars);
    expanded::<C, PreparedAffinePoint<C>>(
        criterion,
        name,
        "fixed_base_cached",
        affine,
        dense,
        &scalars,
    );
    compact::<C, AffinePoint<C>>(criterion, name, "eisenstein", affine, dense, &scalars);
    compact::<C, PreparedAffinePoint<C>>(
        criterion,
        name,
        "eisenstein_cached",
        affine,
        dense,
        &scalars,
    );
}

fn invalid_entries<C: PastaCurve, E: CurveTableEntry<C> + bento::Pod>(
    entries: &[E],
) -> Vec<(String, Vec<E>)> {
    // First and last failures distinguish early rejection from a complete
    // validation scan; the last expanded entry is also its final carry.
    let mut cases = Vec::new();
    for (position, index) in [("first", 0), ("last", entries.len() - 1)] {
        let mut damaged = vec![
            (
                "wrong_multiple",
                E::from_affine(&entries[index].affine().neg()),
            ),
            (
                "unreduced",
                match size_of::<E>() {
                    64 => *bento::AlignedBytes([0xff; 64]).as_value::<E>(),
                    96 => *bento::AlignedBytes([0xff; 96]).as_value::<E>(),
                    _ => unreachable!("the entry trait is sealed to two POD layouts"),
                },
            ),
        ];
        if size_of::<E>() == 96 {
            // Preserve valid x/y and replace only the cached x-coordinate.
            // Bento's typed views require static storage; allocate each small
            // damaged record once during setup, outside all timed iterations.
            let mut raw = bento::AlignedBytes([0; 96]);
            raw.0.copy_from_slice(bento::bytes_of(&entries[index]));
            raw.0[32..64].fill(0);
            damaged.push((
                "inconsistent_cache",
                *Box::leak(Box::new(raw)).as_value::<E>(),
            ));
        }
        for (damage, entry) in damaged {
            let mut invalid = entries.to_vec();
            invalid[index] = entry;
            cases.push((format!("{damage}/{position}"), invalid));
        }
    }
    cases
}

fn expanded<C: PastaCurve, E: CurveTableEntry<C> + bento::Pod>(
    criterion: &mut Criterion,
    name: &str,
    layout: &str,
    affine: &AffinePoint<C>,
    dense: PastaField<C::Scalar>,
    scalars: &[(&str, PastaField<C::Scalar>)],
) {
    let corpus = values::<C::Scalar>();
    let expected = corpus.map(|scalar| affine.mul_projective(&scalar));
    for window_bits in 2..=8 {
        let description = FixedBaseDescription { window_bits };
        let required = description.requirements().unwrap();
        let mut entries =
            vec![E::from_affine(&AffinePoint::<C>::GENERATOR); required.table_entries];
        let mut projective = vec![ProjectivePoint::IDENTITY; required.projective_scratch];
        let mut field = vec![PastaField::ZERO; required.field_scratch];
        // Allocation and buffer initialization are outside timed preparation.
        // Execution then borrows this expanded storage without setup per scalar.
        let mut group = criterion.benchmark_group(format!("{name}/{layout}/w{window_bits}"));
        group.bench_function("prepare", |b| {
            b.iter(|| {
                let table = FixedBaseTable::prepare(
                    black_box(affine),
                    black_box(&mut entries),
                    black_box(&mut projective),
                    black_box(&mut field),
                )
                .unwrap();
                black_box(table.as_slice());
            })
        });
        let table =
            FixedBaseTable::prepare(affine, &mut entries, &mut projective, &mut field).unwrap();
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
        for (case, entries) in invalid_entries::<C, E>(table.as_slice()) {
            assert_eq!(
                FixedBaseTable::bind(description, affine, &entries).unwrap_err(),
                CurveError::InvalidTable
            );
            let invalid = FixedBaseTable::bind_trusted(description, affine, &entries).unwrap();
            assert_eq!(invalid.validate(), Err(CurveError::InvalidTable));
            bench(
                &mut group,
                &format!("bind/{case}"),
                &(description, *affine, entries.as_slice()),
                |&(description, base, entries)| {
                    FixedBaseTable::bind(description, &base, entries).unwrap_err()
                },
            );
            bench(&mut group, &format!("validate/{case}"), &invalid, |table| {
                table.validate().unwrap_err()
            });
        }
        group.bench_function("mul", |b| {
            b.iter(|| black_box(&table).mul(black_box(&dense)))
        });
        // Include short and full-width inputs with different GLV signs.
        for &(case, scalar) in scalars {
            // The original "mul" case already measures this dense scalar.
            if case == "dense" {
                continue;
            }
            assert_eq!(table.mul(&scalar), affine.mul_projective(&scalar));
            bench(
                &mut group,
                &format!("mul/{case}"),
                &(table, scalar),
                |(table, scalar)| table.mul(scalar),
            );
        }
        if window_bits == 2 {
            for (case, bit, negative) in [("positive", 176, false), ("negative", 214, true)] {
                let scalar =
                    PastaField::from_canonical_uint(CanonicalUint::power_of_two(bit).unwrap())
                        .unwrap();
                let (a, b) = glv_decompose::<C>(&scalar);
                // A width-2 carry starts one past the all-ones base-4 number.
                let minimum_carry = u128::MAX / 3 + 1;
                assert!(a.unsigned_abs() < minimum_carry);
                assert!(b.unsigned_abs() >= minimum_carry);
                assert_eq!(b < 0, negative);
                assert_eq!(table.mul(&scalar), affine.mul_projective(&scalar));
                bench(
                    &mut group,
                    &format!("mul/final_carry/{case}"),
                    &(table, scalar),
                    |(table, scalar)| table.mul(scalar),
                );
            }
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

fn compact<C: PastaCurve, E: CurveTableEntry<C> + bento::Pod>(
    criterion: &mut Criterion,
    name: &str,
    layout: &str,
    affine: &AffinePoint<C>,
    dense: PastaField<C::Scalar>,
    scalars: &[(&str, PastaField<C::Scalar>)],
) {
    let mut entries = [E::from_affine(affine); 8];
    let mut projective = [ProjectivePoint::IDENTITY; 8];
    let mut field = [PastaField::ZERO; 8];
    let mut group = criterion.benchmark_group(format!("{name}/{layout}"));
    group.bench_function("prepare", |b| {
        b.iter(|| {
            let table = EisensteinTable::prepare(
                black_box(affine),
                black_box(&mut entries),
                black_box(&mut projective),
                black_box(&mut field),
            )
            .unwrap();
            black_box(table.as_slice());
        })
    });
    let table =
        EisensteinTable::prepare(affine, &mut entries, &mut projective, &mut field).unwrap();
    group.bench_function("bind", |b| {
        b.iter(|| EisensteinTable::bind(black_box(affine), black_box(table.as_slice())).unwrap())
    });
    group.bench_function("bind_trusted", |b| {
        b.iter(|| {
            EisensteinTable::bind_trusted(black_box(affine), black_box(table.as_slice())).unwrap()
        })
    });
    bench(&mut group, "validate", &table, |table| {
        table.validate().unwrap()
    });
    for (case, entries) in invalid_entries::<C, E>(table.as_slice()) {
        assert_eq!(
            EisensteinTable::bind(affine, &entries).unwrap_err(),
            CurveError::InvalidTable
        );
        let invalid = EisensteinTable::bind_trusted(affine, &entries).unwrap();
        assert_eq!(invalid.validate(), Err(CurveError::InvalidTable));
        bench(
            &mut group,
            &format!("bind/{case}"),
            &(*affine, entries.as_slice()),
            |&(base, entries)| EisensteinTable::bind(&base, entries).unwrap_err(),
        );
        bench(&mut group, &format!("validate/{case}"), &invalid, |table| {
            table.validate().unwrap_err()
        });
    }
    assert_eq!(table.mul(&dense), affine.mul_projective(&dense));
    bench(&mut group, "mul", &(table, dense), |(table, scalar)| {
        table.mul(scalar)
    });
    for &(case, scalar) in scalars {
        // Keep the same dense-scalar ID as the expanded tables.
        if case == "dense" {
            continue;
        }
        assert_eq!(table.mul(&scalar), affine.mul_projective(&scalar));
        bench(
            &mut group,
            &format!("mul/{case}"),
            &(table, scalar),
            |(table, scalar)| table.mul(scalar),
        );
    }
    let corpus = values::<C::Scalar>();
    assert_eq!(
        corpus.map(|scalar| table.mul(&scalar)),
        corpus.map(|scalar| affine.mul_projective(&scalar))
    );
    group.throughput(Throughput::Elements(CORPUS_SIZE as u64));
    bench(
        &mut group,
        "mul/corpus",
        &(table, corpus),
        |(table, scalars)| scalars.map(|scalar| table.mul(&scalar)),
    );
    group.finish();
}

fn benchmarks(criterion: &mut Criterion) {
    curve::<Pallas>(criterion, "Pallas");
    curve::<Vesta>(criterion, "Vesta");
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
