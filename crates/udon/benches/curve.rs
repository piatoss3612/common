use std::hint::black_box;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use zakura_udon::{
    curve::{
        AffinePoint, FixedBaseDescription, FixedBaseTable, Pallas, PastaCurve, Point,
        ProjectivePoint, Vesta, batch_normalize,
    },
    field::{CanonicalUint, PastaField},
};

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
    group.finish();

    let mut group = criterion.benchmark_group(format!("{name}/batch_normalize"));
    for size in [1, 8, 64] {
        let mut current = lhs;
        let points: Vec<_> = (0..size)
            .map(|_| {
                current = current.add(&rhs);
                current
            })
            .collect();
        let mut output = vec![Point::IDENTITY; size];
        let mut scratch = vec![PastaField::ZERO; size];
        group.throughput(Throughput::Elements(size as u64));
        group.bench_function(BenchmarkId::from_parameter(size), |b| {
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
    }
    group.finish();

    for window_bits in [4, 8] {
        let description = FixedBaseDescription { window_bits };
        let required = description.requirements().unwrap();
        let mut entries = vec![generator; required.affine_points];
        let mut projective = vec![ProjectivePoint::IDENTITY; required.projective_scratch];
        let mut field = vec![PastaField::ZERO; required.field_scratch];
        // Allocation and buffer initialization are outside timed preparation.
        // Execution then borrows this expanded storage without setup per scalar.
        let mut group = criterion.benchmark_group(format!("{name}/fixed_base/w{window_bits}"));
        group.bench_function("prepare", |b| {
            b.iter(|| {
                let table = FixedBaseTable::prepare(
                    black_box(description),
                    black_box(&affine),
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
            &affine,
            &mut entries,
            &mut projective,
            &mut field,
        )
        .unwrap();
        assert_eq!(table.mul(&scalar), affine.mul_projective(&scalar));
        group.bench_function("bind", |b| {
            b.iter(|| {
                FixedBaseTable::bind(
                    black_box(description),
                    black_box(&affine),
                    black_box(table.as_slice()),
                )
                .unwrap()
            })
        });
        group.bench_function("mul", |b| {
            b.iter(|| black_box(&table).mul(black_box(&scalar)))
        });
        group.finish();
    }
}

fn benchmarks(criterion: &mut Criterion) {
    curve::<Pallas>(criterion, "Pallas");
    curve::<Vesta>(criterion, "Vesta");
}

criterion_group!(benches, benchmarks);
criterion_main!(benches);
