//! Fixed-base products from shared and distinct retained tables.

// Constant-size `chunks_exact` predates `as_chunks`; migration is upstream
// work, and the pinned toolchain's Clippy predates the lint itself.
#![allow(unknown_lints)]
#![allow(clippy::chunks_exact_to_as_chunks)]

use super::*;

fn ladder<C: PastaCurve>(
    base: &AffinePoint<C>,
    scalar: &PastaField<C::Scalar>,
) -> ProjectivePoint<C> {
    let scalar = scalar.to_canonical_uint();
    let mut result = ProjectivePoint::IDENTITY;
    for bit in (0..255).rev() {
        result = result.double();
        if scalar.bit(bit).unwrap() {
            result = result.add_mixed(base);
        }
    }
    result
}

pub(super) fn benchmarks<C: PastaCurve, E: CurveTableEntry<C>>(
    criterion: &mut Criterion,
    name: &str,
    layout: &str,
) {
    let base = AffinePoint::<C>::GENERATOR;
    let other_base = *base
        .to_projective()
        .double()
        .to_point()
        .as_affine()
        .unwrap();
    let full = values::<C::Scalar>();
    let short = core::array::from_fn::<_, CORPUS_SIZE, _>(|i| {
        PastaField::<C::Scalar>::from_u64(0x1234_5678_9abc_def0_u64.wrapping_mul(i as u64 + 1))
    });
    let tiny = core::array::from_fn::<_, CORPUS_SIZE, _>(|i| match i % 4 {
        0 => PastaField::ZERO,
        1 => PastaField::ONE,
        2 => PastaField::<C::Scalar>::ONE.neg(),
        _ => PastaField::from_u64(17),
    });
    for (width, other_width) in [(2, 2), (4, 4), (8, 8), (4, 8)] {
        let description = FixedBaseDescription { window_bits: width };
        let other_description = FixedBaseDescription {
            window_bits: other_width,
        };
        let required = description.requirements().unwrap();
        let other_required = other_description.requirements().unwrap();
        let mut entries = vec![E::from_affine(&base); required.table_entries];
        let mut other_entries = vec![E::from_affine(&other_base); other_required.table_entries];
        let count = required.table_entries.max(other_required.table_entries);
        let mut projective = vec![ProjectivePoint::IDENTITY; count];
        let mut field = vec![PastaField::ZERO; count];
        let table = FixedBaseTable::prepare_with(
            description,
            &base,
            &mut entries,
            &mut projective,
            &mut field,
        )
        .unwrap();
        let other = FixedBaseTable::prepare_with(
            other_description,
            &other_base,
            &mut other_entries,
            &mut projective,
            &mut field,
        )
        .unwrap();
        for (case, right) in [("distinct", other), ("shared", table)] {
            if case == "shared" && width != other_width {
                continue;
            }
            let mut group = criterion.benchmark_group(format!(
                "{name}/expanded_products/{layout}/w{width}_{other_width}/{case}"
            ));
            group.throughput(Throughput::Elements(CORPUS_SIZE as u64));
            for (shape, scalars) in [("full", full), ("short", short), ("tiny", tiny)] {
                for pair in scalars.chunks_exact(2) {
                    let expected = [ladder(&base, &pair[0]), ladder(right.base(), &pair[1])];
                    assert_eq!([table.mul(&pair[0]), right.mul(&pair[1])], expected);
                }
                group.bench_function(shape, |b| {
                    b.iter(|| {
                        let (left, right, scalars) = black_box((&table, &right, &scalars));
                        for pair in scalars.chunks_exact(2) {
                            black_box([left.mul(&pair[0]), right.mul(&pair[1])]);
                        }
                    })
                });
            }
            group.finish();
        }
    }
}
