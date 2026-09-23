use super::*;
use crate::{
    curve::{
        AffinePoint, EisensteinTableBatch, Pallas, Point, PreparedAffinePoint, ProjectivePoint,
        Vesta, msm::tests::Buffers, scalar,
    },
    exec::{ExecutionOptions, SerialExecutor, TaskBudget},
    field::{PrimeModulus, Reduced},
    test_support::field_samples,
};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    vec,
    vec::Vec,
};

fn execute<C: PastaCurve>(input: Input<'_, C>) -> ProjectivePoint<C> {
    let options = ExecutionOptions::default();
    let mut scratch = Buffers::new(input.requirements(options).unwrap());
    input
        .execute(options, &SerialExecutor, scratch.borrow())
        .unwrap()
}

fn check<C: PastaCurve, S: ReductionState>(
    selection: Selection<'_, C>,
    points: &[Point<C>],
    row: &[PastaField<C::Scalar, S>],
) {
    // Canonical bytes independently distinguish both representations of zero.
    let positions: Vec<_> = row
        .iter()
        .enumerate()
        .filter(|(_, s)| s.to_bytes() != [0; 32])
        .map(|(i, _)| i)
        .collect();
    let expected = row
        .iter()
        .enumerate()
        .fold(ProjectivePoint::IDENTITY, |sum, (i, s)| {
            let index = selection.indices.map_or(i, |indices| indices[i] as usize);
            sum.add(&scalar::multiply(&s.into_loose(), |term| {
                term.add(&points[index].to_projective())
            }))
        });
    let loose: Vec<_> = row.iter().map(|s| s.into_loose()).collect();
    assert_eq!(execute(selection.with_scalars(&loose)), expected);
    let k = positions.len();
    let mut indices = vec![u32::MAX; k + 2];
    let mut scalars = vec![PastaField::ONE; k + 3];
    for exact in [true, false, true] {
        let input = selection
            .with_nonzero_scalars(
                row,
                &mut indices[..if exact { k } else { k + 2 }],
                &mut scalars[..if exact { k } else { k + 3 }],
            )
            .unwrap();
        assert_eq!(input.len(), k);
        assert_eq!(
            core::mem::discriminant(&input.bases),
            core::mem::discriminant(&selection.bases)
        );
        assert_eq!(execute(input), expected);
        let retained = input.selection();
        let next: Vec<_> = positions
            .iter()
            .map(|&i| row[i].into_loose().neg())
            .collect();
        assert_eq!(execute(retained.with_scalars(&next)), expected.neg());
        for (output, &position) in positions.iter().enumerate() {
            assert_eq!(
                indices[output],
                selection.indices.map_or(position as u32, |i| i[position])
            );
            assert_eq!(
                scalars[output].montgomery_limbs(),
                row[position].montgomery_limbs()
            );
        }
        assert!(indices[k..].iter().all(|&i| i == u32::MAX));
        assert!(
            scalars[k..]
                .iter()
                .all(|s| s.reduce() == PastaField::<_, Reduced>::ONE)
        );
    }
    // Dirty live prefixes from earlier calls are outside the next empty result.
    let before_indices = indices.clone();
    let before_scalars: Vec<_> = scalars.iter().map(|s| s.montgomery_limbs()).collect();
    let zeros = vec![PastaField::<C::Scalar>::from_montgomery_limbs(C::Scalar::MODULUS); row.len()];
    assert!(
        selection
            .with_nonzero_scalars(&zeros, &mut indices, &mut scalars)
            .unwrap()
            .is_empty()
    );
    assert_eq!(indices, before_indices);
    assert_eq!(
        scalars
            .iter()
            .map(|s| s.montgomery_limbs())
            .collect::<Vec<_>>(),
        before_scalars
    );
}

fn cases<C: PastaCurve>() {
    use super::super::Bases;
    for n in [0, 1, 2, 3, 17, 65] {
        let affine: Vec<_> = field_samples::<C::Scalar>()
            .skip(3)
            .take(n)
            .map(|s| {
                scalar::multiply(&s, |sum| sum.add(&ProjectivePoint::<C>::GENERATOR))
                    .to_point()
                    .as_affine()
                    .copied()
                    .unwrap()
            })
            .collect();
        let ordinary: Vec<_> = affine.iter().map(AffinePoint::to_point).collect();
        let mut points = ordinary.clone();
        if n > 0 {
            points[0] = Point::IDENTITY;
        }
        if n > 2 {
            points[2] = points[1].neg();
        }
        let cached: Vec<_> = affine
            .iter()
            .map(PreparedAffinePoint::from_affine)
            .collect();
        let r = EisensteinTableBatch::<C>::requirements(n).unwrap();
        let mut entries = vec![AffinePoint::GENERATOR; r.table_entries];
        let mut cached_entries =
            vec![PreparedAffinePoint::from_affine(&AffinePoint::GENERATOR); r.table_entries];
        let mut projective = vec![ProjectivePoint::IDENTITY; r.projective_scratch];
        let mut field = vec![PastaField::ZERO; r.field_scratch];
        let tables = EisensteinTableBatch::prepare(
            &affine,
            &mut entries,
            &mut projective,
            &mut field,
            TaskBudget::SERIAL,
            &SerialExecutor,
        );
        let cached_tables = EisensteinTableBatch::prepare(
            &affine,
            &mut cached_entries,
            &mut projective,
            &mut field,
            TaskBudget::SERIAL,
            &SerialExecutor,
        );
        for (bases, reference) in [
            (Bases::Points(&points), &points),
            (Bases::Affine(&affine), &ordinary),
            (Bases::Prepared(&cached), &ordinary),
            (Bases::Compact(tables), &ordinary),
            (Bases::CompactPrepared(cached_tables), &ordinary),
        ] {
            let mapping: Vec<_> = (0..n * 2).map(|i| ((i * 11 + 1) % n) as u32).collect();
            for selection in [
                Selection::new(bases),
                Selection::indexed(bases, &mapping).unwrap(),
            ] {
                for stride in [0, 1, 2, 17] {
                    let row: Vec<_> = field_samples::<C::Scalar>()
                        .take(selection.len())
                        .enumerate()
                        .map(|(i, s)| {
                            if stride != 0 && i % stride == 0 {
                                s
                            } else if i % 2 == 0 {
                                PastaField::ZERO
                            } else {
                                PastaField::from_montgomery_limbs(C::Scalar::MODULUS)
                            }
                        })
                        .collect();
                    check(selection, reference, &row);
                    let reduced: Vec<_> = row.iter().map(|s| s.reduce()).collect();
                    check(selection, reference, &reduced);
                }
            }
        }
    }
}

#[test]
fn nonzero_support_matches_dense_and_binary_sums() {
    cases::<Pallas>();
    cases::<Vesta>();
}

fn rejection<C: PastaCurve>() {
    use super::super::Bases;
    let bases = [Point::<C>::GENERATOR; 3];
    let selection = Selection::new(Bases::Points(&bases));
    let row = [
        PastaField::<C::Scalar>::ONE,
        PastaField::ZERO,
        PastaField::ONE,
        PastaField::ZERO,
    ];
    let mut indices = [u32::MAX; 4];
    let mut scalars = [PastaField::<C::Scalar>::ZERO; 4];
    for (count, index_len, scalar_len) in [(2, 4, 4), (4, 4, 4), (3, 1, 4), (3, 4, 1), (3, 0, 0)] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _ = selection.with_nonzero_scalars(
                    &row[..count],
                    &mut indices[..index_len],
                    &mut scalars[..scalar_len],
                );
            }))
            .is_err()
        );
        assert_eq!(indices, [u32::MAX; 4]);
        assert!(scalars.iter().all(|s| s.montgomery_limbs() == [0; 4]));
    }
    // Invalid known support is rejected even if its coefficient would be zero.
    for mapping in [&[3][..], &[0, u32::MAX][..]] {
        assert_eq!(
            Selection::indexed(Bases::Points(&bases), mapping)
                .err()
                .unwrap(),
            CurveError::BaseIndexOutOfBounds {
                position: mapping.len() - 1,
                index: mapping[mapping.len() - 1],
                bases: 3
            }
        );
    }
}

#[test]
fn nonzero_support_rejects_before_writes() {
    rejection::<Pallas>();
    rejection::<Vesta>();
    assert_eq!(check_dense_index(None), Ok(()));
    assert_eq!(check_dense_index(Some(u32::MAX as usize)), Ok(()));
    if let Some(too_large) = (u32::MAX as usize).checked_add(1) {
        assert_eq!(
            check_dense_index(Some(too_large)),
            Err(CurveError::SizeOverflow)
        );
        assert_eq!(
            check_dense_index(Some(usize::MAX)),
            Err(CurveError::SizeOverflow)
        );
    }
}

fn boundaries<C: PastaCurve>() {
    use super::super::Bases;
    use crate::test_support::modulus;
    use num_bigint::BigUint;
    let p = modulus::<C::Scalar>();
    let one = BigUint::from(1_u8);
    let row: Vec<_> = [&p - &one, &p + &one, (&p << 1_usize) - &one]
        .iter()
        .map(|v| {
            PastaField::<C::Scalar>::from_montgomery_limbs(v.to_u64_digits().try_into().unwrap())
        })
        .chain([
            PastaField::ONE,
            PastaField::<C::Scalar>::ONE.neg(),
            PastaField::ZERO,
            PastaField::from_montgomery_limbs(C::Scalar::MODULUS),
        ])
        .collect();
    let points = [Point::<C>::GENERATOR; 7];
    let selection = Selection::new(Bases::Points(&points));
    check(selection, &points, &row);
    check(
        selection,
        &points,
        &row.iter().map(|s| s.reduce()).collect::<Vec<_>>(),
    );
}

#[test]
fn nonzero_support_preserves_boundary_representations() {
    boundaries::<Pallas>();
    boundaries::<Vesta>();
}
