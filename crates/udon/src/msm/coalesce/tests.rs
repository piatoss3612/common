use super::*;
use crate::msm::test_support::Buffers;
use crate::{
    curve::{Pallas, PreparedAffinePoint, ProjectivePoint, Vesta, pasta::test_reference::multiply},
    exec::{ExecutionOptions, SerialExecutor},
    field::pasta::test_support::{field_samples, modulus},
    field::{PrimeModulus, Reduced},
};
use num_bigint::BigUint;
use std::{
    collections::BTreeMap,
    panic::{AssertUnwindSafe, catch_unwind},
    vec,
    vec::Vec,
};

fn integer<M: PrimeModulus, S: ReductionState>(value: &PastaField<M, S>) -> BigUint {
    BigUint::from_bytes_le(&value.to_bytes())
}

fn reference<C: PastaCurve>(
    bases: &[Point<C>],
    scalars: &[PastaField<C::Scalar>],
) -> ProjectivePoint<C> {
    bases
        .iter()
        .zip(scalars)
        .fold(ProjectivePoint::IDENTITY, |sum, (base, s)| {
            sum.add(&multiply(s, |term| term.add(&base.to_projective())))
        })
}

fn execute<C: PastaCurve>(input: Input<'_, C>) -> ProjectivePoint<C> {
    let options = ExecutionOptions::default();
    let mut scratch = Buffers::new(input.requirements(options).unwrap());
    input
        .execute(options, &SerialExecutor, scratch.borrow())
        .unwrap()
}

fn rows<C: PastaCurve>(n: usize) -> Vec<Vec<PastaField<C::Scalar>>> {
    let p = modulus::<C::Scalar>();
    let upper = ((p << 1_usize) - BigUint::from(1_u8))
        .to_u64_digits()
        .try_into()
        .unwrap();
    vec![
        vec![PastaField::ZERO; n],
        vec![PastaField::from_montgomery_limbs(C::Scalar::MODULUS); n],
        vec![PastaField::ONE; n],
        (0..n)
            .map(|i| {
                if i % 2 == 0 {
                    PastaField::ONE
                } else {
                    PastaField::<C::Scalar>::ONE.neg()
                }
            })
            .collect(),
        field_samples().take(n).collect(),
        vec![PastaField::from_montgomery_limbs(upper); n],
    ]
}

fn points<C: PastaCurve>() {
    let seed: Vec<_> = field_samples::<C::Scalar>()
        .take(33)
        .map(|s| multiply(&s, |sum| sum.add(&ProjectivePoint::<C>::GENERATOR)).to_point())
        .collect();
    let p = modulus::<C::Scalar>();
    for n in [0, 1, 2, 3, 17, 65, 257] {
        for distinct in [1, 7, 33] {
            let bases: Vec<_> = (0..n)
                .map(|i| match i % 7 {
                    0 => Point::IDENTITY,
                    1..=3 => seed[(i * 11) % distinct],
                    _ => seed[(i * 11) % distinct].neg(),
                })
                .collect();
            let mut keys = vec![CoalescingKey::EMPTY; n + 1];
            for coefficients in rows::<C>(n) {
                let plan = CoalescingPlan::prepare(&bases, &mut keys);
                assert_eq!(plan.len(), n);
                assert_eq!(plan.is_empty(), n == 0);
                let mut expected = BTreeMap::<[u8; 32], BigUint>::new();
                for (base, coefficient) in bases.iter().zip(&coefficients) {
                    if let Some((x, y)) = base.coordinates() {
                        let value = integer(coefficient);
                        let signed = if y.is_odd() { (&p - value) % &p } else { value };
                        let entry = expected.entry(x.to_bytes()).or_default();
                        *entry = (&*entry + signed) % &p;
                    }
                }
                assert_eq!(plan.groups(), expected.len());
                expected.retain(|_, value| *value != BigUint::from(0_u8));
                let mut output = vec![Point::GENERATOR; plan.groups() + 2];
                let mut sums = vec![PastaField::ONE; plan.groups() + 3];
                let reduced: Vec<_> = coefficients.iter().map(|s| s.reduce()).collect();
                for pass in 0..2 {
                    output.fill(Point::GENERATOR);
                    sums.fill(PastaField::ONE);
                    let input = if pass == 0 {
                        plan.with_scalars(&coefficients, &mut output, &mut sums)
                    } else {
                        plan.with_scalars(&reduced, &mut output, &mut sums)
                    };
                    let live = input.len();
                    assert_eq!(live, expected.len());
                    assert_eq!(execute(input), reference(&bases, &coefficients));
                    for ((point, sum), (x, value)) in output.iter().zip(&sums).zip(&expected) {
                        let (actual_x, y) = point.coordinates().unwrap();
                        assert_eq!(&actual_x.to_bytes(), x);
                        assert!(!y.is_odd());
                        assert_eq!(&integer(sum), value);
                    }
                    assert!(output[live..].iter().all(|p| *p == Point::GENERATOR));
                    assert!(
                        sums[live..]
                            .iter()
                            .all(|s| s.reduce() == PastaField::<_, Reduced>::ONE)
                    );
                }
            }
            assert_eq!(keys[n], CoalescingKey::EMPTY);
        }
    }
    // Both orientations and repeated points cancel modulo the scalar field.
    let g = Point::<C>::GENERATOR;
    for (bases, coefficients) in [
        ([g, g.neg()], [PastaField::ONE; 2]),
        (
            [g, g],
            [PastaField::ONE, PastaField::<C::Scalar>::ONE.neg()],
        ),
    ] {
        let mut keys = [CoalescingKey::EMPTY; 2];
        let plan = CoalescingPlan::prepare(&bases, &mut keys);
        assert_eq!(plan.groups(), 1);
        let mut points = [g];
        let mut sums = [PastaField::ONE];
        let input = plan.with_scalars(&coefficients, &mut points, &mut sums);
        assert!(input.is_empty());
    }
}

fn indexed<C: PastaCurve>() {
    let affine: Vec<_> = (1..=17)
        .map(|i| {
            multiply::<C>(&PastaField::from_u64(i), |sum| {
                sum.add(&ProjectivePoint::GENERATOR)
            })
            .to_point()
            .as_affine()
            .copied()
            .unwrap()
        })
        .collect();
    let mut points: Vec<_> = affine.iter().map(|p| p.to_point()).collect();
    points[0] = Point::IDENTITY;
    points[2] = points[1];
    points[3] = points[1].neg();
    let prepared: Vec<_> = affine
        .iter()
        .map(PreparedAffinePoint::from_affine)
        .collect();
    let p = modulus::<C::Scalar>();
    for bases in [
        Bases::Points(&points),
        Bases::Affine(&affine),
        Bases::Prepared(&prepared),
    ] {
        for n in [0, 1, 2, 17, 257] {
            let indices: Vec<_> = (0..n).map(|i| ((i * 11 + 3) % 17) as u32).collect();
            let mut order = vec![usize::MAX; n + 1];
            let plan = IndexedCoalescingPlan::prepare(bases, &indices, &mut order).unwrap();
            assert_eq!(plan.len(), n);
            assert_eq!(plan.is_empty(), n == 0);
            for coefficients in rows::<C>(n) {
                let mut expected = BTreeMap::<u32, BigUint>::new();
                for (index, coefficient) in indices.iter().zip(&coefficients) {
                    let entry = expected.entry(*index).or_default();
                    *entry = (&*entry + integer(coefficient)) % &p;
                }
                assert_eq!(plan.groups(), expected.len());
                expected.retain(|_, value| *value != BigUint::from(0_u8));
                let mut output = vec![u32::MAX; plan.groups() + 2];
                let mut sums = vec![PastaField::ONE; plan.groups() + 3];
                let reduced: Vec<_> = coefficients.iter().map(|s| s.reduce()).collect();
                for pass in 0..2 {
                    output.fill(u32::MAX);
                    sums.fill(PastaField::ONE);
                    let input = if pass == 0 {
                        plan.with_scalars(&coefficients, &mut output, &mut sums)
                    } else {
                        plan.with_scalars(&reduced, &mut output, &mut sums)
                    };
                    let live = input.len();
                    assert_eq!(live, expected.len());
                    let selected: Vec<_> = indices
                        .iter()
                        .map(|&i| match bases {
                            Bases::Points(b) => b[i as usize],
                            _ => affine[i as usize].to_point(),
                        })
                        .collect();
                    assert_eq!(execute(input), reference(&selected, &coefficients));
                    for ((index, sum), (expected_index, value)) in
                        output.iter().zip(&sums).zip(&expected)
                    {
                        assert_eq!(index, expected_index);
                        assert_eq!(&integer(sum), value);
                    }
                    assert!(output[live..].iter().all(|i| *i == u32::MAX));
                    assert!(
                        sums[live..]
                            .iter()
                            .all(|s| s.reduce() == PastaField::<_, Reduced>::ONE)
                    );
                }
            }
            assert_eq!(order[n], usize::MAX);
        }
    }
    let plan = IndexedCoalescingPlan::<C>::prepare(Bases::Points(&[]), &[], &mut []).unwrap();
    assert!(
        plan.with_scalars(&[] as &[PastaField<C::Scalar>], &mut [], &mut [])
            .is_empty()
    );
}

fn rejection<C: PastaCurve>() {
    let bases = [Point::<C>::GENERATOR; 3];
    let mut keys = [CoalescingKey::EMPTY; 4];
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            CoalescingPlan::prepare(&bases, &mut keys[..2]);
        }))
        .is_err()
    );
    assert_eq!(keys, [CoalescingKey::EMPTY; 4]);
    let plan = CoalescingPlan::prepare(&bases, &mut keys);
    let mut points = [Point::IDENTITY; 3];
    let mut output = [u32::MAX; 3];
    let mut sums = [PastaField::<C::Scalar>::ONE; 3];
    for (count, point_len, scalar_len) in [(2, 3, 3), (4, 3, 3), (3, 0, 3), (3, 3, 0)] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                plan.with_scalars(
                    &[PastaField::<C::Scalar>::ZERO; 4][..count],
                    &mut points[..point_len],
                    &mut sums[..scalar_len],
                );
            }))
            .is_err()
        );
        assert_eq!(points, [Point::IDENTITY; 3]);
        assert_eq!(
            sums.map(PastaField::reduce),
            [PastaField::<_, Reduced>::ONE; 3]
        );
    }
    let mut order = [usize::MAX; 4];
    for indices in [&[3][..], &[0, u32::MAX][..]] {
        assert_eq!(
            IndexedCoalescingPlan::prepare(Bases::Points(&bases), indices, &mut order)
                .err()
                .unwrap(),
            CurveError::BaseIndexOutOfBounds {
                position: indices.len() - 1,
                index: indices[indices.len() - 1],
                bases: 3
            }
        );
        assert_eq!(order, [usize::MAX; 4]);
    }
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ =
                IndexedCoalescingPlan::prepare(Bases::Points(&bases), &[0, 0, 0], &mut order[..2]);
        }))
        .is_err()
    );
    assert_eq!(order, [usize::MAX; 4]);
    let plan =
        IndexedCoalescingPlan::prepare(Bases::Points(&bases), &[0, 0, 0], &mut order).unwrap();
    for (count, index_len, scalar_len) in [(2, 3, 3), (4, 3, 3), (3, 0, 3), (3, 3, 0)] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                plan.with_scalars(
                    &[PastaField::<C::Scalar>::ZERO; 4][..count],
                    &mut output[..index_len],
                    &mut sums[..scalar_len],
                );
            }))
            .is_err()
        );
        assert_eq!(output, [u32::MAX; 3]);
        assert_eq!(
            sums.map(PastaField::reduce),
            [PastaField::<_, Reduced>::ONE; 3]
        );
    }
}

#[test]
fn coalesced_points_match_integer_groups_and_uncoalesced_binary_sums() {
    points::<Pallas>();
    points::<Vesta>();
}

#[test]
fn coalesced_indices_match_integer_groups_and_uncoalesced_binary_sums() {
    indexed::<Pallas>();
    indexed::<Vesta>();
}

#[test]
fn coalescing_rejects_before_writing_any_storage() {
    rejection::<Pallas>();
    rejection::<Vesta>();
}
