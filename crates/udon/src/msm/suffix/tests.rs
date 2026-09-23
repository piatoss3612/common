use super::*;
use crate::{
    curve::{Pallas, Vesta, pasta::test_reference::multiply},
    exec::{ExecutionOptions, SerialExecutor},
    field::pasta::test_support::{field_samples, modulus},
    field::{CanonicalUint, PrimeModulus},
    msm::test_support::Buffers,
    msm::test_support::Buffers,
};
use num_bigint::BigUint;
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    vec,
    vec::Vec,
};

fn bases<C: PastaCurve>(n: usize) -> Vec<Point<C>> {
    field_samples::<C::Scalar>()
        .take(n)
        .map(|scalar| Point::<C>::GENERATOR.mul_projective(&scalar).to_point())
        .collect()
}

fn reference<C: PastaCurve>(
    bases: &[Point<C>],
    coefficients: &[PastaField<C::Scalar>],
) -> ProjectivePoint<C> {
    bases
        .iter()
        .zip(coefficients)
        .fold(ProjectivePoint::IDENTITY, |sum, (base, coefficient)| {
            sum.add(&multiply(coefficient, |sum| sum.add(&base.to_projective())))
        })
}

fn from_u128<M: PrimeModulus>(value: u128) -> PastaField<M> {
    PastaField::from_canonical_uint(CanonicalUint::from_limbs([
        value as u64,
        (value >> 64) as u64,
        0,
        0,
    ]))
    .unwrap()
}

fn check_fields<C: PastaCurve>() {
    let mut samples = field_samples::<C::Scalar>();
    let modulus = modulus::<C::Scalar>();
    let upper: [u64; 4] = ((&modulus << 1_usize) - BigUint::from(1_u8))
        .to_u64_digits()
        .try_into()
        .unwrap();
    for n in [0, 1, 2, 3, 8, 17, 33, 257] {
        let mut points = bases::<C>(n);
        for index in 0..n {
            if index % 7 == 0 {
                points[index] = Point::IDENTITY;
            }
            if index % 7 == 2 {
                points[index] = points[index - 1].neg();
            }
        }
        let mut rows: Vec<Vec<PastaField<C::Scalar>>> = vec![
            vec![PastaField::ZERO; n],
            vec![PastaField::ONE; n],
            (0..n).map(|_| samples.next().unwrap()).collect(),
            (0..n)
                .map(|i| PastaField::from_i64(2 - (i % 9) as i64))
                .collect(),
        ];
        let repeated = samples.next().unwrap();
        rows.push(
            (0..n)
                .map(|i| {
                    if i % 11 < 8 {
                        repeated
                    } else {
                        samples.next().unwrap()
                    }
                })
                .collect(),
        );
        rows.push(
            (0..n)
                .map(|i| match i % 4 {
                    0 => PastaField::from_montgomery_limbs(C::Scalar::MODULUS),
                    1 => PastaField::<C::Scalar>::ONE.neg(),
                    2 => PastaField::ONE,
                    _ => PastaField::from_montgomery_limbs(upper),
                })
                .collect(),
        );
        for coefficients in rows {
            let expected = reference(&points, &coefficients);
            for (capacity, field_capacity) in [(1, 0), (7, 2), (n.max(1) + 3, n + 5)] {
                let mut output = vec![Point::GENERATOR; n + 1];
                let mut projective = vec![ProjectivePoint::GENERATOR; capacity + 1];
                let mut field = vec![PastaField::ONE; field_capacity + 1];
                let suffix = SuffixBasis::prepare(
                    &points,
                    &mut output,
                    &mut projective[..capacity],
                    &mut field[..field_capacity],
                );
                assert_eq!(suffix.len(), n);
                assert_eq!(suffix.is_empty(), n == 0);
                for index in 0..n {
                    // A separate forward sum verifies every suffix and its order.
                    let sum = points[index..]
                        .iter()
                        .fold(ProjectivePoint::IDENTITY, |sum, point| {
                            sum.add(&point.to_projective())
                        });
                    assert_eq!(suffix.suffix()[index].to_projective(), sum);
                }
                let mut differences = vec![PastaField::ONE; n + 1];
                let input = suffix.with_scalars(&coefficients, &mut differences);
                let options = ExecutionOptions::default();
                let mut scratch = Buffers::new(input.requirements(options).unwrap());
                assert_eq!(
                    input
                        .execute(options, &SerialExecutor, scratch.borrow())
                        .unwrap(),
                    expected
                );
                let mut previous = BigUint::from(0_u8);
                for (index, value) in coefficients.iter().enumerate() {
                    let value = BigUint::from_bytes_le(&value.to_bytes());
                    let difference = (&value + &modulus - previous) % &modulus;
                    assert_eq!(
                        BigUint::from_bytes_le(&differences[index].to_bytes()),
                        difference
                    );
                    previous = value;
                }
                assert_eq!(differences[n].reduce(), PastaField::<_>::ONE.reduce());
                assert_eq!(output[n], Point::GENERATOR);
                assert_eq!(projective[capacity], ProjectivePoint::GENERATOR);
                assert_eq!(
                    field[field_capacity.min(n).min(capacity)].reduce(),
                    PastaField::<_>::ONE.reduce()
                );
            }
        }
    }
}

fn check_unsigned<C: PastaCurve>() {
    let rows = [
        vec![],
        vec![0],
        vec![u128::MAX],
        vec![0, u128::MAX],
        vec![u128::MAX; 9],
        vec![
            0,
            1,
            1,
            u64::MAX as u128,
            1 << 64,
            (1 << 127) - 1,
            1 << 127,
            u128::MAX,
        ],
        (0..513).map(|i| (1 << 100) + (i / 7) as u128).collect(),
    ];
    for coefficients in rows {
        let n = coefficients.len();
        let points = bases::<C>(n);
        let mut suffix = vec![Point::IDENTITY; n];
        let basis = SuffixBasis::prepare(
            &points,
            &mut suffix,
            &mut [ProjectivePoint::IDENTITY; 3],
            &mut [PastaField::ZERO; 2],
        );
        let mut differences = vec![73; n + 1];
        let input = basis
            .with_monotone_unsigned(&coefficients, &mut differences)
            .unwrap();
        let options = ExecutionOptions::default();
        let mut scratch = Buffers::new(input.requirements(options).unwrap());
        let fields: Vec<_> = coefficients.iter().map(|x| from_u128(*x)).collect();
        assert_eq!(
            input
                .execute(options, &SerialExecutor, scratch.borrow())
                .unwrap(),
            reference(&points, &fields)
        );
        let mut previous = 0;
        for (index, coefficient) in coefficients.iter().enumerate() {
            assert_eq!(differences[index], coefficient - previous);
            previous = *coefficient;
        }
        assert_eq!(differences[n], 73);
    }
}

fn check_rejection<C: PastaCurve>() {
    let points = [Point::<C>::GENERATOR; 4];
    let mut output = [Point::GENERATOR; 5];
    let mut projective = [ProjectivePoint::GENERATOR; 5];
    let mut field = [PastaField::ONE; 5];
    for output_len in [0, 3] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                SuffixBasis::prepare(
                    &points,
                    &mut output[..output_len],
                    &mut projective,
                    &mut field,
                );
            }))
            .is_err()
        );
        assert_eq!(output, [Point::GENERATOR; 5]);
        assert_eq!(projective, [ProjectivePoint::GENERATOR; 5]);
        assert_eq!(
            field.map(PastaField::reduce),
            [PastaField::<_>::ONE.reduce(); 5]
        );
    }
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            SuffixBasis::prepare(&points, &mut output, &mut [], &mut field);
        }))
        .is_err()
    );
    assert_eq!(output, [Point::GENERATOR; 5]);
    assert_eq!(
        field.map(PastaField::reduce),
        [PastaField::<_>::ONE.reduce(); 5]
    );
    let basis = SuffixBasis::prepare(&points, &mut output, &mut projective, &mut []);
    let mut differences = [73; 5];
    for (row, position) in [
        ([1, 0, 2, 3], 1),
        ([0, 2, 1, 3], 2),
        ([0, 1, u128::MAX, u128::MAX - 1], 3),
    ] {
        assert_eq!(
            basis
                .with_monotone_unsigned(&row, &mut differences)
                .err()
                .unwrap(),
            CurveError::InvalidScalar { position }
        );
        assert_eq!(differences, [73; 5]);
    }
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = basis.with_monotone_unsigned(&[0; 3], &mut differences);
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = basis.with_monotone_unsigned(&[0; 4], &mut differences[..3]);
        }))
        .is_err()
    );
    assert_eq!(differences, [73; 5]);
    let mut differences = [PastaField::ONE; 5];
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            basis.with_scalars(&[PastaField::ZERO; 3], &mut differences);
        }))
        .is_err()
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            basis.with_scalars(&[PastaField::ZERO; 4], &mut differences[..3]);
        }))
        .is_err()
    );
    assert_eq!(
        differences.map(PastaField::reduce),
        [PastaField::<_>::ONE.reduce(); 5]
    );
}

#[test]
fn suffix_and_field_differences_match_integer_and_binary_oracles() {
    check_fields::<Pallas>();
    check_fields::<Vesta>();
}
#[test]
fn monotone_unsigned_differences_preserve_boundaries() {
    check_unsigned::<Pallas>();
    check_unsigned::<Vesta>();
}
#[test]
fn invalid_rows_and_storage_preserve_outputs() {
    check_rejection::<Pallas>();
    check_rejection::<Vesta>();
}

#[test]
fn empty_basis_accepts_empty_scratch() {
    let basis = SuffixBasis::<Pallas>::prepare(&[], &mut [], &mut [], &mut []);
    assert!(basis.is_empty());
    assert_eq!(basis.with_scalars(&[], &mut []).len(), 0);
    assert_eq!(basis.with_monotone_unsigned(&[], &mut []).unwrap().len(), 0);
}
