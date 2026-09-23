use super::*;
use crate::msm::test_support::Buffers;
use crate::{
    curve::{Pallas, Vesta, msm::test_support::Buffers, pasta::test_reference::multiply},
    exec::{ExecutionOptions, SerialExecutor},
    field::PrimeModulus,
    field::pasta::test_support::{field_samples, modulus},
};
use num_bigint::BigUint;
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    vec,
    vec::Vec,
};

fn reference<C: PastaCurve>(bases: &[Point<C>], coefficients: &[BigUint]) -> ProjectivePoint<C> {
    bases
        .iter()
        .zip(coefficients)
        .fold(ProjectivePoint::IDENTITY, |sum, (base, coefficient)| {
            let mut term = ProjectivePoint::IDENTITY;
            for byte in coefficient.to_bytes_be() {
                for bit in (0..8).rev() {
                    term = term.double();
                    if byte & (1 << bit) != 0 {
                        term = term.add(&base.to_projective());
                    }
                }
            }
            sum.add(&term)
        })
}

fn integer<M: PrimeModulus, S: ReductionState>(value: PastaField<M, S>) -> BigUint {
    BigUint::from_bytes_le(&value.to_bytes())
}

fn execute<C: PastaCurve>(input: Input<'_, C>) -> ProjectivePoint<C> {
    let options = ExecutionOptions::default();
    let mut scratch = Buffers::new(input.requirements(options).unwrap());
    input
        .execute(options, &SerialExecutor, scratch.borrow())
        .unwrap()
}

fn bases<C: PastaCurve>(n: usize) -> Vec<Point<C>> {
    field_samples::<C::Scalar>()
        .take(n)
        .enumerate()
        .map(|(i, s)| {
            if i % 7 == 0 {
                Point::IDENTITY
            } else {
                multiply(&s, |sum| sum.add(&ProjectivePoint::GENERATOR)).to_point()
            }
        })
        .collect()
}

fn sparse<C: PastaCurve>() {
    let modulus = modulus::<C::Scalar>();
    for n in [0, 1, 2, 7, 33, 257] {
        let mut points = bases::<C>(n + 2);
        if n >= 2 {
            points[2] = points[1].neg();
        }
        // Indices address this region, not the enclosing storage.
        let points = &points[1..n + 1];
        let basis = BasisSum::prepare(points);
        assert_eq!(basis.original().as_ptr(), points.as_ptr());
        assert_eq!(
            basis.sum().to_projective(),
            reference(points, &vec![1_u8.into(); n])
        );
        for k in [0, n, n * 2] {
            let indices: Vec<_> = (0..k).map(|i| ((i * 17 + 1) % n) as u32).collect();
            let mut differences: Vec<_> = field_samples::<C::Scalar>().take(k).collect();
            if k >= 2 {
                differences[0] = PastaField::from_montgomery_limbs(C::Scalar::MODULUS);
                differences[1] = PastaField::<C::Scalar>::ONE.neg();
            }
            for constant in [
                PastaField::ZERO,
                PastaField::ONE,
                field_samples().next().unwrap(),
            ] {
                let mut dense = vec![integer(constant); n];
                for (&index, &difference) in indices.iter().zip(&differences) {
                    dense[index as usize] =
                        (&dense[index as usize] + integer(difference)) % &modulus;
                }
                let corrections = basis.corrections(&indices, &differences).unwrap();
                let extra_bases = [Point::GENERATOR, Point::IDENTITY, Point::GENERATOR.neg()];
                let extra_scalars = [PastaField::from_u64(11), constant, PastaField::from_u64(3)];
                let extra = Input::new(Bases::Points(&extra_bases), &extra_scalars);
                let actual = basis
                    .sum()
                    .mul_projective(&constant)
                    .add(&execute(corrections))
                    .add(&execute(extra));
                let expected = reference(points, &dense)
                    .add(&reference(&extra_bases, &extra_scalars.map(integer)));
                assert_eq!(actual, expected);

                // A prior dense combination can have unrelated coefficients.
                let prior: Vec<_> = field_samples::<C::Scalar>().take(n).map(integer).collect();
                let previous = reference(points, &prior);
                let mut updated = prior;
                for (&index, &difference) in indices.iter().zip(&differences) {
                    updated[index as usize] =
                        (&updated[index as usize] + integer(difference)) % &modulus;
                }
                assert_eq!(
                    previous.add(&execute(corrections)).add(&execute(extra)),
                    reference(points, &updated)
                        .add(&reference(&extra_bases, &extra_scalars.map(integer)))
                );
            }
        }
    }
}

fn tails<C: PastaCurve>() {
    let modulus = modulus::<C::Scalar>();
    let upper = ((&modulus << 1_usize) - BigUint::from(1_u8))
        .to_u64_digits()
        .try_into()
        .unwrap();
    for n in [0, 1, 2, 17, 257] {
        let points = bases::<C>(n);
        let basis = BasisSum::prepare(&points);
        for k in [0, n / 2, n] {
            let mut tail: Vec<_> = field_samples::<C::Scalar>().take(k).collect();
            if k > 0 {
                tail[0] = PastaField::from_montgomery_limbs(upper);
                tail[k - 1] = PastaField::from_montgomery_limbs(C::Scalar::MODULUS);
            }
            for constant in [
                PastaField::ZERO,
                PastaField::ONE,
                PastaField::from_montgomery_limbs(upper),
            ] {
                let row = ConstantPrefix::new(n, &constant, &tail).unwrap();
                let mut dense = vec![integer(constant); n - k];
                dense.extend(tail.iter().copied().map(integer));
                let expected = reference(&points, &dense);
                let mut differences = vec![PastaField::ONE; k + 1];
                let input = basis.tail_corrections(row, &mut differences);
                assert_eq!(
                    basis.sum().mul_projective(&constant).add(&execute(input)),
                    expected
                );
                for (&difference, &value) in differences.iter().zip(&tail) {
                    assert_eq!(
                        integer(difference),
                        (integer(value) + &modulus - integer(constant)) % &modulus
                    );
                }
                assert_eq!(differences[k].reduce(), PastaField::<_>::ONE.reduce());
                let reduced: Vec<_> = tail.iter().map(|v| v.reduce()).collect();
                let row = ConstantPrefix::new(n, &constant.reduce(), &reduced).unwrap();
                let input = basis.tail_corrections(row, &mut differences);
                assert_eq!(
                    basis.sum().mul_projective(&constant).add(&execute(input)),
                    expected
                );
                assert_eq!(differences[k].reduce(), PastaField::<_>::ONE.reduce());
            }
        }
    }
}

fn rejection<C: PastaCurve>() {
    let points = [Point::<C>::GENERATOR; 3];
    let basis = BasisSum::prepare(&points);
    for indices in [&[3][..], &[0, u32::MAX][..]] {
        let zeros = vec![PastaField::ZERO; indices.len()];
        let position = indices.len() - 1;
        assert_eq!(
            basis.corrections(indices, &zeros).err().unwrap(),
            CurveError::BaseIndexOutOfBounds {
                position,
                index: indices[position],
                bases: 3,
            }
        );
    }
    let empty = BasisSum::<C>::prepare(&[]);
    assert_eq!(empty.sum(), Point::IDENTITY);
    assert!(empty.corrections(&[0], &[PastaField::ZERO]).is_err());
    assert!(catch_unwind(AssertUnwindSafe(|| basis.corrections(&[0], &[]))).is_err());
    let mut differences = [PastaField::ONE; 4];
    for n in [0, 2, 4] {
        let row = ConstantPrefix::new(
            n,
            &PastaField::<C::Scalar>::ZERO,
            &[] as &[PastaField<C::Scalar>],
        )
        .unwrap();
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                basis.tail_corrections(row, &mut differences);
            }))
            .is_err()
        );
        assert_eq!(
            differences.map(PastaField::reduce),
            [PastaField::<_>::ONE.reduce(); 4]
        );
    }
    let row = ConstantPrefix::new(
        3,
        &PastaField::<C::Scalar>::ZERO,
        &[PastaField::<C::Scalar>::ZERO; 3],
    )
    .unwrap();
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            basis.tail_corrections(row, &mut differences[..2]);
        }))
        .is_err()
    );
    assert_eq!(
        differences.map(PastaField::reduce),
        [PastaField::<_>::ONE.reduce(); 4]
    );
}

#[test]
fn constant_regions_sparse_updates_and_extras_match_dense_integer_oracles() {
    sparse::<Pallas>();
    sparse::<Vesta>();
}

#[test]
fn actual_tails_match_dense_rows_and_integer_differences() {
    tails::<Pallas>();
    tails::<Vesta>();
}

#[test]
fn correction_rejections_preserve_output() {
    rejection::<Pallas>();
    rejection::<Vesta>();
}
