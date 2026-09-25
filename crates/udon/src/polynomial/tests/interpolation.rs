use super::*;
use crate::field::{PallasBase, PallasScalar, batch_invert_groups, count_inversions};
use crate::test_support::{field_samples, integer, modulus};
use num_bigint::BigUint;
use std::{vec, vec::Vec};

fn raw<M: PrimeModulus>(value: &BigUint) -> PastaField<M> {
    let digits = value.to_u64_digits();
    let mut limbs = [0; 4];
    limbs[..digits.len()].copy_from_slice(&digits);
    PastaField::from_montgomery_limbs(limbs)
}

fn canonical<M: PrimeModulus, S: ReductionState>(value: &PastaField<M, S>) -> BigUint {
    let p = modulus::<M>();
    integer(&value.montgomery_limbs()) * (BigUint::from(1u8) << 256usize).modinv(&p).unwrap() % p
}

fn assert_value<M: PrimeModulus>(value: &PastaField<M>, expected: &BigUint) {
    assert!(integer(&value.montgomery_limbs()) < modulus::<M>() * 2u8);
    assert_eq!(canonical(value), *expected);
}

fn assert_untouched<M: PrimeModulus>(values: &[PastaField<M>], sentinel: PastaField<M>) {
    assert!(
        values
            .iter()
            .all(|v| v.montgomery_limbs() == sentinel.montgomery_limbs())
    );
}

// Construct each basis independently by out-of-place integer convolution. This
// does not share vanishing preparation or synthetic division with production.
fn oracle(points: &[BigUint], values: &[BigUint], p: &BigUint) -> (Vec<BigUint>, Vec<BigUint>) {
    let mut coefficients = vec![BigUint::from(0u8); points.len()];
    let mut weights = Vec::new();
    for (i, point) in points.iter().enumerate() {
        let mut basis = vec![BigUint::from(1u8)];
        let mut denominator = BigUint::from(1u8);
        for (j, other) in points.iter().enumerate() {
            if i == j {
                continue;
            }
            let mut next = vec![BigUint::from(0u8); basis.len() + 1];
            for (k, a) in basis.iter().enumerate() {
                next[k] = (&next[k] + a * (p - other)) % p;
                next[k + 1] = (&next[k + 1] + a) % p;
            }
            basis = next;
            denominator = denominator * (point + p - other) % p;
        }
        let weight = denominator.modinv(p).unwrap();
        for (out, coefficient) in coefficients.iter_mut().zip(basis) {
            *out = (&*out + coefficient * &weight * &values[i]) % p;
        }
        weights.push(weight);
    }
    (coefficients, weights)
}

fn evaluate_integer(coefficients: &[BigUint], point: &BigUint, p: &BigUint) -> BigUint {
    coefficients
        .iter()
        .rev()
        .fold(BigUint::from(0u8), |acc, a| (acc * point + a) % p)
}

fn check_case<M: PrimeModulus, S: ReductionState, T: ReductionState>(
    points: &[PastaField<M, S>],
    values: &[PastaField<M, T>],
) {
    let n = points.len();
    let p = modulus::<M>();
    let sentinel = raw::<M>(&(&p * 2u8 - 1u8));
    let point_integers: Vec<_> = points.iter().map(canonical).collect();
    let value_integers: Vec<_> = values.iter().map(canonical).collect();
    let (expected, expected_weights) = oracle(&point_integers, &value_integers, &p);
    let queries: Vec<_> = points
        .iter()
        .copied()
        .map(PastaField::into_loose)
        .chain(field_samples::<M>().skip(40).take(3))
        .chain([PastaField::<M>::ZERO, PastaField::ONE, sentinel])
        .collect();
    for scratch_len in [0, 1, 2, 3, n, n + 3] {
        let mut weights = vec![sentinel; n + 2];
        let mut scratch = vec![sentinel; scratch_len];
        let mut plan = None;
        let inversions = count_inversions(|| {
            plan = Some(InterpolationPlan::prepare(points, &mut weights, &mut scratch).unwrap());
        });
        let plan = plan.unwrap();
        assert_eq!(plan.points().len(), n);
        if n <= 1 {
            assert_eq!(inversions, 0);
            assert_untouched(&scratch, sentinel);
        } else if scratch_len >= n {
            assert_eq!(inversions, 1);
        } else if scratch_len == 0 {
            assert_eq!(inversions, n);
        }
        assert_untouched(&scratch[n.min(scratch_len)..], sentinel);
        for (actual, expected) in plan.weights().iter().zip(&expected_weights) {
            assert_value(actual, expected);
        }
        let bound = InterpolationPlan::bind(points, plan.weights()).unwrap();
        assert!(core::ptr::eq(bound.points(), points));
        assert!(core::ptr::eq(bound.weights(), plan.weights()));
        let mut output = vec![sentinel; n + 2];
        let mut scratch = vec![sentinel; n + 3];
        assert_eq!(
            count_inversions(|| {
                assert_eq!(bound.interpolate(values, &mut output, &mut scratch), Ok(n));
            }),
            0
        );
        for (actual, expected) in output.iter().zip(&expected) {
            assert_value(actual, expected);
        }
        assert_untouched(&output[n..], sentinel);
        assert_untouched(&scratch[n..], sentinel);
        for query in &queries {
            scratch.fill(sentinel);
            let mut actual = PastaField::ZERO;
            assert_eq!(
                count_inversions(|| {
                    actual = plan
                        .evaluate(values, &query.reduce(), &mut scratch)
                        .unwrap();
                }),
                0
            );
            let expected = evaluate_integer(&expected, &canonical(query), &p);
            assert_value(&actual, &expected);
            assert_value(
                &plan.evaluate(values, query, &mut scratch).unwrap(),
                &expected,
            );
            if n <= 1 || point_integers.contains(&canonical(query)) {
                assert_untouched(&scratch, sentinel);
            } else {
                assert_untouched(&scratch[n..], sentinel);
            }
        }
        assert_untouched(&weights[n..], sentinel);
    }
}

fn arithmetic_field<M: PrimeModulus>() {
    let p = modulus::<M>();
    let mut points: Vec<_> = field_samples::<M>().take(33).collect();
    points[..5].copy_from_slice(&[
        raw(&p),
        raw(&BigUint::from(1u8)),
        raw(&(&p * 2u8 - 1u8)),
        PastaField::ONE,
        PastaField::<M>::ONE.neg(),
    ]);
    let mut values: Vec<_> = field_samples::<M>().skip(70).take(33).collect();
    values[..5].copy_from_slice(&[
        raw(&p),
        raw(&(&p * 2u8 - 1u8)),
        PastaField::ONE,
        PastaField::ZERO,
        raw(&(&p + 1u8)),
    ]);
    let reduced_values: Vec<_> = values.iter().copied().map(PastaField::reduce).collect();
    let reduced_points: Vec<_> = points.iter().copied().map(PastaField::reduce).collect();
    for n in [0, 1, 2, 3, 4, 7, 8, 9, 16, 33] {
        check_case(&points[..n], &values[..n]);
        check_case(&reduced_points[..n], &reduced_values[..n]);
    }
    for values in [vec![raw(&p); 8], vec![raw(&(&p * 2u8 - 1u8)); 8]] {
        check_case(&points[..8], &values);
    }
    points.reverse();
    values.reverse();
    check_case(&points[..8], &values[..8]);

    // Known coefficients establish the interpolation degree and untrimmed extent.
    let coefficients: Vec<_> = field_samples::<M>().skip(90).take(9).collect();
    let integers: Vec<_> = coefficients.iter().map(canonical).collect();
    let values: Vec<_> = points[..16]
        .iter()
        .map(|x| {
            let value = evaluate_integer(&integers, &canonical(x), &p);
            raw::<M>(&((value << 256usize) % &p))
        })
        .collect();
    let mut weights = vec![PastaField::ZERO; 16];
    let mut scratch = vec![PastaField::ZERO; 16];
    let plan = InterpolationPlan::prepare(&points[..16], &mut weights, &mut scratch).unwrap();
    let mut output = vec![PastaField::ONE; 16];
    plan.interpolate(&values, &mut output, &mut scratch)
        .unwrap();
    for (actual, expected) in output.iter().zip(&integers) {
        assert_value(actual, expected);
    }
    assert!(output[9..].iter().all(PastaField::is_zero));
    // Binding accepts an alternate loose representation of every weight.
    for weight in &mut weights {
        *weight = raw(&(integer(&weight.reduce().montgomery_limbs()) + &p));
    }
    InterpolationPlan::bind(&points[..16], &weights).unwrap();
}

#[test]
fn interpolants_match_independent_integer_basis_construction() {
    arithmetic_field::<PallasBase>();
    arithmetic_field::<PallasScalar>();
}

fn errors_field<M: PrimeModulus>() {
    let p = modulus::<M>();
    let sentinel = raw::<M>(&(&p * 2u8 - 1u8));
    let points = [0, 2, 5, 8].map(PastaField::<M>::from_u64);
    for n in 1..=4 {
        for capacity in 0..n {
            let mut weights = [sentinel; 6];
            let mut scratch = [sentinel; 6];
            let expected = InterpolationError::WeightsTooShort {
                required: n,
                actual: capacity,
            };
            assert_eq!(
                InterpolationPlan::prepare(&points[..n], &mut weights[..capacity], &mut scratch)
                    .unwrap_err(),
                expected
            );
            assert_eq!(
                InterpolationPlan::prepare_denominators(&points[..n], &mut weights[..capacity])
                    .unwrap_err(),
                expected
            );
            assert_eq!(
                InterpolationPlan::bind(&points[..n], &weights[..capacity]).unwrap_err(),
                expected
            );
            assert_untouched(&weights, sentinel);
            assert_untouched(&scratch, sentinel);
            let descriptor =
                InterpolationPlan::prepare_denominators(&points[..n], &mut weights).unwrap();
            assert_eq!(
                descriptor.complete(&weights[..capacity]).unwrap_err(),
                expected
            );
        }
    }
    for (first, second) in [(0, 1), (0, 3), (1, 3), (2, 3)] {
        let mut duplicate = points;
        duplicate[second] = raw(&(integer(&points[first].reduce().montgomery_limbs()) + &p));
        let mut weights = [sentinel; 6];
        let mut scratch = [sentinel; 6];
        let expected = InterpolationError::DuplicatePoints { first, second };
        assert_eq!(
            InterpolationPlan::prepare(&duplicate, &mut weights, &mut scratch).unwrap_err(),
            expected
        );
        assert_eq!(
            InterpolationPlan::prepare_denominators(&duplicate, &mut weights).unwrap_err(),
            expected
        );
        assert_untouched(&weights, sentinel);
        assert_untouched(&scratch, sentinel);
    }
    let mut weights = [sentinel; 6];
    InterpolationPlan::prepare(&points, &mut weights, &mut []).unwrap();
    let plan = InterpolationPlan::bind(&points, &weights).unwrap();
    for count in [0, 1, 3, 5, 6] {
        let mut output = [sentinel; 6];
        let mut scratch = [sentinel; 6];
        let expected = InterpolationError::ValueCount {
            expected: 4,
            actual: count,
        };
        assert_eq!(
            plan.interpolate(&weights[..count], &mut output, &mut scratch),
            Err(expected)
        );
        assert_eq!(
            plan.evaluate(&weights[..count], &points[0], &mut scratch)
                .unwrap_err(),
            expected
        );
        assert_untouched(&output, sentinel);
        assert_untouched(&scratch, sentinel);
    }
    for capacity in 0..4 {
        let mut output = [sentinel; 6];
        let mut scratch = [sentinel; 6];
        assert_eq!(
            plan.interpolate(&points, &mut output[..capacity], &mut scratch),
            Err(InterpolationError::OutputTooShort {
                required: 4,
                actual: capacity
            })
        );
        let expected = InterpolationError::ScratchTooShort {
            required: 4,
            actual: capacity,
        };
        assert_eq!(
            plan.interpolate(&points, &mut output, &mut scratch[..capacity]),
            Err(expected)
        );
        for query in [points[0], sentinel] {
            assert_eq!(
                plan.evaluate(&points, &query, &mut scratch[..capacity])
                    .unwrap_err(),
                expected
            );
        }
        assert_untouched(&output, sentinel);
        assert_untouched(&scratch, sentinel);
    }
    // Values may borrow retained data; exact output buffers need no spare slot.
    let mut output = [sentinel; 4];
    let mut scratch = [sentinel; 4];
    assert_eq!(
        plan.interpolate(plan.points(), &mut output, &mut scratch),
        Ok(4)
    );
    assert!(output[0].is_zero() && output[1].is_one());
    assert!(output[2..].iter().all(PastaField::is_zero));
    check_case(plan.points(), plan.weights());
}

#[test]
fn invalid_interpolation_data_preserves_caller_storage() {
    errors_field::<PallasBase>();
    errors_field::<PallasScalar>();
}

fn grouped_field<M: PrimeModulus>() {
    let points: Vec<_> = field_samples::<M>().take(9).collect();
    let sentinel = raw::<M>(&(modulus::<M>() * 2u8 - 1u8));
    for scratch_len in [0, 1, 2, 3, 8, 13, 16] {
        let mut a = [sentinel; 11];
        let mut b = [sentinel; 3];
        let mut empty = [sentinel; 2];
        let mut extra = [PastaField::<M>::from_u64(13), PastaField::ZERO];
        let pa = InterpolationPlan::prepare_denominators(&points, &mut a).unwrap();
        let pb = InterpolationPlan::prepare_denominators(&points[..1], &mut b).unwrap();
        let pe = InterpolationPlan::prepare_denominators(&points[..0], &mut empty).unwrap();
        let mut scratch = vec![sentinel; scratch_len];
        let inversions = count_inversions(|| {
            batch_invert_groups(
                &mut [&mut a[..9], &mut empty[..0], &mut extra[..], &mut b[..1]],
                &mut scratch,
            );
        });
        if scratch_len >= 12 {
            assert_eq!(inversions, 1);
            assert_untouched(&scratch[12..], sentinel);
        }
        let a_plan = pa.complete(&a).unwrap();
        let b_plan = pb.complete(&b).unwrap();
        let empty_plan = pe.complete(&empty).unwrap();
        let mut work = [sentinel; 9];
        let query = PastaField::<M>::from_u64(17);
        assert_eq!(
            a_plan
                .evaluate(&points, &query, &mut work)
                .unwrap()
                .reduce(),
            query.reduce()
        );
        assert_eq!(
            b_plan
                .evaluate(&points[..1], &query, &mut work)
                .unwrap()
                .reduce(),
            points[0].reduce()
        );
        assert!(
            empty_plan
                .evaluate(&points[..0], &query, &mut [])
                .unwrap()
                .is_zero()
        );
        InterpolationPlan::bind(&points, a_plan.weights()).unwrap();
        assert_untouched(&a[9..], sentinel);
        assert_untouched(&b[1..], sentinel);
        assert_untouched(&empty, sentinel);
        assert_value(
            &extra[0],
            &BigUint::from(13u8).modinv(&modulus::<M>()).unwrap(),
        );
        assert!(extra[1].is_zero());
    }
}

#[test]
fn interpolation_weights_share_denominator_batches() {
    grouped_field::<PallasBase>();
    grouped_field::<PallasScalar>();
}
