use super::*;
use crate::field::{PallasBase, PallasScalar};
use crate::test_support::{field_samples, integer, modulus};
use num_bigint::BigUint;
use std::{vec, vec::Vec};

fn from_raw<M: PrimeModulus>(raw: &BigUint) -> PastaField<M> {
    let digits = raw.to_u64_digits();
    let mut limbs = [0; 4];
    limbs[..digits.len()].copy_from_slice(&digits);
    PastaField::from_montgomery_limbs(limbs)
}

fn oracle<M: PrimeModulus, S: ReductionState>(
    coefficients: &[PastaField<M, S>],
    point: &PastaField<M>,
) -> BigUint {
    let p = modulus::<M>();
    let inverse_r = (BigUint::from(1u8) << 256usize).modinv(&p).unwrap();
    let point = integer(&point.montgomery_limbs()) * &inverse_r % &p;
    coefficients.iter().rev().fold(BigUint::from(0u8), |y, a| {
        (y * &point + integer(&a.montgomery_limbs()) * &inverse_r) % &p
    })
}

fn assert_value<M: PrimeModulus>(actual: PastaField<M>, expected: &BigUint) {
    let p = modulus::<M>();
    let raw = integer(&actual.montgomery_limbs());
    assert!(raw < &p * 2u8);
    assert_eq!(raw % &p, (expected << 256usize) % &p);
}

fn check_queries<M: PrimeModulus, S: ReductionState>(
    plan: EvaluationPlan<'_, M>,
    inputs: &[&[PastaField<M, S>]],
) {
    let expected: Vec<_> = inputs.iter().map(|a| oracle(a, &plan.point())).collect();
    let sentinel = from_raw::<M>(&(&modulus::<M>() * 2u8 - 1u8));
    let mut output = vec![sentinel; inputs.len() + 3];
    for output_len in [output.len(), inputs.len()] {
        plan.evaluate_many(inputs, &mut output[..output_len])
            .unwrap();
        for ((input, actual), expected) in inputs.iter().zip(&output).zip(&expected) {
            assert_value(*actual, expected);
            assert_value(plan.evaluate(input).unwrap(), expected);
            assert_value(evaluate(input, &plan.point()), expected);
            assert_value(evaluate(input, &plan.point().reduce()), expected);
        }
        assert!(
            output[inputs.len()..]
                .iter()
                .all(|value| { value.montgomery_limbs() == sentinel.montgomery_limbs() })
        );
    }
}

fn check_field<M: PrimeModulus>() {
    let p = modulus::<M>();
    let mut values: Vec<_> = field_samples::<M>().take(1030).collect();
    let mut points = vec![
        PastaField::<M>::ZERO,
        PastaField::ONE,
        PastaField::<M>::ONE.neg(),
        PastaField::ZETA,
        values[0],
        values[1],
    ];
    for raw in [
        BigUint::from(0u8),
        BigUint::from(1u8),
        &p - 1u8,
        p.clone(),
        &p + 1u8,
        &p * 2u8 - 1u8,
        integer(&PastaField::<M>::ONE.montgomery_limbs()) + &p,
    ] {
        points.push(from_raw(&raw));
    }
    values[..points.len()].copy_from_slice(&points);
    let reduced: Vec<_> = values.iter().map(|a| a.reduce()).collect();
    let lengths = [0, 1, 2, 3, 4, 5, 7, 8, 9, 31, 32, 33, 63, 64, 65, 129, 1025];
    for point in points {
        let mut storage = vec![PastaField::ZERO; 1024];
        let plan = EvaluationPlan::prepare(&point, &mut storage);
        let inputs: Vec<_> = lengths.iter().rev().map(|&n| &values[..n]).collect();
        check_queries(plan, &inputs);
        let reduced_inputs: Vec<_> = lengths.iter().map(|&n| &reduced[..n]).collect();
        check_queries(plan, &reduced_inputs);
        // Inputs may share the retained table, including overlapping prefixes.
        check_queries(plan, &[plan.powers(), &plan.powers()[..3], plan.powers()]);

        // Independent integer powers also check the moved preparation helper.
        let point_integer = oracle(&[PastaField::<M>::ZERO, PastaField::ONE], &point);
        let mut expected = point_integer.clone();
        for value in plan.powers() {
            assert_value(*value, &expected);
            expected = expected * &point_integer % &p;
        }
        let bound = EvaluationPlan::bind(&point.reduce(), plan.powers());
        assert_eq!(bound.point().reduce(), point.reduce());
        assert!(core::ptr::eq(bound.powers(), plan.powers()));
        check_queries(bound, &inputs);

        for length in [0, 1, 2, 3, 7, 8, 9, 32, 33, 64, 65] {
            let required = EvaluationPlan::<M>::power_count(length);
            let sentinel = from_raw::<M>(&(&p * 2u8 - 1u8));
            let mut storage = vec![sentinel; required + 2];
            let plan = EvaluationPlan::prepare(&point.reduce(), &mut storage[..required]);
            check_queries(plan, &[&values[..length], &values[..0], &values[..1]]);
            assert!(
                storage[required..]
                    .iter()
                    .all(|value| { value.montgomery_limbs() == sentinel.montgomery_limbs() })
            );
        }
    }
    // Valid loose representatives of every stored power must also bind.
    let mut storage = vec![PastaField::ZERO; 129];
    EvaluationPlan::prepare(&values[15], &mut storage);
    for power in &mut storage {
        *power = from_raw(&(integer(&power.reduce().montgomery_limbs()) + &p));
    }
    let plan = EvaluationPlan::bind(&values[15], &storage);
    let maximal = vec![from_raw::<M>(&(&p * 2u8 - 1u8)); 130];
    check_queries(plan, &[&maximal, &maximal[..33], &[]]);
    // Cancellation and trailing zeros retain the declared coefficient extent.
    let cancellation = [values[15].neg(), PastaField::ONE, PastaField::ZERO];
    assert!(plan.evaluate(&cancellation).unwrap().is_zero());
}

#[test]
fn evaluations_match_integer_horner() {
    check_field::<PallasBase>();
    check_field::<PallasScalar>();
}

fn check_errors<M: PrimeModulus>() {
    assert_eq!(EvaluationPlan::<M>::power_count(0), 0);
    assert_eq!(EvaluationPlan::<M>::power_count(1), 0);
    assert_eq!(EvaluationPlan::<M>::power_count(usize::MAX), usize::MAX - 1);
    let point = PastaField::<M>::from_u64(7);
    let coefficients = [PastaField::<M>::ONE; 35];
    let original = [from_raw::<M>(&modulus::<M>()); 5];
    let mut output = original;
    let mut storage = [PastaField::ZERO; 33];
    let plan = EvaluationPlan::prepare(&point, &mut storage);
    assert_eq!(
        plan.evaluate(&coefficients).unwrap_err(),
        EvaluationError::PowersTooShort {
            required: 34,
            actual: 33
        }
    );
    let inputs = [&coefficients[..1], &coefficients[..0], &coefficients[..]];
    assert_eq!(
        plan.evaluate_many(&inputs, &mut output),
        Err(EvaluationError::PowersTooShort {
            required: 34,
            actual: 33
        })
    );
    assert_eq!(
        output.map(|v| v.montgomery_limbs()),
        original.map(|v| v.montgomery_limbs())
    );
    assert_eq!(
        plan.evaluate_many(&inputs[..2], &mut output[..1]),
        Err(EvaluationError::OutputTooShort {
            required: 2,
            actual: 1
        })
    );
    assert_eq!(
        output.map(|v| v.montgomery_limbs()),
        original.map(|v| v.montgomery_limbs())
    );
    plan.evaluate_many::<crate::field::Loose>(&[], &mut output)
        .unwrap();
    assert_eq!(
        output.map(|v| v.montgomery_limbs()),
        original.map(|v| v.montgomery_limbs())
    );
    let empty = EvaluationPlan::bind(&point, &[]);
    check_queries(empty, &[&coefficients[..0], &coefficients[..1]]);
    assert_eq!(
        empty.evaluate(&coefficients[..2]).unwrap_err(),
        EvaluationError::PowersTooShort {
            required: 1,
            actual: 0
        }
    );
}

#[test]
fn invalid_shapes_preserve_buffers() {
    check_errors::<PallasBase>();
    check_errors::<PallasScalar>();
}
