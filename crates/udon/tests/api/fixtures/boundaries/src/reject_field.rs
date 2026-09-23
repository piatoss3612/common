use arithmetic::field::{PastaField, PrimeModulus};
use arithmetic::{
    fft::Domain,
    field::Reduced,
    polynomial::{
        EvaluationPlan, InterpolationError, InterpolationPlan, InterpolationPreparation,
        MonicDivisionError, VanishingError, divide_linear_in_place, divide_monic_in_place,
        evaluate, vanishing_polynomial,
    },
};

pub fn check<M: PrimeModulus>() {
    polynomial::<M>();
    interpolation::<M>();
    let _two = PastaField::<M>::from_u64(2);
    let _four = _two.square();
    let _reduced = _two.reduce();
    #[cfg(feature = "loose-order")]
    let _ = core::cmp::Ord::cmp(&_two, &_four);
    #[cfg(feature = "loose-sqrt")]
    let _ = _four.sqrt();
    #[cfg(feature = "loose-sqrt-alt")]
    let _ = _four.sqrt_alt();
    #[cfg(feature = "loose-sqrt-ratio")]
    let _ = _four.sqrt_ratio(&_reduced);
    #[cfg(feature = "loose-sqrt-denominator")]
    let _ = _reduced.sqrt_ratio(&_four);
    #[cfg(feature = "roots")]
    let _ = M::ROOTS;
    #[cfg(feature = "inverse-roots")]
    let _ = M::INVERSE_ROOTS;
    #[cfg(feature = "montgomery-inv")]
    let _ = M::MONTGOMERY_INV;
    #[cfg(feature = "r")]
    let _ = M::R;
    #[cfg(feature = "r2")]
    let _ = M::R2;
    #[cfg(feature = "r3")]
    let _ = M::R3;
    #[cfg(feature = "b448")]
    let _ = M::B448;
    #[cfg(feature = "sqrt-exponent")]
    let _ = M::SQRT_EXPONENT;
    #[cfg(feature = "two-inverse")]
    let _ = M::TWO_INVERSE;
    #[cfg(feature = "delta")]
    let _ = M::DELTA;
    #[cfg(feature = "zeta")]
    let _ = M::ZETA;
    #[cfg(feature = "zeta-inverse")]
    let _ = M::ZETA_INVERSE;
    #[cfg(feature = "modulus-signed62")]
    let _ = M::MODULUS_SIGNED62;
    #[cfg(feature = "safegcd-corrections")]
    let _ = M::SAFEGCD_CORRECTIONS;
    #[cfg(feature = "power-of-two-inverses")]
    let _ = M::POWER_OF_TWO_INVERSES;
    #[cfg(feature = "pow-sqrt-exponent")]
    let _ = M::pow_sqrt_exponent(&_four);
    #[cfg(feature = "sqrt-large")]
    let _ = M::sqrt_large(&_four, _four);
    #[cfg(feature = "sqrt-finish-large")]
    let _ = M::sqrt_finish_large(_four, _four);
    #[cfg(feature = "invalid-reduced-limbs")]
    let _ = const {
        arithmetic::field::Fp::<arithmetic::field::Reduced>::from_montgomery_limbs(
            arithmetic::field::PallasBase::MODULUS,
        )
    };
    #[cfg(feature = "invalid-loose-limbs")]
    let _ = const { <arithmetic::field::Fp>::from_montgomery_limbs([u64::MAX; 4]) };
}

#[cfg(any(feature = "foreign-modulus", feature = "foreign-reduction"))]
#[derive(Clone, Copy, Eq, PartialEq)]
enum Foreign {}

#[cfg(feature = "foreign-reduction")]
impl arithmetic::field::ReductionState for Foreign {}

#[cfg(feature = "foreign-modulus")]
impl PrimeModulus for Foreign {
    const MODULUS: [u64; 4] = [97, 0, 0, 0];
}

const fn constant_evaluation<M: PrimeModulus>() -> EvaluationPlan<'static, M> {
    EvaluationPlan::bind(&PastaField::<M, Reduced>::ONE, &[])
}

const EVALUATION_POWERS: usize = EvaluationPlan::<arithmetic::field::PallasBase>::power_count(4);

fn interpolation<M: PrimeModulus>() {
    let points = [0, 2, 5].map(|n| PastaField::<M>::from_u64(n).reduce());
    let values = [3, 11, 38].map(PastaField::<M>::from_u64);
    let mut weights = [PastaField::ZERO; 3];
    let preparation: InterpolationPreparation<'_, M, Reduced> =
        InterpolationPlan::prepare_denominators(&points, &mut weights).unwrap();
    let domain = Domain::<PastaField<M>>::new(2).unwrap().subgroup();
    let query = PastaField::<M>::from_u64(7);
    let mut basis = [PastaField::ZERO; 4];
    let basis_completion = domain.prepare_lagrange(&query, 0..4, &mut basis).unwrap();
    arithmetic::field::batch_invert_groups_scaled(
        &mut [&mut weights[..], &mut basis[..]],
        &PastaField::ONE,
        &mut [PastaField::ZERO; 7],
    );
    basis_completion.complete(&mut basis).unwrap();
    let plan = preparation.complete(&weights).unwrap();
    let mut scratch = [PastaField::ZERO; 3];
    let mut coefficients = [PastaField::ZERO; 3];
    assert_eq!(
        plan.interpolate(&values, &mut coefficients, &mut scratch),
        Ok(3)
    );
    assert_eq!(
        coefficients.map(PastaField::reduce),
        [3, 2, 1].map(|n| PastaField::<M>::from_u64(n).reduce())
    );
    assert_eq!(
        plan.evaluate(&values, &query, &mut scratch)
            .unwrap()
            .reduce(),
        evaluate(&coefficients, &query).reduce()
    );
    assert_eq!(
        plan.evaluate(&values, &points[1], &mut scratch)
            .unwrap()
            .reduce(),
        values[1].reduce()
    );
    InterpolationPlan::bind(plan.points(), plan.weights()).unwrap();
    assert_eq!(
        InterpolationPlan::prepare(&[points[0]; 2], &mut scratch, &mut []).unwrap_err(),
        InterpolationError::DuplicatePoints {
            first: 0,
            second: 1
        }
    );

    #[cfg(feature = "interpolation-points-mutation")]
    {
        let mut points = [<arithmetic::field::Fp>::ZERO, <arithmetic::field::Fp>::ONE];
        let mut weights = [<arithmetic::field::Fp>::ZERO; 2];
        let plan = InterpolationPlan::prepare(&points, &mut weights, &mut []).unwrap();
        points[0] = <arithmetic::field::Fp>::from_u64(3);
        let _ = plan.points();
    }
    #[cfg(feature = "interpolation-weights-mutation")]
    {
        let plan = InterpolationPlan::prepare(&points, &mut weights, &mut []).unwrap();
        weights[0] = PastaField::ZERO;
        let _ = plan.weights();
    }
    #[cfg(feature = "interpolation-field-mismatch")]
    {
        let points = [<arithmetic::field::Fp>::ZERO];
        let mut weights = [<arithmetic::field::Fp>::ZERO];
        let plan = InterpolationPlan::prepare(&points, &mut weights, &mut []).unwrap();
        let _ = plan.evaluate(
            &[<arithmetic::field::Fq>::ONE],
            &<arithmetic::field::Fp>::ONE,
            &mut [<arithmetic::field::Fp>::ZERO],
        );
    }
    #[cfg(feature = "interpolation-reduced-storage")]
    {
        let _ = plan.interpolate(&values, &mut [points[0]; 3], &mut scratch);
    }
    #[cfg(feature = "interpolation-completion-field-mismatch")]
    {
        let _ = InterpolationPlan::prepare_denominators(
            &[<arithmetic::field::Fp>::ZERO],
            &mut [<arithmetic::field::Fp>::ZERO],
        )
        .unwrap()
        .complete(&[<arithmetic::field::Fq>::ONE]);
    }
}

fn polynomial<M: PrimeModulus>() {
    let [half, delta, zeta, zeta_inverse] = super::pass::parameters::<M>();
    let two = PastaField::<M>::from_u64(2);
    let four = two.square();
    let reduced = two.reduce();
    let nonsquare = const { PastaField::<M, Reduced>::SQRT_NONSQUARE };
    assert_eq!(nonsquare, PastaField::root_of_unity(32).unwrap());
    let (is_square, alternate) = nonsquare.sqrt_alt();
    assert!(!is_square);
    assert_eq!(alternate.square().reduce(), nonsquare.square().reduce());
    let (is_square, root) = four.reduce().sqrt_ratio(&PastaField::ONE);
    assert!(is_square);
    assert_eq!(root.square().reduce(), four.reduce());

    let constant_plan = const { constant_evaluation::<M>() };
    assert_eq!(
        constant_plan.evaluate(&[half]).unwrap().reduce(),
        half.reduce()
    );
    let coefficients = [half, delta, zeta, zeta_inverse];
    let mut powers = [PastaField::<M>::ZERO; EVALUATION_POWERS];
    assert_eq!(
        powers.len(),
        EvaluationPlan::<M>::power_count(coefficients.len())
    );
    let plan = EvaluationPlan::prepare(&reduced, &mut powers);
    let plan = EvaluationPlan::bind(&two, plan.powers());
    let expected = evaluate(&coefficients, &two).reduce();
    assert_eq!(plan.point().reduce(), two.reduce());
    assert_eq!(plan.evaluate(&coefficients).unwrap().reduce(), expected);
    let mut output = [half; 4];
    plan.evaluate_many(&[&coefficients[..], &[], &coefficients[..1]], &mut output)
        .unwrap();
    assert_eq!(
        output.map(PastaField::reduce),
        [expected, PastaField::ZERO, half.reduce(), half.reduce()]
    );
    let mut divided = coefficients;
    let split = divide_linear_in_place(&mut divided, &reduced);
    let (remainder, quotient) = divided.split_at(split);
    assert_eq!(split, 1);
    assert_eq!(remainder[0].reduce(), expected);
    assert_eq!(
        evaluate(quotient, &four)
            .mul(&four.sub(&two))
            .add(&remainder[0])
            .reduce(),
        evaluate(&coefficients, &four).reduce()
    );
    #[cfg(feature = "division-field-mismatch")]
    {
        let _ = divide_linear_in_place(
            &mut [<arithmetic::field::Fp>::ONE],
            &<arithmetic::field::Fq>::ONE,
        );
    }
    #[cfg(feature = "division-reduced-storage")]
    {
        let _ = divide_linear_in_place(&mut [reduced], &two);
    }
    let mut divisor = [PastaField::<M>::ZERO; 3];
    assert_eq!(
        vanishing_polynomial(&[reduced, reduced], &mut divisor),
        Ok(3)
    );
    assert!(evaluate(&divisor, &two).is_zero());
    let mut divided = coefficients;
    let split = divide_monic_in_place(&mut divided, &divisor.map(PastaField::reduce)).unwrap();
    assert_eq!(split, 2);
    assert_eq!(
        evaluate(&divided[split..], &four)
            .mul(&evaluate(&divisor, &four))
            .add(&evaluate(&divided[..split], &four))
            .reduce(),
        evaluate(&coefficients, &four).reduce()
    );
    assert_eq!(
        divide_monic_in_place(&mut divided, &[two]),
        Err(MonicDivisionError::NonMonicDivisor)
    );
    assert_eq!(
        vanishing_polynomial(&[two], &mut divisor[..1]),
        Err(VanishingError::OutputTooShort {
            required: 2,
            actual: 1
        })
    );
    #[cfg(feature = "monic-field-mismatch")]
    {
        let _ = divide_monic_in_place(
            &mut [<arithmetic::field::Fp>::ONE],
            &[<arithmetic::field::Fq>::ONE],
        );
    }
    #[cfg(feature = "monic-reduced-storage")]
    {
        let _ = divide_monic_in_place(&mut [reduced], &[two]);
    }
    #[cfg(feature = "vanishing-field-mismatch")]
    {
        let _ = vanishing_polynomial(
            &[<arithmetic::field::Fq>::ONE],
            &mut [<arithmetic::field::Fp>::ONE; 2],
        );
    }
    #[cfg(feature = "vanishing-reduced-storage")]
    {
        let _ = vanishing_polynomial(&[two], &mut [reduced; 2]);
    }
    #[cfg(feature = "evaluation-powers-mutation")]
    {
        powers[0] = PastaField::ZERO;
        let _ = plan.evaluate(&coefficients);
    }
    #[cfg(feature = "evaluation-field-mismatch")]
    {
        let fp_plan = EvaluationPlan::bind(&<arithmetic::field::Fp>::ONE, &[]);
        let _ = fp_plan.evaluate(&[<arithmetic::field::Fq>::ONE]);
    }
}

#[cfg(feature = "foreign-reduction-flag")]
    let _ = S::REDUCED;
}
