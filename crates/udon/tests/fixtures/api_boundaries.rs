#![forbid(unsafe_code)]
#![deny(warnings)]

mod facade {
    pub use arithmetic::{
        curve::{AffinePoint, Pallas, PastaCurve, Vesta, glv_decompose},
        fft::{
            ConstantPrefixExpansion, Domain, LagrangeCompletion, LagrangeError, VanishingDivision,
            VanishingFactors,
        },
        field::{
            ConstantPrefix, ConstantPrefixError, Fp, Fq, PallasBase, PallasScalar, PastaField,
            PrimeModulus, Reduced,
        },
        polynomial::{
            EvaluationPlan, InterpolationError, InterpolationPlan, InterpolationPreparation,
            MonicDivisionError, VanishingError, divide_linear_in_place, divide_monic_in_place,
            evaluate, vanishing_polynomial,
        },
    };
}

use facade::*;

// Generic constant initializers must work through the consumer's facade,
// without access to the sealed implementation parameters.
const fn constant_evaluation<M: PrimeModulus>() -> EvaluationPlan<'static, M> {
    EvaluationPlan::bind(&PastaField::<M, Reduced>::ONE, &[])
}

const fn parameters<M: PrimeModulus>() -> [PastaField<M>; 4] {
    [
        PastaField::<M>::TWO_INVERSE,
        PastaField::<M>::DELTA,
        PastaField::<M>::ZETA,
        PastaField::<M>::ZETA_INVERSE,
    ]
}

const FP_PARAMETERS: [Fp; 4] = parameters::<PallasBase>();
const FQ_PARAMETERS: [Fq; 4] = parameters::<PallasScalar>();
const EVALUATION_POWERS: usize = EvaluationPlan::<PallasBase>::power_count(4);

fn interpolation<M: PrimeModulus>() {
    let points = [0, 2, 5].map(|n| PastaField::<M>::from_u64(n).reduce());
    let values = [3, 11, 38].map(PastaField::<M>::from_u64);
    let mut weights = [PastaField::ZERO; 3];
    let preparation: InterpolationPreparation<'_, M, Reduced> =
        InterpolationPlan::prepare_denominators(&points, &mut weights).unwrap();
    let domain = Domain::<M>::new(2).unwrap().subgroup();
    let query = PastaField::<M>::from_u64(7);
    let mut basis = [PastaField::ZERO; 4];
    let basis_completion = domain.prepare_lagrange(&query, 0..4, &mut basis).unwrap();
    arithmetic::field::batch_invert_groups(
        &mut [&mut weights[..], &mut basis[..]],
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
        let mut points = [<Fp>::ZERO, <Fp>::ONE];
        let mut weights = [<Fp>::ZERO; 2];
        let plan = InterpolationPlan::prepare(&points, &mut weights, &mut []).unwrap();
        points[0] = <Fp>::from_u64(3);
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
        let points = [<Fp>::ZERO];
        let mut weights = [<Fp>::ZERO];
        let plan = InterpolationPlan::prepare(&points, &mut weights, &mut []).unwrap();
        let _ = plan.evaluate(&[<Fq>::ONE], &<Fp>::ONE, &mut [<Fp>::ZERO]);
    }
    #[cfg(feature = "interpolation-reduced-storage")]
    {
        let _ = plan.interpolate(&values, &mut [points[0]; 3], &mut scratch);
    }
    #[cfg(feature = "interpolation-completion-field-mismatch")]
    {
        let _ = InterpolationPlan::prepare_denominators(&[<Fp>::ZERO], &mut [<Fp>::ZERO])
            .unwrap()
            .complete(&[<Fq>::ONE]);
    }
}

fn field<M: PrimeModulus>([half, delta, zeta, zeta_inverse]: [PastaField<M>; 4]) {
    constant_prefix::<M>();
    interpolation::<M>();
    vanishing_division::<M>();
    assert_eq!(half.double().reduce(), PastaField::<M, Reduced>::ONE);
    assert_eq!(
        delta.reduce(),
        PastaField::<M>::from_u64(5).pow_u64(1 << 32).reduce()
    );
    let two = PastaField::<M>::from_u64(2);
    let domain = Domain::<M>::new(2).unwrap().coset();
    let node = domain.shift().mul(&domain.domain().root());
    let mut basis = [PastaField::ZERO; 4];
    domain
        .evaluate_lagrange(&node.reduce(), 0..4, &mut basis, &mut [])
        .unwrap();
    assert_eq!(
        basis.map(PastaField::reduce),
        [
            PastaField::ZERO,
            PastaField::ONE,
            PastaField::ZERO,
            PastaField::ZERO
        ]
    );
    let completion: LagrangeCompletion<M> =
        domain.prepare_lagrange(&two, 0..4, &mut basis).unwrap();
    assert_eq!(completion.value_count(), 4);
    arithmetic::field::batch_invert_groups(&mut [&mut basis[..]], &mut [PastaField::ZERO; 4]);
    completion.complete(&mut basis).unwrap();
    let mut reconstructed = PastaField::ZERO;
    let mut node = domain.shift();
    for value in basis {
        reconstructed = reconstructed.add(&value.mul(&node.square()));
        node = node.mul(&domain.domain().root());
    }
    assert_eq!(reconstructed.reduce(), two.square().reduce());
    assert_eq!(
        domain.evaluate_lagrange(&two, 4..5, &mut basis, &mut []),
        Err(LagrangeError::InvalidRange {
            start: 4,
            end: 5,
            size: 4
        })
    );
    #[cfg(feature = "lagrange-field-mismatch")]
    {
        let _ = Domain::<PallasBase>::new(1)
            .unwrap()
            .subgroup()
            .evaluate_lagrange(&<Fq>::ONE, 0..1, &mut [<Fp>::ZERO], &mut []);
    }
    #[cfg(feature = "lagrange-reduced-storage")]
    {
        let _ = domain.prepare_lagrange(&two, 0..1, &mut [two.reduce()]);
    }
    #[cfg(feature = "lagrange-completion-field-mismatch")]
    {
        let mut values = [<Fp>::ZERO];
        let completion = Domain::<PallasBase>::new(1)
            .unwrap()
            .subgroup()
            .prepare_lagrange(&<Fp>::ZERO, 0..1, &mut values)
            .unwrap();
        let _ = completion.complete(&mut [<Fq>::ZERO]);
    }
    let four = two.square();
    assert_eq!(
        four.reduce().sqrt().unwrap().square().reduce(),
        four.reduce()
    );
    assert_eq!(
        two.mul(&two.invert().unwrap()).reduce(),
        PastaField::<M, Reduced>::ONE
    );
    assert_eq!(M::MODULUS[3], 1 << 62);
    assert_eq!(
        PastaField::<M>::root_of_unity(4)
            .unwrap()
            .mul(&PastaField::<M>::root_of_unity_inverse(4).unwrap())
            .reduce(),
        PastaField::<M, Reduced>::ONE,
    );
    assert_eq!(
        zeta.mul(&zeta_inverse).reduce(),
        PastaField::<M, Reduced>::ONE
    );
    let reduced: PastaField<M, Reduced> = two.reduce();
    assert_eq!(
        reduced.into_loose().montgomery_limbs(),
        reduced.montgomery_limbs()
    );
    assert_eq!(reduced.mul(&two).reduce(), four.reduce());
    assert!(reduced < four.reduce());
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
        let _ = divide_linear_in_place(&mut [<Fp>::ONE], &<Fq>::ONE);
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
        let _ = divide_monic_in_place(&mut [<Fp>::ONE], &[<Fq>::ONE]);
    }
    #[cfg(feature = "monic-reduced-storage")]
    {
        let _ = divide_monic_in_place(&mut [reduced], &[two]);
    }
    #[cfg(feature = "vanishing-field-mismatch")]
    {
        let _ = vanishing_polynomial(&[<Fq>::ONE], &mut [<Fp>::ONE; 2]);
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
        let fp_plan = EvaluationPlan::bind(&<Fp>::ONE, &[]);
        let _ = fp_plan.evaluate(&[<Fq>::ONE]);
    }

    #[cfg(feature = "loose-equality")]
    let _ = two == four;
    #[cfg(feature = "loose-order")]
    let _ = core::cmp::Ord::cmp(&two, &four);
    #[cfg(feature = "loose-sqrt")]
    let _ = four.sqrt();
    #[cfg(feature = "loose-sqrt-alt")]
    let _ = four.sqrt_alt();
    #[cfg(feature = "loose-sqrt-ratio")]
    let _ = four.sqrt_ratio(&reduced);
    #[cfg(feature = "loose-sqrt-denominator")]
    let _ = reduced.sqrt_ratio(&four);

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
    let _ = M::pow_sqrt_exponent(&four);
    #[cfg(feature = "sqrt-large")]
    let _ = M::sqrt_large(&four, four);
    #[cfg(feature = "sqrt-finish-large")]
    let _ = M::sqrt_finish_large(four, four);
}

fn constant_prefix<M: PrimeModulus>() {
    let base = Domain::<M>::new(2).unwrap().subgroup();
    let extended = Domain::<M>::new(3).unwrap().coset();
    let tail = [PastaField::<M>::from_u64(9).reduce()];
    let constant = PastaField::<M>::from_u64(3).reduce();
    let input: ConstantPrefix<'_, M, Reduced> = ConstantPrefix::new(4, &constant, &tail).unwrap();
    let mut dense = [PastaField::ZERO; 4];
    input.write_values(&mut dense).unwrap();
    assert_eq!(
        dense.map(PastaField::reduce),
        [constant, constant, constant, tail[0]]
    );
    assert_eq!(
        ConstantPrefix::new(0, &constant, &tail).unwrap_err(),
        ConstantPrefixError::TailTooLong { length: 0, tail: 1 }
    );
    let mut coefficients = [PastaField::ZERO; 4];
    base.interpolate_constant_prefix(input, &mut coefficients, &mut [PastaField::ZERO])
        .unwrap();
    let mut samples = [PastaField::ZERO; 8];
    let plan =
        ConstantPrefixExpansion::prepare(base, extended, &mut samples, &mut [PastaField::ZERO; 8])
            .unwrap();
    let rebound = ConstantPrefixExpansion::bind(base, extended, plan.samples()).unwrap();
    let mut output = [PastaField::ZERO; 8];
    rebound.evaluate(input, &mut output).unwrap();
    for (i, value) in output.iter().enumerate() {
        let point = extended
            .shift()
            .mul(&extended.domain().root().pow_u64(i as u64));
        assert_eq!(value.reduce(), evaluate(&coefficients, &point).reduce());
    }

    #[cfg(feature = "tail-samples-mutation")]
    {
        samples[0] = PastaField::ZERO;
        let _ = plan.samples();
    }
    #[cfg(feature = "tail-input-mutation")]
    {
        let mut tail = tail;
        let input = ConstantPrefix::new(4, &constant, &tail).unwrap();
        tail[0] = constant;
        let _ = input.tail();
    }
    #[cfg(feature = "tail-field-mismatch")]
    {
        let input = ConstantPrefix::new(4, &<Fq>::ONE, &[<Fq>::ONE]).unwrap();
        let _ = Domain::<PallasBase>::new(2)
            .unwrap()
            .subgroup()
            .interpolate_constant_prefix(input, &mut [<Fp>::ZERO; 4], &mut [<Fp>::ZERO]);
    }
    #[cfg(feature = "tail-reduced-storage")]
    {
        let _ = plan.evaluate(input, &mut [constant; 8]);
    }
}

fn vanishing_division<M: PrimeModulus>() {
    use arithmetic::{
        exec::{ExecutionOptions, SerialExecutor},
        fft::{ElementOrder, Transform},
    };
    let domain = Domain::<M>::new(2).unwrap();
    let shift = PastaField::<M>::from_u64(7);
    let division = VanishingDivision::new(domain, &shift.reduce(), 2).unwrap();
    let mut values = core::array::from_fn::<_, 4, _>(|j| {
        let x = shift.mul(&domain.root().pow_u64(j as u64));
        x.square()
            .sub(&PastaField::<M>::ONE)
            .mul(&PastaField::<M>::from_u64(3).add(&x))
    });
    let mut storage = [PastaField::ZERO; 2];
    let factors: VanishingFactors<'_, M> = division.prepare_factors(&mut storage, &mut []);
    let mut divided = values;
    factors.divide_in_place(&mut divided, ElementOrder::Natural);
    for (j, value) in divided.iter().enumerate() {
        let x = shift.mul(&domain.root().pow_u64(j as u64));
        assert_eq!(
            value.reduce(),
            PastaField::<M>::from_u64(3).add(&x).reduce()
        );
    }
    Transform::new(domain.subgroup())
        .forward(
            &mut values,
            ExecutionOptions::default(),
            &SerialExecutor,
            &mut [],
        )
        .unwrap();
    let mut low = [PastaField::ZERO; 2];
    division.write_pieces(
        &values,
        ElementOrder::Natural,
        &mut [&mut low],
        &mut [PastaField::ZERO; 2],
    );
    assert_eq!(
        low.map(PastaField::reduce),
        [3, 1].map(|x| PastaField::<M>::from_u64(x).reduce())
    );

    #[cfg(feature = "vanishing-factors-mutation")]
    {
        storage[0] = PastaField::ONE;
        let _ = factors.as_slice();
    }
    #[cfg(feature = "vanishing-finish-field-mismatch")]
    {
        let division =
            VanishingDivision::new(Domain::<PallasBase>::new(0).unwrap(), &<Fp>::ZETA, 1).unwrap();
        division.write_pieces(&[<Fq>::ZERO], ElementOrder::Natural, &mut [], &mut []);
    }
    #[cfg(feature = "vanishing-finish-reduced-storage")]
    {
        division.write_pieces(
            &values,
            ElementOrder::Natural,
            &mut [&mut [PastaField::<M, Reduced>::ZERO; 2]],
            &mut [PastaField::ZERO; 2],
        );
    }
}

fn suffix<C: PastaCurve>() {
    use arithmetic::{
        curve::msm::{ScalarStorage, Scratch},
        curve::{Point, ProjectivePoint, msm::SuffixBasis},
        exec::{ExecutionOptions, SerialExecutor},
    };
    let mut sums = [Point::IDENTITY; 3];
    // Original storage can be reused and dropped before using the prepared sums.
    let basis = {
        let mut original = [Point::<C>::GENERATOR; 3];
        let basis = SuffixBasis::prepare(
            &original,
            &mut sums,
            &mut [ProjectivePoint::IDENTITY; 2],
            &mut [PastaField::ZERO; 1],
        );
        original.fill(Point::IDENTITY);
        assert!(original.iter().all(Point::is_identity));
        basis
    };
    #[cfg(feature = "suffix-output-mutation")]
    {
        sums[0] = Point::IDENTITY;
    }
    #[cfg(feature = "suffix-curve-mismatch")]
    let _: SuffixBasis<'_, Vesta> = basis;
    #[cfg(feature = "suffix-field-mismatch")]
    let _ = basis.with_scalars(&[Fp::ONE; 3], &mut [Fp::ZERO; 3]);
    assert_eq!(basis.len(), 3);
    let mut differences = [0; 3];
    // The source coefficients need not outlive the returned input.
    let input = {
        let row = [2, 2, 5];
        basis
            .with_monotone_unsigned(&row, &mut differences)
            .unwrap()
    };
    #[cfg(feature = "suffix-differences-mutation")]
    {
        differences[0] = 17;
    }
    let options = ExecutionOptions::default();
    let r = input.requirements(options).unwrap();
    let actual = input
        .execute(
            options,
            &SerialExecutor,
            Scratch::new(
                &mut vec![ScalarStorage::ZERO; r.scalars()],
                &mut vec![0; r.digits()],
                &mut vec![AffinePoint::GENERATOR; r.affine()],
                &mut vec![ProjectivePoint::IDENTITY; r.projective()],
                &mut vec![PastaField::ZERO; r.field()],
                &mut vec![0; r.indices()],
            ),
        )
        .unwrap();
    assert_eq!(
        actual,
        Point::<C>::GENERATOR.mul_projective(&PastaField::from_u64(9))
    );
    assert_eq!(differences, [2, 0, 3]);
}

fn basis_sum<C: PastaCurve>() {
    use arithmetic::curve::{Point, msm::BasisSum};
    #[allow(unused_mut)]
    let mut original = [Point::<C>::GENERATOR; 3];
    let basis = BasisSum::prepare(&original);
    #[cfg(feature = "sum-original-mutation")]
    {
        original[0] = Point::IDENTITY;
    }
    #[cfg(feature = "sum-curve-mismatch")]
    let _: BasisSum<'_, Vesta> = basis;
    #[cfg(feature = "sum-field-mismatch")]
    let _ = basis.corrections(&[0], &[Fp::ONE]);
    assert_eq!(basis.original().len(), 3);
    #[allow(unused_mut)]
    let mut indices = [2, 0, 2];
    let mut differences = [PastaField::<C::Scalar>::ONE; 3];
    let input = basis.corrections(&indices, &differences).unwrap();
    #[cfg(feature = "sum-indices-mutation")]
    {
        indices[0] = 1;
    }
    assert_eq!(input.len(), 3);
    // Source tail values need not outlive the converted input.
    let input = {
        let tail = [PastaField::<C::Scalar>::from_u64(7)];
        basis.tail_corrections(
            ConstantPrefix::new(3, &PastaField::<C::Scalar>::ONE, &tail).unwrap(),
            &mut differences,
        )
    };
    #[cfg(feature = "sum-differences-mutation")]
    {
        differences[0] = PastaField::ZERO;
    }
    assert_eq!(input.len(), 1);
    assert_eq!(
        differences[0].reduce(),
        PastaField::<C::Scalar>::from_u64(6).reduce()
    );
}

fn coalescing<C: PastaCurve>() {
    use arithmetic::curve::{
        Point,
        msm::{Bases, CoalescingKey, CoalescingPlan, IndexedCoalescingPlan},
    };
    let mut points = [Point::IDENTITY; 1];
    let mut sums = [PastaField::<C::Scalar>::ZERO; 1];
    let input = {
        #[allow(unused_mut)]
        let mut original = [Point::<C>::GENERATOR; 2];
        let mut keys = [CoalescingKey::EMPTY; 2];
        let plan = CoalescingPlan::prepare(&original, &mut keys);
        #[cfg(feature = "coalesce-original-mutation")]
        {
            original[0] = Point::IDENTITY;
        }
        #[cfg(feature = "coalesce-keys-mutation")]
        {
            keys[0] = CoalescingKey::EMPTY;
        }
        #[cfg(feature = "coalesce-curve-mismatch")]
        let _: CoalescingPlan<'_, Vesta> = plan;
        #[cfg(feature = "coalesce-field-mismatch")]
        let _ = plan.with_scalars(&[Fp::ONE; 2], &mut points, &mut sums);
        // The returned points own their representatives and release the plan,
        // original bases, keys and coefficients after this conversion.
        plan.with_scalars(&[PastaField::<C::Scalar>::ONE; 2], &mut points, &mut sums)
    };
    #[cfg(feature = "coalesce-points-mutation")]
    {
        points[0] = Point::IDENTITY;
    }
    assert_eq!(input.len(), 1);
    assert_eq!(
        points[0].mul_projective(&sums[0]),
        Point::<C>::GENERATOR.double()
    );
    let original = [Point::<C>::GENERATOR; 1];
    #[allow(unused_mut)]
    let mut indices = [0, 0];
    let mut order = [0; 2];
    let plan =
        IndexedCoalescingPlan::prepare(Bases::Points(&original), &indices, &mut order).unwrap();
    #[cfg(feature = "coalesce-indices-mutation")]
    {
        indices[0] = 1;
    }
    #[cfg(feature = "coalesce-order-mutation")]
    {
        order[0] = 1;
    }
    let mut output = [0];
    let input = {
        let row = [PastaField::<C::Scalar>::ONE; 2];
        plan.with_scalars(&row, &mut output, &mut sums)
    };
    #[cfg(feature = "coalesce-output-mutation")]
    {
        output[0] = 1;
    }
    assert_eq!(input.len(), 1);
    assert_eq!(output, [0]);
    assert_eq!(
        sums[0].reduce(),
        PastaField::<C::Scalar>::from_u64(2).reduce()
    );
}

fn nonzero_support<C: PastaCurve>() {
    use arithmetic::curve::{
        Point,
        msm::{Bases, Selection},
    };
    #[allow(unused_mut)]
    let mut original = [Point::<C>::GENERATOR; 2];
    #[allow(unused_mut)]
    let mut mapping = [1, 0, 1];
    let selection = Selection::indexed(Bases::Points(&original), &mapping).unwrap();
    let mut indices = [0; 2];
    let mut scalars = [PastaField::<C::Scalar>::ZERO; 2];
    let input = {
        let row = [
            PastaField::<C::Scalar>::ONE,
            PastaField::ZERO,
            PastaField::ONE,
        ];
        #[cfg(feature = "support-field-mismatch")]
        let _ = selection.with_nonzero_scalars(&[Fp::ONE; 3], &mut indices, &mut scalars);
        selection
            .with_nonzero_scalars(&row, &mut indices, &mut scalars)
            .unwrap()
    };
    #[cfg(feature = "support-bases-mutation")]
    {
        original[0] = Point::IDENTITY;
    }
    #[cfg(feature = "support-mapping-mutation")]
    {
        mapping[0] = 0;
    }
    #[cfg(feature = "support-indices-mutation")]
    {
        indices[0] = 0;
    }
    #[cfg(feature = "support-scalars-mutation")]
    {
        scalars[0] = PastaField::ZERO;
    }
    let retained = input.selection();
    let next = [PastaField::<C::Scalar>::from_u64(3); 2];
    assert_eq!(retained.with_scalars(&next).len(), 2);
    assert_eq!(indices, [1, 1]);
}

fn curve<C: PastaCurve>() {
    nonzero_support::<C>();
    coalescing::<C>();
    basis_sum::<C>();
    suffix::<C>();
    use arithmetic::{
        curve::msm::{
            Bases, PreparedScalars, ScalarStorage, Scratch, SharedScalarInput,
            run::{BatchPlan, MsmPlan},
        },
        exec::{ExecutionOptions, SerialExecutor, TaskBudget},
    };
    let generator = AffinePoint::<C>::GENERATOR;
    #[cfg(feature = "loose-coordinates")]
    let _ = AffinePoint::<C>::from_xy(PastaField::<C::Base>::ONE, PastaField::<C::Base>::ONE);
    assert_eq!(
        generator.mul_projective(&PastaField::<C::Scalar>::from_u64(2)),
        generator.to_projective().double(),
    );
    assert_eq!(
        glv_decompose::<C>(&PastaField::<C::Scalar>::ONE.neg()),
        (-1, 0),
    );

    #[cfg(feature = "glv-a")]
    let _ = C::GLV_A;
    #[cfg(feature = "glv-b")]
    let _ = C::GLV_B;

    let options = ExecutionOptions::default()
        .with_task_budget(TaskBudget::new(3).unwrap())
        .with_memory_limit(8192);
    let plan = MsmPlan::<C>::new(1, options).unwrap();
    assert_eq!(BatchPlan::<C>::storage_len(1, options).unwrap(), (1, 1));
    let mut records = [ScalarStorage::<C>::ZERO];
    let prepared = PreparedScalars::prepare(
        &[PastaField::ONE],
        &mut records,
        TaskBudget::SERIAL,
        &SerialExecutor,
    );
    let mut digits = vec![0; prepared.cache_len(&plan)];
    assert_eq!(prepared.cache(&plan, &mut digits).len(), 1);

    #[allow(unused_mut)]
    let mut matrix_bases = [generator, generator.neg()];
    let matrix = SharedScalarInput::new(Bases::Affine(&matrix_bases), prepared, 2, 1, 1).unwrap();
    assert_eq!((matrix.outputs(), matrix.terms()), (2, 1));
    #[cfg(feature = "matrix-bases-mutation")]
    {
        matrix_bases[0] = generator.neg();
    }
    #[cfg(feature = "matrix-scalars-mutation")]
    {
        records[0] = ScalarStorage::ZERO;
    }
    #[cfg(feature = "matrix-curve-mismatch")]
    let _: SharedScalarInput<'_, Vesta> = matrix;
    let r = matrix.requirements(options).unwrap();
    let mut digits = vec![0; r.digits()];
    let mut affine = vec![generator; r.affine()];
    let mut projective = vec![arithmetic::curve::ProjectivePoint::IDENTITY; r.projective()];
    let mut field = vec![PastaField::ZERO; r.field()];
    let mut indices = vec![0; r.indices()];
    let mut output = [arithmetic::curve::ProjectivePoint::IDENTITY; 2];
    matrix
        .execute(
            &mut output,
            options,
            &SerialExecutor,
            Scratch::new(
                &mut [],
                &mut digits,
                &mut affine,
                &mut projective,
                &mut field,
                &mut indices,
            ),
        )
        .unwrap();
    assert_eq!(output, matrix_bases.map(|base| base.to_projective()));

    #[cfg(feature = "eisenstein-length")]
    let _ = arithmetic::curve::EisensteinTable::<C>::bind(&generator, &[generator; 7]);
    #[cfg(feature = "empty-msm-slots")]
    let _ = arithmetic::curve::msm::run::ParallelMsmRun::new(
        plan,
        arithmetic::curve::msm::Input::new(arithmetic::curve::msm::Bases::Affine(&[]), &[]),
        &mut [],
        &mut [[const { arithmetic::exec::run::TaskStorage::EMPTY }; 1]; 0],
    );

    #[cfg(feature = "msm-arithmetic")]
    let _ = arithmetic::curve::msm::ArithmeticOptions::default();
    #[cfg(feature = "msm-kernel")]
    let _ = arithmetic::curve::msm::Algorithm::Auto;
    #[cfg(feature = "msm-accumulation")]
    let _ = arithmetic::curve::msm::Accumulation::Auto;
    #[cfg(feature = "fft-codelet")]
    let _ = arithmetic::fft::Codelet::Radix2;
    #[cfg(feature = "fft-strategy")]
    let _ = arithmetic::fft::Strategy::SERIAL;
    #[cfg(feature = "cache-options")]
    let _ = prepared.cache_len(options);
}

#[cfg(any(
    feature = "foreign-modulus",
    feature = "foreign-curve",
    feature = "foreign-reduction"
))]
#[derive(Clone, Copy, Eq, PartialEq)]
enum Foreign {}

#[cfg(feature = "foreign-reduction")]
impl arithmetic::field::ReductionState for Foreign {}

#[cfg(feature = "foreign-modulus")]
impl PrimeModulus for Foreign {
    const MODULUS: [u64; 4] = [97, 0, 0, 0];
}

#[cfg(feature = "foreign-curve")]
impl PastaCurve for Foreign {
    type Base = PallasBase;
    type Scalar = PallasScalar;
}

fn main() {
    #[cfg(feature = "invalid-reduced-limbs")]
    let _ = const { Fp::<Reduced>::from_montgomery_limbs(PallasBase::MODULUS) };
    #[cfg(feature = "invalid-loose-limbs")]
    let _ = const { <Fp>::from_montgomery_limbs([u64::MAX; 4]) };
    #[cfg(feature = "empty-interpolation")]
    let _ = arithmetic::fft::run::InterpolationPlan::<PallasBase, 0>::new(
        [],
        false,
        arithmetic::fft::StorageLayout::Contiguous,
        arithmetic::exec::ExecutionOptions::default(),
    );
    field::<PallasBase>(FP_PARAMETERS);
    field::<PallasScalar>(FQ_PARAMETERS);
    curve::<Pallas>();
    curve::<Vesta>();
}
