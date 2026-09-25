use arithmetic::{
    curve::{AffinePoint, PastaCurve},
    exec::{ExecutionOptions, SerialExecutor, TaskBudget},
    field::{ConstantPrefix, PastaField},
    msm::{Bases, PreparedScalars, ScalarStorage, Scratch, SharedScalarInput, execution::MsmPlan},
};

pub fn check<C: PastaCurve>() {
    nonzero_support::<C>();
    coalescing::<C>();
    basis_sum::<C>();
    suffix::<C>();
    let generator = AffinePoint::<C>::GENERATOR;
    let options = ExecutionOptions::default()
        .with_task_budget(TaskBudget::new(3).unwrap())
        .with_memory_limit(8192);
    let _plan = MsmPlan::<C>::new(1, options).unwrap();
    let mut records = [ScalarStorage::<C>::ZERO];
    let prepared = PreparedScalars::prepare(
        &[PastaField::ONE],
        &mut records,
        TaskBudget::SERIAL,
        &SerialExecutor,
    );
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
    let _: SharedScalarInput<'_, arithmetic::curve::Vesta> = matrix;
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

    #[cfg(feature = "loose-coordinates")]
    let _ = AffinePoint::<C>::from_xy(PastaField::<C::Base>::ONE, PastaField::<C::Base>::ONE);
    #[cfg(feature = "glv-a")]
    let _ = C::GLV_A;
    #[cfg(feature = "glv-b")]
    let _ = C::GLV_B;
    #[cfg(feature = "eisenstein-length")]
    let _ = arithmetic::curve::EisensteinTable::<C>::bind(&generator, &[generator; 7]);
    #[cfg(feature = "empty-msm-slots")]
    let _ = arithmetic::msm::execution::ParallelMsmRun::new(
        _plan,
        arithmetic::msm::Input::new(arithmetic::msm::Bases::Affine(&[]), &[]),
        &mut [],
        &mut [[const { arithmetic::exec::execution::TaskStorage::EMPTY }; 1]; 0],
    );
    #[cfg(feature = "msm-arithmetic")]
    let _ = arithmetic::msm::ArithmeticOptions::default();
    #[cfg(feature = "msm-kernel")]
    let _ = arithmetic::msm::Kernel::Auto;
    #[cfg(feature = "msm-accumulation")]
    let _ = arithmetic::msm::Accumulation::Auto;
    #[cfg(feature = "cache-options")]
    let _ = prepared.cache_len(options);
}

#[cfg(feature = "foreign-curve")]
#[derive(Clone, Copy, Eq, PartialEq)]
enum Foreign {}

#[cfg(feature = "foreign-curve")]
impl PastaCurve for Foreign {
    type Base = arithmetic::field::PallasBase;
    type Scalar = arithmetic::field::PallasScalar;
}

fn suffix<C: PastaCurve>() {
    use arithmetic::{
        msm::{ScalarStorage, Scratch},
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
    let _: SuffixBasis<'_, arithmetic::curve::Vesta> = basis;
    #[cfg(feature = "suffix-field-mismatch")]
    let _ = basis.with_scalars(
        &[arithmetic::field::Fp::ONE; 3],
        &mut [arithmetic::field::Fp::ZERO; 3],
    );
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
    let _: BasisSum<'_, arithmetic::curve::Vesta> = basis;
    #[cfg(feature = "sum-field-mismatch")]
    let _ = basis.corrections(&[0], &[arithmetic::field::Fp::ONE]);
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
        let _: CoalescingPlan<'_, arithmetic::curve::Vesta> = plan;
        #[cfg(feature = "coalesce-field-mismatch")]
        let _ = plan.with_scalars(&[arithmetic::field::Fp::ONE; 2], &mut points, &mut sums);
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
        let _ = selection.with_nonzero_scalars(
            &[arithmetic::field::Fp::ONE; 3],
            &mut indices,
            &mut scalars,
        );
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
