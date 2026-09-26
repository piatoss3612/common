//! Successful downstream calls through a renamed dependency and facade.

mod facade {
    #[cfg(feature = "traits")]
    pub use arithmetic::{
        curve::{Affine, AffineAdapter, ProjectiveAdapter},
        field::{Field, FieldAdapter},
    };
    pub use arithmetic::{
        curve::{AffinePoint, Pallas, PastaCurve, Vesta, glv_decompose},
        field::{Fp, Fq, PallasBase, PallasScalar, PastaField, PrimeModulus, Reduced},
    };
}

use facade::*;

// Generic constant initializers must work through the consumer's facade,
// without access to the sealed implementation parameters.
pub(super) const fn parameters<M: PrimeModulus>() -> [PastaField<M>; 4] {
    [
        PastaField::<M>::TWO_INVERSE,
        PastaField::<M>::DELTA,
        PastaField::<M>::ZETA,
        PastaField::<M>::ZETA_INVERSE,
    ]
}

const FP_PARAMETERS: [Fp; 4] = parameters::<PallasBase>();
const FQ_PARAMETERS: [Fq; 4] = parameters::<PallasScalar>();

fn field<M: PrimeModulus>([half, delta, zeta, zeta_inverse]: [PastaField<M>; 4]) {
    assert_eq!(half.double().reduce(), PastaField::<M, Reduced>::ONE);
    assert_eq!(
        delta.reduce(),
        PastaField::<M>::from_u64(5).pow_u64(1 << 32).reduce()
    );
    let two = PastaField::<M>::from_u64(2);
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

    // Compare native field values only after explicit reduction.
    assert_eq!(two.double().reduce(), four.reduce());
    assert_ne!(two.reduce(), four.reduce());
    assert_eq!(two.add(&two).reduce(), four.reduce());
    assert_eq!(
        two.mul(&PastaField::<M>::from_u64(3)).reduce(),
        PastaField::<M>::from_u64(6).reduce()
    );
}

fn curve<C: PastaCurve>() {
    use arithmetic::{
        exec::{ExecutionOptions, SerialExecutor, TaskBudget},
        msm::{
            PreparedScalars, ScalarStorage,
            execution::{BatchPlan, MsmPlan},
        },
    };
    let generator = AffinePoint::<C>::GENERATOR;
    assert_eq!(
        generator.mul_projective(&PastaField::<C::Scalar>::from_u64(2)),
        generator.to_projective().double(),
    );
    assert_eq!(
        glv_decompose::<C>(&PastaField::<C::Scalar>::ONE.neg()),
        (-1, 0),
    );
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
    assert_eq!(
        prepared
            .cache(&plan, &mut digits, TaskBudget::SERIAL, &SerialExecutor)
            .len(),
        1
    );
}

#[cfg(feature = "traits")]
mod consumer {
    use super::*;

    pub(super) fn field<M: PrimeModulus>() {
        let two = FieldAdapter::<M>::from(2);
        let four = two.square();
        fn generic<F: Field>(value: F) -> F {
            value.square() * F::root_of_unity(1).unwrap() + F::ONE
        }
        assert_eq!(
            generic(two),
            -FieldAdapter::<M>::ONE * four + FieldAdapter::<M>::ONE
        );

        // Both entry points work with only a Field bound and bounded scratch.
        fn generic_batch<F: arithmetic::field::Field>(value: F) {
            let original = [F::ZERO, value, value.square()];
            let mut values = original;
            let mut scratch = [F::ZERO; 2];
            F::batch_invert(&mut values, &mut scratch);
            assert_eq!(values[0], F::ZERO);
            assert_eq!(values[1] * original[1], F::ONE);
            assert_eq!(values[2] * original[2], F::ONE);
            let (left, right) = values.split_at_mut(1);
            F::batch_invert_groups(&mut [left, right], &mut scratch[..1]);
            assert_eq!(values, original);
        }
        generic_batch(two);

        fn generic_product<F: arithmetic::field::Field>(value: F) {
            let values = [value, F::from(3)];
            assert_eq!(values.iter().product::<F>(), value * F::from(3));
            assert_eq!(values.into_iter().product::<F>(), value * F::from(3));
            assert_eq!(core::iter::empty::<F>().product::<F>(), F::ONE);
            assert_eq!(core::iter::empty::<&F>().product::<F>(), F::ONE);
        }
        generic_product(two);

        // Match consumers that hold only the field trait and a domain descriptor.
        fn generic_transform<F: Field>(value: F) {
            let domain = F::domain(2).unwrap();
            let mut values = [value; 4];
            domain.transform(&mut values);
            assert_eq!(values, [value * F::from(4), F::ZERO, F::ZERO, F::ZERO]);
            domain.inverse_transform(&mut values);
            assert_eq!(values, [value; 4]);
        }
        generic_transform(two);
    }

    pub(super) fn curve<C: PastaCurve>() {
        let generator = AffinePoint::<C>::GENERATOR;
        fn commit<A: Affine>(bases: &[A], scalars: &[A::Scalar]) -> A::Projective {
            A::msm(scalars, bases)
        }
        assert_eq!(
            commit(
                &[AffineAdapter::new(generator.to_point())],
                &[FieldAdapter::<C::Scalar>::from(2)]
            ),
            ProjectiveAdapter::new(generator.to_projective().double()),
        );

        fn generic_group<P: arithmetic::curve::Projective>(point: P, affine: P::Affine) {
            assert_eq!(point.add_mixed(&affine), point.double());
            assert_eq!(point.add_mixed(&P::Affine::identity()), point);
            assert_eq!(point.add_mixed(&affine.negate()), P::identity());
            let points = [point, point];
            assert_eq!(points.iter().sum::<P>(), point.double());
            assert_eq!(points.into_iter().sum::<P>(), point.double());
            assert_eq!(core::iter::empty::<P>().sum::<P>(), P::identity());
            assert_eq!(core::iter::empty::<&P>().sum::<P>(), P::identity());
        }
        generic_group(
            ProjectiveAdapter::new(generator.to_projective()),
            AffineAdapter::new(generator.to_point()),
        );
    }
}

pub fn run() {
    field::<PallasBase>(FP_PARAMETERS);
    field::<PallasScalar>(FQ_PARAMETERS);
    curve::<Pallas>();
    curve::<Vesta>();
    #[cfg(feature = "traits")]
    {
        consumer::field::<PallasBase>();
        consumer::field::<PallasScalar>();
        consumer::curve::<Pallas>();
        consumer::curve::<Vesta>();
    }
}
