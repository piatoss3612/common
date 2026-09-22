#![forbid(unsafe_code)]
#![deny(warnings)]

mod facade {
    pub use arithmetic::{
        curve::{AffinePoint, Pallas, PastaCurve, Vesta, glv_decompose},
        field::{Fp, Fq, PallasBase, PallasScalar, PastaField, PrimeModulus},
    };
}

use facade::*;

// Generic constant initializers must work through the consumer's facade,
// without access to the sealed implementation parameters.
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

fn field<M: PrimeModulus>([half, delta, zeta, zeta_inverse]: [PastaField<M>; 4]) {
    assert_eq!(half.double(), PastaField::<M>::ONE);
    assert_eq!(delta, PastaField::<M>::from_u64(5).pow_u64(1 << 32));
    let two = PastaField::<M>::from_u64(2);
    let four = two.square();
    assert_eq!(four.sqrt().unwrap().square(), four);
    assert_eq!(two.mul(&two.invert().unwrap()), PastaField::<M>::ONE);
    assert_eq!(M::MODULUS[3], 1 << 62);
    assert_eq!(
        PastaField::<M>::root_of_unity(4)
            .unwrap()
            .mul(&PastaField::<M>::root_of_unity_inverse(4).unwrap()),
        PastaField::<M>::ONE,
    );
    assert_eq!(zeta.mul(&zeta_inverse), PastaField::<M>::ONE);

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
}

fn curve<C: PastaCurve>() {
    use arithmetic::{
        curve::msm::{
            PreparedScalars, ScalarStorage,
            run::{BatchPlan, MsmPlan},
        },
        exec::{ExecutionOptions, SerialExecutor, TaskBudget},
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
    let _ = arithmetic::curve::msm::Kernel::Auto;
    #[cfg(feature = "msm-accumulation")]
    let _ = arithmetic::curve::msm::Accumulation::Auto;
    #[cfg(feature = "fft-codelet")]
    let _ = arithmetic::fft::Codelet::Radix2;
    #[cfg(feature = "fft-strategy")]
    let _ = arithmetic::fft::Strategy::SERIAL;
    #[cfg(feature = "cache-options")]
    let _ = prepared.cache_len(options);
}

#[cfg(any(feature = "foreign-modulus", feature = "foreign-curve"))]
#[derive(Clone, Copy, Eq, PartialEq)]
enum Foreign {}

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
