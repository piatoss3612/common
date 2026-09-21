#![forbid(unsafe_code)]
#![deny(warnings)]

mod facade {
    pub use arithmetic::{
        curve::{AffinePoint, Pallas, PastaCurve, Vesta, glv_decompose},
        field::{Fp, Fq, PallasBase, PallasScalar, PastaField, PrimeModulus},
    };
}

use facade::*;

const fn half<M: PrimeModulus>() -> PastaField<M> {
    PastaField::<M>::two_inverse()
}

const FP_HALF: Fp = half::<PallasBase>();
const FQ_HALF: Fq = half::<PallasScalar>();

fn field<M: PrimeModulus>() {
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
    assert_eq!(
        PastaField::<M>::zeta().mul(&PastaField::<M>::zeta_inverse()),
        PastaField::<M>::ONE,
    );

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
            Accumulation, ArithmeticOptions, BatchOptions, Kernel, PreparedScalars, ScalarStorage,
            run::{BatchPlan, MsmPlan},
        },
        exec::{SerialExecutor, TaskBudget},
    };
    use core::num::NonZeroUsize;

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

    let options = ArithmeticOptions::DEFAULT
        .with_kernel(Kernel::Booth {
            width: Some(4),
            accumulation: Accumulation::Auto,
        })
        .unwrap();
    let batch = BatchOptions::new(options)
        .with_task_budget(TaskBudget::new(3).unwrap())
        .with_memory_limit(8192);
    assert_eq!(batch.arithmetic(), options);
    let plan = MsmPlan::<C>::new(1, options, NonZeroUsize::MIN).unwrap();
    assert_eq!(plan.grain(), 1);
    assert_eq!(BatchPlan::<C>::storage_len(1, batch).unwrap(), (1, 1));
    let mut records = [ScalarStorage::<C>::ZERO];
    let prepared = PreparedScalars::prepare(
        &[PastaField::ONE],
        &mut records,
        TaskBudget::SERIAL,
        &SerialExecutor,
    )
    .unwrap();
    let mut digits = vec![0; prepared.cache_len(options).unwrap()];
    assert_eq!(prepared.cache(options, &mut digits).unwrap().len(), 1);

    #[cfg(feature = "arithmetic-budget")]
    let _ = options.with_task_budget(TaskBudget::SERIAL);
    #[cfg(feature = "arithmetic-limit")]
    let _ = options.with_memory_limit(8192);
    #[cfg(feature = "plan-batch-options")]
    let _ = MsmPlan::<C>::new(1, batch, NonZeroUsize::MIN);
    #[cfg(feature = "cache-batch-options")]
    let _ = prepared.cache_len(batch);
    #[cfg(feature = "batch-arithmetic-options")]
    let _ = BatchPlan::<C>::storage_len(1, options);
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
    assert_eq!(FP_HALF.double(), Fp::ONE);
    assert_eq!(FQ_HALF.double(), Fq::ONE);
    field::<PallasBase>();
    field::<PallasScalar>();
    curve::<Pallas>();
    curve::<Vesta>();
}
