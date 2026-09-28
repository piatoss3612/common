#![forbid(unsafe_code)]
#![deny(warnings)]

use arithmetic::{
    curve::{
        EisensteinTable, Pallas, PallasAffine, PallasPoint, PallasProjective, PreparedAffinePoint,
        Vesta, VestaAffine, VestaPoint, VestaProjective, glv_decompose,
    },
    field::{Fp, Fq},
};

mod literals {
    pub use arithmetic::{
        fp_hex as fp, fq_hex as fq, pallas_affine as pallas, vesta_affine as vesta,
    };
}

// Re-exported macros and const lifts must work through the renamed dependency.
const FP_LITERAL: Fp =
    literals::fp!("0x000000000000000000000000000000000000000000000000000000000000002a",);
const FQ_LITERAL: Fq =
    literals::fq!("0x000000000000000000000000000000000000000000000000000000000000002a");
const X: Fp = PallasAffine::GENERATOR.coordinates().0.into_loose();
const Y: Fp = PallasAffine::GENERATOR.coordinates().1.into_loose();
const PALLAS: PallasAffine = literals::pallas!(X, Y,);
const VESTA: VestaAffine = literals::vesta!(
    *VestaAffine::GENERATOR.coordinates().0,
    *VestaAffine::GENERATOR.coordinates().1,
);
const PALLAS_POINT: PallasPoint = PALLAS.to_point();
const VESTA_POINT: VestaPoint = VESTA.to_point();
const PALLAS_PROJECTIVE: PallasProjective = PALLAS_POINT.to_projective();
const VESTA_PROJECTIVE: VestaProjective = VESTA_POINT.to_projective();

fn main() {
    assert_eq!(FP_LITERAL.reduce(), Fp::from_u64(42));
    assert_eq!(FQ_LITERAL.reduce(), Fq::from_u64(42));
    assert_eq!(PALLAS, PallasAffine::GENERATOR);
    assert_eq!(VESTA, VestaAffine::GENERATOR);
    assert_eq!(PALLAS_PROJECTIVE, PallasProjective::GENERATOR);
    assert_eq!(VESTA_PROJECTIVE, VestaProjective::GENERATOR);
    assert_eq!(literals::pallas!(X, Y), PALLAS);
    assert_eq!(
        literals::vesta!(*VESTA.coordinates().0, *VESTA.coordinates().1),
        VESTA
    );
    assert_eq!(PALLAS.mul_projective(&Fq::ONE), PALLAS_PROJECTIVE);
    assert_eq!(VESTA.mul_projective(&Fp::ONE), VESTA_PROJECTIVE);

    assert_eq!(glv_decompose::<Pallas>(&<Fq>::ONE.neg()), (-1, 0));
    assert_eq!(glv_decompose::<Vesta>(&<Fp>::ONE.neg()), (-1, 0));
    assert_eq!(
        PALLAS.endomorphism(),
        PALLAS
            .mul_projective(&Fq::ZETA)
            .to_point()
            .as_affine()
            .copied()
            .unwrap()
    );
    assert_eq!(
        VESTA.endomorphism(),
        VESTA
            .mul_projective(&Fp::ZETA)
            .to_point()
            .as_affine()
            .copied()
            .unwrap()
    );
    let mut entries = [PreparedAffinePoint::from_affine(&PALLAS); 8];
    let mut projective = [PallasProjective::IDENTITY; 8];
    let mut field = [Fp::ZERO; 8];
    let table = EisensteinTable::prepare(&PALLAS, &mut entries, &mut projective, &mut field);
    assert_eq!(table.mul(&<Fq>::ONE.neg()), PALLAS_PROJECTIVE.neg());

    #[cfg(feature = "invalid-pallas")]
    let _ = literals::pallas!(<Fp>::ZERO, <Fp>::ZERO);
    #[cfg(feature = "invalid-vesta")]
    let _ = literals::vesta!(<Fq>::ONE, <Fq>::ONE);

    #[cfg(feature = "runtime-pallas")]
    {
        let runtime = std::hint::black_box(X);
        let _ = literals::pallas!(runtime, Y);
    }
    #[cfg(feature = "runtime-vesta")]
    {
        let runtime = std::hint::black_box(*VESTA.coordinates().0);
        let _ = literals::vesta!(runtime, *VESTA.coordinates().1);
    }

    #[cfg(feature = "wrong-pallas-field")]
    let _ = literals::pallas!(<Fq>::ZERO, <Fq>::ZERO);
    #[cfg(feature = "wrong-vesta-field")]
    let _ = literals::vesta!(<Fp>::ZERO, <Fp>::ZERO);
    #[cfg(feature = "wrong-curve")]
    let _ = PALLAS_PROJECTIVE.add(&VESTA_PROJECTIVE);
    #[cfg(feature = "wrong-scalar")]
    let _ = PALLAS.mul_projective(&Fp::ONE);
}
