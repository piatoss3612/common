//! Native arithmetic stays explicit with or without consumer traits.
#![allow(unused_imports)]
use arithmetic::{
    curve::{PallasAffine, PallasPoint, PallasProjective},
    field::{Fp, Fq},
};
fn main() {
    assert_eq!(
        <Fp>::ONE.add(&<Fp>::ONE).reduce(),
        <Fp>::from_u64(2).reduce()
    );
    #[cfg(feature = "field-eq")]
    {
        let _ = <Fp>::ONE == <Fp>::ONE;
    }
    #[cfg(feature = "field-ne")]
    {
        let _ = <Fp>::ZERO != <Fp>::ONE;
    }
    #[cfg(feature = "field-add")]
    {
        let _ = <Fp>::ONE + <Fp>::ONE;
    }
    #[cfg(feature = "field-sub")]
    {
        let _ = <Fp>::ONE - <Fp>::ONE;
    }
    #[cfg(feature = "field-mul")]
    {
        let _ = <Fp>::ONE * <Fp>::ONE;
    }
    #[cfg(feature = "field-neg")]
    {
        let _ = -<Fp>::ONE;
    }
    #[cfg(feature = "field-add-assign")]
    {
        let mut a = <Fp>::ONE;
        a += <Fp>::ONE;
    }
    #[cfg(feature = "field-sub-assign")]
    {
        let mut a = <Fp>::ONE;
        a -= <Fp>::ONE;
    }
    #[cfg(feature = "field-mul-assign")]
    {
        let mut a = <Fp>::ONE;
        a *= <Fp>::ONE;
    }
    #[cfg(feature = "field-reduced-add")]
    {
        let _ = <Fp>::ONE.reduce() + <Fp>::ONE.reduce();
    }
    #[cfg(feature = "field-sum")]
    {
        let _: Fp = [<Fp>::ONE].into_iter().sum();
    }
    #[cfg(feature = "field-product")]
    {
        let _: Fp = [<Fp>::ONE].into_iter().product();
    }
    #[cfg(feature = "point-neg")]
    {
        let _ = -PallasPoint::GENERATOR;
    }
    #[cfg(feature = "point-mul")]
    {
        let _ = PallasPoint::GENERATOR * <Fq>::ONE;
    }
    #[cfg(feature = "affine-neg")]
    {
        let _ = PallasAffine::GENERATOR.neg();
        let _ = -PallasAffine::GENERATOR;
    }
    #[cfg(feature = "affine-mul")]
    {
        let _ = PallasAffine::GENERATOR * <Fq>::ONE;
    }
    #[cfg(feature = "projective-add")]
    {
        let _ = PallasProjective::GENERATOR + PallasProjective::GENERATOR;
    }
    #[cfg(feature = "projective-sub")]
    {
        let _ = PallasProjective::GENERATOR - PallasProjective::GENERATOR;
    }
    #[cfg(feature = "projective-neg")]
    {
        let _ = -PallasProjective::GENERATOR;
    }
    #[cfg(feature = "projective-mul")]
    {
        let _ = PallasProjective::GENERATOR * <Fq>::ONE;
    }
    #[cfg(feature = "projective-add-assign")]
    {
        let mut p = PallasProjective::GENERATOR;
        p += p;
    }
    #[cfg(feature = "projective-sub-assign")]
    {
        let mut p = PallasProjective::GENERATOR;
        p -= p;
    }
    #[cfg(feature = "projective-sum")]
    {
        let _: PallasProjective = [PallasProjective::GENERATOR].into_iter().sum();
    }
}
