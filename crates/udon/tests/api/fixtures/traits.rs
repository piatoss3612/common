//! Uses a renamed dependency with and without the consumer interfaces.

use arithmetic as udon;
use udon::{
    curve::{PallasPoint, PallasProjective},
    exec::{ExecutionOptions, SerialExecutor},
    fft::{Domain, Transform, reference},
    field::{Fp, Fq},
};

fn main() {
    let value = Fp::from_u64(7);
    assert_eq!(value * value.invert().unwrap(), Fp::ONE);
    assert_eq!([value, Fp::ONE].into_iter().product::<Fp>(), value);
    assert_eq!(
        PallasPoint::GENERATOR * Fq::from_u64(2),
        PallasPoint::GENERATOR.double(),
    );
    let lifted = PallasProjective::from(PallasPoint::GENERATOR);
    assert_eq!(PallasPoint::from(lifted), PallasPoint::GENERATOR);
    assert_eq!(udon::poseidon::PALLAS_BASE.rounds(), 64);

    let domain = Domain::<Fp>::new(2).unwrap();
    let input = [value, Fp::ONE, Fp::ZERO, Fp::DELTA];
    let mut expected = input;
    reference::transform(&mut expected, &domain.root());
    let mut actual = input;
    let transform = Transform::new(domain.subgroup());
    transform
        .forward(
            &mut actual,
            ExecutionOptions::default(),
            &SerialExecutor,
            &mut [],
        )
        .unwrap();
    assert_eq!(actual, expected);

    #[cfg(feature = "field")]
    field();
    #[cfg(feature = "curve")]
    curve();
    #[cfg(feature = "domain")]
    {
        let mut generic = input;
        domain.transform(&mut generic);
        assert_eq!(generic, expected);
    }
    #[cfg(feature = "poly")]
    assert_eq!(
        udon::poly::evaluate([&Fp::ONE, &value], Fp::ONE),
        value + Fp::ONE
    );
    #[cfg(feature = "cycle")]
    cycle::<udon::cycle::Pasta>();
    #[cfg(feature = "poseidon")]
    poseidon::<udon::poseidon::PoseidonFp>();
}

#[cfg(feature = "field")]
fn field() {
    fn generic<F: udon::field::Field>(values: &mut [F]) {
        F::batch_invert(values, &mut [F::ZERO; 2]);
        assert_eq!(values[0] * F::from(7), F::ONE);
        assert_eq!(values[1], F::ZERO);
    }
    generic(&mut [Fp::from_u64(7), Fp::ZERO]);
}

#[cfg(feature = "curve")]
fn curve() {
    fn generic<A: udon::curve::Affine>() {
        assert_eq!(A::msm(&[], &[]), A::identity().to_projective());
    }
    generic::<PallasPoint>();
}

#[cfg(feature = "cycle")]
fn cycle<C: udon::cycle::Cycle>() {}

#[cfg(feature = "poseidon")]
fn poseidon<P: udon::poseidon::PoseidonPermutation<Fp>>() {
    assert_eq!(P::T, 5);
}
