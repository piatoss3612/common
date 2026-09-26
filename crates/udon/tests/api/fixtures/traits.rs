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

    let mut inverses = [value, Fp::ZERO];
    udon::field::batch_invert(&mut inverses, &mut [Fp::ZERO; 2]);
    assert_eq!(inverses, [value.invert().unwrap(), Fp::ZERO]);
    udon::field::batch_invert_groups(&mut [&mut inverses[..]], &mut [Fp::ZERO; 1]);
    assert_eq!(inverses, [value, Fp::ZERO]);

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

    let mut basis = [Fp::ZERO; 4];
    domain
        .subgroup()
        .evaluate_lagrange(&<Fp>::ZERO, 0..4, &mut basis, &mut [])
        .unwrap();
    assert_eq!(basis, [domain.size_inverse(); 4]);

    // Native polynomial arithmetic remains available without consumer traits.
    let coefficients = [Fp::ONE, value, Fp::ONE];
    let point = <Fp>::from_u64(2);
    assert_eq!(
        udon::polynomial::evaluate(&coefficients, &point),
        Fp::from_u64(19)
    );
    let mut divided = coefficients;
    let split = udon::polynomial::divide_linear_in_place(&mut divided, &point);
    assert_eq!(split, 1);
    assert_eq!(divided, [19, 9, 1].map(Fp::from_u64));

    #[cfg(feature = "field")]
    field();
    #[cfg(feature = "curve")]
    curve();
    #[cfg(feature = "domain")]
    {
        let mut generic = input;
        domain.transform(&mut generic);
        assert_eq!(generic, expected);
        let mut values = [Fp::ZERO; 4];
        assert_eq!(
            domain.lagrange_evaluations(Fp::ZERO, &mut values, &mut []),
            None
        );
        assert_eq!(values, basis);
    }
    #[cfg(feature = "polynomial")]
    {
        assert_eq!(
            udon::polynomial::evaluate_iter(&coefficients, point),
            divided[0]
        );
        let mut quotient: Vec<_> =
            udon::polynomial::divide_linear_rev(coefficients, point).collect();
        quotient.reverse();
        assert_eq!(quotient, divided[split..]);
        assert_eq!(udon::polynomial::geometric_sum(Fp::ONE, 3), Fp::from_u64(3));
        let mut product = [Fp::ZERO; 4];
        udon::polynomial::multiply(&coefficients, &[Fp::ONE; 2], &mut product, &mut []);
        assert_eq!(product, [1, 8, 8, 1].map(Fp::from_u64));
    }
    #[cfg(feature = "cycle")]
    cycle::<udon::cycle::Pasta>();
    #[cfg(feature = "poseidon")]
    poseidon::<udon::poseidon::PoseidonFp>();
}

#[cfg(feature = "field")]
fn field() {
    fn generic<F: udon::field::Field>(values: &mut [F]) {
        let value = F::from(7);
        let repr: F::Repr = value.to_bytes();
        assert_eq!(F::from_bytes(repr), Some(value));
        assert_eq!(F::ZETA.pow_u64(3), F::ONE);
        let mut accumulator = F::Accumulator::default();
        F::mul_accumulate(&mut accumulator, &value, &value);
        F::mul_accumulate(&mut accumulator, &F::ONE, &F::from(2));
        assert_eq!(F::reduce(accumulator), F::from(51));

        let domain = F::domain(1).unwrap();
        let mut coefficients = [value, F::ONE];
        domain.transform(&mut coefficients);
        assert_eq!(coefficients, [F::from(8), F::from(6)]);
        domain.inverse_transform(&mut coefficients);
        assert_eq!(coefficients, [value, F::ONE]);

        assert_eq!(value.mul_add(&F::from(3), &F::from(2)), F::from(23));
        assert_eq!(udon::field::dot(&[value], &[value]), F::from(49));
        assert_eq!(
            udon::field::dot_iter(
                [value, F::ONE].iter(),
                [F::from(2), F::from(3)].iter().rev()
            ),
            F::from(23)
        );
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
fn poseidon<P: udon::poseidon::PoseidonPermutation<Fp> + Default>() {
    let instance = P::default();
    let native = udon::poseidon::PALLAS_BASE;
    assert_eq!(P::T, 5);
    assert_eq!(P::ALPHA, native.alpha);
    assert_eq!(instance.round_constants().len(), native.rounds());
    assert_eq!(instance.mds_matrix().len(), native.width());
    for (row, expected) in instance
        .round_constants()
        .iter()
        .zip(native.round_constants)
    {
        assert_eq!(row.as_ref(), expected);
    }
    for (row, expected) in instance.mds_matrix().iter().zip(native.mds) {
        assert_eq!(row.as_ref(), expected);
    }
}
