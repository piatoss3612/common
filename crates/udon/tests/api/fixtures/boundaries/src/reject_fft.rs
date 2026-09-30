use arithmetic::{
    fft::{
        ConstantPrefixExpansion, Domain, LagrangeCompletion, LagrangeError, VanishingDivision,
        VanishingFactors,
    },
    field::{ConstantPrefix, ConstantPrefixError, PastaField, PrimeModulus, Reduced},
    polynomial::evaluate,
};
pub fn check() {
    field::<arithmetic::field::PallasBase>();
    field::<arithmetic::field::PallasScalar>();
    #[cfg(feature = "fft-codelet")]
    let _ = arithmetic::fft::Codelet::Radix2;
    #[cfg(feature = "fft-strategy")]
    let _ = arithmetic::fft::Strategy::SERIAL;
    #[cfg(feature = "empty-interpolation")]
    let _ = arithmetic::fft::execution::InterpolationPlan::<arithmetic::field::PallasBase, 0>::new(
        [],
        false,
        arithmetic::fft::StorageLayout::Contiguous,
        arithmetic::exec::ExecutionOptions::default(),
    );
}

fn constant_prefix<M: PrimeModulus>() {
    let base = Domain::<PastaField<M>>::new(2).unwrap().subgroup();
    let extended = Domain::<PastaField<M>>::new(3).unwrap().coset();
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
        let input = ConstantPrefix::new(
            4,
            &<arithmetic::field::Fq>::ONE,
            &[<arithmetic::field::Fq>::ONE],
        )
        .unwrap();
        let _ = Domain::<PastaField<arithmetic::field::PallasBase>>::new(2)
            .unwrap()
            .subgroup()
            .interpolate_constant_prefix(
                input,
                &mut [<arithmetic::field::Fp>::ZERO; 4],
                &mut [<arithmetic::field::Fp>::ZERO],
            );
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
    let domain = Domain::<PastaField<M>>::new(2).unwrap();
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
        let division = VanishingDivision::new(
            Domain::<PastaField<arithmetic::field::PallasBase>>::new(0).unwrap(),
            &<arithmetic::field::Fp>::ZETA,
            1,
        )
        .unwrap();
        division.write_pieces(
            &[<arithmetic::field::Fq>::ZERO],
            ElementOrder::Natural,
            &mut [],
            &mut [],
        );
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

fn field<M: PrimeModulus>() {
    constant_prefix::<M>();
    vanishing_division::<M>();
    let two = PastaField::<M>::from_u64(2);
    let domain = Domain::<PastaField<M>>::new(2).unwrap().coset();
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
    arithmetic::field::batch_invert_groups_scaled(
        &mut [&mut basis[..]],
        &PastaField::ONE,
        &mut [PastaField::ZERO; 4],
    );
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
        let _ = Domain::<PastaField<arithmetic::field::PallasBase>>::new(1)
            .unwrap()
            .subgroup()
            .evaluate_lagrange(
                &<arithmetic::field::Fq>::ONE,
                0..1,
                &mut [<arithmetic::field::Fp>::ZERO],
                &mut [],
            );
    }
    #[cfg(feature = "lagrange-reduced-storage")]
    {
        let _ = domain.prepare_lagrange(&two, 0..1, &mut [two.reduce()]);
    }
    #[cfg(feature = "lagrange-completion-field-mismatch")]
    {
        let mut values = [<arithmetic::field::Fp>::ZERO];
        let completion = Domain::<PastaField<arithmetic::field::PallasBase>>::new(1)
            .unwrap()
            .subgroup()
            .prepare_lagrange(&<arithmetic::field::Fp>::ZERO, 0..1, &mut values)
            .unwrap();
        let _ = completion.complete(&mut [<arithmetic::field::Fq>::ZERO]);
    }
}
