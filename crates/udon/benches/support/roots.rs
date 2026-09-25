use super::{
    Criterion, PastaField, PrimeModulus, Reduced, SEED_A, SEED_B, Throughput, bench, values,
};

fn retry<M: PrimeModulus>(value: PastaField<M, Reduced>) -> (bool, PastaField<M, Reduced>) {
    match value.sqrt() {
        Some(root) => (true, root),
        None => (
            false,
            value
                .mul(&PastaField::SQRT_NONSQUARE)
                .reduce()
                .sqrt()
                .unwrap(),
        ),
    }
}

fn invert_then_sqrt<M: PrimeModulus>(
    numerator: PastaField<M, Reduced>,
    denominator: PastaField<M, Reduced>,
) -> Option<PastaField<M, Reduced>> {
    if numerator.is_zero() {
        return Some(PastaField::ZERO);
    }
    numerator.mul(&denominator.invert()?).reduce().sqrt()
}

fn invert_then_retry<M: PrimeModulus>(
    numerator: PastaField<M, Reduced>,
    denominator: PastaField<M, Reduced>,
) -> (bool, PastaField<M, Reduced>) {
    if numerator.is_zero() {
        return (true, PastaField::ZERO);
    }
    match denominator.invert() {
        Some(inverse) => retry(numerator.mul(&inverse).reduce()),
        None => (false, PastaField::ZERO),
    }
}

pub(super) fn benchmarks<M: PrimeModulus>(criterion: &mut Criterion, name: &str) {
    let squares = values::<M, 32>(SEED_A).map(|value| value.square().reduce());
    let nonsquares = squares.map(|value| value.mul(&PastaField::SQRT_NONSQUARE).reduce());
    let mixed = core::array::from_fn(|i| {
        if i % 2 == 0 {
            squares[i]
        } else {
            nonsquares[i]
        }
    });
    let denominators = values::<M, 32>(SEED_B).map(PastaField::reduce);
    assert!(denominators.iter().all(|value| !value.is_zero()));
    let mut group = criterion.benchmark_group(format!("{name}/sqrt_alt"));
    group.throughput(Throughput::Elements(32));
    for (label, inputs) in [
        ("square", squares),
        ("nonsquare", nonsquares),
        ("mixed", mixed),
    ] {
        for value in inputs {
            let actual = value.sqrt_alt();
            let reference = retry(value);
            assert_eq!(actual.0, reference.0);
            assert_eq!(actual.1.square().reduce(), reference.1.square().reduce());
        }
        bench(&mut group, &format!("{label}/direct"), &inputs, |inputs| {
            inputs.map(|value| value.sqrt_alt())
        });
        bench(&mut group, &format!("{label}/retry"), &inputs, |inputs| {
            inputs.map(retry)
        });
    }
    group.finish();

    let mut group = criterion.benchmark_group(format!("{name}/sqrt_ratio"));
    group.throughput(Throughput::Elements(32));
    for (label, ratios) in [
        ("square", squares),
        ("nonsquare", nonsquares),
        ("mixed", mixed),
    ] {
        let inputs: [_; 32] =
            core::array::from_fn(|i| (ratios[i].mul(&denominators[i]).reduce(), denominators[i]));
        for (numerator, denominator) in inputs {
            let actual = numerator.sqrt_ratio(&denominator);
            let reference = invert_then_retry(numerator, denominator);
            assert_eq!(actual.0, reference.0);
            assert_eq!(actual.1.square().reduce(), reference.1.square().reduce());
            assert_eq!(actual.0, invert_then_sqrt(numerator, denominator).is_some());
        }
        bench(&mut group, &format!("{label}/direct"), &inputs, |inputs| {
            inputs.map(|(numerator, denominator)| numerator.sqrt_ratio(&denominator))
        });
        bench(
            &mut group,
            &format!("{label}/invert_then_sqrt"),
            &inputs,
            |inputs| {
                inputs.map(|(numerator, denominator)| invert_then_sqrt(numerator, denominator))
            },
        );
        bench(
            &mut group,
            &format!("{label}/invert_then_retry"),
            &inputs,
            |inputs| {
                inputs.map(|(numerator, denominator)| invert_then_retry(numerator, denominator))
            },
        );
    }
    group.throughput(Throughput::Elements(1));
    for (label, input) in [
        ("zero_numerator", (PastaField::ZERO, denominators[0])),
        ("zero_denominator", (squares[0], PastaField::ZERO)),
        ("both_zero", (PastaField::ZERO, PastaField::ZERO)),
    ] {
        assert_eq!(
            input.0.sqrt_ratio(&input.1),
            invert_then_retry(input.0, input.1)
        );
        bench(&mut group, &format!("{label}/direct"), &input, |(n, d)| {
            n.sqrt_ratio(d)
        });
        bench(
            &mut group,
            &format!("{label}/invert_then_retry"),
            &input,
            |(n, d)| invert_then_retry(*n, *d),
        );
    }
    group.finish();
}
