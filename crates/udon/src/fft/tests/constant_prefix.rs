use super::*;
use crate::field::{ConstantPrefix, ReductionState, count_inversions};
use crate::test_support::{integer, modulus};
use num_bigint::BigUint;

fn canonical<M: PrimeModulus, S: ReductionState>(value: &PastaField<M, S>) -> BigUint {
    let p = modulus::<M>();
    integer(&value.montgomery_limbs()) * (BigUint::from(1u8) << 256usize).modinv(&p).unwrap() % p
}

fn from_raw<M: PrimeModulus>(raw: &BigUint) -> PastaField<M> {
    let digits = raw.to_u64_digits();
    let mut limbs = [0; 4];
    limbs[..digits.len()].copy_from_slice(&digits);
    PastaField::from_montgomery_limbs(limbs)
}

fn assert_raw<M: PrimeModulus>(values: &[PastaField<M>], expected: PastaField<M>) {
    assert!(
        values
            .iter()
            .all(|value| value.montgomery_limbs() == expected.montgomery_limbs())
    );
}

fn assert_integer<M: PrimeModulus>(values: &[PastaField<M>], expected: &[BigUint]) {
    assert_eq!(values.len(), expected.len());
    assert_loose_bound(values);
    for (value, expected) in values.iter().zip(expected) {
        assert_eq!(&canonical(value), expected);
    }
}

fn interpolate_integer<M: PrimeModulus>(
    values: &[PastaField<M>],
    domain: CosetDomain<M>,
) -> Vec<BigUint> {
    let p = modulus::<M>();
    let root = canonical(&domain.domain().root()).modinv(&p).unwrap();
    let shift = canonical(&domain.shift()).modinv(&p).unwrap();
    let size_inverse = BigUint::from(values.len()).modinv(&p).unwrap();
    let values: Vec<_> = values.iter().map(canonical).collect();
    // A dense integer inverse DFT has no knowledge of constant prefixes or
    // the short delta polynomial used by the implementation.
    (0..values.len())
        .map(|k| {
            let step = root.modpow(&BigUint::from(k), &p);
            let mut power = BigUint::from(1u8);
            let mut sum = BigUint::from(0u8);
            for value in &values {
                sum = (sum + value * &power) % &p;
                power = power * &step % &p;
            }
            sum * &size_inverse * shift.modpow(&BigUint::from(k), &p) % &p
        })
        .collect()
}

fn evaluate_integer<M: PrimeModulus>(
    coefficients: &[BigUint],
    domain: CosetDomain<M>,
) -> Vec<BigUint> {
    let p = modulus::<M>();
    let mut point = canonical(&domain.shift());
    let root = canonical(&domain.domain().root());
    (0..domain.size())
        .map(|_| {
            let value = coefficients
                .iter()
                .rev()
                .fold(BigUint::from(0u8), |sum, coefficient| {
                    (sum * &point + coefficient) % &p
                });
            point = &point * &root % &p;
            value
        })
        .collect()
}

fn transforms_field<M: PrimeModulus>() {
    let p = modulus::<M>();
    let sentinel = from_raw::<M>(&(&p * 2u8 - 1u8));
    let constant = from_raw::<M>(&(integer(&PastaField::<M>::from_u64(7).montgomery_limbs()) + &p));
    for log in 0..=4 {
        let subgroup = Domain::<M>::new(log).unwrap();
        let n = subgroup.size();
        let mut tail = inputs::<M>(n);
        tail[0] = from_raw(&p); // Loose zero must be a value, not a skipped entry.
        if n > 1 {
            tail[1] = constant; // A zero correction still occupies its index.
        }
        if n > 2 {
            tail[2] = sentinel;
        }
        for base in [subgroup.subgroup(), subgroup.coset()] {
            for t in 0..=n {
                let input = ConstantPrefix::new(n, &constant, &tail[..t]).unwrap();
                let mut dense = vec![constant; n];
                dense[n - t..].copy_from_slice(&tail[..t]);
                let expected = interpolate_integer(&dense, base);
                let mut output = vec![sentinel; n + 2];
                let mut scratch = vec![sentinel; t + 2];
                assert_eq!(
                    count_inversions(|| {
                        base.interpolate_constant_prefix(input, &mut output, &mut scratch)
                            .unwrap();
                    }),
                    0
                );
                assert_integer(&output[..n], &expected);
                assert_raw(&output[n..], sentinel);
                assert_raw(&scratch[t..], sentinel);
                let tail_reduced = reduced(&tail[..t]);
                let reduced_input =
                    ConstantPrefix::new(n, &constant.reduce(), &tail_reduced).unwrap();
                base.interpolate_constant_prefix(reduced_input, &mut output, &mut scratch)
                    .unwrap();
                assert_integer(&output[..n], &expected);
                for prepared in [false, true] {
                    let tables = Prepared::new(base);
                    let transform = if prepared {
                        tables.tables().bind(base)
                    } else {
                        Transform::new(base)
                    };
                    let mut ordinary = dense.clone();
                    transform
                        .inverse(&mut ordinary, Default::default(), &SerialExecutor, &mut [])
                        .unwrap();
                    assert_integer(&ordinary, &expected);
                }
                for extra in 0..=2 {
                    let target = Domain::<M>::new(log + extra).unwrap();
                    for extended in [target.subgroup(), target.coset()] {
                        let size = extended.size();
                        let expected = evaluate_integer(&expected, extended);
                        let mut samples = vec![sentinel; size + 2];
                        let mut inverse_scratch = vec![sentinel; size + 2];
                        let mut result = vec![sentinel; size + 2];
                        let mut plan = None;
                        let inversions = count_inversions(|| {
                            plan = Some(
                                ConstantPrefixExpansion::prepare(
                                    base,
                                    extended,
                                    &mut samples,
                                    &mut inverse_scratch,
                                )
                                .unwrap(),
                            );
                        });
                        assert!(inversions <= 1);
                        let plan = plan.unwrap();
                        assert_eq!(plan.base(), base);
                        assert_eq!(plan.extended(), extended);
                        ConstantPrefixExpansion::bind(base, extended, plan.samples()).unwrap();
                        assert_eq!(
                            count_inversions(|| plan.evaluate(input, &mut result).unwrap()),
                            0
                        );
                        assert_integer(&result[..size], &expected);
                        plan.evaluate(reduced_input, &mut result).unwrap();
                        assert_integer(&result[..size], &expected);
                        assert_raw(&result[size..], sentinel);
                        assert_raw(&samples[size..], sentinel);
                        assert_raw(&inverse_scratch[size..], sentinel);
                        if base.is_subgroup() {
                            let expansion =
                                Expansion::new(Transform::new(base), extended, None).unwrap();
                            let mut ordinary = vec![PastaField::ZERO; size];
                            expansion
                                .evaluations(
                                    &dense,
                                    &mut ordinary,
                                    Default::default(),
                                    &SerialExecutor,
                                    &mut [],
                                )
                                .unwrap();
                            let mut natural = vec![PastaField::ZERO; size];
                            expansion.layout().copy_to_natural(&ordinary, &mut natural);
                            assert_integer(&natural, &expected);
                        }
                    }
                }
            }
        }
    }
    // A full tail makes every choice of constant immaterial, including loose
    // zero and boundary representations. Equal tail values need not be trimmed.
    let domain = Domain::<M>::new(2).unwrap().coset();
    let values = [sentinel; 4];
    for constant in [PastaField::ZERO, from_raw(&p), sentinel, PastaField::ONE] {
        let input = ConstantPrefix::new(4, &constant, &values).unwrap();
        let mut output = [PastaField::ZERO; 4];
        domain
            .interpolate_constant_prefix(input, &mut output, &mut [PastaField::ZERO; 4])
            .unwrap();
        assert_eq!(output[0].reduce(), sentinel.reduce());
        assert!(output[1..].iter().all(PastaField::is_zero));
    }
}

#[test]
fn constant_prefix_transforms_match_integer_and_dense_oracles() {
    transforms_field::<PallasBase>();
    transforms_field::<PallasScalar>();
}

fn contracts_field<M: PrimeModulus>() {
    let sentinel = from_raw::<M>(&(modulus::<M>() * 2u8 - 1u8));
    let tail = [PastaField::<M>::ONE; 2];
    let input = ConstantPrefix::new(4, &sentinel, &tail).unwrap();
    let empty = ConstantPrefix::new(0, &sentinel, &tail[..0]).unwrap();
    let base = Domain::<M>::new(2).unwrap().subgroup();
    let target = Domain::<M>::new(3).unwrap().coset();
    let mut out = [sentinel; 8];
    let mut scratch = [sentinel; 10];
    for (output_len, scratch_len) in [(3, 2), (4, 1)] {
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _ = base.interpolate_constant_prefix(
                    input,
                    &mut out[..output_len],
                    &mut scratch[..scratch_len],
                );
            }))
            .is_err()
        );
        assert_raw(&out, sentinel);
        assert_raw(&scratch, sentinel);
    }
    assert_eq!(
        base.interpolate_constant_prefix(empty, &mut out, &mut scratch),
        Err(FftError::InvalidLayout)
    );
    assert_raw(&out, sentinel);
    assert_raw(&scratch, sentinel);
    assert!(ConstantPrefixExpansion::prepare(target, base, &mut out, &mut scratch).is_err());
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = ConstantPrefixExpansion::prepare(base, target, &mut out[..7], &mut scratch);
        }))
        .is_err()
    );
    assert_raw(&out, sentinel);
    assert_raw(&scratch, sentinel);
    let plan = ConstantPrefixExpansion::prepare(base, target, &mut out, &mut scratch).unwrap();
    let correct = plan.samples().to_vec();
    let mut result = [sentinel; 10];
    assert_eq!(
        plan.evaluate(empty, &mut result),
        Err(FftError::InvalidLayout)
    );
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let _ = plan.evaluate(input, &mut result[..7]);
        }))
        .is_err()
    );
    assert_raw(&result, sentinel);
    assert!(ConstantPrefixExpansion::bind(base, target, &correct[..7]).is_err());
    assert!(ConstantPrefixExpansion::bind(base, target, &scratch).is_err());
    assert!(ConstantPrefixExpansion::bind(target, base, &correct).is_err());
    for target in [target, target.domain().subgroup(), base] {
        let size = target.size();
        let mut correct = vec![sentinel; size];
        ConstantPrefixExpansion::prepare(base, target, &mut correct, &mut scratch).unwrap();
        let loose: Vec<_> = correct
            .iter()
            .map(|value| from_raw(&(integer(&value.reduce().montgomery_limbs()) + modulus::<M>())))
            .collect();
        ConstantPrefixExpansion::bind(base, target, &loose).unwrap();
        for len in [0, 1, 2, 3, size, size + 2] {
            let mut bounded = vec![sentinel; len];
            let plan =
                ConstantPrefixExpansion::prepare(base, target, &mut out, &mut bounded).unwrap();
            assert_eq!(reduced(plan.samples()), reduced(&correct));
            if len > size {
                assert_raw(&bounded[size..], sentinel);
            }
        }
    }
}

#[test]
fn constant_prefix_domain_and_buffer_contracts() {
    contracts_field::<PallasBase>();
    contracts_field::<PallasScalar>();
}
