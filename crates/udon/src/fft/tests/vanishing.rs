use super::*;
use crate::exec::ExecutionOptions;
use crate::field::{ReductionState, count_inversions};
use crate::test_support::{integer, modulus};
use num_bigint::BigUint;

fn canonical<M: PrimeModulus, S: ReductionState>(value: &PastaField<M, S>) -> BigUint {
    let p = modulus::<M>();
    let raw = integer(&value.montgomery_limbs());
    assert!(raw < &p * 2u8);
    raw * (BigUint::from(1u8) << 256usize).modinv(&p).unwrap() % p
}

fn from_integer<M: PrimeModulus>(value: &BigUint) -> PastaField<M> {
    let mut bytes = [0; 32];
    let encoded = value.to_bytes_le();
    bytes[..encoded.len()].copy_from_slice(&encoded);
    PastaField::from_bytes(bytes).unwrap()
}

fn loose<M: PrimeModulus>(value: PastaField<M>) -> PastaField<M> {
    let raw = integer(&value.reduce().montgomery_limbs()) + modulus::<M>();
    let mut limbs = [0; 4];
    let digits = raw.to_u64_digits();
    limbs[..digits.len()].copy_from_slice(&digits);
    PastaField::from_montgomery_limbs(limbs)
}

fn integer_evaluate(coefficients: &[BigUint], x: &BigUint, p: &BigUint) -> BigUint {
    coefficients
        .iter()
        .rev()
        .fold(BigUint::from(0u8), |a, b| (a * x + b) % p)
}

fn integer_cases<M: PrimeModulus>() {
    let p = modulus::<M>();
    for log in 0..=4 {
        let domain = Domain::<M>::new(log).unwrap();
        let size = domain.size();
        let root = canonical(&domain.root());
        for shift in [
            PastaField::ZETA,
            PastaField::ZETA_INVERSE,
            PastaField::from_u64(7),
        ] {
            let g = canonical(&shift);
            let c = g.modpow(&BigUint::from(size), &p);
            for piece_log in 0..=log {
                let n = 1 << piece_log;
                let plan = VanishingDivision::new(domain, &loose(shift), n).unwrap();
                let m = plan.piece_count();
                for exact in [false, true] {
                    // Construct h independently, then multiply by X^n - 1 and
                    // reduce X^N to c using integer arithmetic. The full case
                    // deliberately has nonzero discarded high coefficients.
                    let mut h: Vec<_> = inputs::<M>(size).iter().map(canonical).collect();
                    if exact {
                        h[size - n..].fill(BigUint::from(0u8));
                    }
                    let mut a: Vec<_> = h.iter().map(|v| (&p - v) % &p).collect();
                    for (i, v) in h.iter().enumerate() {
                        let index = (i + n) % size;
                        a[index] = (&a[index]
                            + v * if i + n >= size { c.clone() } else { 1u8.into() })
                            % &p;
                    }
                    let mut node = g.clone();
                    let evaluations: Vec<PastaField<M>> = (0..size)
                        .map(|_| {
                            let value = from_integer(&integer_evaluate(&a, &node, &p));
                            node = &node * &root % &p;
                            loose(value)
                        })
                        .collect();
                    // Independent pointwise division followed by a direct
                    // integer inverse DFT; this does not use column recurrence.
                    let divided: Vec<_> = evaluations
                        .iter()
                        .enumerate()
                        .map(|(i, value)| {
                            let node = &g * root.modpow(&BigUint::from(i), &p) % &p;
                            let divisor = (node.modpow(&BigUint::from(n), &p) + &p - 1u8) % &p;
                            canonical(value) * divisor.modinv(&p).unwrap() % &p
                        })
                        .collect();
                    let inverse_root = root.modinv(&p).unwrap();
                    let inverse_size = BigUint::from(size).modinv(&p).unwrap();
                    for (degree, expected) in h.iter().enumerate() {
                        let sum: BigUint = divided
                            .iter()
                            .enumerate()
                            .map(|(i, value)| {
                                value * inverse_root.modpow(&BigUint::from(i * degree), &p)
                            })
                            .sum();
                        let result = sum
                            * &inverse_size
                            * g.modpow(&BigUint::from(degree), &p).modinv(&p).unwrap()
                            % &p;
                        assert_eq!(&result, expected);
                    }
                    let mut transformed = evaluations.clone();
                    reference::transform(&mut transformed, &domain.root());
                    // Force every supplied transform value into the upper half
                    // of the allowed loose range, including zero as p.
                    transformed
                        .iter_mut()
                        .for_each(|value| *value = loose(*value));
                    for order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                        let input: Vec<_> = (0..size)
                            .map(|i| {
                                transformed[match order {
                                    ElementOrder::Natural => i,
                                    ElementOrder::BitReversed => reverse(i, log),
                                }]
                            })
                            .collect();
                        let original = bytes_of_slice(&input).to_vec();
                        for count in 0..=m {
                            let sentinel = loose(PastaField::<M>::ONE.neg());
                            let mut pieces = vec![vec![sentinel; n + 2]; count];
                            let mut column = vec![sentinel; if count == 0 { 0 } else { m + 2 }];
                            plan.write_pieces(
                                &input,
                                order,
                                &mut pieces.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(),
                                &mut column,
                            );
                            for (j, piece) in pieces.iter().enumerate() {
                                assert_eq!(
                                    piece[..n].iter().map(canonical).collect::<Vec<_>>(),
                                    h[j * n..(j + 1) * n]
                                );
                                assert_eq!(
                                    bytes_of_slice(&piece[n..]),
                                    bytes_of_slice(&[sentinel; 2])
                                );
                            }
                            if count != 0 {
                                assert_eq!(
                                    bytes_of_slice(&column[m..]),
                                    bytes_of_slice(&[sentinel; 2])
                                );
                            }
                            assert_eq!(bytes_of_slice(&input), original);
                        }
                        for scratch_len in [0, 1, 2, m] {
                            let sentinel = loose(PastaField::ONE);
                            let mut storage = vec![sentinel; m + 1];
                            let factors = plan
                                .prepare_factors(&mut storage, &mut vec![sentinel; scratch_len]);
                            assert_eq!(factors.as_slice().len(), m);
                            assert_eq!(factors.division().piece_size(), n);
                            assert_eq!(factors.division().shift().reduce(), shift.reduce());
                            let mut values: Vec<_> = (0..size)
                                .map(|i| {
                                    evaluations[match order {
                                        ElementOrder::Natural => i,
                                        ElementOrder::BitReversed => reverse(i, log),
                                    }]
                                })
                                .collect();
                            let inversions =
                                count_inversions(|| factors.divide_in_place(&mut values, order));
                            assert_eq!(inversions, 0);
                            for (i, value) in values.iter().enumerate() {
                                let row = match order {
                                    ElementOrder::Natural => i,
                                    ElementOrder::BitReversed => reverse(i, log),
                                };
                                assert_eq!(canonical(value), divided[row]);
                            }
                            assert_eq!(bytes_of_slice(&storage[m..]), bytes_of_slice(&[sentinel]));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn integer_division_and_piece_prefixes() {
    integer_cases::<PallasBase>();
    integer_cases::<PallasScalar>();
}

fn composition<M: PrimeModulus>() {
    for size in [1, 8, 64, 1024] {
        let domain = Domain::<M>::for_size(size).unwrap();
        let n = (size / 4).max(1);
        let division = VanishingDivision::new(domain, &PastaField::<M>::ZETA.reduce(), n).unwrap();
        let h = inputs::<M>(size - n);
        let mut a = vec![PastaField::ZERO; size];
        for (i, value) in h.iter().enumerate() {
            a[i] = a[i].sub(value);
            a[i + n] = a[i + n].add(value);
        }
        let evaluations = reference_coset(&a, domain.coset());
        let tables = Prepared::new(domain.subgroup());
        for table in [None, Some(tables.tables())] {
            let transform = table.map_or_else(
                || Transform::new(domain.subgroup()),
                |tables| tables.bind(domain.subgroup()),
            );
            for input_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                for output_order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                    for tasks in [1, 4] {
                        let options = ExecutionOptions::default()
                            .with_task_budget(crate::exec::TaskBudget::new(tasks).unwrap());
                        let plan = run::FftPlan::new(
                            transform,
                            TransformRequest {
                                input_order,
                                output_order,
                                ..TransformRequest::new(Direction::Forward)
                            },
                            StorageLayout::Contiguous,
                            options,
                        )
                        .unwrap();
                        let mut values: Vec<_> = (0..size)
                            .map(|i| {
                                evaluations[match input_order {
                                    ElementOrder::Natural => i,
                                    ElementOrder::BitReversed => reverse(i, domain.log_size()),
                                }]
                            })
                            .collect();
                        let mut scratch = vec![
                            PastaField::ZERO;
                            plan.retained_fields().max(division.piece_count())
                        ];
                        plan.execute(None, &mut values, None, &mut scratch, &SerialExecutor);
                        let mut output = vec![PastaField::ZERO; size - n];
                        division.write_pieces(
                            &values,
                            output_order,
                            &mut output.chunks_exact_mut(n).collect::<Vec<_>>(),
                            &mut scratch,
                        );
                        assert_eq!(reduced(&output), reduced(&h));
                    }
                }
            }
        }
    }
}

#[test]
fn transform_orders_tables_and_scratch_reuse() {
    composition::<PallasBase>();
    composition::<PallasScalar>();
}

fn contracts<M: PrimeModulus>() {
    let domain = Domain::<M>::new(3).unwrap();
    for n in [0, 3, 16, usize::MAX] {
        assert!(matches!(
            VanishingDivision::new(domain, &PastaField::<M>::ZETA, n),
            Err(FftError::InvalidLayout)
        ));
    }
    for shift in [
        PastaField::ZERO,
        loose(PastaField::ZERO),
        PastaField::ONE,
        loose(domain.root()),
        domain.root().neg(),
    ] {
        assert!(matches!(
            VanishingDivision::new(domain, &shift, 2),
            Err(FftError::InvalidLayout)
        ));
    }
    let division = VanishingDivision::new(domain, &PastaField::<M>::ZETA, 2).unwrap();
    let sentinel = loose(PastaField::<M>::ONE.neg());
    let input = [sentinel; 8];
    for (input_len, count, short, scratch_len) in [
        (7, 1, false, 4),
        (8, 5, false, 4),
        (8, 2, true, 4),
        (8, 2, false, 3),
        (7, 0, false, 0),
    ] {
        let mut pieces = vec![vec![sentinel; 2]; count];
        if short {
            pieces.last_mut().unwrap().pop();
        }
        let before = pieces.clone();
        let mut scratch = vec![sentinel; scratch_len];
        assert!(
            catch_unwind(AssertUnwindSafe(|| division.write_pieces(
                &input[..input_len],
                ElementOrder::Natural,
                &mut pieces.iter_mut().map(Vec::as_mut_slice).collect::<Vec<_>>(),
                &mut scratch
            )))
            .is_err()
        );
        for (a, b) in pieces.iter().zip(&before) {
            assert_eq!(bytes_of_slice(a), bytes_of_slice(b));
        }
        assert_eq!(
            bytes_of_slice(&scratch),
            bytes_of_slice(&vec![sentinel; scratch_len])
        );
    }
    let mut short = [sentinel; 3];
    let mut scratch = [sentinel; 4];
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            division.prepare_factors(&mut short, &mut scratch);
        }))
        .is_err()
    );
    assert_eq!(bytes_of_slice(&short), bytes_of_slice(&[sentinel; 3]));
    assert_eq!(bytes_of_slice(&scratch), bytes_of_slice(&[sentinel; 4]));
    let mut storage = [sentinel; 4];
    let factors = division.prepare_factors(&mut storage, &mut scratch);
    assert!(
        catch_unwind(AssertUnwindSafe(
            || factors.divide_in_place(&mut short, ElementOrder::Natural)
        ))
        .is_err()
    );
    assert_eq!(bytes_of_slice(&short), bytes_of_slice(&[sentinel; 3]));
    let largest = (0..=32)
        .rev()
        .find_map(|log| Domain::<M>::new(log).ok())
        .unwrap();
    let plan = VanishingDivision::new(largest, &PastaField::<M>::ZETA, 1).unwrap();
    assert_eq!(plan.piece_count(), largest.size());
    let plan = VanishingDivision::new(largest, &PastaField::<M>::ZETA, largest.size()).unwrap();
    assert_eq!(plan.piece_count(), 1);
}

#[test]
fn dimensions_and_validation_before_mutation() {
    contracts::<PallasBase>();
    contracts::<PallasScalar>();
}
