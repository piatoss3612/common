use super::*;
use crate::field::{ReductionState, batch_invert_groups, count_inversions};
use crate::test_support::{integer, modulus};
use num_bigint::BigUint;

fn from_raw<M: PrimeModulus>(raw: &BigUint) -> PastaField<M> {
    let digits = raw.to_u64_digits();
    let mut limbs = [0; 4];
    limbs[..digits.len()].copy_from_slice(&digits);
    PastaField::from_montgomery_limbs(limbs)
}

fn canonical<M: PrimeModulus, S: ReductionState>(value: &PastaField<M, S>) -> BigUint {
    let p = modulus::<M>();
    let raw = integer(&value.montgomery_limbs());
    assert!(raw < &p * 2u8);
    raw * (BigUint::from(1u8) << 256usize).modinv(&p).unwrap() % p
}

fn nodes<M: PrimeModulus>(domain: CosetDomain<M>) -> Vec<BigUint> {
    let p = modulus::<M>();
    let root = canonical(&domain.domain().root());
    let mut value = canonical(&domain.shift());
    (0..domain.size())
        .map(|_| {
            let result = value.clone();
            value = &value * &root % &p;
            result
        })
        .collect()
}

fn basis<M: PrimeModulus>(domain: CosetDomain<M>, point: &PastaField<M>) -> Vec<BigUint> {
    let p = modulus::<M>();
    let x = canonical(point);
    let nodes = nodes(domain);
    // Evaluate each defining product of linear factors. This uses neither
    // the root-of-unity quotient identity nor the production inversion engine.
    nodes
        .iter()
        .enumerate()
        .map(|(i, node)| {
            let mut numerator = BigUint::from(1u8);
            let mut denominator = BigUint::from(1u8);
            for (j, other) in nodes.iter().enumerate() {
                if i != j {
                    numerator = numerator * (&x + &p - other) % &p;
                    denominator = denominator * (node + &p - other) % &p;
                }
            }
            numerator * denominator.modinv(&p).unwrap() % &p
        })
        .collect()
}

fn assert_values<M: PrimeModulus>(actual: &[PastaField<M>], expected: &[BigUint]) {
    assert_eq!(actual.len(), expected.len());
    for (actual, expected) in actual.iter().zip(expected) {
        assert_eq!(&canonical(actual), expected);
    }
}

fn ranges_field<M: PrimeModulus>() {
    let p = modulus::<M>();
    let sentinel = from_raw::<M>(&(&p * 2u8 - 1u8));
    for log in 0..=5 {
        let subgroup = Domain::<M>::new(log).unwrap();
        let n = subgroup.size();
        for domain in [subgroup.subgroup(), subgroup.coset()] {
            let mut points: Vec<_> = field_samples::<M>().take(3).collect();
            points.extend(
                [
                    BigUint::from(0u8),
                    BigUint::from(1u8),
                    &p - 1u8,
                    p.clone(),
                    &p + 1u8,
                    &p * 2u8 - 1u8,
                ]
                .map(|raw| from_raw(&raw)),
            );
            let mut node = domain.shift();
            for _ in 0..n {
                points.push(from_raw(&(integer(&node.reduce().montgomery_limbs()) + &p)));
                node = node.mul(&subgroup.root());
            }
            let mut ranges = vec![0..0, n..n, 0..n, 0..1, n - 1..n, n / 2..n];
            if n > 2 {
                ranges.push(1..n - 1);
            }
            for point in points {
                let expected = basis(domain, &point);
                assert_eq!(expected.iter().sum::<BigUint>() % &p, BigUint::from(1u8));
                let node_hit = nodes(domain).contains(&canonical(&point));
                for range in &ranges {
                    let count = range.len();
                    let mut split = vec![sentinel; count + 2];
                    let completion = domain
                        .prepare_lagrange(&point, range.clone(), &mut split)
                        .unwrap();
                    if node_hit || n == 1 {
                        assert!(split[..count].iter().all(PastaField::is_zero));
                    }
                    let mut prefix = vec![sentinel; count];
                    let inversions = count_inversions(|| {
                        batch_invert_groups(&mut [&mut split[..count]], &mut prefix);
                    });
                    assert_eq!(inversions, usize::from(count != 0 && !node_hit && n != 1));
                    completion.complete(&mut split).unwrap();
                    assert_values(&split[..count], &expected[range.clone()]);
                    assert!(
                        split[count..]
                            .iter()
                            .all(|x| { x.montgomery_limbs() == sentinel.montgomery_limbs() })
                    );
                    for scratch_len in [0, 1, 2, 3, count, count + 3] {
                        let mut output = vec![sentinel; count + 2];
                        let mut scratch = vec![sentinel; scratch_len];
                        let inversions = count_inversions(|| {
                            domain
                                .evaluate_lagrange(&point, range.clone(), &mut output, &mut scratch)
                                .unwrap();
                        });
                        let expected_inversions = if node_hit || n == 1 {
                            0
                        } else {
                            count.div_ceil(scratch_len.max(1))
                        };
                        assert_eq!(inversions, expected_inversions);
                        assert_values(&output[..count], &expected[range.clone()]);
                        let scratch_used = if node_hit || n == 1 { 0 } else { count };
                        assert!(
                            output[count..]
                                .iter()
                                .chain(scratch.iter().skip(scratch_used))
                                .all(|x| x.montgomery_limbs() == sentinel.montgomery_limbs())
                        );
                        // Reduced query points have the same public contract.
                        domain
                            .evaluate_lagrange(
                                &point.reduce(),
                                range.clone(),
                                &mut output,
                                &mut scratch,
                            )
                            .unwrap();
                        assert_values(&output[..count], &expected[range.clone()]);
                    }
                }
            }
        }
    }
}

#[test]
fn lagrange_ranges_match_integer_basis_products() {
    ranges_field::<PallasBase>();
    ranges_field::<PallasScalar>();
}

fn grouped_field<M: PrimeModulus>() {
    let subgroup = Domain::<M>::new(4).unwrap();
    let domains = [subgroup.subgroup(), subgroup.coset(), subgroup.coset()];
    let points = [
        PastaField::from_u64(7),
        PastaField::ZERO,
        domains[2].shift().mul(&subgroup.root().pow_u64(5)),
    ];
    let ranges = [2..9, 8..16, 4..7];
    let expected = core::array::from_fn::<_, 3, _>(|i| basis(domains[i], &points[i]));
    let sentinel = from_raw::<M>(&(modulus::<M>() * 2u8 - 1u8));
    for scratch_len in [0, 1, 3, 10, 22, 25] {
        let mut values = ranges.each_ref().map(|r| vec![sentinel; r.len() + 2]);
        let completions = core::array::from_fn::<_, 3, _>(|i| {
            domains[i]
                .prepare_lagrange(&points[i], ranges[i].clone(), &mut values[i])
                .unwrap()
        });
        for i in 0..3 {
            assert_eq!(completions[i].value_count(), ranges[i].len());
            let p = modulus::<M>();
            let domain_nodes = nodes(domains[i]);
            for (value, node) in values[i].iter().zip(&domain_nodes[ranges[i].clone()]) {
                let expected = if i == 2 {
                    BigUint::from(0u8)
                } else {
                    (canonical(&points[i]) * node.modinv(&p).unwrap() + &p - 1u8) % &p
                };
                assert_eq!(canonical(value), expected);
            }
        }
        let mut extra = [PastaField::from_u64(11), PastaField::ZERO];
        let mut scratch = vec![sentinel; scratch_len];
        let [a, b, c] = &mut values;
        let inversions = count_inversions(|| {
            batch_invert_groups(
                &mut [&mut a[..7], &mut b[..8], &mut c[..3], &mut extra],
                &mut scratch,
            );
        });
        if scratch_len >= 20 {
            assert_eq!(inversions, 1);
            assert!(
                scratch[20..]
                    .iter()
                    .all(|x| x.montgomery_limbs() == sentinel.montgomery_limbs())
            );
        }
        for i in 0..3 {
            assert_eq!(
                count_inversions(|| completions[i].complete(&mut values[i]).unwrap()),
                0
            );
            assert_values(
                &values[i][..ranges[i].len()],
                &expected[i][ranges[i].clone()],
            );
            assert!(
                values[i][ranges[i].len()..]
                    .iter()
                    .all(|x| { x.montgomery_limbs() == sentinel.montgomery_limbs() })
            );
        }
        let p = modulus::<M>();
        assert_eq!(
            canonical(&extra[0]),
            BigUint::from(11u8).modinv(&p).unwrap()
        );
        assert!(extra[1].is_zero());
    }
}

#[test]
fn lagrange_completion_shares_unscaled_inversion_batches() {
    grouped_field::<PallasBase>();
    grouped_field::<PallasScalar>();
}

fn errors_field<M: PrimeModulus>() {
    let domain = Domain::<M>::new(3).unwrap().coset();
    let original = [from_raw::<M>(&(modulus::<M>() * 2u8 - 1u8)); 12];
    for range in [
        core::ops::Range { start: 2, end: 1 },
        0..9,
        9..9,
        usize::MAX..usize::MAX,
        0..usize::MAX,
    ] {
        let mut values = original;
        let mut scratch = original;
        let expected = Err(LagrangeError::InvalidRange {
            start: range.start,
            end: range.end,
            size: 8,
        });
        assert_eq!(
            domain.evaluate_lagrange(
                &PastaField::<M>::ONE,
                range.clone(),
                &mut values,
                &mut scratch
            ),
            expected
        );
        assert_eq!(
            domain
                .prepare_lagrange(&PastaField::<M>::ONE, range, &mut values)
                .map(|_| ()),
            expected
        );
        assert_eq!(
            values.map(|value| value.montgomery_limbs()),
            original.map(|value| value.montgomery_limbs())
        );
        assert_eq!(
            scratch.map(|value| value.montgomery_limbs()),
            original.map(|value| value.montgomery_limbs())
        );
    }
    for point in [PastaField::ZERO, domain.shift()] {
        let mut values = original;
        let mut scratch = original;
        let expected = Err(LagrangeError::BufferTooShort {
            required: 5,
            actual: 4,
        });
        assert_eq!(
            domain.evaluate_lagrange(&point, 2..7, &mut values[..4], &mut scratch),
            expected
        );
        assert_eq!(
            domain
                .prepare_lagrange(&point, 2..7, &mut values[..4])
                .map(|_| ()),
            expected
        );
        assert_eq!(
            values.map(|value| value.montgomery_limbs()),
            original.map(|value| value.montgomery_limbs())
        );
        assert_eq!(
            scratch.map(|value| value.montgomery_limbs()),
            original.map(|value| value.montgomery_limbs())
        );
        let completion = domain.prepare_lagrange(&point, 2..7, &mut values).unwrap();
        let prepared = values;
        assert_eq!(completion.complete(&mut values[..4]), expected);
        assert_eq!(
            values.map(|value| value.montgomery_limbs()),
            prepared.map(|value| value.montgomery_limbs())
        );
    }
    // Exercise the largest addressable domain without allocating its full size.
    let subgroup = (0..=32)
        .rev()
        .find_map(|log| Domain::<M>::new(log).ok())
        .unwrap();
    let n = subgroup.size();
    for domain in [subgroup.subgroup(), subgroup.coset()] {
        let mut values = [PastaField::ZERO; 2];
        let node = domain.shift().mul(&subgroup.inverse_root());
        domain
            .evaluate_lagrange(&node, n - 2..n, &mut values, &mut [])
            .unwrap();
        assert!(values[0].is_zero());
        assert!(values[1].is_one());
        domain
            .evaluate_lagrange(&PastaField::<M>::ZERO, n - 2..n, &mut values, &mut [])
            .unwrap();
        let p = modulus::<M>();
        let inverse_size = BigUint::from(n).modinv(&p).unwrap();
        assert_values(&values, &[inverse_size.clone(), inverse_size]);
    }
}

#[test]
fn lagrange_validates_before_writes_and_bounds_large_ranges() {
    errors_field::<PallasBase>();
    errors_field::<PallasScalar>();
}
