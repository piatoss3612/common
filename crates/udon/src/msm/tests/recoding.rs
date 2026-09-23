use super::*;

#[test]
fn packed_midpoint_carries_reconstruct_signed_extremes() {
    use crate::curve::pasta::scalar::centered_digit;
    use num_bigint::BigInt;
    for width in [2, 4, 8] {
        for negative in [false, true] {
            for value in [0, 1, 127, 128, 129, 255, 256, u128::MAX, i128::MAX as u128] {
                let mut carry = 0;
                let mut magnitude = value;
                let mut digits = Vec::new();
                for _ in 0..128 / width {
                    let digit = centered_digit(
                        (magnitude & ((1 << width) - 1)) as u16,
                        negative,
                        &mut carry,
                        width,
                    );
                    assert!((-(1 << (width - 1))..1 << (width - 1)).contains(&digit));
                    assert_eq!(digit as i8 as i16, digit);
                    digits.push(digit);
                    magnitude >>= width;
                }
                let mut actual = BigInt::from(if negative { -carry } else { carry });
                for d in digits.into_iter().rev() {
                    actual = (actual << width) + d;
                }
                let expected = BigInt::from(value);
                assert_eq!(actual, if negative { -expected } else { expected });
            }
        }
    }
}

#[test]
fn affine_reducer_and_weighted_collapse_match_biguint() {
    fn check<C: PastaCurve>() {
        use crate::{curve::pasta::test_reference::Reference, field::pasta::test_support::modulus};
        use num_bigint::BigUint;
        let modulus = modulus::<C::Base>();
        let g = AffinePoint::<C>::GENERATOR;
        let pool: Vec<_> = (1..=13)
            .map(|i| {
                *g.mul_projective(&PastaField::<_>::from_u64(i))
                    .to_point()
                    .as_affine()
                    .unwrap()
            })
            .collect();
        for case in 0..96 {
            let mut points = Vec::new();
            let mut starts = Vec::new();
            let mut lens = Vec::new();
            let mut expected = Vec::new();
            for bucket in 0..7 {
                starts.push(points.len());
                let n = (case * 7 + bucket * 3) % 19;
                lens.push(n);
                let mut sum = Reference::identity();
                for i in 0..n {
                    // Includes all-cancelling levels, odd survivors, and equal
                    // operands alongside distinct points.
                    let mut p = pool[if case % 3 == 0 {
                        bucket
                    } else {
                        (case + i / 2) % pool.len()
                    }];
                    if case % 2 == 0 && i % 2 == 1 {
                        p = p.neg();
                    }
                    sum = sum.add(&Reference::from_point(&p.to_point()), &modulus);
                    points.push(p);
                }
                expected.push(sum);
            }
            let mut control = points.clone();
            let mut control_lens = lens.clone();
            let pairs = points.len() / 2;
            let mut fused = points.clone();
            let mut fused_lens = lens.clone();
            let mut fields = vec![PastaField::ONE; pairs * 2 + 3];
            while fused_lens.iter().any(|&n| n > 1) {
                buckets::reduce_fused::<C, false>(
                    &mut fused,
                    &starts,
                    &mut fused_lens,
                    &mut fields[..pairs * 2],
                );
            }
            buckets::reduce_original(
                &mut control,
                &starts,
                &mut control_lens,
                &mut vec![PastaField::ZERO; pairs * 6],
                &mut vec![0; pairs],
            );
            buckets::reduce(
                &mut points,
                &starts,
                &mut lens,
                &mut vec![PastaField::ONE; pairs * 2],
            );
            assert_eq!(lens, control_lens);
            assert_eq!(lens, fused_lens);
            assert!(
                fields[pairs * 2..]
                    .iter()
                    .all(|x| x.reduce() == PastaField::ONE)
            );
            let mut survivors = vec![g; starts.len()];
            let mut weighted = Reference::identity();
            for i in 0..starts.len() {
                let result = if lens[i] == 0 {
                    Point::IDENTITY
                } else {
                    survivors[i] = points[starts[i]];
                    assert_eq!(points[starts[i]], control[starts[i]]);
                    assert_eq!(points[starts[i]], fused[starts[i]]);
                    points[starts[i]].to_point()
                };
                expected[i].assert_point(&result);
                weighted =
                    weighted.add(&expected[i].mul(&BigUint::from(i + 1), &modulus), &modulus);
            }
            weighted.assert_point(&buckets::collapse(&survivors, &lens).to_point());
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}

#[test]
fn production_booth_rows_reconstruct_both_glv_halves() {
    fn check<C: PastaCurve>() {
        use num_bigint::BigInt;
        for tail in 1..=recode::CHUNK {
            let n = recode::CHUNK + tail;
            let scalars: Vec<_> = field_samples::<C::Scalar>().take(n).collect();
            let mut storage = vec![ScalarStorage::<C>::ZERO; n];
            let retained = PreparedScalars::prepare(
                &scalars,
                &mut storage,
                TaskBudget::SERIAL,
                &SerialExecutor,
            );
            for width in 4..=12 {
                let geometry = recode::Geometry::Booth(width);
                let mut digits = vec![73; geometry.storage_len(n).unwrap()];
                recode::write(retained.records, geometry, &mut digits);
                let mut values = vec![[BigInt::from(0), BigInt::from(0)]; n];
                for window in (0..geometry.windows()).rev() {
                    recode::rows(&digits, n, 0..n, geometry, window, |term, a, b| {
                        values[term][0] = (&values[term][0] << width) + BigInt::from(a);
                        values[term][1] = (&values[term][1] << width) + BigInt::from(b);
                    });
                }
                for (term, record) in retained.records.iter().enumerate() {
                    assert_eq!(values[term], record.halves.map(BigInt::from));
                }
            }
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}

#[test]
fn production_booth_bounds_and_partial_row_visits() {
    fn check<C: PastaCurve>() {
        use crate::curve::parameters::GlvParameters;
        use num_bigint::BigInt;
        let mut records = vec![ScalarStorage::<C>::ZERO; 2049];
        for (i, record) in records.iter_mut().enumerate() {
            for (half, bound) in GlvParameters::<C>::BOUNDS.into_iter().enumerate() {
                let magnitude = match i % 6 {
                    0 => bound,
                    1 => bound - 1,
                    2 => 0,
                    3 => 1,
                    4 => 128,
                    _ => 255,
                };
                record.halves[half] = if (i / 6 + half) % 2 == 0 {
                    magnitude as i128
                } else {
                    -(magnitude as i128)
                };
            }
        }
        for width in 4..=12 {
            let geometry = recode::Geometry::Booth(width);
            let mut digits = vec![73; geometry.storage_len(records.len()).unwrap() + 1];
            recode::write_parallel(
                &records,
                geometry,
                &mut digits[..geometry.storage_len(records.len()).unwrap()],
                TaskBudget::new(7).unwrap(),
                &Pool,
            );
            assert_eq!(*digits.last().unwrap(), 73);
            for range in [0..2049, 1..255, 255..257, 256..513, 511..2049, 2049..2049] {
                // The midpoint conventions can yield different digit sequences;
                // reconstruct integers to compare their mathematical meaning.
                for direct in [false, true] {
                    let mut values = vec![[BigInt::from(0), BigInt::from(0)]; records.len()];
                    let mut visits = vec![0; records.len()];
                    for window in (0..geometry.windows()).rev() {
                        let visit = |i: usize, a: i16, b: i16| {
                            visits[i] += 1;
                            values[i][0] = (&values[i][0] << width) + a;
                            values[i][1] = (&values[i][1] << width) + b;
                        };
                        if direct {
                            recode::window_rows::<C, true>(
                                &records,
                                &[],
                                range.clone(),
                                geometry,
                                window,
                                visit,
                            );
                        } else {
                            recode::rows(
                                &digits,
                                records.len(),
                                range.clone(),
                                geometry,
                                window,
                                visit,
                            );
                        }
                    }
                    for (i, record) in records.iter().enumerate() {
                        assert_eq!(
                            visits[i],
                            if range.contains(&i) {
                                geometry.windows()
                            } else {
                                0
                            }
                        );
                        assert_eq!(
                            values[i],
                            if range.contains(&i) {
                                record.halves.map(BigInt::from)
                            } else {
                                [BigInt::from(0), BigInt::from(0)]
                            }
                        );
                    }
                }
            }
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}
