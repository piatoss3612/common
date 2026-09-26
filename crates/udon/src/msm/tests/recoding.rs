use super::*;

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
        use crate::curve::pasta::parameters::GlvParameters;
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
