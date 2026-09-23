use super::*;

fn differentials<C: PastaCurve>() {
    let g = AffinePoint::<C>::GENERATOR;
    let affine: Vec<_> = (1..=1030)
        .map(|i| {
            *g.mul_projective(&PastaField::<_>::from_u64(i))
                .to_point()
                .as_affine()
                .unwrap()
        })
        .collect();
    let prepared: Vec<_> = affine
        .iter()
        .map(PreparedAffinePoint::from_affine)
        .collect();
    let points: Vec<_> = affine
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i % 11 == 0 {
                Point::IDENTITY
            } else if i % 3 == 0 {
                p.neg().to_point()
            } else {
                p.to_point()
            }
        })
        .collect();
    let indices: Vec<_> = (0..1030).map(|i| (i * 13 % 37) as u32).collect();
    let mut full: Vec<_> = field_samples::<C::Scalar>().take(1030).collect();
    for (i, k) in full.iter_mut().enumerate() {
        if i % 19 == 0 {
            *k = PastaField::ZERO;
        }
        if i % 23 == 0 {
            *k = PastaField::<_>::ONE.neg();
        }
    }
    let short: Vec<_> = (0..1030)
        .map(|i| PastaField::from_u64((i * 137) as u64))
        .collect();
    for n in [
        0, 1, 7, 8, 15, 16, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257, 513, 1030,
    ] {
        for scalars in [&full[..n], &short[..n]] {
            let mut storage =
                vec![ScalarStorage::ZERO; PreparedScalars::<C>::storage_len(n).unwrap() + 1];
            let retained = PreparedScalars::<C>::prepare(
                scalars,
                &mut storage,
                TaskBudget::new(3).unwrap(),
                &Pool,
            );
            assert_eq!(retained.len(), n);
            assert_eq!(retained.is_empty(), n == 0);
            for bases in [
                Bases::Affine(&affine[..n]),
                Bases::Prepared(&prepared[..n]),
                Bases::Points(&points[..n]),
            ] {
                let dense = Input::new(bases, scalars);
                let ix: Vec<_> = indices[..n].iter().map(|i| i % n.max(1) as u32).collect();
                let indexed = Input::indexed(bases, &ix, scalars).unwrap();
                for input in [dense, indexed] {
                    let reused = match input.indices {
                        Some(indices) => Input::indexed_prepared(bases, indices, retained),
                        None => Ok(Input::new_prepared(bases, retained)),
                    }
                    .unwrap();
                    let expected = reference(&input);
                    for (tasks, cap) in [
                        (1, None),
                        (3, NonZeroUsize::new(17)),
                        (65, NonZeroUsize::new(67)),
                    ] {
                        let options = BatchOptions::new(
                            ArithmeticOptions::DEFAULT.with_max_terms_per_pass(cap),
                        )
                        .with_task_budget(TaskBudget::new(tasks).unwrap());
                        let r = input.requirements_with(options).unwrap();
                        assert_eq!(r, batch_requirements(&[input], options).unwrap());
                        let mut buffers = Buffers::new(r);
                        assert_eq!(
                            input
                                .execute_with(options, &SerialExecutor, buffers.borrow())
                                .unwrap(),
                            expected,
                            "n={n}, tasks={tasks}"
                        );
                        // Reuse dirty working storage while allowing parallel joins.
                        assert_eq!(
                            input
                                .execute_with(options, &Pool, buffers.borrow())
                                .unwrap(),
                            expected
                        );
                        buffers.tails(r);
                        let r = reused.requirements_with(options).unwrap();
                        assert_eq!(r.scalars, 0);
                        assert_eq!(r, batch_requirements(&[reused], options).unwrap());
                        let mut buffers = Buffers::new(r);
                        assert_eq!(
                            reused
                                .execute_with(options, &Pool, buffers.borrow())
                                .unwrap(),
                            expected
                        );
                        buffers.tails(r);
                    }
                }
            }
            assert!(*storage.last().unwrap() == ScalarStorage::ZERO);
        }
    }
}

#[test]
fn pallas_differentials() {
    differentials::<Pallas>();
}

#[test]
fn vesta_differentials() {
    differentials::<Vesta>();
}

fn scalar_boundaries<C: PastaCurve>() {
    use crate::field::CanonicalUint;

    let bases = [
        Point::<C>::GENERATOR,
        Point::GENERATOR.neg(),
        Point::IDENTITY,
    ];
    for bits in [0, 1, 63, 64, 65, 120, 121, 127, 128, 129, 254] {
        let mut limbs = [0; 4];
        for bit in 0..bits {
            limbs[bit / 64] |= 1 << (bit % 64);
        }
        let scalar = PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs)).unwrap();
        for n in [1, 7, 8, 16, 31, 32, 47, 48, 49, 63, 127, 128, 129] {
            let scalars: Vec<_> = (0..n)
                .map(|i| if i % 5 == 0 { PastaField::ZERO } else { scalar })
                .collect();
            let indices: Vec<_> = (0..n).map(|i| (i % bases.len()) as u32).collect();
            let input = Input::indexed(Bases::Points(&bases), &indices, &scalars).unwrap();
            let expected = reference(&input);
            let mut storage =
                vec![ScalarStorage::ZERO; PreparedScalars::<C>::storage_len(n).unwrap()];
            let retained = PreparedScalars::<C>::prepare(
                &scalars,
                &mut storage,
                TaskBudget::SERIAL,
                &SerialExecutor,
            );
            let reused =
                Input::indexed_prepared(Bases::Points(&bases), &indices, retained).unwrap();
            for tasks in [1, 4] {
                for cap in [1, 7, 8, 17, n] {
                    let options = BatchOptions::new(
                        ArithmeticOptions::DEFAULT.with_max_terms_per_pass(NonZeroUsize::new(cap)),
                    )
                    .with_task_budget(TaskBudget::new(tasks).unwrap());
                    let r = input.requirements_with(options).unwrap();
                    let mut buffers = Buffers::new(r);
                    assert_eq!(
                        input
                            .execute_with(options, &SerialExecutor, buffers.borrow())
                            .unwrap(),
                        expected,
                        "bits={bits}, n={n}, tasks={tasks}, cap={cap}"
                    );
                    buffers.tails(r);
                    assert_eq!(
                        reused
                            .execute_with(options, &SerialExecutor, buffers.borrow())
                            .unwrap(),
                        expected,
                        "prepared bits={bits}, n={n}, tasks={tasks}, cap={cap}"
                    );
                    buffers.tails(r);
                }
            }
        }
    }
}

#[test]
fn short_scalar_dispatch_and_highest_windows() {
    scalar_boundaries::<Pallas>();
    scalar_boundaries::<Vesta>();
}

#[test]
fn dense_and_sparse_bounded_rows_cross_chunk_policies() {
    fn check<C: PastaCurve>() {
        let bases = [
            Point::<C>::GENERATOR,
            Point::GENERATOR.neg(),
            Point::IDENTITY,
        ];
        for n in [511, 512, 513] {
            for bits in [32, 64, 128] {
                for dense in [false, true] {
                    let magnitude = if dense {
                        u128::MAX >> (128 - bits)
                    } else {
                        1_u128 << (bits - 1)
                    };
                    let scalar =
                        PastaField::from_canonical_uint(crate::field::CanonicalUint::from_limbs([
                            magnitude as u64,
                            (magnitude >> 64) as u64,
                            0,
                            0,
                        ]))
                        .unwrap();
                    let scalars: Vec<_> = (0..n)
                        .map(|i| if i % 2 == 0 { scalar } else { scalar.neg() })
                        .collect();
                    let indices: Vec<_> = (0..n).map(|i| (i % 3) as u32).collect();
                    let raw = Input::indexed(Bases::Points(&bases), &indices, &scalars).unwrap();
                    let expected = reference(&raw);
                    let mut records = vec![ScalarStorage::ZERO; n];
                    let prepared = PreparedScalars::prepare(
                        &scalars,
                        &mut records,
                        TaskBudget::SERIAL,
                        &SerialExecutor,
                    );
                    let cache_options = ArithmeticOptions::DEFAULT;
                    let mut digits = vec![73; prepared.cache_len_with(cache_options).unwrap()];
                    let cached = prepared.cache_with(cache_options, &mut digits).unwrap();
                    let reused = raw.selection().with_prepared_scalars(cached);
                    for chunk in [n, 257] {
                        let options = BatchOptions::new(
                            ArithmeticOptions::DEFAULT
                                .with_chunk_size(NonZeroUsize::new(chunk).unwrap()),
                        )
                        .with_task_budget(TaskBudget::new(3).unwrap());
                        for input in [raw, reused] {
                            let r = input.requirements_with(options).unwrap();
                            let mut buffers = Buffers::new(r);
                            assert_eq!(
                                input
                                    .execute_with(options, &Pool, buffers.borrow())
                                    .unwrap(),
                                expected
                            );
                            buffers.tails(r);
                        }
                    }
                }
            }
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}

fn collisions<C: PastaCurve>() {
    for n in [8, 31, 32, 127, 128, 129, 255, 256, 257] {
        let g = Point::<C>::GENERATOR;
        let bases: Vec<_> = (0..n)
            .map(|i| match i % 4 {
                0 | 1 => g,
                2 => g.neg(),
                _ => Point::IDENTITY,
            })
            .collect();
        for value in [
            PastaField::ZERO,
            PastaField::ONE,
            PastaField::from_u64(128),
            PastaField::<_>::ONE.neg(),
        ] {
            let scalars = vec![value; n];
            let input = Input::new(Bases::Points(&bases), &scalars);
            let expected = reference(&input);
            for cap in [1, 2, 3, 37, n] {
                let options = BatchOptions::new(
                    ArithmeticOptions::DEFAULT.with_max_terms_per_pass(NonZeroUsize::new(cap)),
                );
                let mut buffers = Buffers::new(input.requirements_with(options).unwrap());
                assert_eq!(
                    input
                        .execute_with(options, &SerialExecutor, buffers.borrow())
                        .unwrap(),
                    expected
                );
            }
        }
    }
}

#[test]
fn doubling_cancellation_identity_and_pass_survivors() {
    collisions::<Pallas>();
    collisions::<Vesta>();
}

/// The trait entry point runs the planned kernel over bounded stack scratch.
/// It must agree with the retained ladder across the streaming geometry,
/// with identity bases and zero scalars mixed in, for both curves.
fn affine_trait_msm<C: PastaCurve>() {
    use crate::curve::Affine;

    let generator = Point::<C>::GENERATOR;
    for size in [
        0usize, 1, 2, 5, 15, 16, 17, 63, 64, 65, 100, 255, 256, 257, 600, 1030,
    ] {
        let scalars: Vec<PastaField<C::Scalar>> = (0..size)
            .map(|index| match index % 7 {
                0 => PastaField::ZERO,
                1 => PastaField::ONE,
                2 => PastaField::<C::Scalar>::ONE.neg(),
                _ => PastaField::<C::Scalar>::from_u64(index as u64 + 3)
                    .mul(&PastaField::<C::Scalar>::DELTA),
            })
            .collect();
        let bases: Vec<Point<C>> = (0..size)
            .map(|index| {
                if index % 11 == 4 {
                    Point::IDENTITY
                } else {
                    generator
                        .mul_projective(&PastaField::<C::Scalar>::from_u64(index as u64 + 1))
                        .to_point()
                }
            })
            .collect();
        let mut expected = ProjectivePoint::IDENTITY;
        for (scalar, base) in scalars.iter().zip(&bases) {
            let base = base.to_projective();
            expected = expected.add(&test_reference::multiply(scalar, |sum| sum.add(&base)));
        }
        assert_eq!(
            <Point<C> as Affine>::msm(&scalars, &bases),
            expected,
            "{size} terms"
        );
    }
}

#[test]
fn affine_trait_msm_matches_the_retained_ladder() {
    affine_trait_msm::<Pallas>();
    affine_trait_msm::<Vesta>();
}
