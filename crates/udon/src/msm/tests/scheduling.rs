use super::*;

#[test]
fn grouped_jobs_share_workers_with_side_work_and_reuse_dirty_scratch() {
    let bases = [AffinePoint::<Pallas>::GENERATOR; 700];
    let cached = [PreparedAffinePoint::from_affine(&bases[0]); 700];
    let points = [Point::<Pallas>::IDENTITY; 700];
    let scalars: Vec<_> = field_samples().take(700).collect();
    let indices: Vec<_> = (0..700).map(|i| i as u32 % 13).collect();
    let mut jobs = [
        Input::new(Bases::Affine(&[]), &[]),
        Input::indexed(Bases::Prepared(&cached), &indices[..700], &scalars).unwrap(),
        Input::new(Bases::Affine(&bases[..17]), &scalars[..17]),
        Input::indexed(Bases::Points(&points), &indices[..99], &scalars[..99]).unwrap(),
        Input::new(Bases::Affine(&bases[..257]), &scalars[..257]),
    ];
    let expected = jobs.map(|input| reference(&input));
    let mut records = vec![ScalarStorage::ZERO; scalars.len()];
    let prepared =
        PreparedScalars::prepare(&scalars, &mut records, TaskBudget::SERIAL, &SerialExecutor);
    let mut digits = vec![0; prepared.cache_len_with(ArithmeticOptions::DEFAULT).unwrap()];
    let retained = prepared
        .cache_with(ArithmeticOptions::DEFAULT, &mut digits)
        .unwrap();
    jobs[1] = jobs[1].selection().with_prepared_scalars(retained);
    for workers in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        for tasks in [1, 3, 7, 32, 128] {
            let options = BatchOptions::new(
                ArithmeticOptions::DEFAULT.with_max_terms_per_pass(NonZeroUsize::new(63)),
            )
            .with_task_budget(TaskBudget::new(tasks).unwrap());
            let r = batch_requirements(&jobs, options).unwrap();
            let mut buffers = Buffers::new(r);
            let mut output = [ProjectivePoint::IDENTITY; 5];
            pool.install(|| {
                Pool.join(
                    || execute_batch(&jobs, &mut output, options, &Pool, buffers.borrow()).unwrap(),
                    || (),
                )
            });
            assert_eq!(output, expected);
            buffers.tails(r);
            let (j, w) = BatchPlan::<Pallas>::storage_len_with(jobs.len(), options).unwrap();
            let mut metadata = vec![JobStorage::EMPTY; j + 1];
            let mut ranges = vec![WorkerStorage::EMPTY; w + 1];
            let plan = BatchPlan::new_with(&jobs, options, &mut metadata, &mut ranges).unwrap();
            assert_eq!(plan.requirements(), r);
            // Dirty scratch checks that workers clear unclaimed result slots.
            for _ in 0..2 {
                output.fill(ProjectivePoint::GENERATOR);
                pool.install(|| plan.execute(&mut output, &Pool, buffers.borrow()));
                assert_eq!(output, expected);
                buffers.tails(r);
            }
            let width = JoinWidth(core::sync::atomic::AtomicUsize::new(1));
            let ((), peak) = width.measure(|| plan.execute(&mut output, &width, buffers.borrow()));
            assert_eq!(output, expected);
            assert!(
                peak <= tasks,
                "{peak} concurrent leaves exceed {tasks} tasks"
            );
            assert_eq!(metadata[j], JobStorage::EMPTY);
            assert_eq!(ranges[w], WorkerStorage::EMPTY);
        }
    }
}

#[test]
fn forced_kernels_chunks_and_incompatible_caches() {
    fn check<C: PastaCurve>() {
        let n = 259;
        let scalars: Vec<_> = field_samples::<C::Scalar>().take(n).collect();
        let bases: Vec<_> = (0..n)
            .map(|i| {
                if i % 5 == 0 {
                    Point::IDENTITY
                } else if i % 2 == 0 {
                    AffinePoint::<C>::GENERATOR.neg().to_point()
                } else {
                    AffinePoint::<C>::GENERATOR.to_point()
                }
            })
            .collect();
        let raw = Input::new(Bases::Points(&bases), &scalars);
        let expected = reference(&raw);
        let mut records = vec![ScalarStorage::ZERO; n];
        let prepared =
            PreparedScalars::prepare(&scalars, &mut records, TaskBudget::SERIAL, &SerialExecutor);
        let cached_options = ArithmeticOptions::DEFAULT
            .with_algorithm(Algorithm::Booth {
                width: Some(8),
                accumulation: Accumulation::Auto,
            })
            .unwrap();
        let mut bytes = vec![73; prepared.cache_len_with(cached_options).unwrap() + 1];
        let cached = prepared.cache_with(cached_options, &mut bytes).unwrap();
        let reused = raw.selection().with_prepared_scalars(cached);
        assert_eq!(
            reused
                .requirements_with(BatchOptions::new(cached_options))
                .unwrap()
                .digits(),
            0
        );
        for width in 4..=12 {
            for accumulation in [
                Accumulation::Affine,
                Accumulation::Projective,
                Accumulation::Hybrid,
            ] {
                for (chunk, pass) in [
                    (None, None),
                    (NonZeroUsize::new(67), NonZeroUsize::new(17)),
                    (NonZeroUsize::new(2), NonZeroUsize::new(1)),
                ] {
                    // The widest cap-one cases repeat thousands of empty bucket
                    // collapses; the smaller widths cover that lifetime boundary.
                    if chunk.is_some_and(|c| c.get() == 2) && width > 5 {
                        continue;
                    }
                    let options = BatchOptions::new(
                        ArithmeticOptions::DEFAULT
                            .with_algorithm(Algorithm::Booth {
                                width: Some(width),
                                accumulation,
                            })
                            .unwrap()
                            .with_chunk_size(chunk.unwrap_or(NonZeroUsize::MAX))
                            .with_max_terms_per_pass(pass),
                    )
                    .with_task_budget(TaskBudget::new(3).unwrap());
                    let r = raw.requirements_with(options).unwrap();
                    let mut buffers = Buffers::new(r);
                    assert_eq!(
                        raw.execute_with(options, &Pool, buffers.borrow()).unwrap(),
                        expected
                    );
                    buffers.tails(r);
                    let r = reused.requirements_with(options).unwrap();
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
        assert_eq!(bytes[n * 32], 73);
    }
    check::<Pallas>();
    check::<Vesta>();
}

#[test]
fn memory_ceiling_and_reusable_weighted_plans() {
    type C = Pallas;
    let scalars: Vec<_> = field_samples::<<C as PastaCurve>::Scalar>()
        .take(1025)
        .collect();
    let bases = vec![AffinePoint::<C>::GENERATOR; scalars.len()];
    let sizes = [3, 1025, 0, 17, 65, 2];
    let inputs: Vec<_> = sizes
        .iter()
        .map(|&n| Input::new(Bases::Affine(&bases[..n]), &scalars[..n]))
        .collect();
    let expected: Vec<_> = inputs.iter().map(reference).collect();
    let serial = batch_requirements(&inputs, BatchOptions::default()).unwrap();
    assert_eq!(
        serial.digits(),
        inputs[1]
            .requirements_with(BatchOptions::default())
            .unwrap()
            .digits()
    );
    for tasks in [1, 2, 3, 5, 17] {
        for limit in [8192, 32768, 262144, 8 * 1024 * 1024] {
            let options = BatchOptions::default()
                .with_task_budget(TaskBudget::new(tasks).unwrap())
                .with_memory_limit(limit);
            let r = batch_requirements(&inputs, options).unwrap();
            assert!(r.bytes::<C>().unwrap() <= limit);
            let conservative = Input::<C>::requirements_for_len(1025, options).unwrap();
            assert!(conservative.bytes::<C>().unwrap() <= limit);
            let (j, w) = BatchPlan::<C>::storage_len_with(inputs.len(), options).unwrap();
            let mut jobs = vec![JobStorage::EMPTY; j + 1];
            let mut workers = vec![WorkerStorage::EMPTY; w + 1];
            let plan = BatchPlan::new_with(&inputs, options, &mut jobs, &mut workers).unwrap();
            assert!(plan.temporary_bytes() <= limit);
            assert!(plan.worker_ranges() <= tasks);
            let r = plan.requirements();
            let mut buffers = Buffers::new(r);
            let mut output = vec![ProjectivePoint::GENERATOR; inputs.len()];
            for _ in 0..2 {
                plan.execute(&mut output, &Pool, buffers.borrow());
                assert_eq!(output, expected);
                buffers.tails(r);
            }
            assert_eq!(jobs[j], JobStorage::EMPTY);
            assert_eq!(workers[w], WorkerStorage::EMPTY);
        }
    }
    let options = BatchOptions::default().with_memory_limit(0);
    let mut jobs = [JobStorage::EMPTY; 6];
    let mut workers = [WorkerStorage::EMPTY; 1];
    assert!(matches!(
        BatchPlan::new_with(&inputs, options, &mut jobs, &mut workers),
        Err(CurveError::MemoryLimit { .. })
    ));
    assert!(jobs.iter().all(|j| *j == JobStorage::EMPTY));
    assert!(workers.iter().all(|w| *w == WorkerStorage::EMPTY));
    // A fixed ceiling bounds every buffer even at sizing-only stress lengths.
    for n in [32768, 1 << 20] {
        assert!(
            Input::<C>::requirements_for_len(n, BatchOptions::default().with_memory_limit(32768))
                .unwrap()
                .bytes::<C>()
                .unwrap()
                <= 32768
        );
    }
}

#[test]
fn streaming_buckets_match_complete_chunks_and_reuse() {
    fn check<C: PastaCurve>() {
        let scalars: Vec<_> = field_samples::<C::Scalar>().take(513).collect();
        let bases: Vec<_> = (0..17)
            .map(|i| {
                if i % 5 == 0 {
                    Point::IDENTITY
                } else {
                    AffinePoint::<C>::GENERATOR.to_point()
                }
            })
            .collect();
        let indices: Vec<_> = (0..scalars.len()).map(|i| (i % 17) as u32).collect();
        let raw = Input::indexed(Bases::Points(&bases), &indices, &scalars).unwrap();
        let expected = reference(&raw);
        let mut records = vec![ScalarStorage::ZERO; scalars.len()];
        let prepared =
            PreparedScalars::prepare(&scalars, &mut records, TaskBudget::SERIAL, &SerialExecutor);
        for width in 4..=12 {
            for chunk in [1, 2, 17, 256, 513] {
                let options = BatchOptions::new(
                    ArithmeticOptions::DEFAULT
                        .with_algorithm(Algorithm::StreamingBooth { width: Some(width) })
                        .unwrap()
                        .with_chunk_size(NonZeroUsize::new(chunk).unwrap()),
                );
                for input in [raw, raw.selection().with_prepared_scalars(prepared)] {
                    let r = input.requirements_with(options).unwrap();
                    let mut buffers = Buffers::new(r);
                    for _ in 0..2 {
                        assert_eq!(
                            input
                                .execute_with(options, &SerialExecutor, buffers.borrow())
                                .unwrap(),
                            expected
                        );
                        buffers.tails(r);
                    }
                }
            }
        }
        let o = BatchOptions::new(
            ArithmeticOptions::DEFAULT
                .with_algorithm(Algorithm::StreamingBooth { width: Some(4) })
                .unwrap(),
        )
        .with_memory_limit(32768);
        let r = raw.requirements_with(o).unwrap();
        assert!(r.bytes::<C>().unwrap() <= 32768);
        let mut buffers = Buffers::new(r);
        assert_eq!(
            raw.execute_with(o, &SerialExecutor, buffers.borrow())
                .unwrap(),
            expected
        );
    }
    check::<Pallas>();
    check::<Vesta>();
}
