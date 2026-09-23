use super::*;

#[test]
fn prepared_scalars_validate_before_writes_and_release_originals() {
    type C = Pallas;
    let mut scalars: Vec<_> = field_samples::<<C as PastaCurve>::Scalar>()
        .take(257)
        .collect();
    let g = AffinePoint::<C>::GENERATOR;
    let bases = vec![g; scalars.len()];
    let raw = Input::new(Bases::Affine(&bases), &scalars);
    let expected = reference(&raw);
    let bytes = PreparedScalars::<C>::storage_len(scalars.len()).unwrap();
    let mut storage = vec![ScalarStorage::ZERO; bytes + 1];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = PreparedScalars::<C>::prepare(
                &scalars,
                &mut storage[..bytes - 1],
                TaskBudget::SERIAL,
                &SerialExecutor,
            );
        }))
        .is_err()
    );
    assert!(storage.iter().all(|b| *b == ScalarStorage::ZERO));
    assert_eq!(
        PreparedScalars::<C>::storage_len(usize::MAX),
        Err(CurveError::SizeOverflow)
    );
    let retained =
        PreparedScalars::<C>::prepare(&scalars, &mut storage, TaskBudget::SERIAL, &SerialExecutor);
    scalars.fill(PastaField::ZERO);
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = Input::new_prepared(Bases::Affine(&bases[..1]), retained);
        }))
        .is_err()
    );
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = Input::indexed_prepared(Bases::Affine(&bases), &[], retained);
        }))
        .is_err()
    );
    let mut indices = vec![0; bases.len()];
    indices[13] = bases.len() as u32;
    assert!(matches!(
        Input::indexed_prepared(Bases::Affine(&bases), &indices, retained),
        Err(CurveError::BaseIndexOutOfBounds { position: 13, .. })
    ));
    let negative = vec![g.neg(); bases.len()];
    // Mix ordinary and prepared jobs, sharing one preparation across changing
    // bases and indices. The source scalar vector has already been overwritten.
    indices.fill(0);
    let inputs = [
        Input::new(Bases::Affine(&bases), &scalars),
        Input::new_prepared(Bases::Affine(&bases), retained),
        Input::indexed_prepared(Bases::Affine(&negative[..1]), &indices, retained).unwrap(),
    ];
    let options = BatchOptions::new(
        ArithmeticOptions::DEFAULT.with_max_terms_per_pass(NonZeroUsize::new(17)),
    )
    .with_task_budget(TaskBudget::new(4).unwrap());
    let r = batch_requirements(&inputs, options).unwrap();
    let mut buffers = Buffers::new(r);
    let mut output = [ProjectivePoint::GENERATOR; 3];
    execute_batch(&inputs, &mut output, options, &Pool, buffers.borrow()).unwrap();
    assert_eq!(
        output,
        [ProjectivePoint::IDENTITY, expected, expected.neg()]
    );
    buffers.tails(r);
    assert!(storage[bytes] == ScalarStorage::ZERO);
}

#[test]
fn compact_tables_and_selection_rebind_across_scalar_rows() {
    fn check<C: PastaCurve>() {
        use crate::curve::EisensteinTableBatch;
        let n = 35;
        let bases = vec![AffinePoint::<C>::GENERATOR; n];
        let r = EisensteinTableBatch::<C>::requirements(n).unwrap();
        let mut entries = vec![AffinePoint::GENERATOR; r.table_entries];
        let mut projective = vec![ProjectivePoint::IDENTITY; r.projective_scratch];
        let mut field = vec![PastaField::ZERO; r.field_scratch];
        let tables = EisensteinTableBatch::prepare(
            &bases,
            &mut entries,
            &mut projective,
            &mut field,
            TaskBudget::new(3).unwrap(),
            &Pool,
        );
        let cached_entries: Vec<_> = tables
            .as_slice()
            .iter()
            .map(PreparedAffinePoint::from_affine)
            .collect();
        let cached_tables = EisensteinTableBatch::bind(&cached_entries);
        let cached_bases: Vec<_> = bases.iter().map(PreparedAffinePoint::from_affine).collect();
        let indices: Vec<_> = (0..259).map(|i| (i % 7) as u32).collect();
        for basis in [
            Bases::Prepared(&cached_bases),
            Bases::Compact(tables),
            Bases::CompactPrepared(cached_tables),
        ] {
            let selection = Selection::indexed(basis, &indices).unwrap();
            for row in 0..3 {
                let scalars: Vec<_> = field_samples::<C::Scalar>()
                    .skip(row * indices.len())
                    .take(indices.len())
                    .collect();
                let input = selection.with_scalars(&scalars);
                let expected = reference(&input);
                for options in [
                    BatchOptions::default(),
                    BatchOptions::new(
                        ArithmeticOptions::DEFAULT.with_chunk_size(NonZeroUsize::new(31).unwrap()),
                    ),
                    BatchOptions::default().with_memory_limit(8192),
                ] {
                    let mut buffers = Buffers::new(input.requirements_with(options).unwrap());
                    assert_eq!(
                        input
                            .execute_with(options, &Pool, buffers.borrow())
                            .unwrap(),
                        expected
                    );
                }
            }
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}

#[test]
fn optional_compact_batch_certificate_preserves_fallbacks() {
    use crate::curve::{EisensteinScalar, EisensteinTableBatch};
    fn check<C: PastaCurve>() {
        let bases = [AffinePoint::<C>::GENERATOR; 32];
        let r = EisensteinTableBatch::<C>::requirements(bases.len()).unwrap();
        let mut entries = vec![AffinePoint::GENERATOR; r.table_entries];
        let mut projective = vec![ProjectivePoint::IDENTITY; r.projective_scratch];
        let mut fields = vec![PastaField::ZERO; r.field_scratch];
        let tables = EisensteinTableBatch::prepare(
            &bases,
            &mut entries,
            &mut projective,
            &mut fields,
            TaskBudget::SERIAL,
            &SerialExecutor,
        );
        let mut fields = vec![
            PastaField::ZERO;
            EisensteinTableBatch::<C>::multiplication_scratch(bases.len())
                .unwrap()
        ];
        for scalar in [
            PastaField::<_>::ZERO,
            PastaField::<_>::ONE,
            PastaField::<_>::ONE.neg(),
        ]
        .into_iter()
        .chain(field_samples::<C::Scalar>().take(64))
        {
            let prepared = EisensteinScalar::new(&scalar);
            let certified = prepared;
            assert_eq!(prepared.digits(), certified.digits());
            assert_eq!(prepared.batch_safe(), certified.batch_safe());
            let mut plain = [ProjectivePoint::IDENTITY; 32];
            let mut cached = plain;
            tables.mul_prepared(
                &prepared,
                &mut plain,
                &mut fields,
                TaskBudget::SERIAL,
                &SerialExecutor,
            );
            tables.mul_prepared(
                &certified,
                &mut cached,
                &mut fields,
                TaskBudget::new(3).unwrap(),
                &Pool,
            );
            assert_eq!(plain, cached);
            assert!(
                cached
                    .iter()
                    .all(|p| *p == bases[0].mul_projective(&scalar))
            );
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}

#[test]
fn parallel_cache_matches_serial_and_preserves_tails() {
    use core::sync::atomic::{AtomicUsize, Ordering};

    fn check<C: PastaCurve>() {
        for n in [0, 5, 1023, 1024, 1025] {
            let scalars: Vec<_> = field_samples::<C::Scalar>().take(n).collect();
            let mut records = vec![ScalarStorage::ZERO; n];
            let prepared = PreparedScalars::<C>::prepare(
                &scalars,
                &mut records,
                TaskBudget::SERIAL,
                &SerialExecutor,
            );
            let plan =
                execution::MsmPlan::<C>::new(n, crate::exec::ExecutionOptions::default()).unwrap();
            let len = prepared.cache_len(&plan);
            let mut serial = vec![73; len + 1];
            let expected = prepared
                .cache(&plan, &mut serial, TaskBudget::SERIAL, &SerialExecutor)
                .cached
                .map(|c| c.digits.to_vec());
            for workers in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                let mut bytes = vec![73; len + 1];
                let cached = pool.install(|| {
                    prepared.cache(&plan, &mut bytes, TaskBudget::new(4).unwrap(), &Pool)
                });
                assert!(cached.records == prepared.records);
                assert_eq!(cached.cached.map(|c| c.digits), expected.as_deref());
                assert_eq!(bytes, serial);
            }
            for tasks in [1, 2, 3, 8] {
                let width = JoinWidth(AtomicUsize::new(1));
                let mut bytes = vec![73; len + 1];
                let cached =
                    prepared.cache(&plan, &mut bytes, TaskBudget::new(tasks).unwrap(), &width);
                assert_eq!(cached.cached.map(|c| c.digits), expected.as_deref());
                assert_eq!(bytes, serial);
                assert!(width.0.load(Ordering::Relaxed) <= tasks);
            }
            if len != 0 {
                let width = JoinWidth(AtomicUsize::new(0));
                let mut short = vec![73; len - 1];
                assert!(
                    std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                        let _ =
                            prepared.cache(&plan, &mut short, TaskBudget::new(4).unwrap(), &width);
                    }))
                    .is_err()
                );
                assert_eq!(width.0.load(Ordering::Relaxed), 0);
                assert!(short.iter().all(|&byte| byte == 73));
            }
            assert_eq!(serial[len], 73);
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}
