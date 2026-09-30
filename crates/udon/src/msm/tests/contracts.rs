use super::*;

#[test]
fn scratch_can_be_reused_after_executor_unwind() {
    use std::{
        panic::{AssertUnwindSafe, catch_unwind},
        sync::atomic::{AtomicUsize, Ordering},
    };
    struct Panics {
        calls: AtomicUsize,
        at: usize,
    }
    impl Executor for Panics {
        fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
        where
            L: FnOnce() -> A + Send,
            R: FnOnce() -> B + Send,
            A: Send,
            B: Send,
        {
            let fail = self.calls.fetch_add(1, Ordering::SeqCst) == self.at;
            SerialExecutor.join(
                || {
                    let value = left();
                    assert!(!fail, "injected executor failure");
                    value
                },
                right,
            )
        }
    }
    let bases = [AffinePoint::<Pallas>::GENERATOR; 700];
    let scalars: Vec<_> = field_samples().take(bases.len()).collect();
    let input = Input::new(Bases::Affine(&bases), &scalars);
    let expected = reference(&input);
    let options = BatchOptions::new(
        ArithmeticOptions::DEFAULT.with_max_terms_per_pass(NonZeroUsize::new(37)),
    )
    .with_task_budget(TaskBudget::new(4).unwrap());
    let r = input.requirements_with(options).unwrap();
    let mut buffers = Buffers::new(r);
    // The first two joins prepare three scalar-record chunks; later joins evaluate
    // windows. Exercise an unwind after writes in either scoped phase.
    for at in [0, 2] {
        let executor = Panics {
            calls: AtomicUsize::new(0),
            at,
        };
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                input
                    .execute_with(options, &executor, buffers.borrow())
                    .unwrap();
            }))
            .is_err()
        );
        buffers.tails(r);
        assert_eq!(
            input
                .execute_with(options, &Pool, buffers.borrow())
                .unwrap(),
            expected
        );
        buffers.tails(r);
    }
    let inputs = [input, input];
    let (j, w) = BatchPlan::<Pallas>::storage_len_with(inputs.len(), options).unwrap();
    let mut jobs = vec![JobStorage::EMPTY; j];
    let mut workers = vec![WorkerStorage::EMPTY; w];
    let plan = BatchPlan::new_with(&inputs, options, &mut jobs, &mut workers).unwrap();
    let mut planned_buffers = Buffers::new(plan.requirements());
    let mut output = [ProjectivePoint::IDENTITY; 2];
    for at in [0, 2] {
        let executor = Panics {
            calls: AtomicUsize::new(0),
            at,
        };
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                plan.execute(&mut output, &executor, planned_buffers.borrow());
            }))
            .is_err()
        );
        plan.execute(&mut output, &Pool, planned_buffers.borrow());
        assert_eq!(output, [expected; 2]);
        planned_buffers.tails(plan.requirements());
    }
    let mut storage =
        vec![ScalarStorage::ZERO; PreparedScalars::<Pallas>::storage_len(bases.len()).unwrap() + 1];
    let executor = Panics {
        calls: AtomicUsize::new(0),
        at: 0,
    };
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            PreparedScalars::<Pallas>::prepare(
                &scalars,
                &mut storage,
                options.task_budget,
                &executor,
            );
        }))
        .is_err()
    );
    assert!(*storage.last().unwrap() == ScalarStorage::ZERO);
    let retained =
        PreparedScalars::<Pallas>::prepare(&scalars, &mut storage, options.task_budget, &Pool);
    let reused = Input::new_prepared(Bases::Affine(&bases), retained);
    assert_eq!(
        reused
            .execute_with(options, &Pool, buffers.borrow())
            .unwrap(),
        expected
    );
    buffers.tails(r);
    assert!(*storage.last().unwrap() == ScalarStorage::ZERO);

    let scalars: Vec<_> = field_samples().take(1025).collect();
    let mut records = vec![ScalarStorage::<Pallas>::ZERO; scalars.len()];
    let prepared =
        PreparedScalars::prepare(&scalars, &mut records, TaskBudget::SERIAL, &SerialExecutor);
    let plan =
        execution::MsmPlan::new(scalars.len(), crate::exec::ExecutionOptions::DEFAULT).unwrap();
    let len = prepared.cache_len(&plan);
    let mut serial = vec![73; len + 1];
    let expected = prepared
        .cache(&plan, &mut serial, TaskBudget::SERIAL, &SerialExecutor)
        .cached
        .unwrap()
        .digits
        .to_vec();
    let mut bytes = vec![73; len + 1];
    for at in [0, 2] {
        let executor = Panics {
            calls: AtomicUsize::new(0),
            at,
        };
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                let _ = prepared.cache(&plan, &mut bytes, options.task_budget, &executor);
            }))
            .is_err()
        );
        assert_eq!(bytes[len], 73);
        let cached = prepared.cache(&plan, &mut bytes, options.task_budget, &Pool);
        assert_eq!(cached.cached.unwrap().digits, expected);
        assert_eq!(bytes, serial);
    }
}

#[test]
fn validation_precedes_writes_and_sizing_rejects_overflow() {
    type C = Pallas;
    let bases = [AffinePoint::<C>::GENERATOR; 256];
    let scalars = [PastaField::ONE; 256];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = Input::new(Bases::Affine(&bases), &scalars[..255]);
        }))
        .is_err()
    );
    assert!(matches!(
        Input::indexed(Bases::Affine(&bases), &[256], &scalars[..1]),
        Err(CurveError::BaseIndexOutOfBounds {
            position: 0,
            index: 256,
            bases: 256
        })
    ));
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = Input::indexed(Bases::Affine(&bases), &[], &scalars[..1]);
        }))
        .is_err()
    );
    for n in [usize::MAX, isize::MAX as usize, usize::MAX / 64] {
        assert_eq!(
            Input::<C>::requirements_for_len(n, BatchOptions::default()),
            Err(CurveError::SizeOverflow)
        );
    }
    const R: Requirements = match Input::<C>::requirements_for_len(
        256,
        BatchOptions::new(ArithmeticOptions::DEFAULT),
    ) {
        Ok(r) => r,
        Err(_) => panic!("valid length"),
    };
    let input = Input::new(Bases::Affine(&bases), &scalars);
    assert_eq!(input.requirements_with(BatchOptions::default()).unwrap(), R);
    for short in 0..7 {
        let mut b = Buffers::<C>::new(R);
        let mut scratch = b.borrow();
        match short {
            0 => scratch.digits = &mut scratch.digits[..R.digits - 1],
            1 => scratch.affine = &mut scratch.affine[..R.affine - 1],
            2 => scratch.projective = &mut scratch.projective[..R.projective - 1],
            3 => scratch.field = &mut scratch.field[..R.field - 1],
            4 => scratch.indices = &mut scratch.indices[..R.indices - 1],
            5 => scratch.scalars = &mut scratch.scalars[..R.scalars - 1],
            _ => (),
        }
        let mut output = [ProjectivePoint::GENERATOR; 2];
        let len = if short == 6 { 2 } else { 1 };
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = execute_batch(
                    &[input],
                    &mut output[..len],
                    BatchOptions::default(),
                    &SerialExecutor,
                    scratch,
                );
            }))
            .is_err()
        );
        assert_eq!(output, [ProjectivePoint::GENERATOR; 2]);
        assert!(b.digits.iter().all(|&x| x == 73));
        assert!(b.affine.iter().all(|&x| x == AffinePoint::GENERATOR));
        assert!(
            b.projective
                .iter()
                .all(|&x| x == ProjectivePoint::GENERATOR)
        );
        assert!(b.field.iter().all(|&x| x.reduce() == PastaField::ONE));
        assert!(b.indices.iter().all(|&x| x == 73));
    }
}

#[test]
fn typed_sources_validate_bounds_and_signed_extremes() {
    fn check<C: PastaCurve>() {
        use crate::field::PrimeModulus;
        let g = AffinePoint::<C>::GENERATOR;
        let bases = [g; 6];
        let indices = [5, 3, 1, 2, 2, 0];
        let selection = Selection::indexed(Bases::Affine(&bases), &indices).unwrap();
        let signed = [i128::MIN, i128::MAX, -1, 0, 1, -129];
        let unsigned = [0, 1, 129, u128::MAX, 1 << 127, 17];
        let raw_signed: Vec<_> = signed
            .iter()
            .map(|s| {
                let n = s.unsigned_abs();
                let value =
                    PastaField::from_canonical_uint(crate::field::CanonicalUint::from_limbs([
                        n as u64,
                        (n >> 64) as u64,
                        0,
                        0,
                    ]))
                    .unwrap();
                if *s < 0 { value.neg() } else { value }
            })
            .collect();
        let raw_unsigned: Vec<_> = unsigned
            .iter()
            .map(|s| {
                PastaField::from_canonical_uint(crate::field::CanonicalUint::from_limbs([
                    *s as u64,
                    (s >> 64) as u64,
                    0,
                    0,
                ]))
                .unwrap()
            })
            .collect();
        for (typed, raw) in [
            (selection.with_signed(&signed), &raw_signed),
            (selection.with_unsigned(&unsigned), &raw_unsigned),
        ] {
            let expected = reference(&selection.with_scalars(raw));
            for width in [None, Some(5), Some(12)] {
                let options = width.map_or(BatchOptions::default(), |w| {
                    BatchOptions::new(
                        ArithmeticOptions::DEFAULT
                            .with_algorithm(Algorithm::Booth {
                                width: Some(w),
                                accumulation: Accumulation::Auto,
                            })
                            .unwrap(),
                    )
                });
                let mut buffers = Buffers::new(typed.requirements_with(options).unwrap());
                assert_eq!(
                    typed
                        .execute_with(options, &SerialExecutor, buffers.borrow())
                        .unwrap(),
                    expected
                );
            }
        }
        let mut records = [ScalarStorage::<C>::ZERO; 7];
        let mut canonical: Vec<_> = raw_unsigned.iter().map(|s| s.to_canonical_uint()).collect();
        assert!(selection.with_canonical(&canonical, 127).is_err());
        assert!(
            PreparedScalars::canonical(
                &canonical,
                127,
                &mut records,
                TaskBudget::SERIAL,
                &SerialExecutor
            )
            .is_err()
        );
        assert!(records.iter().all(|r| *r == ScalarStorage::ZERO));
        canonical[2] = crate::field::CanonicalUint::from_limbs(C::Scalar::MODULUS);
        assert!(selection.with_canonical(&canonical, 256).is_err());
        assert!(
            PreparedScalars::canonical(
                &canonical,
                256,
                &mut records,
                TaskBudget::SERIAL,
                &SerialExecutor
            )
            .is_err()
        );
        assert!(records.iter().all(|r| *r == ScalarStorage::ZERO));
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = selection.with_unsigned(&unsigned[..5]);
            }))
            .is_err()
        );
    }
    check::<Pallas>();
    check::<Vesta>();
}
