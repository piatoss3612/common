use super::*;
use crate::{
    curve::{Pallas, Vesta, tests::multiply},
    exec::SerialExecutor,
    test_support::field_samples,
};
use std::{vec, vec::Vec};

pub(super) struct Buffers<C: PastaCurve> {
    scalars: Vec<ScalarStorage<C>>,
    digits: Vec<u8>,
    affine: Vec<AffinePoint<C>>,
    projective: Vec<ProjectivePoint<C>>,
    field: Vec<PastaField<C::Base>>,
    indices: Vec<usize>,
}

impl<C: PastaCurve> Buffers<C> {
    pub(super) fn new(r: Requirements) -> Self {
        Self {
            scalars: vec![ScalarStorage::ZERO; r.scalars + 1],
            digits: vec![73; r.digits + 1],
            affine: vec![AffinePoint::GENERATOR; r.affine + 1],
            projective: vec![ProjectivePoint::GENERATOR; r.projective + 1],
            field: vec![PastaField::ONE; r.field + 1],
            indices: vec![73; r.indices + 1],
        }
    }
    pub(super) fn borrow(&mut self) -> Scratch<'_, C> {
        Scratch {
            scalars: &mut self.scalars,
            digits: &mut self.digits,
            affine: &mut self.affine,
            projective: &mut self.projective,
            field: &mut self.field,
            indices: &mut self.indices,
        }
    }
    fn tails(&self, r: Requirements) {
        assert!(self.scalars[r.scalars] == ScalarStorage::ZERO);
        assert_eq!(self.digits[r.digits], 73);
        assert_eq!(self.affine[r.affine], AffinePoint::GENERATOR);
        assert_eq!(self.projective[r.projective], ProjectivePoint::GENERATOR);
        assert_eq!(
            (self.field[r.field]).reduce(),
            (PastaField::<_>::ONE).reduce()
        );
        assert_eq!(self.indices[r.indices], 73);
    }
}

struct Pool;
impl Executor for Pool {
    fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
    where
        L: FnOnce() -> A + Send,
        R: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        rayon::join(left, right)
    }
}

/// Measures the widest set of independent leaves exposed by a join tree.
///
/// Jobs run sequentially: consecutive joins take a maximum; joined branches add.
/// This checks the allowance without relying on OS scheduling or worker counts.
pub(super) struct JoinWidth(pub(super) core::sync::atomic::AtomicUsize);
impl JoinWidth {
    pub(super) fn measure<R>(&self, work: impl FnOnce() -> R) -> (R, usize) {
        use core::sync::atomic::{AtomicUsize, Ordering};
        struct Restore<'a>(&'a AtomicUsize, usize);
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                self.0.store(self.1, Ordering::Relaxed);
            }
        }
        let _restore = Restore(&self.0, self.0.swap(1, Ordering::Relaxed));
        let result = work();
        (result, self.0.load(Ordering::Relaxed))
    }
}
impl Executor for JoinWidth {
    fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
    where
        L: FnOnce() -> A + Send,
        R: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        let ((a, left), (b, right)) =
            SerialExecutor.join(|| self.measure(left), || self.measure(right));
        self.0
            .fetch_max(left + right, core::sync::atomic::Ordering::Relaxed);
        (a, b)
    }
}

fn reference<C: PastaCurve>(input: &Input<'_, C>) -> ProjectivePoint<C> {
    let mut sum = ProjectivePoint::IDENTITY;
    let Scalars::Raw(scalars) = input.scalars else {
        panic!("reference needs original scalars")
    };
    for (i, k) in scalars.iter().enumerate() {
        let j = input.indices.map_or(i, |indices| indices[i] as usize);
        let base = match input.bases {
            Bases::Affine(b) => b[j].to_projective(),
            Bases::Prepared(b) => b[j].to_affine().to_projective(),
            Bases::Points(b) => b[j].to_projective(),
            Bases::Compact(b) => b.get(j).unwrap().base().to_projective(),
            Bases::CompactPrepared(b) => b.get(j).unwrap().base().to_projective(),
        };
        sum = sum.add(&multiply(k, |sum| sum.add(&base)));
    }
    sum
}

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
            .with_kernel(Kernel::Booth {
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
                            .with_kernel(Kernel::Booth {
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
                            .with_kernel(Kernel::Booth {
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
                        .with_kernel(Kernel::StreamingBooth { width: Some(width) })
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
                .with_kernel(Kernel::StreamingBooth { width: Some(4) })
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
