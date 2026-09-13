use super::*;
use crate::{
    curve::{Pallas, Vesta, scalar},
    exec::SerialExecutor,
    test_support::field_samples,
};
use std::{vec, vec::Vec};

struct Buffers<C: PastaCurve> {
    digits: Vec<u8>,
    affine: Vec<AffinePoint<C>>,
    projective: Vec<ProjectivePoint<C>>,
    field: Vec<PastaField<C::Base>>,
    indices: Vec<usize>,
}

impl<C: PastaCurve> Buffers<C> {
    fn new(r: Requirements) -> Self {
        Self {
            digits: vec![73; r.digits + 1],
            affine: vec![AffinePoint::GENERATOR; r.affine + 1],
            projective: vec![ProjectivePoint::GENERATOR; r.projective + 1],
            field: vec![PastaField::ONE; r.field + 1],
            indices: vec![73; r.indices + 1],
        }
    }
    fn borrow(&mut self) -> Scratch<'_, C> {
        Scratch {
            digits: &mut self.digits,
            affine: &mut self.affine,
            projective: &mut self.projective,
            field: &mut self.field,
            indices: &mut self.indices,
        }
    }
    fn tails(&self, r: Requirements) {
        assert_eq!(self.digits[r.digits], 73);
        assert_eq!(self.affine[r.affine], AffinePoint::GENERATOR);
        assert_eq!(self.projective[r.projective], ProjectivePoint::GENERATOR);
        assert_eq!(self.field[r.field], PastaField::ONE);
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
        };
        sum = sum.add(&scalar::multiply(k, |sum| sum.add(&base)));
    }
    sum
}

fn differentials<C: PastaCurve>() {
    let g = AffinePoint::<C>::GENERATOR;
    let affine: Vec<_> = (1..=1030)
        .map(|i| {
            *g.mul_projective(&PastaField::from_u64(i))
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
            *k = PastaField::ONE.neg();
        }
    }
    let short: Vec<_> = (0..1030)
        .map(|i| PastaField::from_u64((i * 137) as u64))
        .collect();
    for n in [
        0, 1, 7, 8, 15, 31, 32, 33, 127, 128, 129, 255, 256, 257, 513, 1030,
    ] {
        for scalars in [&full[..n], &short[..n]] {
            let mut storage = vec![73; PreparedScalars::<C>::storage_len(n).unwrap() + 1];
            let retained = PreparedScalars::<C>::prepare(
                scalars,
                &mut storage,
                TaskBudget::new(3).unwrap(),
                &Pool,
            )
            .unwrap();
            assert_eq!(retained.len(), n);
            assert_eq!(retained.is_empty(), n == 0);
            for bases in [
                Bases::Affine(&affine[..n]),
                Bases::Prepared(&prepared[..n]),
                Bases::Points(&points[..n]),
            ] {
                let dense = Input::new(bases, scalars).unwrap();
                let ix: Vec<_> = indices[..n].iter().map(|i| i % n.max(1) as u32).collect();
                let indexed = Input::indexed(bases, &ix, scalars).unwrap();
                for input in [dense, indexed] {
                    let reused = match input.indices {
                        Some(indices) => Input::indexed_prepared(bases, indices, retained),
                        None => Input::new_prepared(bases, retained),
                    }
                    .unwrap();
                    let expected = reference(&input);
                    for (tasks, cap) in [
                        (1, None),
                        (3, NonZeroUsize::new(17)),
                        (65, NonZeroUsize::new(67)),
                    ] {
                        let options = ExecutionOptions {
                            task_budget: TaskBudget::new(tasks).unwrap(),
                            max_terms_per_pass: cap,
                        };
                        let r = input.requirements(options).unwrap();
                        assert_eq!(r, batch_requirements(&[input], options).unwrap());
                        let mut buffers = Buffers::new(r);
                        assert_eq!(
                            input
                                .execute(options, &SerialExecutor, buffers.borrow())
                                .unwrap(),
                            expected,
                            "n={n}, tasks={tasks}"
                        );
                        // Reuse dirty working storage while allowing parallel joins.
                        assert_eq!(
                            input.execute(options, &Pool, buffers.borrow()).unwrap(),
                            expected
                        );
                        buffers.tails(r);
                        let r = reused.requirements(options).unwrap();
                        assert_eq!(r.digits, 0);
                        assert_eq!(r, batch_requirements(&[reused], options).unwrap());
                        let mut buffers = Buffers::new(r);
                        assert_eq!(
                            reused.execute(options, &Pool, buffers.borrow()).unwrap(),
                            expected
                        );
                        buffers.tails(r);
                    }
                }
            }
            assert_eq!(*storage.last().unwrap(), 73);
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
    let raw = Input::new(Bases::Affine(&bases), &scalars).unwrap();
    let expected = reference(&raw);
    let bytes = PreparedScalars::<C>::storage_len(scalars.len()).unwrap();
    let mut storage = vec![73; bytes + 1];
    assert!(matches!(
        PreparedScalars::<C>::prepare(
            &scalars,
            &mut storage[..bytes - 1],
            TaskBudget::SERIAL,
            &SerialExecutor
        ),
        Err(CurveError::ScratchTooSmall { .. })
    ));
    assert!(storage.iter().all(|b| *b == 73));
    assert_eq!(
        PreparedScalars::<C>::storage_len(usize::MAX),
        Err(CurveError::SizeOverflow)
    );
    let retained =
        PreparedScalars::<C>::prepare(&scalars, &mut storage, TaskBudget::SERIAL, &SerialExecutor)
            .unwrap();
    scalars.fill(PastaField::ZERO);
    assert!(matches!(
        Input::new_prepared(Bases::Affine(&bases[..1]), retained),
        Err(CurveError::LengthMismatch { .. })
    ));
    assert!(matches!(
        Input::indexed_prepared(Bases::Affine(&bases), &[], retained),
        Err(CurveError::LengthMismatch { .. })
    ));
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
        Input::new(Bases::Affine(&bases), &scalars).unwrap(),
        Input::new_prepared(Bases::Affine(&bases), retained).unwrap(),
        Input::indexed_prepared(Bases::Affine(&negative[..1]), &indices, retained).unwrap(),
    ];
    let options = ExecutionOptions {
        task_budget: TaskBudget::new(4).unwrap(),
        max_terms_per_pass: NonZeroUsize::new(17),
    };
    let r = batch_requirements(&inputs, options).unwrap();
    let mut buffers = Buffers::new(r);
    let mut output = [ProjectivePoint::GENERATOR; 3];
    execute_batch(&inputs, &mut output, options, &Pool, buffers.borrow()).unwrap();
    assert_eq!(
        output,
        [ProjectivePoint::IDENTITY, expected, expected.neg()]
    );
    buffers.tails(r);
    assert_eq!(storage[bytes], 73);
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
            let mut storage = vec![73; PreparedScalars::<C>::storage_len(n).unwrap()];
            let retained = PreparedScalars::<C>::prepare(
                &scalars,
                &mut storage,
                TaskBudget::SERIAL,
                &SerialExecutor,
            )
            .unwrap();
            let reused =
                Input::indexed_prepared(Bases::Points(&bases), &indices, retained).unwrap();
            for tasks in [1, 4] {
                for cap in [1, 7, 8, 17, n] {
                    let options = ExecutionOptions {
                        task_budget: TaskBudget::new(tasks).unwrap(),
                        max_terms_per_pass: NonZeroUsize::new(cap),
                    };
                    let r = input.requirements(options).unwrap();
                    let mut buffers = Buffers::new(r);
                    assert_eq!(
                        input
                            .execute(options, &SerialExecutor, buffers.borrow())
                            .unwrap(),
                        expected,
                        "bits={bits}, n={n}, tasks={tasks}, cap={cap}"
                    );
                    buffers.tails(r);
                    assert_eq!(
                        reused
                            .execute(options, &SerialExecutor, buffers.borrow())
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
    let input = Input::new(Bases::Affine(&bases), &scalars).unwrap();
    let expected = reference(&input);
    let options = ExecutionOptions {
        task_budget: TaskBudget::new(4).unwrap(),
        max_terms_per_pass: NonZeroUsize::new(37),
    };
    let r = input.requirements(options).unwrap();
    let mut buffers = Buffers::new(r);
    // The first two joins prepare three digit chunks; later joins evaluate
    // windows. Exercise an unwind after writes in either scoped phase.
    for at in [0, 2] {
        let executor = Panics {
            calls: AtomicUsize::new(0),
            at,
        };
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                input.execute(options, &executor, buffers.borrow()).unwrap();
            }))
            .is_err()
        );
        buffers.tails(r);
        assert_eq!(
            input.execute(options, &Pool, buffers.borrow()).unwrap(),
            expected
        );
        buffers.tails(r);
    }
    let mut storage = vec![73; PreparedScalars::<Pallas>::storage_len(bases.len()).unwrap() + 1];
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
            )
            .unwrap();
        }))
        .is_err()
    );
    assert_eq!(*storage.last().unwrap(), 73);
    let retained =
        PreparedScalars::<Pallas>::prepare(&scalars, &mut storage, options.task_budget, &Pool)
            .unwrap();
    let reused = Input::new_prepared(Bases::Affine(&bases), retained).unwrap();
    assert_eq!(
        reused.execute(options, &Pool, buffers.borrow()).unwrap(),
        expected
    );
    buffers.tails(r);
    assert_eq!(*storage.last().unwrap(), 73);
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
            PastaField::ONE.neg(),
        ] {
            let scalars = vec![value; n];
            let input = Input::new(Bases::Points(&bases), &scalars).unwrap();
            let expected = reference(&input);
            for cap in [1, 2, 3, 37, n] {
                let options = ExecutionOptions {
                    max_terms_per_pass: NonZeroUsize::new(cap),
                    ..ExecutionOptions::SERIAL
                };
                let mut buffers = Buffers::new(input.requirements(options).unwrap());
                assert_eq!(
                    input
                        .execute(options, &SerialExecutor, buffers.borrow())
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
fn grouped_jobs_and_nested_single_worker() {
    let bases = [AffinePoint::<Pallas>::GENERATOR; 700];
    let cached = [PreparedAffinePoint::from_affine(&bases[0]); 700];
    let points = [Point::<Pallas>::IDENTITY; 700];
    let scalars: Vec<_> = field_samples().take(700).collect();
    let indices: Vec<_> = (0..700).map(|i| i as u32 % 13).collect();
    let jobs = [
        Input::new(Bases::Affine(&[]), &[]).unwrap(),
        Input::indexed(Bases::Prepared(&cached), &indices[..700], &scalars).unwrap(),
        Input::new(Bases::Affine(&bases[..17]), &scalars[..17]).unwrap(),
        Input::indexed(Bases::Points(&points), &indices[..99], &scalars[..99]).unwrap(),
        Input::new(Bases::Affine(&bases[..257]), &scalars[..257]).unwrap(),
    ];
    let expected = jobs.map(|input| reference(&input));
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap();
    for tasks in [1, 3, 7, 32, 128] {
        let options = ExecutionOptions {
            task_budget: TaskBudget::new(tasks).unwrap(),
            max_terms_per_pass: NonZeroUsize::new(63),
        };
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
    }
}

#[test]
fn validation_precedes_writes_and_sizing_rejects_overflow() {
    type C = Pallas;
    let bases = [AffinePoint::<C>::GENERATOR; 256];
    let scalars = [PastaField::ONE; 256];
    assert!(matches!(
        Input::new(Bases::Affine(&bases), &scalars[..255]),
        Err(CurveError::LengthMismatch { .. })
    ));
    assert!(matches!(
        Input::indexed(Bases::Affine(&bases), &[256], &scalars[..1]),
        Err(CurveError::BaseIndexOutOfBounds {
            position: 0,
            index: 256,
            bases: 256
        })
    ));
    assert!(matches!(
        Input::indexed(Bases::Affine(&bases), &[], &scalars[..1]),
        Err(CurveError::LengthMismatch { .. })
    ));
    for n in [usize::MAX, isize::MAX as usize, usize::MAX / 64] {
        assert_eq!(
            Input::<C>::requirements_for_len(n, ExecutionOptions::SERIAL),
            Err(CurveError::SizeOverflow)
        );
    }
    const R: Requirements = match Input::<C>::requirements_for_len(256, ExecutionOptions::SERIAL) {
        Ok(r) => r,
        Err(_) => panic!("valid length"),
    };
    let input = Input::new(Bases::Affine(&bases), &scalars).unwrap();
    assert_eq!(input.requirements(ExecutionOptions::SERIAL).unwrap(), R);
    for short in 0..6 {
        let mut b = Buffers::<C>::new(R);
        let mut scratch = b.borrow();
        match short {
            0 => scratch.digits = &mut scratch.digits[..R.digits - 1],
            1 => scratch.affine = &mut scratch.affine[..R.affine - 1],
            2 => scratch.projective = &mut scratch.projective[..R.projective - 1],
            3 => scratch.field = &mut scratch.field[..R.field - 1],
            4 => scratch.indices = &mut scratch.indices[..R.indices - 1],
            _ => (),
        }
        let mut output = [ProjectivePoint::GENERATOR; 2];
        let len = if short == 5 { 2 } else { 1 };
        assert!(
            execute_batch(
                &[input],
                &mut output[..len],
                ExecutionOptions::SERIAL,
                &SerialExecutor,
                scratch
            )
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
        assert!(b.field.iter().all(|&x| x == PastaField::ONE));
        assert!(b.indices.iter().all(|&x| x == 73));
    }
}

#[test]
fn packed_midpoint_carries_reconstruct_signed_extremes() {
    use crate::curve::scalar::centered_digit;
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
