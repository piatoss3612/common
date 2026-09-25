use super::*;
use crate::{
    curve::{AffinePoint, Pallas, Point, Vesta, tests::multiply},
    exec::{SerialExecutor, TaskBudget},
    field::PastaField,
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
        assert_eq!(r.scalars(), 0);
        Self {
            digits: vec![37; r.digits() + 1],
            affine: vec![AffinePoint::GENERATOR; r.affine() + 1],
            projective: vec![ProjectivePoint::GENERATOR; r.projective() + 1],
            field: vec![PastaField::ONE; r.field() + 1],
            indices: vec![37; r.indices() + 1],
        }
    }
    fn borrow(&mut self) -> Scratch<'_, C> {
        Scratch::new(
            &mut [],
            &mut self.digits,
            &mut self.affine,
            &mut self.projective,
            &mut self.field,
            &mut self.indices,
        )
    }
    fn snapshot(&self) -> Vec<u64> {
        let mut words: Vec<_> = self.digits.iter().map(|d| u64::from(*d)).collect();
        for p in &self.affine {
            words.extend(p.x.montgomery_limbs());
            words.extend(p.y.montgomery_limbs());
        }
        for p in &self.projective {
            words.extend(p.x.montgomery_limbs());
            words.extend(p.y.montgomery_limbs());
            words.extend(p.z.montgomery_limbs());
        }
        for f in &self.field {
            words.extend(f.montgomery_limbs());
        }
        words.extend(self.indices.iter().map(|i| *i as u64));
        words
    }
    fn tails(&self, r: Requirements) {
        assert_eq!(self.digits[r.digits()], 37);
        assert_eq!(self.affine[r.affine()], AffinePoint::GENERATOR);
        assert_eq!(self.projective[r.projective()], ProjectivePoint::GENERATOR);
        assert_eq!(
            self.field[r.field()].reduce(),
            PastaField::<C::Base>::ONE.reduce()
        );
        assert_eq!(self.indices[r.indices()], 37);
    }
}

fn differential<C: PastaCurve>() {
    let samples: Vec<_> = field_samples::<C::Scalar>().take(513).collect();
    for n in [0, 1, 7, 31, 32, 127, 128, 257, 513] {
        for outputs in [0, 1, 2, 5] {
            let g = AffinePoint::<C>::GENERATOR;
            let points: Vec<_> = (0..n * outputs)
                .map(|i| match i % 7 {
                    0 => Point::IDENTITY,
                    1 | 3 => g.neg().to_point(),
                    _ => g.to_point(),
                })
                .collect();
            for short in [false, true] {
                let scalars: Vec<_> = (0..n)
                    .map(|i| {
                        if short {
                            if i % 2 == 0 {
                                PastaField::<C::Scalar>::from_u64((i % 19) as u64)
                            } else {
                                PastaField::<C::Scalar>::from_u64((i % 19) as u64).neg()
                            }
                        } else {
                            samples[i % samples.len()]
                        }
                    })
                    .collect();
                let mut records = vec![crate::curve::msm::ScalarStorage::ZERO; n];
                let prepared = PreparedScalars::prepare(
                    &scalars,
                    &mut records,
                    TaskBudget::SERIAL,
                    &SerialExecutor,
                );
                for (row_stride, term_stride) in [(n, 1), (1, outputs), (0, 0)] {
                    let matrix = SharedScalarInput::new(
                        Bases::Points(&points),
                        prepared,
                        outputs,
                        row_stride,
                        term_stride,
                    )
                    .unwrap();
                    let expected: Vec<_> = (0..outputs)
                        .map(|row| {
                            scalars.iter().enumerate().fold(
                                ProjectivePoint::IDENTITY,
                                |sum, (i, k)| {
                                    let base =
                                        points[row * row_stride + i * term_stride].to_projective();
                                    sum.add(&multiply(k, |p| p.add(&base)))
                                },
                            )
                        })
                        .collect();
                    for tasks in [1, 3] {
                        for limit in [8192, 65536, usize::MAX] {
                            let options = ExecutionOptions::DEFAULT
                                .with_task_budget(TaskBudget::new(tasks).unwrap())
                                .with_memory_limit(limit);
                            let r = matrix.requirements(options).unwrap();
                            assert!(r.bytes::<C>().unwrap() <= limit);
                            let mut buffers = Buffers::new(r);
                            let mut output = vec![ProjectivePoint::GENERATOR; outputs];
                            matrix
                                .execute(&mut output, options, &SerialExecutor, buffers.borrow())
                                .unwrap();
                            assert_eq!(
                                output, expected,
                                "n={n} outputs={outputs} short={short} strides={row_stride}/{term_stride} tasks={tasks} limit={limit}"
                            );
                            buffers.tails(r);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn matrices_match_binary_ladders() {
    differential::<Pallas>();
    differential::<Vesta>();
}

#[test]
fn extent_and_empty_shapes() {
    assert_eq!(extent(0, usize::MAX, usize::MAX, usize::MAX), Ok(0));
    assert_eq!(extent(usize::MAX, 0, usize::MAX, usize::MAX), Ok(0));
    assert_eq!(extent(1, 1, usize::MAX, usize::MAX), Ok(1));
    for shape in [
        (2, 1, usize::MAX, 1),
        (1, 2, 1, usize::MAX),
        (2, 2, usize::MAX / 2, usize::MAX / 2 + 1),
    ] {
        assert_eq!(
            extent(shape.0, shape.1, shape.2, shape.3),
            Err(CurveError::SizeOverflow)
        );
    }
    let mut records = [crate::curve::msm::ScalarStorage::<Pallas>::ZERO; 3];
    let prepared = PreparedScalars::unsigned(
        &[1, 2, 3],
        &mut records,
        TaskBudget::SERIAL,
        &SerialExecutor,
    );
    assert!(matches!(
        SharedScalarInput::new(
            Bases::Affine(&[AffinePoint::GENERATOR; 5]),
            prepared,
            2,
            3,
            1
        ),
        Err(CurveError::MatrixTooSmall {
            required: 6,
            provided: 5
        })
    ));
    assert!(matches!(
        SharedScalarInput::new(Bases::Affine(&[]), prepared, usize::MAX, 0, 0),
        Err(CurveError::SizeOverflow)
    ));
    let bases = [AffinePoint::GENERATOR];
    let options = ExecutionOptions::DEFAULT.with_task_budget(TaskBudget::new(3).unwrap());
    let tiled = SharedScalarInput::new(Bases::Affine(&bases), prepared, 9, 0, 0).unwrap();
    let many = SharedScalarInput::new(Bases::Affine(&bases), prepared, 1_000_000, 0, 0).unwrap();
    assert_eq!(tiled.requirements(options), many.requirements(options));
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

fn layouts<C: PastaCurve>(n: usize, limit: usize) {
    use crate::curve::{EisensteinTableBatch, PreparedAffinePoint, msm::ScalarStorage};
    let outputs = 5;
    let samples: Vec<_> = field_samples::<C::Scalar>().take(513).collect();
    let scalars: Vec<_> = (0..n).map(|i| samples[i % samples.len()]).collect();
    let mut records = vec![ScalarStorage::ZERO; n];
    let prepared = PreparedScalars::prepare(
        &scalars,
        &mut records,
        crate::exec::TaskBudget::SERIAL,
        &SerialExecutor,
    );
    let affine: Vec<_> = (0..n * outputs + 13)
        .map(|i| {
            *AffinePoint::<C>::GENERATOR
                .mul_projective(&PastaField::<C::Scalar>::from_u64((i + 1) as u64))
                .to_point()
                .as_affine()
                .unwrap()
        })
        .collect();
    let cached: Vec<_> = affine
        .iter()
        .map(PreparedAffinePoint::from_affine)
        .collect();
    let tr = EisensteinTableBatch::<C>::requirements(affine.len()).unwrap();
    let mut entries = vec![AffinePoint::GENERATOR; tr.table_entries];
    let mut cached_entries =
        vec![PreparedAffinePoint::from_affine(&AffinePoint::GENERATOR); tr.table_entries];
    let mut projective = vec![ProjectivePoint::IDENTITY; tr.projective_scratch];
    let mut field = vec![PastaField::ZERO; tr.field_scratch];
    let tables = EisensteinTableBatch::prepare(
        &affine,
        &mut entries,
        &mut projective,
        &mut field,
        TaskBudget::SERIAL,
        &SerialExecutor,
    );
    let cached_tables = EisensteinTableBatch::prepare(
        &cached,
        &mut cached_entries,
        &mut projective,
        &mut field,
        TaskBudget::SERIAL,
        &SerialExecutor,
    );
    let options = ExecutionOptions::DEFAULT
        .with_task_budget(TaskBudget::new(3).unwrap())
        .with_memory_limit(limit);
    for bases in [
        Bases::Affine(&affine),
        Bases::Prepared(&cached),
        Bases::Compact(tables),
        Bases::CompactPrepared(cached_tables),
    ] {
        let row = Input::new_prepared(
            match bases {
                Bases::Compact(table) => {
                    Bases::Compact(EisensteinTableBatch::bind(&table.as_slice()[..8 * n]))
                }
                Bases::CompactPrepared(table) => {
                    Bases::CompactPrepared(EisensteinTableBatch::bind(&table.as_slice()[..8 * n]))
                }
                _ => Bases::Affine(&affine[..n]),
            },
            prepared,
        );
        let plan = super::super::run::MsmPlan::for_input(&row, options).unwrap();
        let mut digits = vec![0; prepared.cache_len(&plan)];
        let cached_scalars = prepared.cache(&plan, &mut digits);
        for scalars in [prepared, cached_scalars] {
            let matrix = SharedScalarInput::new(bases, scalars, outputs, n + 2, 1).unwrap();
            let expected: Vec<_> = (0..outputs)
                .map(|j| {
                    let input = Input::new_prepared(
                        Bases::Affine(&affine[j * (n + 2)..j * (n + 2) + n]),
                        prepared,
                    );
                    let mut buffers = Buffers::new(input.requirements(options).unwrap());
                    input
                        .execute(options, &SerialExecutor, buffers.borrow())
                        .unwrap()
                })
                .collect();
            let r = matrix.requirements(options).unwrap();
            let mut buffers = Buffers::new(r);
            for workers in [1, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(workers)
                    .build()
                    .unwrap();
                let mut actual = vec![ProjectivePoint::IDENTITY; outputs];
                pool.install(|| {
                    matrix
                        .execute(&mut actual, options, &Pool, buffers.borrow())
                        .unwrap()
                });
                assert_eq!(actual, expected);
                buffers.tails(r);
            }
        }
    }
}

#[test]
fn layouts_caches_and_worker_pools() {
    for n in [33, 513] {
        layouts::<Pallas>(n, usize::MAX);
        layouts::<Vesta>(n, usize::MAX);
    }
    layouts::<Pallas>(2048, 98304);
    layouts::<Vesta>(2048, 98304);
}

#[test]
fn validation_capacity_and_unwind() {
    use crate::curve::msm::ScalarStorage;
    use std::panic::{AssertUnwindSafe, catch_unwind};
    let scalars: Vec<_> = field_samples::<<Pallas as PastaCurve>::Scalar>()
        .take(257)
        .collect();
    let mut records = vec![ScalarStorage::<Pallas>::ZERO; scalars.len()];
    let prepared =
        PreparedScalars::prepare(&scalars, &mut records, TaskBudget::SERIAL, &SerialExecutor);
    let matrix =
        SharedScalarInput::new(Bases::Affine(&[AffinePoint::GENERATOR]), prepared, 7, 0, 0)
            .unwrap();
    let options = ExecutionOptions::DEFAULT.with_task_budget(TaskBudget::new(5).unwrap());
    let tight = options.with_memory_limit(8192);
    let r = matrix.requirements(tight).unwrap();
    let mut buffers = Buffers::new(r);
    let mut output = [ProjectivePoint::GENERATOR; 7];
    matrix
        .execute(&mut output, options, &SerialExecutor, buffers.borrow())
        .unwrap();
    let expected = output;
    buffers.tails(r);
    output.fill(ProjectivePoint::GENERATOR);
    let before = buffers.snapshot();
    assert!(matches!(
        matrix.execute(
            &mut output,
            options.with_memory_limit(0),
            &SerialExecutor,
            buffers.borrow()
        ),
        Err(CurveError::MemoryLimit { .. })
    ));
    assert_eq!(output, [ProjectivePoint::GENERATOR; 7]);
    assert_eq!(buffers.snapshot(), before);
    let mut insufficient = Buffers::new(r);
    insufficient.projective.clear();
    let before_insufficient = insufficient.snapshot();
    assert!(matches!(
        matrix.execute(&mut output, options, &SerialExecutor, insufficient.borrow()),
        Err(CurveError::ScratchTooSmall { .. })
    ));
    assert_eq!(output, [ProjectivePoint::GENERATOR; 7]);
    assert_eq!(insufficient.snapshot(), before_insufficient);
    assert!(
        catch_unwind(AssertUnwindSafe(|| matrix.execute(
            &mut output[..6],
            options,
            &SerialExecutor,
            buffers.borrow()
        )))
        .is_err()
    );
    assert_eq!(output, [ProjectivePoint::GENERATOR; 7]);

    assert_eq!(buffers.snapshot(), before);

    struct Fail(core::sync::atomic::AtomicBool);
    impl Executor for Fail {
        fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
        where
            L: FnOnce() -> A + Send,
            R: FnOnce() -> B + Send,
            A: Send,
            B: Send,
        {
            SerialExecutor.join(
                || {
                    let value = left();
                    assert!(
                        self.0.swap(true, core::sync::atomic::Ordering::Relaxed),
                        "injected executor unwind"
                    );
                    value
                },
                right,
            )
        }
    }
    let r = matrix.requirements(options).unwrap();
    let mut buffers = Buffers::new(r);
    assert!(
        catch_unwind(AssertUnwindSafe(|| matrix.execute(
            &mut output,
            options,
            &Fail(core::sync::atomic::AtomicBool::new(false)),
            buffers.borrow()
        )))
        .is_err()
    );
    matrix
        .execute(&mut output, options, &SerialExecutor, buffers.borrow())
        .unwrap();
    assert_eq!(output, expected);
    buffers.tails(r);
}

#[test]
fn scalar_boundaries_chunks_and_task_ceilings() {
    fn check<C: PastaCurve>() {
        use crate::curve::msm::{ScalarStorage, tests::JoinWidth};
        use crate::field::PrimeModulus;
        let scalar_cases = [
            PastaField::ZERO,
            PastaField::ONE,
            PastaField::<C::Scalar>::ONE.neg(),
            PastaField::from_montgomery_limbs(C::Scalar::MODULUS),
            PastaField::from_montgomery_limbs([u64::MAX, u64::MAX, u64::MAX, (1 << 63) - 1]),
        ];
        let g = AffinePoint::<C>::GENERATOR;
        for n in [1, 129, 8193] {
            let raw: Vec<_> = (0..n)
                .map(|i| scalar_cases[i % scalar_cases.len()])
                .collect();
            let mut records = vec![ScalarStorage::ZERO; n];
            let prepared =
                PreparedScalars::prepare(&raw, &mut records, TaskBudget::SERIAL, &SerialExecutor);
            let base = [g];
            let matrix = SharedScalarInput::new(Bases::Affine(&base), prepared, 9, 0, 0).unwrap();
            let total = raw.iter().fold(PastaField::ZERO, |sum, s| sum.add(s));
            let expected = multiply(&total, |sum| sum.add(&g.to_projective()));
            for tasks in [1, 3, 5] {
                for limit in [8192, usize::MAX] {
                    let options = ExecutionOptions::DEFAULT
                        .with_task_budget(TaskBudget::new(tasks).unwrap())
                        .with_memory_limit(limit);
                    let r = matrix.requirements(options).unwrap();
                    assert!(r.bytes::<C>().unwrap() <= limit);
                    let mut buffers = Buffers::new(r);
                    let mut output = [ProjectivePoint::IDENTITY; 9];
                    let width = JoinWidth(core::sync::atomic::AtomicUsize::new(1));
                    let (result, peak) = width
                        .measure(|| matrix.execute(&mut output, options, &width, buffers.borrow()));
                    result.unwrap();
                    assert!(peak <= tasks, "{peak} exposed leaves exceed {tasks} tasks");
                    assert_eq!(output, [expected; 9]);
                    buffers.tails(r);
                }
            }
        }
        let signed = [i128::MIN, i128::MAX, -1, 0, 1];
        let mut records = [ScalarStorage::ZERO; 5];
        let prepared =
            PreparedScalars::signed(&signed, &mut records, TaskBudget::SERIAL, &SerialExecutor);
        let bases = [g; 5];
        let matrix = SharedScalarInput::new(Bases::Affine(&bases), prepared, 9, 0, 1).unwrap();
        let options = ExecutionOptions::DEFAULT;
        let mut buffers = Buffers::new(matrix.requirements(options).unwrap());
        let mut output = [ProjectivePoint::IDENTITY; 9];
        matrix
            .execute(&mut output, options, &SerialExecutor, buffers.borrow())
            .unwrap();
        assert_eq!(output, [g.neg().to_projective(); 9]);
        for scalar in [0, 1] {
            let prepared = PreparedScalars::unsigned(
                &[scalar; 5],
                &mut records,
                TaskBudget::SERIAL,
                &SerialExecutor,
            );
            let matrix = SharedScalarInput::new(Bases::Affine(&bases), prepared, 9, 0, 1).unwrap();
            let mut buffers = Buffers::new(matrix.requirements(options).unwrap());
            matrix
                .execute(&mut output, options, &SerialExecutor, buffers.borrow())
                .unwrap();
            let expected = g.mul_projective(&PastaField::from_u64(5 * scalar as u64));
            assert_eq!(output, [expected; 9]);
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}
