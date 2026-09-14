use super::{
    msm_run::{Arena, Work},
    run_pool,
};
use spin::RwLock;
use std::num::NonZeroUsize;
use zakura_udon::{
    curve::{
        AffinePoint, EisensteinTableBatch, Pallas, PastaCurve, Point, PreparedAffinePoint,
        ProjectivePoint, Vesta,
        msm::{
            Accumulation, Bases, ExecutionOptions, Input, PreparedScalars, ScalarStorage, Scratch,
            Selection,
            run::{MsmPlan, MsmRun, ParallelMsmRun, WorkKind},
        },
    },
    exec::{
        SerialExecutor, TaskBudget,
        run::{Identity, TaskStorage},
    },
    field::{CanonicalUint, PastaField},
};

fn check<C: PastaCurve>() {
    const TERMS: usize = 385;
    let bases = vec![AffinePoint::<C>::GENERATOR; TERMS];
    let scalars: Vec<_> = (0..TERMS)
        .map(|i| {
            PastaField::<C::Scalar>::from_canonical_uint(CanonicalUint::from_limbs([
                i as u64 * 17 + 3,
                0xabcd,
                i as u64 * 11,
                0x1234_5678,
            ]))
            .unwrap()
        })
        .collect();
    let mut expected = ProjectivePoint::<C>::IDENTITY;
    let scalar = scalars.iter().fold(PastaField::ZERO, |sum, s| sum.add(s));
    for byte in scalar.to_bytes().iter().rev() {
        for bit in (0..8).rev() {
            expected = expected.double();
            if byte & (1 << bit) != 0 {
                expected = expected.add(&ProjectivePoint::GENERATOR);
            }
        }
    }
    let options = [
        ExecutionOptions::SERIAL,
        ExecutionOptions::SERIAL.with_joint_tables(),
        ExecutionOptions::SERIAL
            .with_booth_width(11)
            .unwrap()
            .with_accumulation(Accumulation::Projective),
        ExecutionOptions::SERIAL
            .with_booth_width(7)
            .unwrap()
            .with_accumulation(Accumulation::Hybrid),
        ExecutionOptions::SERIAL.with_streaming_buckets(),
    ];
    for options in options {
        for workers in [1, 3, 4] {
            let plan = MsmPlan::new(
                TERMS,
                options.with_task_budget(TaskBudget::new(workers).unwrap()),
                NonZeroUsize::new(256).unwrap(),
            )
            .unwrap();
            let serial =
                MsmPlan::<C>::new(TERMS, options, NonZeroUsize::new(256).unwrap()).unwrap();
            assert_eq!(plan.retained(), serial.retained());
            assert_eq!(plan.temporary(), serial.temporary());
            let arena = Arena::new(plan);
            let work = [RwLock::new(Work::new(core::iter::once(plan.temporary())))];
            let mut identity = Identity::new();
            let mut slots = [const { TaskStorage::EMPTY }; 3];
            let mut run = MsmRun::new(
                plan,
                Input::new(Bases::Affine(&bases), &scalars).unwrap(),
                &mut identity,
                &mut slots,
            )
            .unwrap();
            assert!(arena.bytes() + work[0].read().bytes() < 4 * 1024 * 1024);
            run_pool::scoped(workers, 3, |pool| {
                let mut ready = [None; 3];
                while run.result().is_none() {
                    let count = run.ready(&mut ready);
                    for request in ready[..count].iter().flatten() {
                        if !pool.available() {
                            break;
                        }
                        if let Some(task) = run
                            .try_claim(*request, || arena.acquire(*request, &work))
                            .unwrap()
                        {
                            assert!(pool.submit(task).is_ok());
                        }
                    }
                    let completed = run
                        .complete(pool.receive().expect("admitted MSM must make progress"))
                        .unwrap();
                    assert_eq!(completed.error, None);
                    assert!(!run.is_failed());
                    drop(completed);
                }
            });
            assert_eq!(run.result(), Some(expected));
            let leases = NonZeroUsize::new(workers).unwrap();
            let required = plan.requirements(leases).unwrap();
            let result = plan
                .execute(
                    Input::new(Bases::Affine(&bases), &scalars).unwrap(),
                    leases,
                    &SerialExecutor,
                    Scratch::new(
                        &mut vec![ScalarStorage::ZERO; required.scalars()],
                        &mut vec![0; required.digits()],
                        &mut vec![AffinePoint::GENERATOR; required.affine()],
                        &mut vec![ProjectivePoint::IDENTITY; required.projective()],
                        &mut vec![PastaField::ZERO; required.field()],
                        &mut vec![0; required.indices()],
                    ),
                )
                .unwrap();
            assert_eq!(result, expected);
        }
    }
}

#[test]
fn bounded_windows_streaming_and_partial_reductions_match_ladder() {
    check::<Pallas>();
    check::<Vesta>();
}

fn ladder<C: PastaCurve>(scalar: PastaField<C::Scalar>) -> ProjectivePoint<C> {
    let mut result = ProjectivePoint::IDENTITY;
    for byte in scalar.to_bytes().iter().rev() {
        for bit in (0..8).rev() {
            result = result.double();
            if byte & (1 << bit) != 0 {
                result = result.add(&ProjectivePoint::GENERATOR);
            }
        }
    }
    result
}

fn chunks<C: PastaCurve>(
    input: Input<'_, C>,
    expected: ProjectivePoint<C>,
    options: ExecutionOptions,
    grain: usize,
    prepared: bool,
    cached: bool,
    skew: bool,
) {
    let plan = MsmPlan::new(input.len(), options, NonZeroUsize::new(grain).unwrap()).unwrap();
    let arenas: [_; 3] = core::array::from_fn(|_| Arena::new(plan));
    let work = [RwLock::new(Work::new(core::iter::once(plan.temporary())))];
    let mut identities = core::array::from_fn(|_| Identity::new());
    let mut slots = [const { [const { TaskStorage::EMPTY }; 3] }; 3];
    let mut run = ParallelMsmRun::new(plan, input, &mut identities, &mut slots).unwrap();
    run_pool::scoped(4, 3, |pool| {
        for repetition in 0..2 {
            if repetition > 0 {
                run.rebind(plan, input).unwrap();
            }
            let mut ready = [None; 32];
            let mut held = None;
            let mut delayed = false;
            while run.result().is_none() {
                let count = run.ready(&mut ready);
                for request in ready[..count].iter().flatten() {
                    if !pool.available() {
                        break;
                    }
                    if prepared {
                        assert_eq!(
                            request.task.scratch.scalars() + request.task.read_scalars,
                            0
                        );
                    }
                    if cached {
                        assert_eq!(request.task.scratch.digits() + request.task.read_digits, 0);
                    }
                    if let Some(task) = run
                        .try_claim(*request, || {
                            arenas[request.slot].acquire(request.task, &work)
                        })
                        .unwrap()
                    {
                        if skew
                            && !delayed
                            && request.slot == 0
                            && request.task.kind == WorkKind::Prepare
                        {
                            held = Some(task);
                            delayed = true;
                        } else {
                            assert!(pool.submit((request.slot, task)).is_ok());
                        }
                    }
                }
                if let Some((slot, receipt)) = pool.receive() {
                    let completed = run.complete(slot, receipt).unwrap();
                    assert_eq!(completed.error, None);
                    drop(completed);
                } else {
                    // All later chunks finished while the first preparation
                    // was suspended. Their retained results cannot consume its
                    // reserved slot or trigger unbounded successor production.
                    let mut task = held.take().expect("admitted chunks must make progress");
                    assert_eq!(run.inflight(), 1);
                    assert_eq!(run.ready(&mut ready), 0);
                    task.execute().unwrap();
                    drop(run.complete(0, task.finish()).unwrap());
                }
                assert!(!run.is_failed());
            }
            assert_eq!(run.result(), Some(expected));
            assert_eq!(run.inflight(), 0);
        }
    });
}

fn representations<C: PastaCurve>() {
    let bases: Vec<_> = (1..=19)
        .map(|i| {
            *ladder::<C>(PastaField::from_u64(i))
                .to_point()
                .as_affine()
                .unwrap()
        })
        .collect();
    let prepared_bases: Vec<_> = bases.iter().map(PreparedAffinePoint::from_affine).collect();
    let points: Vec<_> = bases
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if i % 7 == 0 {
                Point::IDENTITY
            } else {
                p.to_projective().to_point()
            }
        })
        .collect();
    let r = EisensteinTableBatch::<C>::requirements(bases.len()).unwrap();
    let mut entries = vec![AffinePoint::GENERATOR; r.table_entries];
    let mut cached_entries =
        vec![PreparedAffinePoint::from_affine(&AffinePoint::GENERATOR); r.table_entries];
    let compact = EisensteinTableBatch::prepare(
        &bases,
        &mut entries,
        &mut vec![ProjectivePoint::IDENTITY; r.projective_scratch],
        &mut vec![PastaField::ZERO; r.field_scratch],
        TaskBudget::SERIAL,
        &SerialExecutor,
    )
    .unwrap();
    let compact_cached = EisensteinTableBatch::prepare(
        &bases,
        &mut cached_entries,
        &mut vec![ProjectivePoint::IDENTITY; r.projective_scratch],
        &mut vec![PastaField::ZERO; r.field_scratch],
        TaskBudget::SERIAL,
        &SerialExecutor,
    )
    .unwrap();
    let indices: Vec<_> = (0..59).map(|i| (i * 7 % bases.len()) as u32).collect();
    for source in [
        Bases::Affine(&bases),
        Bases::Prepared(&prepared_bases),
        Bases::Points(&points),
        Bases::Compact(compact),
        Bases::CompactPrepared(compact_cached),
    ] {
        for indexed in [false, true] {
            let selection = if indexed {
                Selection::indexed(source, &indices).unwrap()
            } else {
                Selection::new(source)
            };
            let scalars: Vec<_> = (0..selection.len())
                .map(|i| {
                    PastaField::<C::Scalar>::from_u64(i as u64 + 2)
                        .invert()
                        .unwrap()
                })
                .collect();
            let unsigned: Vec<_> = (0..selection.len())
                .map(|i| [0, 1, u128::MAX, 1 << 127, i as u128][i % 5])
                .collect();
            let signed: Vec<_> = (0..selection.len())
                .map(|i| [0, -1, i128::MIN, i128::MAX, i as i128][i % 5])
                .collect();
            let uint: Vec<_> = unsigned
                .iter()
                .map(|&i| CanonicalUint::from_limbs([i as u64, (i >> 64) as u64, 0, 0]))
                .collect();
            let unsigned_fields: Vec<_> = uint
                .iter()
                .map(|&i| PastaField::<C::Scalar>::from_canonical_uint(i).unwrap())
                .collect();
            let signed_fields: Vec<_> =
                signed
                    .iter()
                    .map(|i| {
                        let i_abs = i.unsigned_abs();
                        let f = PastaField::<C::Scalar>::from_canonical_uint(
                            CanonicalUint::from_limbs([i_abs as u64, (i_abs >> 64) as u64, 0, 0]),
                        )
                        .unwrap();
                        if *i < 0 { f.neg() } else { f }
                    })
                    .collect();
            let expected = |scalars: &[PastaField<C::Scalar>]| {
                ladder::<C>(scalars.iter().enumerate().fold(
                    PastaField::ZERO,
                    |sum, (i, scalar)| {
                        let index = if indexed { indices[i] as usize } else { i };
                        let weight = if matches!(source, Bases::Points(_)) && index % 7 == 0 {
                            0
                        } else {
                            index + 1
                        };
                        sum.add(&scalar.mul(&PastaField::from_u64(weight as u64)))
                    },
                ))
            };
            let mut records = vec![ScalarStorage::ZERO; selection.len()];
            let prepared = PreparedScalars::prepare(
                &scalars,
                &mut records,
                TaskBudget::SERIAL,
                &SerialExecutor,
            )
            .unwrap();
            let mut digits = vec![0; prepared.cache_len(ExecutionOptions::SERIAL).unwrap()];
            let cached = prepared
                .cache(ExecutionOptions::SERIAL, &mut digits)
                .unwrap();
            for (input, answer, retained) in [
                (
                    selection.with_scalars(&scalars).unwrap(),
                    expected(&scalars),
                    false,
                ),
                (
                    selection.with_unsigned(&unsigned).unwrap(),
                    expected(&unsigned_fields),
                    false,
                ),
                (
                    selection.with_signed(&signed).unwrap(),
                    expected(&signed_fields),
                    false,
                ),
                (
                    selection.with_canonical(&uint, 128).unwrap(),
                    expected(&unsigned_fields),
                    false,
                ),
                (
                    selection.with_prepared_scalars(prepared).unwrap(),
                    expected(&scalars),
                    true,
                ),
                (
                    selection.with_prepared_scalars(cached).unwrap(),
                    expected(&scalars),
                    true,
                ),
            ] {
                chunks(
                    input,
                    answer,
                    ExecutionOptions::SERIAL,
                    8,
                    retained,
                    false,
                    true,
                );
            }
            chunks(
                selection.with_prepared_scalars(cached).unwrap(),
                expected(&scalars),
                ExecutionOptions::SERIAL,
                64,
                true,
                true,
                false,
            );
        }
    }
}

#[test]
fn independent_chunks_reuse_all_scalar_and_base_representations() {
    representations::<Pallas>();
    representations::<Vesta>();
}

#[test]
fn independent_booth_chunks_and_empty_runs() {
    let bases = [AffinePoint::<Pallas>::GENERATOR; 385];
    let scalars: Vec<_> = (0..bases.len())
        .map(|i| PastaField::from_u64(i as u64 + 2).invert().unwrap())
        .collect();
    let expected = ladder::<Pallas>(scalars.iter().fold(PastaField::ZERO, |sum, s| sum.add(s)));
    for width in 4..=12 {
        let options = ExecutionOptions::SERIAL.with_booth_width(width).unwrap();
        chunks(
            Input::new(Bases::Affine(&bases), &scalars).unwrap(),
            expected,
            options,
            128,
            false,
            false,
            true,
        );
        let empty = Input::new(Bases::Affine(&bases[..0]), &scalars[..0]).unwrap();
        let plan = MsmPlan::<Pallas>::new(0, options, NonZeroUsize::new(128).unwrap()).unwrap();
        assert_eq!(plan.retained().bytes::<Pallas>().unwrap(), 0);
        assert_eq!(plan.temporary().bytes::<Pallas>().unwrap(), 0);
        chunks(
            empty,
            ProjectivePoint::IDENTITY,
            options,
            128,
            false,
            false,
            false,
        );
    }
}
