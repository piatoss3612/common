use super::{Arena, Work, run_pool};
use spin::RwLock;
use std::num::NonZeroUsize;
use std::{vec, vec::Vec};
use zakura_udon::{
    curve::{
        Pallas, PastaCurve, Point, ProjectivePoint, Vesta,
        msm::{
            Bases, Input,
            execution::{MsmPlan, ParallelMsmRun, ProducedInput},
        },
    },
    exec::{
        ExecutionOptions, TaskBudget,
        execution::{Identity, TaskStorage},
    },
    field::{CanonicalUint, PastaField},
};

fn ladder<C: PastaCurve>(scalar: PastaField<C::Scalar>) -> ProjectivePoint<C> {
    let limbs = scalar.to_canonical_uint().limbs();
    let mut sum = ProjectivePoint::IDENTITY;
    for bit in (0..256).rev() {
        sum = sum.double();
        if (limbs[bit / 64] >> (bit % 64)) & 1 != 0 {
            sum = sum.add(&ProjectivePoint::GENERATOR);
        }
    }
    sum
}

fn chunks<C: PastaCurve>(input: Input<'_, C>, expected: ProjectivePoint<C>, bases: &[Point<C>]) {
    let plan = MsmPlan::for_produced(
        ProducedInput::dense(Bases::Points(bases)),
        NonZeroUsize::new(255).unwrap(),
        ExecutionOptions::default().with_task_budget(TaskBudget::new(4).unwrap()),
    )
    .unwrap();
    assert!(plan.grain() <= 255);
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
                    if let Some(task) = run
                        .try_claim(*request, || {
                            arenas[request.slot].acquire(request.task, &work)
                        })
                        .unwrap()
                    {
                        if !delayed && request.slot == 0 && request.task.scratch.scalars() != 0 {
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

fn suffix_chunks<C: PastaCurve>() {
    use zakura_udon::msm::SuffixBasis;
    const N: usize = 513;
    let bases = [Point::<C>::GENERATOR; N];
    let mut sums = [Point::IDENTITY; N];
    let basis = SuffixBasis::prepare(
        &bases,
        &mut sums,
        &mut [ProjectivePoint::IDENTITY; 7],
        &mut [PastaField::ZERO; 3],
    );
    let fields: Vec<_> = (0..N)
        .map(|i| {
            PastaField::<C::Scalar>::from_u64((i / 11 + 2) as u64)
                .invert()
                .unwrap()
        })
        .collect();
    let unsigned: Vec<_> = (0..N).map(|i| (1_u128 << 96) + (i / 13) as u128).collect();
    let unsigned_sum = unsigned
        .iter()
        .fold(PastaField::<C::Scalar>::ZERO, |sum, n| {
            sum.add(
                &PastaField::<C::Scalar>::from_canonical_uint(CanonicalUint::from_limbs([
                    *n as u64,
                    (n >> 64) as u64,
                    0,
                    0,
                ]))
                .unwrap(),
            )
        });
    let mut integer_differences = vec![0; N];
    let mut field_differences = vec![PastaField::ZERO; N];
    for (input, answer) in [
        (
            basis.with_scalars(&fields, &mut field_differences),
            ladder::<C>(fields.iter().fold(PastaField::ZERO, |sum, n| sum.add(n))),
        ),
        (
            basis
                .with_monotone_unsigned(&unsigned, &mut integer_differences)
                .unwrap(),
            ladder::<C>(unsigned_sum),
        ),
    ] {
        chunks(input, answer, basis.suffix());
    }
}

#[test]
fn suffix_differences_survive_independent_chunks_and_delayed_preparation() {
    suffix_chunks::<Pallas>();
    suffix_chunks::<Vesta>();
}
