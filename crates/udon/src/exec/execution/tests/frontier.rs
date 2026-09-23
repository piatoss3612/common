use super::admission;
use crate::exec::execution::frontier::ReadyRange;
use crate::exec::execution::test_pool as run_pool;
use admission::*;
use std::vec::Vec;

use std::panic::{AssertUnwindSafe, catch_unwind};
use zakura_udon::exec::execution::{Frontier, Identity, Kernel, Outcome, TaskError, TaskStorage};

struct Add {
    value: u64,
    fail: bool,
}
impl Kernel<&mut [u64]> for Add {
    type Output = u64;
    fn execute(&mut self, values: &mut &mut [u64]) -> u64 {
        values[0] += self.value;
        assert!(!self.fail, "injected kernel panic");
        values.iter().sum()
    }
}

#[test]
fn detached_scoped_claims_complete_independently_and_drain_panics() {
    let mut data = [1, 2, 3, 4];
    let mut identity = Identity::new();
    let mut slots = [const { TaskStorage::EMPTY }; 2];
    let mut frontier = Frontier::new(&mut identity, &mut slots, 3);
    let mut ready = [ReadyRange::EMPTY];
    assert_eq!(frontier.ready(&mut ready), 1);
    let keys: Vec<_> = ready[0].tasks().collect();
    assert_eq!(keys.len(), 2);
    let (left, right) = data.split_at_mut(2);
    let mut left = Some(left);
    let mut right = Some(right);
    assert!(
        frontier
            .try_claim(
                keys[0],
                Add {
                    value: 10,
                    fail: false
                },
                || None::<&mut [u64]>
            )
            .unwrap()
            .is_none()
    );
    let first = frontier
        .try_claim(
            keys[0],
            Add {
                value: 10,
                fail: true,
            },
            || left.take(),
        )
        .unwrap()
        .unwrap();
    let second = frontier
        .try_claim(
            keys[1],
            Add {
                value: 20,
                fail: false,
            },
            || right.take(),
        )
        .unwrap()
        .unwrap();
    assert!(matches!(
        frontier.try_claim(
            keys[0],
            Add {
                value: 0,
                fail: false
            },
            || panic!("duplicate acquired resources")
        ),
        Err(TaskError::Claimed)
    ));
    std::thread::scope(|scope| {
        let (release, wait) = std::sync::mpsc::sync_channel(0);
        let first = scope.spawn(move || {
            let mut task = first;
            wait.recv().unwrap();
            assert!(catch_unwind(AssertUnwindSafe(|| task.execute())).is_err());
            assert_eq!(task.execute(), Err(TaskError::Executed));
            task.finish()
        });
        let second = scope.spawn(move || {
            let mut task = second;
            task.execute().unwrap();
            task.finish()
        });
        let completed = frontier.complete(second.join().unwrap()).unwrap();
        assert_eq!(completed.output, Some(27));
        assert_eq!(completed.retired, 0..0);
        assert_eq!(frontier.inflight(), 1);
        assert_eq!(
            frontier.ready(&mut ready),
            0,
            "partial frontier must not overtake missing first result"
        );
        release.send(()).unwrap();
        let completed = frontier.complete(first.join().unwrap()).unwrap();
        assert_eq!(completed.outcome, Outcome::Failed);
        assert_eq!(completed.output, None);
        assert_eq!(frontier.inflight(), 0);
        assert!(frontier.is_failed());
        assert!(!frontier.is_complete());
        assert_eq!(frontier.ready(&mut ready), 0);
    });
    assert_eq!(data, [11, 2, 23, 4]);
}

#[test]
fn frontier_is_bounded_rejects_foreign_receipts_and_reuses_epochs() {
    let mut identity = Identity::new();
    let mut slots = [const { TaskStorage::EMPTY }; 3];
    let mut frontier = Frontier::new(&mut identity, &mut slots, 100);
    let mut foreign_identity = Identity::new();
    let mut foreign_slots = [TaskStorage::EMPTY];
    let mut foreign = Frontier::new(&mut foreign_identity, &mut foreign_slots, 100);
    let mut output = [0];
    let mut ready = [ReadyRange::EMPTY];
    let mut old_key = None;
    for i in 0..100 {
        assert_eq!(frontier.ready(&mut ready), 1);
        assert!(ready[0].len() <= 3);
        let key = ready[0].tasks().next().unwrap();
        old_key.get_or_insert(key);
        assert_eq!(key.index(), i);
        let mut task = frontier
            .try_claim(
                key,
                Add {
                    value: 1,
                    fail: false,
                },
                || Some(&mut output[..]),
            )
            .unwrap()
            .unwrap();
        task.execute().unwrap();
        let (error, receipt) = foreign.complete(task.finish()).unwrap_err();
        assert_eq!(error, TaskError::Stale);
        assert_eq!(frontier.complete(receipt).unwrap().retired, i..i + 1);
    }
    assert_eq!(output, [100]);
    assert!(frontier.is_complete());
    frontier.restart(1).unwrap();
    assert!(matches!(
        frontier.try_claim(
            old_key.unwrap(),
            Add {
                value: 1,
                fail: false
            },
            || Some(&mut output[..])
        ),
        Err(TaskError::Stale)
    ));
    assert!(!frontier.is_complete());
}

#[test]
fn admission_counts_typed_capacity_pending_outputs_and_exact_fit() {
    let layout = ArenaLayout {
        classes: [
            BlockClass {
                blocks: 5,
                block_bytes: 32,
            },
            BlockClass {
                blocks: 2,
                block_bytes: 96,
            },
        ],
        metadata_bytes: 64,
    };
    assert_eq!(layout.check(416), Ok(416));
    assert_eq!(layout.check(415), Err(ResourceError::Capacity));
    let mut identity = Identity::new();
    let mut storage = [const { SegmentStorage::EMPTY }; 2];
    let mut admission = Admission::new(&mut identity, &mut storage, layout.capacity());
    let profile = Profile {
        retained: Resources([1, 1]),
        temporary: Resources([3, 0]),
    };
    let first = admission.admit(profile).unwrap();
    let second = admission.admit(profile).unwrap();
    let work = admission
        .try_task(&first, Resources([3, 0]), Resources([1, 1]))
        .unwrap();
    assert!(matches!(
        admission.try_task(&second, Resources([3, 0]), Resources([1, 1])),
        Err(ResourceError::Capacity)
    ));
    assert_eq!(admission.used(), Resources([4, 1]));
    assert_eq!(
        admission.release_retained(&first, Resources([1, 1])),
        Err(ResourceError::InvalidRelease)
    );
    admission.finish_task(work, true).unwrap();
    let work = admission
        .try_task(&second, Resources([3, 0]), Resources([1, 1]))
        .unwrap();
    assert_eq!(admission.used(), Resources([5, 2]));
    admission.finish_task(work, false).unwrap();
    admission
        .release_retained(&first, Resources([1, 1]))
        .unwrap();
    admission.retire(first).unwrap();
    admission.retire(second).unwrap();
    assert_eq!(admission.used(), Resources::ZERO);
}

#[test]
fn all_small_admitted_profiles_have_an_escape_bundle() {
    // Exhaust small two-class, two-segment profiles with all retained maxima
    // live. This dominates every smaller live retained count componentwise.
    // Once running tasks return, each declared temporary bundle must fit.
    for a in 0..3_usize.pow(8) {
        let mut encoded = a;
        let values: [usize; 8] = core::array::from_fn(|_| {
            let n = encoded % 3;
            encoded /= 3;
            n
        });
        let profiles = [
            Profile {
                retained: Resources([values[0], values[1]]),
                temporary: Resources([values[2], values[3]]),
            },
            Profile {
                retained: Resources([values[4], values[5]]),
                temporary: Resources([values[6], values[7]]),
            },
        ];
        let capacity = Resources([
            values[0] + values[4] + values[2].max(values[6]),
            values[1] + values[5] + values[3].max(values[7]),
        ]);
        let mut identity = Identity::new();
        let mut storage = [const { SegmentStorage::EMPTY }; 2];
        let mut admission = Admission::new(&mut identity, &mut storage, capacity);
        let segments = profiles.map(|profile| admission.admit(profile).unwrap());
        for (segment, profile) in segments.iter().zip(profiles) {
            let task = admission
                .try_task(segment, Resources::ZERO, profile.retained)
                .unwrap();
            admission.finish_task(task, true).unwrap();
        }
        for (segment, profile) in segments.iter().zip(profiles) {
            let task = admission
                .try_task(segment, profile.temporary, Resources::ZERO)
                .unwrap();
            admission.finish_task(task, true).unwrap();
        }
        for (segment, profile) in segments.into_iter().zip(profiles) {
            admission
                .release_retained(&segment, profile.retained)
                .unwrap();
            admission.retire(segment).unwrap();
        }
    }
}

#[test]
fn shared_read_leases_move_between_workers_then_release_for_writing() {
    use spin::{RwLock, RwLockReadGuard, RwLockWriteGuard};
    enum Lease<'a> {
        Read(RwLockReadGuard<'a, [u64; 4]>),
        Write(RwLockWriteGuard<'a, [u64; 4]>),
    }
    struct Use;
    impl Kernel<Lease<'_>> for Use {
        type Output = u64;
        fn execute(&mut self, lease: &mut Lease<'_>) -> u64 {
            match lease {
                Lease::Read(values) => values.iter().sum(),
                Lease::Write(values) => {
                    values.fill(9);
                    values.iter().sum()
                }
            }
        }
    }
    for workers in [1, 3, 4] {
        let block = RwLock::new([1, 2, 3, 4]);
        let mut identity = Identity::new();
        let mut slots = [const { TaskStorage::EMPTY }; 2];
        let mut run = Frontier::new(&mut identity, &mut slots, 2);
        let mut ready = [ReadyRange::EMPTY];
        run_pool::scoped(workers, 2, |pool| {
            assert!(pool.queue_bytes() > 0);
            run.ready(&mut ready);
            for key in ready[0].tasks() {
                assert!(pool.available());
                let task = run
                    .try_claim(key, Use, || block.try_read().map(Lease::Read))
                    .unwrap()
                    .unwrap();
                assert!(pool.submit(task).is_ok());
            }
            assert!(block.try_write().is_none());
            let first = run.complete(pool.receive().unwrap()).unwrap();
            assert_eq!(first.output, Some(10));
            drop(first);
            assert!(
                block.try_write().is_none(),
                "last reader owns its guard through publication"
            );
            let second = run.complete(pool.receive().unwrap()).unwrap();
            assert_eq!(second.output, Some(10));
            drop(second);
            run.restart(1).unwrap();
            run.ready(&mut ready);
            let task = run
                .try_claim(ready[0].tasks().next().unwrap(), Use, || {
                    block.try_write().map(Lease::Write)
                })
                .unwrap()
                .unwrap();
            assert!(pool.submit(task).is_ok());
            assert_eq!(
                run.complete(pool.receive().unwrap()).unwrap().output,
                Some(36)
            );
        });
        assert_eq!(*block.read(), [9; 4]);
    }
}
