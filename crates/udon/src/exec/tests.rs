use super::*;
use core::{
    cell::Cell,
    sync::atomic::{AtomicUsize, Ordering},
};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    vec,
    vec::Vec,
};

struct RayonExecutor;

impl Executor for RayonExecutor {
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

struct DropCount<'a>(&'a AtomicUsize);

#[test]
fn independent_borrowed_jobs_keep_indices_and_complete_on_panic() {
    fn check(executor: &impl Executor) {
        for len in [0, 1, 3, 128] {
            let mut values = vec![Cell::new(0); len];
            let mut borrowed: Vec<_> = values.iter_mut().collect();
            for_each_task_mut(&mut borrowed, executor, |index, value| {
                value.set(value.get() + index + 1);
            });
            assert!(
                values
                    .iter()
                    .enumerate()
                    .all(|(i, value)| value.get() == i + 1)
            );
        }
        for failing in [0, 7, 16] {
            let mut values = [0; 17];
            let panic = catch_unwind(AssertUnwindSafe(|| {
                for_each_task_mut(&mut values, executor, |index, value| {
                    *value += 1;
                    assert_ne!(index, failing, "independent job failure");
                });
            }));
            assert!(panic.is_err());
            assert_eq!(values, [1; 17]);
        }
    }
    check(&SerialExecutor);
    for workers in [1, 2, 4] {
        rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap()
            .install(|| check(&RayonExecutor));
    }
}

impl Drop for DropCount<'_> {
    fn drop(&mut self) {
        self.0.fetch_add(1, Ordering::SeqCst);
    }
}

fn join_results(executor: &impl Executor) {
    let mut values = [1, 2, 3, 4];
    let (left, right) = values.split_at_mut(2);
    let (changed, sum) = executor.join(
        || {
            left[0] += 10;
            left
        },
        || {
            right[1] *= 2;
            right.iter().sum::<i32>()
        },
    );
    assert_eq!(changed, &[11, 2]);
    assert_eq!(sum, 11);
    changed[1] += sum;
    assert_eq!(values, [11, 13, 3, 8]);

    // Cell is Send but not Sync, both as a moved capture and as a result.
    let input = Cell::new(3);
    let (cell, array) = executor.join(
        move || {
            input.set(input.get() + 1);
            input
        },
        || [5u8, 6],
    );
    assert_eq!(cell.get(), 4);
    assert_eq!(array, [5, 6]);

    let consumed = AtomicUsize::new(0);
    let retained = AtomicUsize::new(0);
    let (left_input, right_input) = (DropCount(&consumed), DropCount(&consumed));
    let (left_output, right_output) = (DropCount(&retained), DropCount(&retained));
    let results = executor.join(
        move || {
            drop(left_input);
            left_output
        },
        move || {
            drop(right_input);
            right_output
        },
    );
    assert_eq!(consumed.load(Ordering::SeqCst), 2);
    assert_eq!(retained.load(Ordering::SeqCst), 0);
    drop(results);
    assert_eq!(retained.load(Ordering::SeqCst), 2);
}

fn join_panics(executor: &impl Executor) {
    for failing_side in 0..2 {
        let calls = [AtomicUsize::new(0), AtomicUsize::new(0)];
        let captures = AtomicUsize::new(0);
        let results = AtomicUsize::new(0);
        let left_capture = DropCount(&captures);
        let right_capture = DropCount(&captures);
        let run = |side: usize| {
            calls[side].fetch_add(1, Ordering::SeqCst);
            let result = DropCount(&results);
            assert_ne!(side, failing_side, "job failure");
            result
        };
        let panic = catch_unwind(AssertUnwindSafe(|| {
            executor.join(
                move || {
                    let _capture = left_capture;
                    run(0)
                },
                move || {
                    let _capture = right_capture;
                    run(1)
                },
            )
        }));
        assert!(panic.is_err());
        assert_eq!(
            calls.each_ref().map(|count| count.load(Ordering::SeqCst)),
            [1, 1]
        );
        assert_eq!(captures.load(Ordering::SeqCst), 2);
        // The successful result must be dropped before the other branch's panic
        // escapes, regardless of which branch finished first.
        assert_eq!(results.load(Ordering::SeqCst), 2);
    }
}

#[test]
fn serial_join_returns_values_and_completes_both_branches_on_panic() {
    join_results(&SerialExecutor);
    join_panics(&SerialExecutor);
}

#[test]
fn budgets_are_nonzero_and_partition_without_overflow() {
    const ODD: TaskBudget = TaskBudget::new(7).unwrap();
    const SHARES: (TaskBudget, TaskBudget) = ODD.split_at(3).unwrap();
    const PARTITION: (usize, TaskBudget) = ODD.partition(3).unwrap();
    const MAX: TaskBudget = TaskBudget::new(usize::MAX).unwrap();
    const MAX_SHARES: (TaskBudget, TaskBudget) = MAX.split_at(usize::MAX - 1).unwrap();
    const MAX_PARTITION: (usize, TaskBudget) = MAX.partition(2).unwrap();

    assert_eq!(TaskBudget::new(0), None);
    assert_eq!(TaskBudget::SERIAL.get(), 1);
    assert_eq!((SHARES.0.get(), SHARES.1.get()), (3, 4));
    assert_eq!((PARTITION.0, PARTITION.1.get()), (3, 2));
    assert_eq!(ODD.partition(0), None);
    assert_eq!(ODD.partition(usize::MAX), Some((7, TaskBudget::SERIAL)));
    assert_eq!(
        TaskBudget::SERIAL.partition(usize::MAX),
        Some((1, TaskBudget::SERIAL))
    );
    for invalid in [0, 7, usize::MAX] {
        assert_eq!(ODD.split_at(invalid), None);
    }
    assert_eq!(TaskBudget::SERIAL.split_at(1), None);
    assert_eq!(
        (MAX_SHARES.0.get(), MAX_SHARES.1.get()),
        (usize::MAX - 1, 1)
    );
    assert_eq!(
        (MAX_PARTITION.0, MAX_PARTITION.1.get()),
        (2, usize::MAX / 2)
    );
    assert_eq!(
        MAX.partition(usize::MAX),
        Some((usize::MAX, TaskBudget::SERIAL))
    );
}

#[test]
fn chunks_cover_inputs_with_stable_indices_and_uniform_allowances() {
    for len in 0..=33 {
        for chunk_len in [1, 2, 3, 7, 64, usize::MAX] {
            for tasks in [1, 2, 3, 5, 64, usize::MAX] {
                let mut values = vec![0; len];
                let chunks = len.div_ceil(chunk_len);
                let calls = AtomicUsize::new(0);
                for_each_chunk_mut(
                    &mut values,
                    chunk_len,
                    TaskBudget::new(tasks).unwrap(),
                    &SerialExecutor,
                    |index, values, inner| {
                        assert_eq!(inner.get(), tasks / chunks.min(tasks));
                        assert_eq!(values.len(), chunk_len.min(len - index * chunk_len));
                        calls.fetch_add(1, Ordering::SeqCst);
                        for (offset, value) in values.iter_mut().enumerate() {
                            *value += index * chunk_len + offset + 1;
                        }
                    },
                );
                assert_eq!(values, (1..=len).collect::<Vec<_>>());
                assert_eq!(calls.load(Ordering::SeqCst), chunks);
            }
        }
    }
}

#[test]
fn zero_chunk_length_panics_before_work_even_for_empty_inputs() {
    for len in [0, 2] {
        let mut values = [0; 2];
        let calls = AtomicUsize::new(0);
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                for_each_chunk_mut(
                    &mut values[..len],
                    0,
                    TaskBudget::SERIAL,
                    &SerialExecutor,
                    |_, _, _| {
                        calls.fetch_add(1, Ordering::SeqCst);
                    },
                );
            }))
            .is_err()
        );
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }
}

#[test]
fn items_keep_separate_owners_and_accept_send_only_contents() {
    let mut tiles = [vec![Cell::new(0); 3], vec![Cell::new(0); 1], vec![]];
    let addresses = tiles.each_ref().map(|tile| tile.as_ptr());
    for_each_mut(
        &mut tiles,
        TaskBudget::new(7).unwrap(),
        &SerialExecutor,
        |index, tile, inner| {
            assert_eq!(inner.get(), 2);
            for value in tile {
                value.set(index + 1);
            }
        },
    );
    for (index, tile) in tiles.iter().enumerate() {
        assert_eq!(tile.as_ptr(), addresses[index]);
        assert!(tile.iter().all(|value| value.get() == index + 1));
    }
    for_each_mut::<usize, _, _>(&mut [], TaskBudget::SERIAL, &SerialExecutor, |_, _, _| {
        panic!("empty input invoked work");
    });
}

struct Active<'a> {
    current: &'a AtomicUsize,
    tasks: usize,
}

impl Drop for Active<'_> {
    fn drop(&mut self) {
        self.current.fetch_sub(self.tasks, Ordering::SeqCst);
    }
}

fn nested_work(executor: &impl Executor) {
    for tasks in [1, 2, 3, 7, 19] {
        for count in [1, 3, 8] {
            let mut tiles: Vec<Vec<usize>> = (0..count).map(|i| vec![0; 7 + i]).collect();
            let current = AtomicUsize::new(0);
            let peak = AtomicUsize::new(0);
            for_each_mut(
                &mut tiles,
                TaskBudget::new(tasks).unwrap(),
                executor,
                |index, tile, inner| {
                    assert_eq!(inner.get(), tasks / count.min(tasks));
                    let chunks = tile.len().div_ceil(2);
                    for_each_chunk_mut(tile, 2, inner, executor, |chunk, values, leaf| {
                        assert_eq!(leaf.get(), inner.get() / chunks.min(inner.get()));
                        let active = current.fetch_add(leaf.get(), Ordering::SeqCst) + leaf.get();
                        peak.fetch_max(active, Ordering::SeqCst);
                        let _active = Active {
                            current: &current,
                            tasks: leaf.get(),
                        };
                        std::thread::yield_now();
                        for (offset, value) in values.iter_mut().enumerate() {
                            *value += 100 * index + 2 * chunk + offset + 1;
                        }
                    });
                },
            );
            assert_eq!(current.load(Ordering::SeqCst), 0);
            assert!((1..=tasks).contains(&peak.load(Ordering::SeqCst)));
            for (index, tile) in tiles.iter().enumerate() {
                assert_eq!(
                    *tile,
                    (1..=tile.len())
                        .map(|i| 100 * index + i)
                        .collect::<Vec<_>>()
                );
            }
        }
    }
}

fn helper_panic_cleanup(executor: &impl Executor) {
    let mut tiles = [[0; 8]; 3];
    let calls = AtomicUsize::new(0);
    let completed = AtomicUsize::new(0);
    let work_drops = AtomicUsize::new(0);
    let work_capture = DropCount(&work_drops);
    let panic = catch_unwind(AssertUnwindSafe(|| {
        let (calls, completed) = (&calls, &completed);
        for_each_mut(
            &mut tiles,
            TaskBudget::new(7).unwrap(),
            executor,
            move |index, tile, inner| {
                let _ = &work_capture;
                for_each_mut(tile, inner, executor, |offset, value, _| {
                    calls.fetch_add(1, Ordering::SeqCst);
                    let _completion = DropCount(completed);
                    assert!(index != 0 || offset != 0, "callback failure");
                    *value += 1;
                });
            },
        );
    }));
    assert!(panic.is_err());
    assert!(calls.load(Ordering::SeqCst) > 0);
    assert_eq!(
        completed.load(Ordering::SeqCst),
        calls.load(Ordering::SeqCst)
    );
    assert_eq!(work_drops.load(Ordering::SeqCst), 1);
    assert!(tiles.iter().flatten().all(|value| *value <= 1));
}

#[test]
fn serial_helpers_compose_and_clean_up_on_panic() {
    nested_work(&SerialExecutor);
    helper_panic_cleanup(&SerialExecutor);
}

#[test]
fn rayon_adapter_completes_nested_work_on_one_and_multiple_workers() {
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        // Enter the pool from a worker, exercising joins when no idle worker is
        // available as well as when branches can be stolen.
        pool.install(|| {
            join_results(&RayonExecutor);
            join_panics(&RayonExecutor);
            nested_work(&RayonExecutor);
            helper_panic_cleanup(&RayonExecutor);
        });
    }
}
