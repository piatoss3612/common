use crate::bridge::executor::{RayonExecutor, with_side_work};
use std::{
    panic::{AssertUnwindSafe, catch_unwind},
    sync::atomic::{AtomicUsize, Ordering},
};
use zakura_udon::exec::{Executor, TaskBudget, for_each_mut};

#[test]
fn selected_pool_borrowing_nested_progress_and_cleanup() {
    for threads in [1, 3, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let executor = RayonExecutor(&pool);
        let mut values = [0usize; 7];
        let budget = TaskBudget::new(threads).unwrap();
        for_each_mut(&mut values, budget, &executor, |index, value, inner| {
            assert!(inner.get() * budget.partition(7).unwrap().0 <= threads);
            // Explicit nesting even on one worker must make progress. The
            // results borrow the caller's stack and stay in left/right order.
            let (borrowed, count) = executor.join(
                || {
                    assert!(pool.current_thread_index().is_some());
                    *value = index;
                    value
                },
                || {
                    executor.join(
                        || {
                            assert!(pool.current_thread_index().is_some());
                            1
                        },
                        || 2,
                    )
                },
            );
            *borrowed += count.0 + count.1;
        });
        assert_eq!(values, [3, 4, 5, 6, 7, 8, 9]);

        // Even entry from a different pool must select this adapter's pool.
        let other = rayon::ThreadPoolBuilder::new()
            .num_threads(1)
            .build()
            .unwrap();
        other.install(|| {
            executor.join(
                || assert!(pool.current_thread_index().is_some()),
                || assert!(pool.current_thread_index().is_some()),
            )
        });

        let finished = AtomicUsize::new(0);
        let dropped = AtomicUsize::new(0);
        struct Guard<'a>(&'a AtomicUsize);
        impl Drop for Guard<'_> {
            fn drop(&mut self) {
                self.0.fetch_add(1, Ordering::SeqCst);
            }
        }
        let panic = catch_unwind(AssertUnwindSafe(|| {
            with_side_work(
                &executor,
                budget,
                |main| {
                    assert_eq!(main.get(), threads.saturating_sub(1).max(1));
                    panic!("main work failed")
                },
                |side| {
                    assert_eq!(side, TaskBudget::SERIAL);
                    finished.fetch_add(1, Ordering::SeqCst);
                    Guard(&dropped)
                },
            )
        }));
        assert!(panic.is_err());
        assert_eq!(finished.load(Ordering::SeqCst), 1);
        assert_eq!(dropped.load(Ordering::SeqCst), 1);
    }
}
