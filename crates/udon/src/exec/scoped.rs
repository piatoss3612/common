//! Borrowed item and chunk scheduling over a caller-supplied executor.

use super::{Executor, TaskBudget};

/// Exposes each item as an independent scoped job with its original index.
///
/// Items may own buffers or borrow disjoint mutable data. Each callback receives
/// the item's zero-based position in `values`, regardless of execution order.
/// A cooperative executor can schedule items separately even when their work
/// takes different amounts of time.
///
/// This helper does not allocate or impose a task budget. Callers supply each
/// job's resources and any nested budget; the executor controls worker
/// concurrency. Use [`for_each_mut`] to divide one allowance among outer jobs
/// and their nested work.
///
/// On success, every item is visited exactly once, in unspecified order. Empty
/// input invokes no callbacks.
///
/// # Panics
///
/// A callback panic may leave partial changes. All callbacks are still invoked,
/// and all jobs finish or unwind before the panic propagates, following
/// [`Executor::join`]. A second panic during unwinding may abort the process.
pub(crate) fn for_each_task_mut<T, E, F>(values: &mut [T], executor: &E, work: F)
where
    T: Send,
    E: Executor + ?Sized,
    F: Fn(usize, &mut T) + Sync,
{
    fn visit<T, E, F>(values: &mut [T], offset: usize, executor: &E, work: &F)
    where
        T: Send,
        E: Executor + ?Sized,
        F: Fn(usize, &mut T) + Sync,
    {
        match values {
            [] => (),
            [value] => work(offset, value),
            _ => {
                let mid = values.len() / 2;
                let (left, right) = values.split_at_mut(mid);
                executor.join(
                    || visit(left, offset, executor, work),
                    || visit(right, offset + mid, executor, work),
                );
            }
        }
    }
    visit(values, 0, executor, &work);
}

/// Applies `work` to each item with its index and nested task allowance.
///
/// Items may own separate buffers, or be mutable references to separate tiles.
/// Each index is the item's zero-based position in `values`, regardless of
/// execution order. Scheduling, budget division, empty input, and panic behavior
/// follow [`for_each_chunk_mut`] with one item per chunk.
pub fn for_each_mut<T, E, F>(values: &mut [T], budget: TaskBudget, executor: &E, work: F)
where
    T: Send,
    E: Executor + ?Sized,
    F: Fn(usize, &mut T, TaskBudget) + Sync,
{
    for_each_chunk_mut(values, 1, budget, executor, |index, chunk, inner| {
        work(index, &mut chunk[0], inner);
    });
}

/// Applies `work` to each consecutive chunk with its index and nested allowance.
///
/// Chunks have `chunk_len` items, except the last, which may be shorter. Callback
/// index `i` identifies the chunk starting at `i * chunk_len` in `values`,
/// regardless of execution order.
///
/// For `chunk_count` chunks, [`TaskBudget::partition`] chooses at most
/// `min(chunk_count, budget.get())` concurrent jobs, each processing one or more
/// chunks sequentially. Every callback receives the same inner allowance.
/// Callbacks must honor it to keep nested execution within `budget`.
///
/// On success, each chunk is visited exactly once. Order is unspecified. An empty
/// slice invokes no callbacks.
///
/// # Panics
///
/// Panics if `chunk_len` is zero, including for an empty slice.
/// A callback panic may leave partial changes and skip subsequent chunks in its
/// job; all scheduled jobs finish or unwind before the panic propagates.
pub fn for_each_chunk_mut<T, E, F>(
    values: &mut [T],
    chunk_len: usize,
    budget: TaskBudget,
    executor: &E,
    work: F,
) where
    T: Send,
    E: Executor + ?Sized,
    F: Fn(usize, &mut [T], TaskBudget) + Sync,
{
    assert_ne!(chunk_len, 0, "chunk length must be nonzero");
    let Some((jobs, inner)) = budget.partition(values.len().div_ceil(chunk_len)) else {
        return;
    };

    fn visit<T, E, F>(
        values: &mut [T],
        chunk_len: usize,
        offset: usize,
        jobs: usize,
        inner: TaskBudget,
        executor: &E,
        work: &F,
    ) where
        T: Send,
        E: Executor + ?Sized,
        F: Fn(usize, &mut [T], TaskBudget) + Sync,
    {
        if jobs == 1 {
            for (index, chunk) in values.chunks_mut(chunk_len).enumerate() {
                work(offset + index, chunk, inner);
            }
        } else {
            // Halving both counts leaves at least one chunk per job in each
            // branch. Only the final chunk can be partial, so the left split
            // stays within the slice and preserves the chunk boundaries.
            let left_chunks = values.len().div_ceil(chunk_len) / 2;
            let (left, right) = values.split_at_mut(left_chunks * chunk_len);
            let left_jobs = jobs / 2;
            // Repartitioning inner here would divide the same budget twice.
            executor.join(
                || visit(left, chunk_len, offset, left_jobs, inner, executor, work),
                || {
                    visit(
                        right,
                        chunk_len,
                        offset + left_chunks,
                        jobs - left_jobs,
                        inner,
                        executor,
                        work,
                    )
                },
            );
        }
    }
    visit(values, chunk_len, 0, jobs, inner, executor, &work);
}
