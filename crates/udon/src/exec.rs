//! Scoped execution and task budgets for caller-owned work.
//!
//! [`Executor`] lets a caller supply a runtime for two jobs that complete before
//! the call returns. [`TaskBudget`] divides a concurrency allowance between
//! outer jobs and their nested work. [`for_each_mut`] schedules separate buffer
//! owners or borrowed tiles; [`for_each_chunk_mut`] schedules contiguous chunks.
//! [`for_each_task_mut`] exposes independent jobs without dividing their nested
//! allowances. These helpers and [`SerialExecutor`] do not allocate. A caller's
//! executor and jobs may use their own resources.
//!
//! Jobs can return different types, including values borrowed from their inputs:
//!
//! ```
//! use zakura_udon::exec::{Executor, SerialExecutor};
//!
//! let mut values = [1, 2, 3, 4];
//! let (left, right) = values.split_at_mut(2);
//! let (first, sum) = SerialExecutor.join(
//!     || { left[0] += 10; &mut left[0] },
//!     || right.iter().sum::<i32>(),
//! );
//! assert_eq!((*first, sum), (11, 7));
//! ```
//!
//! Independent tiles can share a total task budget with nested FFTs. Each
//! callback receives its own allowance, passed here to
//! [`ExecutionOptions::with_task_budget`].
//! A slice of `Vec`s or other owners works the same way:
//!
//! ```
//! use zakura_udon::{
//!     exec::{ExecutionOptions, SerialExecutor, TaskBudget, for_each_mut},
//!     fft::{Domain, Transform},
//!     field::Fp,
//! };
//!
//! let plan = Transform::new(Domain::new(2).unwrap().subgroup());
//! let mut first = [Fp::ONE; 4];
//! let mut second = [Fp::from_u64(2); 4];
//! let mut tiles = [&mut first[..], &mut second[..]];
//! let budget = TaskBudget::new(4).unwrap();
//! for_each_mut(&mut tiles, budget, &SerialExecutor, |_, tile, inner| {
//!     let options = ExecutionOptions::default().with_task_budget(inner);
//!     let mut scratch = [Fp::ZERO; 4];
//!     plan.forward(tile, options, &SerialExecutor, &mut scratch).unwrap();
//!     plan.inverse(tile, options, &SerialExecutor, &mut scratch).unwrap();
//! });
//! assert_eq!(first, [Fp::ONE; 4]);
//! assert_eq!(second, [Fp::from_u64(2); 4]);
//! ```

use core::num::NonZeroUsize;

pub mod run;

/// Resource constraints for one arithmetic operation.
///
/// Udon selects the implementation within these limits. The byte ceiling covers
/// the used prefixes of arithmetic scratch and retained intermediates. Inputs,
/// outputs, persistent preparation, scheduling metadata, unused buffer tails,
/// alignment padding, stack frames, and executor resources are separate.
/// Supplied buffer capacities are always hard limits, even without a byte
/// ceiling. Reusing these options for concurrent invocations gives each its own
/// allowance; callers must account for their combined storage and concurrency.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionOptions {
    task_budget: TaskBudget,
    memory_limit: Option<usize>,
}

impl ExecutionOptions {
    /// One task with no additional workspace ceiling.
    pub const DEFAULT: Self = Self {
        task_budget: TaskBudget::SERIAL,
        memory_limit: None,
    };

    /// Sets the total allowance, including nested arithmetic work.
    ///
    /// Scoped execution divides this budget among its jobs. Incremental run
    /// callers control dispatch and must limit simultaneous task leases.
    pub const fn with_task_budget(mut self, budget: TaskBudget) -> Self {
        self.task_budget = budget;
        self
    }

    /// Limits arithmetic workspace to `bytes`; planning may use less.
    ///
    /// Zero permits implementations that need no arithmetic workspace. Planning
    /// returns a memory-limit error if required intermediates cannot fit.
    pub const fn with_memory_limit(mut self, bytes: usize) -> Self {
        self.memory_limit = Some(bytes);
        self
    }

    /// Requested concurrency allowance.
    pub const fn task_budget(self) -> TaskBudget {
        self.task_budget
    }

    /// Optional arithmetic workspace ceiling in bytes.
    pub const fn memory_limit(self) -> Option<usize> {
        self.memory_limit
    }

    pub(crate) fn for_scratch<T>(self, fields: usize) -> Self {
        let bytes = fields.saturating_mul(core::mem::size_of::<T>());
        self.with_memory_limit(self.memory_limit.map_or(bytes, |limit| limit.min(bytes)))
    }
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

/// Caller-supplied scoped fork/join execution.
///
/// Each job must be invoked exactly once. [`join`](Self::join) must wait until
/// both jobs have returned or unwound, including when propagating a panic.
/// No job may outlive the call. Job captures must be released before the call
/// returns or unwinds, except where the job transfers ownership into its result or
/// other storage. Jobs and results can borrow non-`'static` data; an
/// implementation may run jobs sequentially or on a caller's pool.
///
/// # Nested joins
///
/// Jobs can invoke further joins. An implementation must support these nested
/// calls even when every pool worker is executing a job, including in a pool
/// with only one worker. Task budgets limit work partitions; they do not reserve
/// idle workers for nested calls.
///
/// Queuing child jobs and blocking their parent workers can deadlock when all
/// workers are waiting for queued children. Increasing the pool size does not
/// satisfy the progress requirement.
///
/// Cooperative fork/join as provided by
/// [`rayon::join`](https://docs.rs/rayon/latest/rayon/fn.join.html) is suitable.
/// An adapter can delegate directly to it: workers can execute available work
/// while waiting for stolen branches. [`SerialExecutor`] satisfies the same
/// requirement by completing each branch synchronously.
pub trait Executor: Sync {
    /// Runs both jobs and returns their results in left, right order.
    ///
    /// Invocation order is unspecified. Nested calls must satisfy the trait's
    /// [progress requirement](Executor#nested-joins).
    ///
    /// # Panics
    ///
    /// If either job panics, the implementation must complete both jobs, drop
    /// any successful result, and propagate the panic. A second panic during
    /// unwinding may abort the process.
    fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
    where
        L: FnOnce() -> A + Send,
        R: FnOnce() -> B + Send,
        A: Send,
        B: Send;
}

/// Executes jobs synchronously without allocation.
///
/// Runs the left job followed by the right, including if the left job unwinds.
/// A second panic during unwinding aborts the process.
#[derive(Clone, Copy, Debug, Default)]
pub struct SerialExecutor;

impl Executor for SerialExecutor {
    fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
    where
        L: FnOnce() -> A + Send,
        R: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        // Keep the right job pending until the left finishes, so unwinding
        // still fulfills the executor's obligation to invoke both jobs.
        struct Pending<R: FnOnce()>(Option<R>);
        impl<R: FnOnce()> Drop for Pending<R> {
            fn drop(&mut self) {
                if let Some(right) = self.0.take() {
                    right();
                }
            }
        }
        let mut right_result = None;
        let pending = Pending(Some(|| right_result = Some(right())));
        let left_result = left();
        drop(pending);
        (left_result, right_result.unwrap())
    }
}

/// A nonzero allowance for concurrent work partitions, including nested work.
///
/// This is a copyable planning value, not a reservation of threads or a global
/// concurrency limiter. Divide it between simultaneous operations when their
/// combined work partitions must fit one allowance. Independent operations with
/// separate scratch can each use a full budget on a bounded, cooperative pool;
/// the pool limits executing workers, while each operation plans its own scratch.
/// Serial execution may use any budget.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TaskBudget(NonZeroUsize);

impl TaskBudget {
    /// One work partition, with no concurrency.
    pub const SERIAL: Self = Self(NonZeroUsize::MIN);

    /// Creates an allowance, returning `None` for zero tasks.
    pub const fn new(tasks: usize) -> Option<Self> {
        match NonZeroUsize::new(tasks) {
            Some(tasks) => Some(Self(tasks)),
            None => None,
        }
    }

    /// Returns the number of allowed work partitions.
    pub const fn get(self) -> usize {
        self.0.get()
    }

    /// Divides the allowance into two nonzero shares that sum to this budget.
    ///
    /// Returns `None` if `left` is zero or at least the entire allowance.
    pub const fn split_at(self, left: usize) -> Option<(Self, Self)> {
        if left == 0 || left >= self.get() {
            return None;
        }
        match (Self::new(left), Self::new(self.get() - left)) {
            (Some(left), Some(right)) => Some((left, right)),
            _ => unreachable!(),
        }
    }

    /// Chooses concurrent outer jobs and an equal allowance for each job.
    ///
    /// Returns `(jobs, inner)`, where `jobs` is the smaller of `max_jobs` and
    /// [`self.get()`](Self::get), and `inner` allows `self.get() / jobs` tasks.
    /// Each outer job receives that allowance for its nested work; any remainder
    /// is unused. Their combined allowances never exceed this budget. Returns
    /// `None` when `max_jobs` is zero.
    pub const fn partition(self, max_jobs: usize) -> Option<(usize, Self)> {
        if max_jobs == 0 {
            return None;
        }
        let jobs = if max_jobs < self.get() {
            max_jobs
        } else {
            self.get()
        };
        match Self::new(self.get() / jobs) {
            Some(inner) => Some((jobs, inner)),
            None => unreachable!(),
        }
    }
}

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
/// ```
/// use zakura_udon::exec::{SerialExecutor, for_each_task_mut};
///
/// let mut first = [0; 2];
/// let mut second = [0; 5];
/// let mut buffers = [&mut first[..], &mut second[..]];
/// for_each_task_mut(&mut buffers, &SerialExecutor, |index, buffer| {
///     buffer.fill(index + 1);
/// });
/// assert_eq!(first, [1; 2]);
/// assert_eq!(second, [2; 5]);
/// ```
///
/// # Panics
///
/// A callback panic may leave partial changes. All callbacks are still invoked,
/// and all jobs finish or unwind before the panic propagates, following
/// [`Executor::join`]. A second panic during unwinding may abort the process.
pub fn for_each_task_mut<T, E, F>(values: &mut [T], executor: &E, work: F)
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
/// Use [`for_each_task_mut`] to expose every item as a separate job without
/// dividing a combined task budget.
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

#[cfg(test)]
mod tests;
