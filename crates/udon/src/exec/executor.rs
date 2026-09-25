//! Scoped fork/join contract and its synchronous implementation.

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
