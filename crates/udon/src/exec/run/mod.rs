//! Incremental work and typed resource accounting for application schedulers.
//!
//! A scheduler owns synchronization, storage, worker selection, and wakeups.
//! Udon supplies bounded task descriptions and coordinator-owned state. Claiming
//! work transfers an owned resource bundle into a [`Task`]; the task does not
//! borrow the frontier. A worker executes one kernel and returns its
//! [`Completion`] before selecting more work.
//!
//! The application owns capacity accounting and actual storage leases. Acquire
//! the complete resource bundle before dispatch, including queue capacity.
//! A kernel must not wait for additional scarce resources or submit children
//! and wait for them.
//!
//! Dropping or forgetting a task does not publish completion. Its frontier and
//! accounting remain occupied until the application drains or abandons that
//! run. Catch unwinding around [`Task::execute`] while retaining the task, then
//! return its failed completion. An in-place failed result must be refilled
//! before reuse. No destructor is responsible for establishing memory safety.
//!
pub(crate) mod frontier;
mod task;
pub(crate) use task::Reserved;

pub(crate) use frontier::Frontier;
pub use frontier::{Identity, TaskKey, TaskStorage};
pub use task::{Completion, Kernel, Outcome, PublishError, Task, TaskError};

/// Shared indexed access to safely fragmented retained storage.
///
/// Implementations typically own or borrow a collection of read guards. They
/// must preserve logical indexing across fragments. Returning references from
/// this view keeps each guard borrowed for the reference's lifetime.
pub trait ReadView<T> {
    /// Number of logical elements in the view.
    fn len(&self) -> usize;
    /// Shared access to a logical element, or `None` for an out-of-range index.
    fn get(&self, index: usize) -> Option<&T>;
    /// Returns a contiguous range when the provider can expose it directly.
    ///
    /// `None` is permitted for any range; kernels then use indexed access.
    /// A returned slice must contain exactly the requested logical elements.
    fn contiguous(&self, _range: core::ops::Range<usize>) -> Option<&[T]> {
        None
    }
    /// Returns a nonempty contiguous prefix of a nonempty requested range.
    ///
    /// Providers may stop at their next fragment boundary. `None` permits an
    /// indexed fallback. A slice must start at `range.start`, contain at most
    /// `range.len()` elements, and preserve the view's logical indexing.
    fn contiguous_prefix(&self, range: core::ops::Range<usize>) -> Option<&[T]> {
        self.contiguous(range)
    }
    /// Whether the view contains no elements.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, const N: usize> ReadView<T> for [T; N] {
    fn len(&self) -> usize {
        N
    }
    fn get(&self, index: usize) -> Option<&T> {
        self.as_slice().get(index)
    }
    fn contiguous(&self, range: core::ops::Range<usize>) -> Option<&[T]> {
        self.as_slice().get(range)
    }
}

impl<T> ReadView<T> for &[T] {
    fn len(&self) -> usize {
        <[T]>::len(self)
    }
    fn get(&self, index: usize) -> Option<&T> {
        <[T]>::get(self, index)
    }
    fn contiguous(&self, range: core::ops::Range<usize>) -> Option<&[T]> {
        <[T]>::get(self, range)
    }
}
