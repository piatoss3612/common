//! Incremental work and typed resource accounting for application schedulers.
//!
//! A scheduler owns synchronization, storage, worker selection, and wakeups.
//! Udon supplies bounded task descriptions and coordinator-owned state. Claiming
//! work transfers an owned resource bundle into a [`Task`]; the task does not
//! borrow the frontier. A worker executes one kernel and returns its
//! [`Completion`] before selecting more work.
//!
//! [`Admission`] checks bounded pipeline segments against typed block capacity.
//! It accounts for resources; it does not create references to storage. Actual
//! exclusive or shared access must come from safe borrowed fragments or the
//! application's lease provider. Acquire the complete bundle before dispatch,
//! including dispatch and completion queue capacity. A kernel must not wait for
//! additional scarce resources or submit children and wait for them.
//!
//! Dropping or forgetting a task does not publish completion. Its frontier and
//! accounting remain occupied until the application drains or abandons that
//! run. Catch unwinding around [`Task::execute`] while retaining the task, then
//! return its failed completion. An in-place failed result must be refilled
//! before reuse. No destructor is responsible for establishing memory safety.
//!
//! Application work uses the same envelope, including scoped mutable borrows:
//!
//! ```
//! use zakura_udon::exec::run::{Frontier, Identity, Kernel, TaskStorage};
//! struct Increment;
//! impl Kernel<&mut u64> for Increment {
//!     type Output = ();
//!     fn execute(&mut self, value: &mut &mut u64) { **value += 1; }
//! }
//! let mut value = 7;
//! let mut identity = Identity::new();
//! let mut slots = [TaskStorage::EMPTY];
//! let mut run = Frontier::new(&mut identity, &mut slots, 1).unwrap();
//! let key = run.tasks().next().unwrap();
//! let mut task = run.try_claim(key, Increment, || Some(&mut value))
//!     .unwrap().unwrap();
//! std::thread::scope(|scope| scope.spawn(|| task.execute().unwrap()).join().unwrap());
//! let completed = run.complete(task.finish()).unwrap();
//! assert_eq!(*completed.resources, 8);
//! assert!(run.is_complete());
//! ```
//!
//! The task's lease prevents access through the original owner, even while
//! suspended in a queue:
//!
//! ```compile_fail
//! use zakura_udon::exec::run::{Frontier, Identity, Kernel, TaskStorage};
//! struct Increment;
//! impl Kernel<&mut u64> for Increment {
//!     type Output = ();
//!     fn execute(&mut self, value: &mut &mut u64) { **value += 1; }
//! }
//! let mut value = 7;
//! let mut identity = Identity::new();
//! let mut slots = [TaskStorage::EMPTY];
//! let mut run = Frontier::new(&mut identity, &mut slots, 1).unwrap();
//! let key = run.tasks().next().unwrap();
//! let mut task = run.try_claim(key, Increment, || Some(&mut value))
//!     .unwrap().unwrap();
//! value = 9; // Still exclusively borrowed by the detached task.
//! task.execute().unwrap();
//! ```

mod admission;
mod frontier;
mod task;
pub(crate) use task::Reserved;

pub use admission::{
    Admission, ArenaLayout, BlockClass, Profile, ResourceError, Resources, Segment, SegmentStorage,
    TaskPermit,
};
pub use frontier::{Completed, Frontier, Identity, ReadyRange, TaskKey, TaskStorage};
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

/// Estimates supplied to optional application scheduling policies.
///
/// These are hints, not hardware reservations or timing guarantees. A scheduler
/// may translate them into additional bandwidth or cache admission tokens.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorkEstimate {
    /// Estimated arithmetic work in kernel-defined units.
    pub arithmetic: usize,
    /// Estimated bytes transferred between the kernel and its storage.
    pub traffic_bytes: usize,
    /// Estimated actively reused working set.
    pub cache_bytes: usize,
}
