#[cfg(test)]
use core::ops::Range;

use super::{Completion, Kernel, Outcome, Task, TaskError};

/// Exclusive identity storage for a run.
///
/// Construct a separate value for each simultaneously bound owner. Binding
/// borrows it exclusively; detached tasks retain shared references obtained
/// from that borrow. It cannot be rebound while those tasks remain accessible.
#[derive(Debug, Default)]
pub struct Identity {
    // Nonzero size makes pointer identity meaningful for distinct owners.
    _occupied: u8,
}

impl Identity {
    /// Unbound identity storage.
    pub fn new() -> Self {
        Self::default()
    }
}

/// The position of one task in a particular frontier epoch.
#[derive(Clone, Copy, Debug)]
pub struct TaskKey<'a> {
    identity: &'a Identity,
    epoch: usize,
    index: usize,
}

impl TaskKey<'_> {
    /// Zero-based task index within the current epoch.
    pub fn index(self) -> usize {
        self.index
    }
}

/// A compact consecutive range of currently claimable tasks.
#[derive(Clone, Debug)]
#[cfg(test)]
pub(super) struct ReadyRange<'a> {
    identity: Option<&'a Identity>,
    epoch: usize,
    range: Range<usize>,
}

#[cfg(test)]
impl<'a> ReadyRange<'a> {
    /// Empty output storage for [`Frontier::ready`].
    pub(super) const EMPTY: Self = Self {
        identity: None,
        epoch: 0,
        range: 0..0,
    };

    /// Task identities without materializing a graph or task array.
    pub(super) fn tasks(&self) -> impl Iterator<Item = TaskKey<'a>> + '_ {
        self.range.clone().map(|index| TaskKey {
            identity: self.identity.unwrap(),
            epoch: self.epoch,
            index,
        })
    }

    /// Number of represented tasks.
    fn len(&self) -> usize {
        self.range.len()
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum State {
    Ready,
    Claimed,
    Complete,
}

/// One slot in a bounded frontier, independent of total task count.
#[derive(Debug)]
pub struct TaskStorage {
    state: State,
}

impl TaskStorage {
    /// Unbound metadata.
    pub const EMPTY: Self = Self {
        state: State::Ready,
    };
}

#[derive(Debug)]
pub(super) struct Ticket<'a> {
    identity: &'a Identity,
    pub(super) key: TaskKey<'a>,
}

/// Resources returned by a successfully accepted completion.
#[derive(Debug)]
pub(crate) struct Completed<R, O> {
    /// The entire owned bundle, including retained input guards and scratch.
    pub(crate) resources: R,
    /// Kernel output; absent for cancellation or unwinding.
    pub(crate) output: Option<O>,
    /// Whether execution returned normally, unwound, or never started.
    pub(crate) outcome: Outcome,
    /// Consecutive completed task indices retired by this publication.
    ///
    /// Reduce or otherwise consume retained partials in this range before
    /// claiming work that reuses their slots. Its length is bounded by the
    /// number of frontier slots.
    #[cfg(test)]
    retired: Range<usize>,
}

/// A bounded ordered frontier with out-of-order task completion.
///
/// Only the next `storage.len()` task indices can be claimed. This prevents
/// later partial results from filling all slots while the earliest missing
/// partial has no space. Completing consecutive indices advances the frontier
/// and incrementally exposes successors. A failed task poisons the epoch;
/// outstanding completions can still be drained, but no new claim succeeds.
///
/// Metadata and identity are caller owned. All methods run on the coordinator;
/// detached tasks can move to scoped workers without borrowing `&mut Frontier`.
#[derive(Debug)]
pub(crate) struct Frontier<'a> {
    identity: &'a Identity,
    storage: &'a mut [TaskStorage],
    epoch: usize,
    start: usize,
    total: usize,
    inflight: usize,
    failed: bool,
}

impl<'a> Frontier<'a> {
    pub(crate) fn reserve<K>(
        &mut self,
        key: TaskKey<'a>,
        kernel: K,
    ) -> Result<super::task::Reserved<'a, K>, TaskError> {
        self.check_key(key)?;
        self.storage[key.index % self.storage.len()].state = State::Claimed;
        self.inflight += 1;
        Ok(super::task::Reserved::new(
            Ticket {
                identity: self.identity,
                key,
            },
            kernel,
        ))
    }

    /// Binds a frontier, panicking if no storage is supplied.
    pub(crate) fn new(
        identity: &'a mut Identity,
        storage: &'a mut [TaskStorage],
        total: usize,
    ) -> Self {
        assert!(!storage.is_empty(), "frontier storage must be nonempty");
        for slot in storage.iter_mut() {
            *slot = TaskStorage::EMPTY;
        }
        Self {
            identity,
            storage,
            epoch: 0,
            start: 0,
            total,
            inflight: 0,
            failed: false,
        }
    }

    /// Writes at most `output.len()` ready ranges and returns the written count.
    ///
    /// Does not claim work. Repeated calls may report the same tasks; resource
    /// availability and arbitration belong to the application scheduler.
    #[cfg(test)]
    pub(super) fn ready(&self, output: &mut [ReadyRange<'a>]) -> usize {
        if self.failed {
            return 0;
        }
        let mut count = 0;
        let end = self
            .start
            .saturating_add(self.storage.len())
            .min(self.total);
        let mut index = self.start;
        while index < end && count < output.len() {
            if self.storage[index % self.storage.len()].state != State::Ready {
                index += 1;
                continue;
            }
            let start = index;
            while index < end && self.storage[index % self.storage.len()].state == State::Ready {
                index += 1;
            }
            output[count] = ReadyRange {
                identity: Some(self.identity),
                epoch: self.epoch,
                range: start..index,
            };
            count += 1;
        }
        count
    }

    /// Iterates every ready key in the bounded frontier, including ranges
    /// separated by claimed tasks. The scan visits at most `capacity()` slots.
    /// A scheduler can skip resource-blocked keys without hiding later work.
    pub(crate) fn tasks(&self) -> impl Iterator<Item = TaskKey<'a>> + '_ {
        let end = if self.failed {
            self.start
        } else {
            self.start
                .saturating_add(self.storage.len())
                .min(self.total)
        };
        (self.start..end)
            .filter(|&index| self.storage[index % self.storage.len()].state == State::Ready)
            .map(|index| TaskKey {
                identity: self.identity,
                epoch: self.epoch,
                index,
            })
    }

    /// Attempts to acquire a complete bundle, then detaches one task.
    ///
    /// `acquire` must use nonblocking, atomic bundle acquisition. Returning
    /// `None` leaves this frontier unchanged; partial acquisitions must be
    /// rolled back by the provider. A stale or already claimed key returns an
    /// error without calling `acquire`. A kernel receives no executor.
    pub(crate) fn try_claim<K: Kernel<R>, R>(
        &mut self,
        key: TaskKey<'a>,
        kernel: K,
        acquire: impl FnOnce() -> Option<R>,
    ) -> Result<Option<Task<'a, K, R>>, TaskError> {
        self.check_key(key)?;
        let index = key.index % self.storage.len();
        let Some(resources) = acquire() else {
            return Ok(None);
        };
        self.storage[index].state = State::Claimed;
        self.inflight += 1;
        Ok(Some(Task::new(
            Ticket {
                identity: self.identity,
                key,
            },
            kernel,
            resources,
        )))
    }

    /// Checks claim identity before deriving task indices or resource ranges.
    /// This does not reserve the task. The coordinator must still claim it.
    pub(crate) fn check_key(&self, key: TaskKey<'a>) -> Result<(), TaskError> {
        if self.failed {
            return Err(TaskError::Failed);
        }
        if !core::ptr::eq(key.identity, self.identity)
            || key.epoch != self.epoch
            || key.index < self.start
            || key.index >= self.total
            || key.index - self.start >= self.storage.len()
        {
            return Err(TaskError::Stale);
        }
        let index = key.index % self.storage.len();
        if self.storage[index].state != State::Ready {
            return Err(TaskError::Claimed);
        }
        Ok(())
    }

    /// Publishes a completion exactly once and returns its owned resources.
    ///
    /// Foreign completions are returned intact in the error. Tasks cannot forge
    /// or clone completion tickets. Accepts failed completions after poisoning
    /// so an application can drain every outstanding task before cancellation.
    pub(crate) fn complete<R, O>(
        &mut self,
        completion: Completion<'a, R, O>,
    ) -> Result<Completed<R, O>, (TaskError, Completion<'a, R, O>)> {
        let key = completion.ticket.key;
        if !core::ptr::eq(completion.ticket.identity, self.identity)
            || key.epoch != self.epoch
            || key.index < self.start
            || key.index >= self.total
            || key.index - self.start >= self.storage.len()
        {
            return Err((TaskError::Stale, completion));
        }
        let slot = &mut self.storage[key.index % self.storage.len()];
        if slot.state != State::Claimed {
            return Err((TaskError::Stale, completion));
        }
        slot.state = State::Complete;
        self.inflight -= 1;
        self.failed |= completion.outcome != Outcome::Success;
        #[cfg(test)]
        let start = self.start;
        while self.start < self.total
            && self.storage[self.start % self.storage.len()].state == State::Complete
        {
            self.storage[self.start % self.storage.len()].state = State::Ready;
            self.start += 1;
        }
        Ok(Completed {
            resources: completion.resources,
            output: completion.output,
            outcome: completion.outcome,
            #[cfg(test)]
            retired: start..self.start,
        })
    }

    /// Starts the next dependency epoch after all current work succeeded.
    pub(crate) fn restart(&mut self, total: usize) -> Result<(), TaskError> {
        if !self.is_complete() {
            return Err(if self.failed {
                TaskError::Failed
            } else {
                TaskError::Busy
            });
        }
        let epoch = self.epoch.checked_add(1).ok_or(TaskError::Overflow)?;
        self.epoch = epoch;
        self.start = 0;
        self.total = total;
        Ok(())
    }

    /// Whether every task in the epoch succeeded and was published.
    pub(crate) fn is_complete(&self) -> bool {
        !self.failed && self.start == self.total
    }

    /// Whether any accepted completion failed or was cancelled.
    #[cfg(test)]
    fn is_failed(&self) -> bool {
        self.failed
    }

    /// Number of detached tasks whose completion has not been published.
    pub(crate) fn inflight(&self) -> usize {
        self.inflight
    }
}
