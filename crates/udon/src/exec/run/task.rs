use super::frontier::Ticket;

/// One bounded kernel operating only on a pre-acquired bundle.
///
/// Implementations must finish without waiting for additional scarce resources.
/// They may retain intermediate state by returning it with their resource
/// bundle. Dependencies on further tasks become coordinator continuations.
pub trait Kernel<R> {
    /// Result published after the kernel returns normally.
    type Output;

    /// Executes one bounded arithmetic or application work unit.
    fn execute(&mut self, resources: &mut R) -> Self::Output;
}

/// State of a returned task envelope.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Outcome {
    /// The kernel returned normally, including when its output is an error value.
    Success,
    /// Execution began but did not return normally.
    Failed,
    /// The task was returned without executing.
    Cancelled,
}

/// Invalid run request, storage capacity, or task/frontier transition.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TaskError {
    /// The input shape, range, or configuration is unsupported by the run.
    InvalidRequest,
    /// No retained slot or frontier storage was supplied.
    Storage,
    /// The key or completion does not belong to the current frontier epoch.
    Stale,
    /// The task has already been claimed or completed.
    Claimed,
    /// The kernel has already started; a task can execute only once.
    Executed,
    /// The epoch has failed and accepts only draining completions.
    Failed,
    /// Current work has not finished.
    Busy,
    /// Storage sizing or epoch identity arithmetic overflowed.
    Overflow,
}

impl core::fmt::Display for TaskError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::InvalidRequest => "invalid run input, range, or configuration",
            Self::Storage => "insufficient run metadata storage",
            Self::Stale => "task does not belong to the current frontier epoch",
            Self::Claimed => "task has already been claimed or completed",
            Self::Executed => "task execution has already started",
            Self::Failed => "run has failed and accepts only draining completions",
            Self::Busy => "current work has not finished",
            Self::Overflow => "storage sizing or epoch identity overflow",
        })
    }
}

impl core::error::Error for TaskError {}

/// Detached ownership of a kernel, all of its resources, and its completion.
///
/// Sendability follows its contents. Neither the kernel nor resource bundle
/// borrows the coordinator. Call [`execute`](Self::execute) by mutable reference
/// so a scoped worker can catch unwinding while keeping this envelope alive.
pub struct Task<'a, K: Kernel<R>, R> {
    ticket: Ticket<'a>,
    kernel: K,
    resources: R,
    output: Option<K::Output>,
    outcome: Outcome,
}

// Structured drivers already own the whole retained bank and every scratch
// bundle for a join. Reserve its ready kernels on the coordinator, then bind
// each complete borrowed bundle immediately before running that kernel. This
// private reservation cannot be dispatched through the public Task protocol.
pub(crate) struct Reserved<'a, K> {
    ticket: Ticket<'a>,
    kernel: K,
}

impl<'a, K> Reserved<'a, K> {
    pub(super) fn new(ticket: Ticket<'a>, kernel: K) -> Self {
        Self { ticket, kernel }
    }

    pub(crate) fn bind<R>(self, resources: R) -> Task<'a, K, R>
    where
        K: Kernel<R>,
    {
        Task::new(self.ticket, self.kernel, resources)
    }
}

impl<'a, K: Kernel<R>, R> Task<'a, K, R> {
    pub(super) fn new(ticket: Ticket<'a>, kernel: K, resources: R) -> Self {
        Self {
            ticket,
            kernel,
            resources,
            output: None,
            outcome: Outcome::Cancelled,
        }
    }

    /// Runs the kernel once. A second attempt returns [`TaskError::Executed`].
    ///
    /// If the kernel unwinds, the owned bundle remains in this task and its
    /// eventual completion is failed. The worker must retain this task outside
    /// the unwind-catching closure and return it to the coordinator.
    pub fn execute(&mut self) -> Result<(), TaskError> {
        if self.outcome != Outcome::Cancelled {
            return Err(TaskError::Executed);
        }
        self.outcome = Outcome::Failed;
        self.output = Some(self.kernel.execute(&mut self.resources));
        self.outcome = Outcome::Success;
        Ok(())
    }

    /// Returns the envelope for publication, including after caught unwinding.
    pub fn finish(self) -> Completion<'a, R, K::Output> {
        Completion {
            ticket: self.ticket,
            resources: self.resources,
            output: self.output,
            outcome: self.outcome,
        }
    }
}

/// An owned, noncloneable task receipt accepted by its originating frontier.
pub struct Completion<'a, R, O> {
    pub(super) ticket: Ticket<'a>,
    pub(super) resources: R,
    pub(super) output: Option<O>,
    pub(super) outcome: Outcome,
}

/// A rejected publication with its intact receipt for routing or draining.
pub type PublishError<'a, R, O> = (TaskError, Completion<'a, R, O>);

impl<R, O> Completion<'_, R, O> {
    /// Execution status before publication; resources remain owned by the receipt.
    pub fn outcome(&self) -> Outcome {
        self.outcome
    }
}

impl<'a, R, O> Completion<'a, R, O> {
    /// Original task identity, for routing its owned result to retained storage.
    pub fn key(&self) -> super::TaskKey<'a> {
        self.ticket.key
    }

    // The structured driver keeps the backing owners until its join returns.
    // Release a completed kernel's temporary borrows so the next bounded
    // kernel can use that same explicitly owned bundle. Publish this receipt
    // before observing any dependency successor.
    pub(crate) fn release_resources(self) -> Completion<'a, (), O> {
        Completion {
            ticket: self.ticket,
            resources: (),
            output: self.output,
            outcome: self.outcome,
        }
    }
}

impl<R, O> core::fmt::Debug for Completion<'_, R, O> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Completion")
            .field("ticket", &self.ticket)
            .field("outcome", &self.outcome)
            .finish_non_exhaustive()
    }
}
