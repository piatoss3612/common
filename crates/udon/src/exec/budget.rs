//! Per-operation concurrency and arithmetic workspace limits.

use core::num::NonZeroUsize;

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
