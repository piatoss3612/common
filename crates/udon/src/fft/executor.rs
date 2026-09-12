use super::{FftError, check_domain_size, check_field_count, min};

/// Caller-supplied scoped fork/join execution.
///
/// Each job must be invoked exactly once. [`join`](Self::join) must wait until
/// both jobs have returned or unwound and their captures have been dropped,
/// including when propagating a panic. Jobs can borrow non-`'static` data; an
/// implementation may run them sequentially or on a caller's pool.
///
/// Udon never allocates job descriptors. An executor's own resource use belongs
/// to its caller and is not included in FFT scratch requirements.
///
/// # Nested joins
///
/// Udon recursively subdivides work with `join`, and residue jobs can invoke
/// further joins for their transforms. An implementation must support these
/// nested calls even when every pool worker is executing a Udon job, including
/// in a pool with only one worker. Task budgets limit work partitions; they do
/// not reserve idle workers for nested calls.
///
/// A bounded pool adapter that queues a child job and blocks its worker until
/// that child completes can deadlock: all workers may be waiting while the
/// children remain queued with no worker available to execute them. Merely
/// increasing the pool size does not satisfy the progress requirement.
///
/// Cooperative fork/join as provided by
/// [`rayon::join`](https://docs.rs/rayon/latest/rayon/fn.join.html) is suitable.
/// A worker runs one branch locally and makes the other available for stealing;
/// while waiting for a stolen branch, it executes available work. A downstream
/// adapter can delegate directly to `rayon::join`. Synchronous execution, as
/// provided by [`SerialExecutor`], also satisfies the nested-join requirement.
pub trait Executor: Sync {
    /// Runs both jobs and waits for their completion.
    ///
    /// Invocation order is unspecified. Implementations must propagate job
    /// panics after completing both jobs, subject to abort on a second panic
    /// during unwinding. Silently skipping a job or swallowing its panic can
    /// cause a transform to return success with incomplete results.
    /// Calls made from either job must satisfy the trait's
    /// [nested-join requirement](Executor#nested-joins).
    fn join<L: FnOnce() + Send, R: FnOnce() + Send>(&self, left: L, right: R);
}

/// Executes jobs synchronously without allocation.
///
/// Runs the left job followed by the right, including if the left job unwinds.
/// A second panic during unwinding aborts the process.
#[derive(Clone, Copy, Debug, Default)]
pub struct SerialExecutor;

impl Executor for SerialExecutor {
    fn join<L: FnOnce() + Send, R: FnOnce() + Send>(&self, left: L, right: R) {
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
        let right = Pending(Some(right));
        left();
        drop(right);
    }
}

/// Scheduling and scratch bounds for a transform.
///
/// Tile lengths are powers of two and are clamped to the transform size.
/// Column widths and task counts must be nonzero. More tasks permit additional
/// concurrency but do not require the executor to create that many threads.
/// Invalid settings produce [`FftError::InvalidExecution`] when requirements
/// are queried or execution begins. The default uses 1,024-element tiles,
/// 64 columns per task, and one task. Increasing `max_tasks` permits concurrency
/// when the transform exceeds one tile. Use [`Self::serial`] for a single
/// whole-transform tile with no scratch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionOptions {
    /// Number of consecutive elements processed by each local tile transform.
    pub tile_len: usize,
    /// Maximum columns in each cross-tile scratch partition, clamped to a tile.
    pub columns_per_task: usize,
    /// Maximum independent partitions in one execution wave.
    pub max_tasks: usize,
}

impl ExecutionOptions {
    /// One whole-transform tile, with no transform scratch or fork/join calls.
    ///
    /// Increasing only `max_tasks` retains the whole-transform tile. Start from
    /// [`Self::default`] or also set `tile_len` to enable tiled execution.
    pub const fn serial() -> Self {
        Self {
            tile_len: 1usize << (usize::BITS - 1),
            columns_per_task: 1,
            max_tasks: 1,
        }
    }

    pub(super) const fn validate(self) -> Result<(), FftError> {
        if !self.tile_len.is_power_of_two() || self.columns_per_task == 0 || self.max_tasks == 0 {
            Err(FftError::InvalidExecution)
        } else {
            Ok(())
        }
    }

    pub(super) const fn geometry(self, size: usize) -> Geometry {
        let tile_len = min(self.tile_len, size);
        let columns = min(self.columns_per_task, tile_len);
        Geometry {
            tile_len,
            tiles: size / tile_len,
            columns,
            jobs: min(self.max_tasks, tile_len.div_ceil(columns)),
        }
    }

    /// Required temporary field storage for a transform of `size` elements.
    ///
    /// This const query needs no domain or plan and applies to both Pasta
    /// fields, with or without tables and any coset shift. Use its result to
    /// size arrays as well as runtime buffers. Whole-transform tiles need no
    /// scratch. The count follows the current execution implementation.
    ///
    /// Size limits and errors are those of [`super::Domain::for_size`]. Returns
    /// [`FftError::InvalidExecution`] for invalid settings, or
    /// [`FftError::SizeOverflow`] if the scratch count overflows `usize` or its
    /// field slice would exceed `isize::MAX` bytes.
    pub const fn requirements(self, size: usize) -> Result<ScratchRequirements, FftError> {
        if let Err(error) = self.validate() {
            return Err(error);
        }
        if let Err(error) = check_domain_size(size) {
            return Err(error);
        }
        let geometry = self.geometry(size);
        let fields = if geometry.tiles == 1 {
            0
        } else {
            // Each concurrent job gathers a rectangular group of columns,
            // with one field per tile and column, into its own partition.
            let partition = match geometry.tiles.checked_mul(geometry.columns) {
                Some(fields) => fields,
                None => return Err(FftError::SizeOverflow),
            };
            match partition.checked_mul(geometry.jobs) {
                Some(fields) => fields,
                None => return Err(FftError::SizeOverflow),
            }
        };
        match check_field_count(fields) {
            Ok(field_elements) => Ok(ScratchRequirements { field_elements }),
            Err(error) => Err(error),
        }
    }
}

impl Default for ExecutionOptions {
    fn default() -> Self {
        Self {
            tile_len: 1024,
            columns_per_task: 64,
            max_tasks: 1,
        }
    }
}

/// Exact temporary field storage for a particular operation and configuration.
///
/// Inputs, outputs, immutable tables, and caller-owned class descriptors are
/// separate. No additional dynamically sized storage is used by Udon.
/// Scratch contents are disposable; see the [working-storage contract](super).
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct ScratchRequirements {
    /// Number of initialized field elements to lend as mutable scratch.
    pub field_elements: usize,
}

impl ScratchRequirements {
    pub(super) fn check(self, provided: usize) -> Result<(), FftError> {
        if provided < self.field_elements {
            Err(FftError::ScratchTooSmall {
                required: self.field_elements,
                provided,
            })
        } else {
            Ok(())
        }
    }
}

pub(super) struct Geometry {
    pub tile_len: usize,
    pub tiles: usize,
    pub columns: usize,
    pub jobs: usize,
}

// Recursion splits both ownership and the concurrency budget. Only a bounded
// control frame lives on the stack; there is no queue or list of borrowed jobs.
pub(super) fn for_chunks<T: Send, E: Executor, F: Fn(usize, &mut [T]) + Sync>(
    values: &mut [T],
    chunk_len: usize,
    budget: usize,
    executor: &E,
    work: &F,
) {
    fn visit<T: Send, E: Executor, F: Fn(usize, &mut [T]) + Sync>(
        values: &mut [T],
        chunk_len: usize,
        offset: usize,
        budget: usize,
        executor: &E,
        work: &F,
    ) {
        let chunks = values.len().div_ceil(chunk_len);
        if chunks <= 1 || budget <= 1 {
            for (index, chunk) in values.chunks_mut(chunk_len).enumerate() {
                work(offset + index, chunk);
            }
        } else {
            let left_chunks = chunks / 2;
            let (left, right) = values.split_at_mut(left_chunks * chunk_len);
            let left_budget = budget / 2;
            executor.join(
                || visit(left, chunk_len, offset, left_budget, executor, work),
                || {
                    visit(
                        right,
                        chunk_len,
                        offset + left_chunks,
                        budget - left_budget,
                        executor,
                        work,
                    )
                },
            );
        }
    }
    visit(values, chunk_len, 0, budget, executor, work);
}
