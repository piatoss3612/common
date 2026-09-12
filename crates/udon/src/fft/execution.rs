use super::{FftError, check_domain_size, check_field_count, min};

/// Scheduling and scratch bounds for a transform.
///
/// Tile lengths are powers of two and are clamped to the transform size.
/// Column widths and task counts must be nonzero. More tasks permit additional
/// concurrency but do not require the executor to create that many threads.
/// Invalid settings produce [`FftError::InvalidExecution`] when requirements
/// are queried or execution begins. The default uses 1,024-element tiles,
/// 64 columns per task, and one task. Increasing [`Self::max_tasks`] permits
/// concurrency when the transform exceeds one tile. Use [`Self::serial`] for a
/// single whole-transform tile with no scratch.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExecutionOptions {
    /// Number of consecutive elements processed by each local tile transform.
    pub tile_len: usize,
    /// Maximum columns in each cross-tile scratch partition, clamped to a tile.
    pub columns_per_task: usize,
    /// Maximum concurrent work partitions within the transform.
    pub max_tasks: usize,
}

impl ExecutionOptions {
    /// One whole-transform tile, with no transform scratch or fork/join calls.
    ///
    /// Increasing only [`Self::max_tasks`] retains the whole-transform tile.
    /// Start from [`Self::default`] or also set [`Self::tile_len`] to enable
    /// tiled execution.
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
    /// Size limits and errors follow
    /// [`Domain::for_size`](super::Domain::for_size). Returns
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
