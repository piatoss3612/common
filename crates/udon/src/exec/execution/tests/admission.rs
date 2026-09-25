//! Application capacity accounting used by scheduler tests.
//!
//! Permits track compatible block counts; the fixture's resource provider owns
//! actual storage and leases. Admission and acquisition must succeed together
//! before a task is dispatched.

use core::array;

/// Counts of compatible, initialized blocks in application-defined classes.
///
/// Each index denotes one exact block type and capacity. Fp and Fq blocks, or
/// differently sized blocks of the same field, must use different classes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Resources<const N: usize>(pub(crate) [usize; N]);

impl<const N: usize> Resources<N> {
    /// No blocks.
    pub(crate) const ZERO: Self = Self([0; N]);

    /// Whether every requested count fits this capacity.
    pub(crate) fn fits(self, capacity: Self) -> bool {
        self.0
            .iter()
            .zip(capacity.0)
            .all(|(&need, have)| need <= have)
    }

    fn add(self, other: Self) -> Result<Self, ResourceError> {
        let mut result = Self::ZERO;
        for (i, value) in result.0.iter_mut().enumerate() {
            *value = self.0[i]
                .checked_add(other.0[i])
                .ok_or(ResourceError::Overflow)?;
        }
        Ok(result)
    }

    fn sub(self, other: Self) -> Result<Self, ResourceError> {
        let mut result = Self::ZERO;
        for (i, value) in result.0.iter_mut().enumerate() {
            *value = self.0[i]
                .checked_sub(other.0[i])
                .ok_or(ResourceError::InvalidRelease)?;
        }
        Ok(result)
    }

    fn max(self, other: Self) -> Self {
        Self(array::from_fn(|i| self.0[i].max(other.0[i])))
    }
}

/// A typed block class, including idle provisioned capacity.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub(crate) struct BlockClass {
    /// Number of initialized compatible blocks in the arena.
    pub(crate) blocks: usize,
    /// Charged bytes per block, including padding and per-block lease metadata.
    pub(crate) block_bytes: usize,
}

/// Provisioned working storage under a byte ceiling.
///
/// Count full capacities, including idle blocks, retained intermediate handoffs,
/// run state, task envelopes, and bounded scheduler queues. Original inputs,
/// external final outputs, and persistent plans and tables are separate. Worker
/// runtime overhead is also separate; this is not a process RSS limit.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ArenaLayout<const N: usize> {
    /// Typed storage classes; bytes cannot substitute for compatible blocks.
    pub(crate) classes: [BlockClass; N],
    /// All other provisioned run, frontier, queue, and scheduler metadata.
    pub(crate) metadata_bytes: usize,
}

impl<const N: usize> ArenaLayout<N> {
    /// Checks the full provisioned capacity, returning its charged byte count.
    pub(crate) fn check(self, ceiling: usize) -> Result<usize, ResourceError> {
        let bytes = self
            .classes
            .iter()
            .try_fold(self.metadata_bytes, |bytes, class| {
                bytes.checked_add(class.blocks.checked_mul(class.block_bytes)?)
            })
            .ok_or(ResourceError::Overflow)?;
        if bytes > ceiling {
            Err(ResourceError::Capacity)
        } else {
            Ok(bytes)
        }
    }

    /// Exact compatible block counts.
    pub(crate) fn capacity(self) -> Resources<N> {
        Resources(self.classes.map(|class| class.blocks))
    }
}

/// Maximum retained storage and largest single-task bundle for a segment.
///
/// The segment extends through the last consumers of its intermediates, possibly
/// across several operations. For each class, `temporary` bounds every one of
/// its tasks, and `retained` bounds all simultaneously live retained blocks.
/// Temporary leases must be released when a bounded task completes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct Profile<const N: usize> {
    /// Maximum simultaneously retained blocks through the release frontier.
    pub(crate) retained: Resources<N>,
    /// Componentwise maximum temporary bundle of any single task.
    pub(crate) temporary: Resources<N>,
}

/// Admission or accounting failure. Failed methods leave accounting unchanged.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum ResourceError {
    /// Arithmetic overflow in a count or byte requirement.
    Overflow,
    /// The complete reservation or lease bundle does not currently fit.
    Capacity,
    /// No segment descriptor is available.
    Metadata,
    /// A task exceeds its declared segment profile.
    Profile,
    /// The segment token is stale or belongs to another admission arena.
    Stale,
    /// A release exceeds the segment's retained resources.
    InvalidRelease,
    /// The segment still owns retained blocks or outstanding task leases.
    Busy,
}

/// Caller-owned storage for one admitted segment.
#[derive(Debug)]
pub(crate) struct SegmentStorage<const N: usize> {
    generation: usize,
    profile: Option<Profile<N>>,
    live: Resources<N>,
    pending: Resources<N>,
    tasks: usize,
}

impl<const N: usize> SegmentStorage<N> {
    /// Unused metadata. Storage must remain bound until admission is drained.
    pub(crate) const EMPTY: Self = Self {
        generation: 0,
        profile: None,
        live: Resources::ZERO,
        pending: Resources::ZERO,
        tasks: 0,
    };
}

/// A nonforgeable token for an admitted segment.
#[derive(Debug)]
pub struct Segment<'a> {
    owner: &'a zakura_udon::exec::execution::Identity,
    slot: usize,
    generation: usize,
}

/// Accounting for one atomic task bundle.
///
/// Return this token with [`Admission::finish_task`], even after a kernel panic.
/// Dropping it deliberately leaves the accounting occupied.
#[derive(Debug)]
pub(crate) struct TaskPermit<'a, const N: usize> {
    segment: Segment<'a>,
    temporary: Resources<N>,
    retained: Resources<N>,
}

/// Single-coordinator admission and task resource accounting.
///
/// Admission maintains `sum(retained maxima) + max(temporary bundles) <= capacity`
/// componentwise. Once running tasks release their temporary leases, at least
/// one declared task bundle fits even if every admitted segment holds its full
/// retained reservation. Progress additionally requires finite bounded kernels,
/// acyclic dependencies, fair dispatch, and eventual external fence completion.
/// Ordered reductions and streaming consumers must include enough retained
/// capacity to reach their next release frontier; byte accounting cannot infer
/// these arithmetic dependencies.
///
/// Actual storage leases remain the application's responsibility. Perform their
/// nonblocking acquisition and this accounting as one coordinator transaction,
/// rolling back acquired guards if accounting fails.
#[derive(Debug)]
pub struct Admission<'a, const N: usize> {
    identity: &'a zakura_udon::exec::execution::Identity,
    slots: &'a mut [SegmentStorage<N>],
    capacity: Resources<N>,
    used: Resources<N>,
    peak: Resources<N>,
}

impl<'a, const N: usize> Admission<'a, N> {
    /// Binds exclusive metadata and identity storage, resetting prior accounting.
    ///
    /// Obtain `capacity` from a checked [`ArenaLayout`]. Outstanding tokens keep
    /// the identity borrowed, preventing rebinding that arena before they end.
    pub fn new(
        identity: &'a mut zakura_udon::exec::execution::Identity,
        slots: &'a mut [SegmentStorage<N>],
        capacity: Resources<N>,
    ) -> Self {
        for slot in slots.iter_mut() {
            *slot = SegmentStorage::EMPTY;
        }
        Self {
            identity,
            slots,
            capacity,
            used: Resources::ZERO,
            peak: Resources::ZERO,
        }
    }

    /// Admits a segment if its full progress reservation fits.
    pub(crate) fn admit(&mut self, profile: Profile<N>) -> Result<Segment<'a>, ResourceError> {
        let mut retained = profile.retained;
        let mut temporary = profile.temporary;
        for slot in self.slots.iter() {
            if let Some(other) = slot.profile {
                retained = retained.add(other.retained)?;
                temporary = temporary.max(other.temporary);
            }
        }
        if !retained.add(temporary)?.fits(self.capacity) {
            return Err(ResourceError::Capacity);
        }
        let index = self
            .slots
            .iter()
            .position(|slot| slot.profile.is_none())
            .ok_or(ResourceError::Metadata)?;
        let slot = &mut self.slots[index];
        let generation = slot
            .generation
            .checked_add(1)
            .ok_or(ResourceError::Overflow)?;
        slot.profile = Some(profile);
        slot.generation = generation;
        Ok(Segment {
            owner: self.identity,
            slot: index,
            generation,
        })
    }

    fn index(&self, segment: &Segment<'_>) -> Result<usize, ResourceError> {
        if !core::ptr::eq(segment.owner, self.identity) {
            return Err(ResourceError::Stale);
        }
        let slot = self.slots.get(segment.slot).ok_or(ResourceError::Stale)?;
        if slot.profile.is_none() || slot.generation != segment.generation {
            return Err(ResourceError::Stale);
        }
        Ok(segment.slot)
    }

    /// Acquires a task's temporary and new retained blocks atomically.
    ///
    /// Existing retained input blocks remain charged until their last consumer
    /// completes. Temporary and retained requests must fit the declared profile.
    pub(crate) fn try_task(
        &mut self,
        segment: &Segment<'a>,
        temporary: Resources<N>,
        retained: Resources<N>,
    ) -> Result<TaskPermit<'a, N>, ResourceError> {
        let index = self.index(segment)?;
        let slot = &self.slots[index];
        let profile = slot.profile.unwrap();
        let live = slot.live.add(retained)?;
        if !temporary.fits(profile.temporary) || !live.fits(profile.retained) {
            return Err(ResourceError::Profile);
        }
        let used = self.used.add(temporary)?.add(retained)?;
        if !used.fits(self.capacity) {
            return Err(ResourceError::Capacity);
        }
        let tasks = slot.tasks.checked_add(1).ok_or(ResourceError::Overflow)?;
        let pending = slot.pending.add(retained)?;
        self.slots[index].live = live;
        self.slots[index].pending = pending;
        self.slots[index].tasks = tasks;
        self.used = used;
        self.peak = self.peak.max(used);
        Ok(TaskPermit {
            segment: Segment {
                owner: segment.owner,
                slot: segment.slot,
                generation: segment.generation,
            },
            temporary,
            retained,
        })
    }

    /// Releases temporary accounting after all actual task leases have returned.
    ///
    /// `retain_outputs` commits newly retained blocks to the segment; `false`
    /// rolls back those blocks too. A failed kernel must not publish its output.
    /// A stale token is returned intact in the error so the owner can drain it.
    pub(crate) fn finish_task(
        &mut self,
        permit: TaskPermit<'a, N>,
        retain_outputs: bool,
    ) -> Result<(), (ResourceError, TaskPermit<'a, N>)> {
        let index = match self.index(&permit.segment) {
            Ok(index) => index,
            Err(error) => return Err((error, permit)),
        };
        self.used = self.used.sub(permit.temporary).unwrap();
        self.slots[index].pending = self.slots[index].pending.sub(permit.retained).unwrap();
        if !retain_outputs {
            self.used = self.used.sub(permit.retained).unwrap();
            self.slots[index].live = self.slots[index].live.sub(permit.retained).unwrap();
        }
        self.slots[index].tasks -= 1;
        Ok(())
    }

    /// Releases retained accounting after its final consumer and actual guard.
    pub(crate) fn release_retained(
        &mut self,
        segment: &Segment<'_>,
        released: Resources<N>,
    ) -> Result<(), ResourceError> {
        let index = self.index(segment)?;
        if !released.fits(self.slots[index].live.sub(self.slots[index].pending)?) {
            return Err(ResourceError::InvalidRelease);
        }
        let live = self.slots[index].live.sub(released)?;
        self.slots[index].live = live;
        self.used = self.used.sub(released).unwrap();
        Ok(())
    }

    /// Releases an empty segment reservation. Errors return the token intact.
    pub(crate) fn retire(
        &mut self,
        segment: Segment<'a>,
    ) -> Result<(), (ResourceError, Segment<'a>)> {
        let index = match self.index(&segment) {
            Ok(index) => index,
            Err(error) => return Err((error, segment)),
        };
        let slot = &mut self.slots[index];
        if slot.tasks != 0 || slot.live != Resources::ZERO {
            return Err((ResourceError::Busy, segment));
        }
        slot.profile = None;
        Ok(())
    }

    /// Currently charged live retained and temporary blocks.
    pub(crate) fn used(&self) -> Resources<N> {
        self.used
    }

    /// Componentwise peak usage since construction (not a simultaneous vector).
    pub(crate) fn peak(&self) -> Resources<N> {
        self.peak
    }
}
