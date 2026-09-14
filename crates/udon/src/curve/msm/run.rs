//! Bounded MSM preparation, window, and reduction tasks.
//!
//! [`MsmPlan`] fixes legal term subdivisions independently of worker count.
//! [`MsmRun`] binds one invocation and incrementally exposes [`Request`]s. The
//! application acquires the requested typed leases, claims a task, executes it
//! on any worker, and publishes its completion. Preparation and window storage
//! are retained until their consumers finish; arithmetic scratch belongs to
//! each executing task and can serve other operations between tasks.

use core::{marker::PhantomData, num::NonZeroUsize};

use super::{
    CurveError, ExecutionOptions, Input, PastaCurve, ProjectivePoint, Requirements, ScalarStorage,
    Scalars, Scratch, check_scratch, kernels, prepared,
    recode::{self, Geometry, Shape},
    schedule,
};
use crate::exec::{
    TaskBudget,
    run::{
        Completion, Frontier, Identity, Kernel, Outcome, ReadView, Task, TaskError, TaskKey,
        TaskStorage, WorkEstimate,
    },
};

mod chunks;
mod driver;
pub(super) mod storage;
pub use chunks::{ChunkRequest, ParallelMsmRun};

/// Arithmetic and storage requirements for bounded MSM tasks.
///
/// `grain` is a maximum number of terms processed by one preparation or window
/// task. It is independent of available workers. Smaller chunks add window
/// collapses and partial reductions; explicit streaming instead retains every
/// window's buckets and deposits one chunk per task. The caller must reserve
/// all those buckets before admitting streaming work.
#[derive(Clone, Copy, Debug)]
pub struct MsmPlan<C: PastaCurve> {
    terms: usize,
    cap: usize,
    options: ExecutionOptions,
    job: schedule::JobStorage,
    retained: Requirements,
    marker: PhantomData<C>,
}

impl<C: PastaCurve> MsmPlan<C> {
    /// Plans bounded tasks without binding inputs or allocating storage.
    ///
    /// The existing accumulation, width, joint, pass, and streaming options
    /// apply. `task_budget` does not affect this plan. The effective grain is
    /// the smaller of `grain`, the optional chunk cap, and the term count.
    /// A memory limit checks retained storage plus one complete task's scratch;
    /// application admission must additionally charge run and queue metadata.
    /// Returns sizing errors before binding or writing any storage.
    pub fn new(
        terms: usize,
        mut options: ExecutionOptions,
        grain: NonZeroUsize,
    ) -> Result<Self, CurveError> {
        options.task_budget = TaskBudget::SERIAL;
        let cap = terms
            .min(grain.get())
            .min(options.chunk_size.map_or(usize::MAX, NonZeroUsize::get));
        options.chunk_size = NonZeroUsize::new(cap.max(1));
        if cap >= 4096 && options.window_bits.is_none() && !options.joint_tables {
            options.window_bits = Some(11);
        }
        let geometry = Geometry::for_len(cap, options);
        let job = schedule::layout::<C>(cap, geometry, false, false, false, options)?;
        let retained = Requirements {
            scalars: cap,
            digits: geometry.storage_len(cap)?,
            projective: if terms == 0 {
                0
            } else {
                geometry
                    .windows()
                    .checked_mul(if options.streaming {
                        geometry.buckets() + 1
                    } else {
                        1
                    })
                    .ok_or(CurveError::SizeOverflow)?
            },
            ..Requirements::default()
        };
        let temporary = if options.streaming {
            Requirements::default()
        } else {
            job.work
        };
        let bytes = retained
            .bytes::<C>()?
            .checked_add(temporary.bytes::<C>()?)
            .ok_or(CurveError::SizeOverflow)?;
        if let Some(limit) = options.memory_limit
            && bytes > limit
        {
            return Err(CurveError::MemoryLimit {
                limit,
                required: bytes,
            });
        }
        Ok(Self {
            terms,
            cap,
            options,
            job,
            retained,
            marker: PhantomData,
        })
    }

    pub(super) fn from_job(
        terms: usize,
        options: ExecutionOptions,
        job: schedule::JobStorage,
    ) -> Self {
        let retained = Requirements {
            scalars: job.requirements.scalars,
            digits: job.requirements.digits,
            projective: if terms == 0 {
                0
            } else {
                job.geometry.windows()
                    + if options.streaming {
                        job.work.projective
                    } else {
                        0
                    }
            },
            ..Requirements::default()
        };
        Self {
            terms,
            cap: job.cap,
            options,
            job,
            retained,
            marker: PhantomData,
        }
    }

    /// Maximum operation-retained typed storage, excluding run metadata.
    pub fn retained(&self) -> Requirements {
        self.retained
    }

    /// Retained arithmetic storage for `slots` independent term chunks.
    ///
    /// Each slot owns preparation and one partial per window. The shared
    /// temporary bundle is unchanged. Metadata, idle provider capacity, and
    /// alignment still belong in application admission accounting. Streaming
    /// uses [`MsmRun`] with one complete set of retained window buckets.
    /// Returns sizing or memory-limit errors including all requested slots.
    pub fn retained_for_slots(&self, slots: NonZeroUsize) -> Result<Requirements, CurveError> {
        let mut r = self.retained;
        r.scalars = r
            .scalars
            .checked_mul(slots.get())
            .ok_or(CurveError::SizeOverflow)?;
        r.digits = r
            .digits
            .checked_mul(slots.get())
            .ok_or(CurveError::SizeOverflow)?;
        r.projective = r
            .projective
            .checked_mul(slots.get())
            .ok_or(CurveError::SizeOverflow)?;
        let bytes = r
            .bytes::<C>()?
            .checked_add(self.temporary().bytes::<C>()?)
            .ok_or(CurveError::SizeOverflow)?;
        if let Some(limit) = self.options.memory_limit
            && bytes > limit
        {
            return Err(CurveError::MemoryLimit {
                limit,
                required: bytes,
            });
        }
        Ok(r)
    }

    /// Largest single executing task's temporary arithmetic bundle.
    pub fn temporary(&self) -> Requirements {
        if self.options.streaming {
            Requirements::default()
        } else {
            self.job.work
        }
    }

    /// Maximum terms in a task; zero only for an empty operation.
    pub fn grain(&self) -> usize {
        self.cap
    }

    /// Number of ordinary window tasks per full-width chunk.
    pub fn windows(&self) -> usize {
        if self.terms == 0 {
            0
        } else {
            self.job.geometry.windows()
        }
    }

    // Reusing scalar records is independent of digit geometry. A digit cache
    // remains valid only for the complete, unsplit input and selected geometry.
    fn cached<'i>(&self, input: Input<'i, C>, geometry: Geometry) -> Option<&'i [u8]> {
        if self.options.streaming || input.len() > self.cap {
            return None;
        }
        match input.scalars {
            Scalars::Prepared(s) => s
                .cached
                .filter(|c| c.geometry == geometry)
                .map(|c| c.digits),
            _ => None,
        }
    }

    fn initial_geometry(&self, input: Input<'_, C>) -> Geometry {
        if !self.options.streaming
            && let Scalars::Prepared(s) = input.scalars
        {
            let actual = Geometry::for_shape(input.len(), s.shape, self.options);
            if matches!(actual, Geometry::Short(_)) {
                return actual;
            }
        }
        self.job.geometry
    }
}

/// Work kind with distinct storage lifetime requirements.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum WorkKind {
    /// Writes one retained scalar and recoding chunk.
    Prepare,
    /// Reads a prepared chunk and computes or deposits one window.
    Window,
    /// Collapses one retained streaming bucket array.
    Collapse,
    /// Recombines completed windows in a bounded ordered reduction.
    Reduce,
}

/// One ready MSM task and the complete resource bundle it requires.
#[derive(Clone, Copy, Debug)]
pub struct Request<'a> {
    /// Claim identity, valid only for this run and dependency epoch.
    pub key: TaskKey<'a>,
    /// Kernel kind.
    pub kind: WorkKind,
    /// First input term processed by this task.
    pub offset: usize,
    /// Terms in this chunk.
    pub terms: usize,
    /// First retained scalar written during preparation, relative to the chunk.
    pub scalar_start: usize,
    /// First retained recoding byte written during preparation.
    pub digit_start: usize,
    /// Window index, or zero for preparation and reduction.
    pub window: usize,
    /// Exclusive task scratch. During preparation its scalar and digit fields
    /// describe writes into retained storage, not additional temporary blocks.
    pub scratch: Requirements,
    /// Shared retained scalar prefix for a window task.
    pub read_scalars: usize,
    /// Shared retained digit prefix for a window task.
    pub read_digits: usize,
    /// Exclusive retained projective bucket prefix for streaming tasks.
    pub buckets: usize,
    /// Exclusive single logical partial output for window or collapse tasks.
    pub write_partial: bool,
    /// Number of shared logical partials read by a reduction task.
    pub read_partials: usize,
    /// Optional scheduling estimates.
    pub estimate: WorkEstimate,
}

/// Borrowed views obtained from an owned task resource bundle.
pub struct Buffers<'a, C: PastaCurve> {
    /// Shared prepared scalar records; empty for a preparation task.
    pub records: &'a dyn ReadView<ScalarStorage<C>>,
    /// Shared recoding bytes; empty for a preparation task.
    pub digits: &'a dyn ReadView<u8>,
    /// Exclusive typed scratch or preparation destination prefixes.
    pub scratch: Scratch<'a, C>,
    /// Exclusive retained buckets for one streaming window.
    pub buckets: &'a mut [ProjectivePoint<C>],
    /// One exclusive logical result slot for a window or collapse task.
    pub output: &'a mut [ProjectivePoint<C>],
    /// Fragmented shared results for the reduction task.
    pub partials: &'a dyn ReadView<ProjectivePoint<C>>,
}

/// An owned bundle of safe fragments or application-provided lease guards.
///
/// The views must represent the prefixes and window named by the claimed
/// [`Request`]. Lengths are checked before writes; semantic slot identity and
/// initialized retained contents belong to the application provider's contract.
/// Incorrect contents can produce incorrect arithmetic but cannot violate Rust
/// memory safety. Guards remain owned by the task through unwinding.
pub trait Resources<C: PastaCurve> {
    /// Borrows all disjoint mutable and shared views for this bounded kernel.
    fn buffers(&mut self) -> Buffers<'_, C>;
}

impl<C: PastaCurve> Resources<C> for Buffers<'_, C> {
    fn buffers(&mut self) -> Buffers<'_, C> {
        Buffers {
            records: self.records,
            digits: self.digits,
            scratch: self.scratch.reborrow(),
            buckets: self.buckets,
            output: self.output,
            partials: self.partials,
        }
    }
}

/// Detached arithmetic for one request; constructed only by [`MsmRun`].
pub struct MsmKernel<'i, C: PastaCurve> {
    input: Input<'i, C>,
    plan: MsmPlan<C>,
    kind: WorkKind,
    offset: usize,
    terms: usize,
    window: usize,
    geometry: Geometry,
}

/// Opaque kernel output, published through [`MsmRun::complete`].
pub struct MsmOutput<C: PastaCurve>(Result<(Shape, ProjectivePoint<C>), CurveError>);

impl<C: PastaCurve> MsmKernel<'_, C> {
    fn run(&self, buffers: Buffers<'_, C>) -> Result<(Shape, ProjectivePoint<C>), CurveError> {
        let Buffers {
            records: buffers_records,
            digits,
            scratch,
            buckets,
            output,
            partials,
        } = buffers;
        let geometry = self.geometry;
        let mut shape = Shape { bits: 0, weight: 0 };
        let mut result = ProjectivePoint::IDENTITY;
        match self.kind {
            WorkKind::Prepare => {
                let source = self
                    .input
                    .scalars
                    .slice(self.offset..self.offset + self.terms);
                let records = match source {
                    Scalars::Prepared(s) => s.records,
                    _ => {
                        check_scratch("scalars", self.terms, scratch.scalars.len())?;
                        let records = &mut scratch.scalars[..self.terms];
                        prepared::prepare_chunk(source, records);
                        records
                    }
                };
                shape = Shape::of(records);
                if self.plan.cached(self.input, geometry).is_none() {
                    let len = geometry.storage_len(self.terms)?;
                    check_scratch("digits", len, scratch.digits.len())?;
                    recode::write(records, geometry, &mut scratch.digits[..len]);
                }
            }
            WorkKind::Window => {
                let records = match self.input.scalars {
                    Scalars::Prepared(s) => Some(&s.records[self.offset..self.offset + self.terms]),
                    _ => None,
                };
                let records: &dyn ReadView<ScalarStorage<C>> = records
                    .as_ref()
                    .map_or(buffers_records, |r| r as &dyn ReadView<_>);
                let cache = self.plan.cached(self.input, geometry);
                let digits: &dyn ReadView<u8> =
                    cache.as_ref().map_or(digits, |d| d as &dyn ReadView<_>);
                check_scratch("scalars", self.terms, records.len())?;
                check_scratch("digits", geometry.storage_len(self.terms)?, digits.len())?;
                let task = kernels::Task {
                    offset: self.offset,
                    window: self.window,
                    pass: self.plan.job.pass.min(self.terms),
                    geometry,
                    accumulation: self.plan.job.accumulation,
                };
                let digit_len = geometry.storage_len(self.terms)?;
                if self.plan.options.streaming {
                    check_scratch("projective", geometry.buckets(), buckets.len())?;
                    let buckets = &mut buckets[..geometry.buckets()];
                    if self.offset == 0 {
                        buckets.fill(ProjectivePoint::IDENTITY);
                    }
                    if let Some(digits) = digits.contiguous(0..digit_len) {
                        kernels::stream(&self.input, self.terms, digits, task, buckets);
                    } else {
                        kernels::stream_view(
                            &self.input,
                            self.terms,
                            storage::Fragmented::new(digits, digit_len),
                            task,
                            buckets,
                        );
                    }
                } else {
                    check_scratch("partial output", 1, output.len())?;
                    let scratch = scratch.checked(self.plan.temporary())?;
                    let work = &mut kernels::Work {
                        affine: scratch.affine,
                        projective: scratch.projective,
                        field: scratch.field,
                        indices: scratch.indices,
                    };
                    output[0] = match (
                        records.contiguous(0..self.terms),
                        digits.contiguous(0..digit_len),
                    ) {
                        (Some(records), Some(digits)) => {
                            kernels::run(&self.input, records, digits, task, work)
                        }
                        _ => kernels::run_view(
                            &self.input,
                            storage::Fragmented::new(records, self.terms),
                            storage::Fragmented::new(digits, digit_len),
                            task,
                            work,
                        ),
                    };
                }
            }
            WorkKind::Collapse => {
                check_scratch("projective", geometry.buckets(), buckets.len())?;
                check_scratch("partial output", 1, output.len())?;
                output[0] = kernels::collapse_projective(&buckets[..geometry.buckets()]);
            }
            WorkKind::Reduce => {
                check_scratch("partial inputs", geometry.windows(), partials.len())?;
                for window in (0..geometry.windows()).rev() {
                    let partial = partials
                        .get(window)
                        .expect("invalid fragmented partial view");
                    for _ in 0..geometry.width() {
                        result = result.double();
                    }
                    result = result.add(partial);
                }
            }
        }
        Ok((shape, result))
    }
}

impl<C: PastaCurve, R: Resources<C>> Kernel<R> for MsmKernel<'_, C> {
    type Output = MsmOutput<C>;
    fn execute(&mut self, resources: &mut R) -> Self::Output {
        MsmOutput(self.run(resources.buffers()))
    }
}

/// Returned leases and operation publication after one task completes.
pub struct Published<C: PastaCurve, R> {
    /// All task resources, ready for provider release or retained handoff.
    pub resources: R,
    /// Arithmetic validation error, if any; the run is then failed.
    pub error: Option<CurveError>,
    /// Kernel execution status, including panic or cancellation.
    pub outcome: Outcome,
    /// Final result, present only on the completion that finishes this MSM.
    pub result: Option<ProjectivePoint<C>>,
}

/// One bound MSM invocation with incremental local dependencies.
///
/// Caller-owned frontier storage bounds detached tasks. Its size controls
/// dispatch capacity, not geometry or scratch provisioning. Preparation readies
/// this input's windows immediately, and its final reduction releases a result
/// without waiting for other operations. Retained chunk storage can be reused
/// after every window consuming it has completed. Streaming bucket storage
/// remains retained through the final collapse.
pub struct MsmRun<'a, 'i, C: PastaCurve> {
    input: Input<'i, C>,
    plan: MsmPlan<C>,
    frontier: Frontier<'a>,
    kind: WorkKind,
    offset: usize,
    end: usize,
    geometry: Geometry,
    shape: Shape,
    result: ProjectivePoint<C>,
    complete: bool,
    failed: bool,
}

impl<'a, 'i, C: PastaCurve> MsmRun<'a, 'i, C> {
    /// Binds an input and exclusive run metadata without touching arithmetic
    /// buffers. Returns an error if its term count differs from the plan or no
    /// frontier storage is supplied.
    pub fn new(
        plan: MsmPlan<C>,
        input: Input<'i, C>,
        identity: &'a mut Identity,
        storage: &'a mut [TaskStorage],
    ) -> Result<Self, TaskError> {
        if input.len() != plan.terms {
            return Err(TaskError::Storage);
        }
        Self::bind_range(plan, input, 0..input.len(), identity, storage)
    }

    fn bind_range(
        plan: MsmPlan<C>,
        input: Input<'i, C>,
        range: core::ops::Range<usize>,
        identity: &'a mut Identity,
        storage: &'a mut [TaskStorage],
    ) -> Result<Self, TaskError> {
        let complete = range.is_empty();
        let frontier = Frontier::new(
            identity,
            storage,
            range.len().min(plan.cap).div_ceil(recode::CHUNK),
        )?;
        Ok(Self {
            input,
            plan,
            frontier,
            kind: WorkKind::Prepare,
            offset: range.start,
            end: range.end,
            geometry: plan.initial_geometry(input),
            shape: Shape { bits: 0, weight: 0 },
            result: ProjectivePoint::IDENTITY,
            complete,
            failed: false,
        })
    }

    /// Writes a bounded number of ready requests, with no task allocation.
    pub fn ready(&self, output: &mut [Option<Request<'a>>]) -> usize {
        self.ready_from(0, output)
    }

    /// Continues a bounded readiness scan at a task index in the current
    /// epoch. Pass one past the last returned key's index to inspect later
    /// requests when the output buffer is smaller than the frontier. Restart
    /// at zero after publishing a completion, which may change dependencies.
    pub fn ready_from(&self, start: usize, output: &mut [Option<Request<'a>>]) -> usize {
        if self.complete || self.failed {
            return 0;
        }
        let mut written = 0;
        for key in self
            .frontier
            .tasks()
            .filter(|key| key.index() >= start)
            .take(output.len())
        {
            let chunk_terms = self.plan.cap.min(self.end - self.offset);
            let scalar_start = if self.kind == WorkKind::Prepare {
                key.index() * recode::CHUNK
            } else {
                0
            };
            let digit_start = scalar_start * self.geometry.stride();
            let terms = if self.kind == WorkKind::Prepare {
                (chunk_terms - scalar_start).min(recode::CHUNK)
            } else {
                chunk_terms
            };
            let mut scratch = Requirements::default();
            let mut read_scalars = 0;
            let mut read_digits = 0;
            let mut buckets = 0;
            match self.kind {
                WorkKind::Prepare => {
                    if !matches!(self.input.scalars, Scalars::Prepared(_)) {
                        scratch.scalars = terms;
                    }
                    if self.plan.cached(self.input, self.geometry).is_none() {
                        scratch.digits = self.geometry.storage_len(terms).unwrap();
                    }
                }
                WorkKind::Window => {
                    if !matches!(self.input.scalars, Scalars::Prepared(_)) {
                        read_scalars = terms;
                    }
                    if self.plan.cached(self.input, self.geometry).is_none() {
                        read_digits = self.geometry.storage_len(terms).unwrap();
                    }
                    scratch = self.plan.temporary();
                    if self.plan.options.streaming {
                        buckets = self.geometry.buckets();
                    }
                }
                WorkKind::Collapse => buckets = self.geometry.buckets(),
                WorkKind::Reduce => {}
            }
            output[written] = Some(Request {
                key,
                kind: self.kind,
                offset: self.offset + scalar_start,
                terms,
                scalar_start,
                digit_start,
                window: key.index(),
                scratch,
                read_scalars,
                read_digits,
                buckets,
                write_partial: self.kind == WorkKind::Collapse
                    || (self.kind == WorkKind::Window && !self.plan.options.streaming),
                read_partials: if self.kind == WorkKind::Reduce {
                    self.geometry.windows()
                } else {
                    0
                },
                estimate: WorkEstimate {
                    arithmetic: terms,
                    traffic_bytes: terms.saturating_mul(size_of::<ScalarStorage<C>>()),
                    cache_bytes: self.plan.temporary().bytes::<C>().unwrap(),
                },
            });
            written += 1;
        }
        written
    }

    /// Detaches a task after atomic acquisition of its complete resource bundle.
    /// Failed acquisition leaves the request ready. Use only requests obtained
    /// from this run; stale and duplicate keys are rejected before acquisition.
    pub fn try_claim<R: Resources<C>>(
        &mut self,
        request: Request<'a>,
        acquire: impl FnOnce() -> Option<R>,
    ) -> Result<Option<Task<'a, MsmKernel<'i, C>, R>>, TaskError> {
        let kernel = self.kernel(request)?;
        self.frontier.try_claim(request.key, kernel, acquire)
    }

    fn kernel(&self, request: Request<'a>) -> Result<MsmKernel<'i, C>, TaskError> {
        if self.failed {
            return Err(TaskError::Failed);
        }
        self.frontier.check_key(request.key)?;
        let start = if self.kind == WorkKind::Prepare {
            request.key.index() * recode::CHUNK
        } else {
            0
        };
        let terms = self.plan.cap.min(self.end - self.offset) - start;
        let terms = if self.kind == WorkKind::Prepare {
            terms.min(recode::CHUNK)
        } else {
            terms
        };
        Ok(MsmKernel {
            input: self.input,
            plan: self.plan,
            kind: self.kind,
            offset: self.offset + start,
            terms,
            window: request.key.index(),
            geometry: self.geometry,
        })
    }

    /// Publishes completion, returns owned leases, and readies local successors.
    /// Failed or foreign receipts follow [`Frontier::complete`]. Resource
    /// validation failure poisons the run and publishes no arithmetic result.
    #[expect(
        clippy::result_large_err,
        reason = "return owned receipts without allocation"
    )]
    pub fn complete<R>(
        &mut self,
        completion: Completion<'a, R, MsmOutput<C>>,
    ) -> Result<Published<C, R>, crate::exec::run::PublishError<'a, R, MsmOutput<C>>> {
        let completed = self.frontier.complete(completion)?;
        let mut error = None;
        self.failed |= completed.outcome != Outcome::Success;
        if let Some(MsmOutput(output)) = completed.output {
            match output {
                Ok((shape, partial)) => {
                    if self.kind == WorkKind::Prepare {
                        self.shape.bits = self.shape.bits.max(shape.bits);
                        self.shape.weight = self.shape.weight.saturating_add(shape.weight);
                    } else if self.kind == WorkKind::Reduce {
                        self.result = self.result.add(&partial);
                    }
                }
                Err(e) => {
                    self.failed = true;
                    error = Some(e);
                }
            }
        }
        if !self.failed && self.frontier.is_complete() {
            let total = match self.kind {
                WorkKind::Prepare => {
                    if !self.plan.options.streaming {
                        let actual = Geometry::for_shape(
                            self.plan.cap.min(self.end - self.offset),
                            self.shape,
                            self.plan.options,
                        );
                        if matches!(actual, Geometry::Short(_)) {
                            self.geometry = actual;
                        }
                    }
                    self.kind = WorkKind::Window;
                    self.geometry.windows()
                }
                WorkKind::Window if !self.plan.options.streaming => {
                    self.kind = WorkKind::Reduce;
                    1
                }
                WorkKind::Window => {
                    if self.end - self.offset > self.plan.cap {
                        self.offset += self.plan.cap;
                        self.kind = WorkKind::Prepare;
                        self.shape = Shape { bits: 0, weight: 0 };
                        self.plan
                            .cap
                            .min(self.end - self.offset)
                            .div_ceil(recode::CHUNK)
                    } else {
                        self.kind = WorkKind::Collapse;
                        self.geometry.windows()
                    }
                }
                WorkKind::Collapse => {
                    self.kind = WorkKind::Reduce;
                    1
                }
                WorkKind::Reduce => {
                    if !self.plan.options.streaming && self.end - self.offset > self.plan.cap {
                        self.offset += self.plan.cap;
                        self.geometry = self.plan.initial_geometry(self.input);
                        self.kind = WorkKind::Prepare;
                        self.shape = Shape { bits: 0, weight: 0 };
                        self.plan
                            .cap
                            .min(self.end - self.offset)
                            .div_ceil(recode::CHUNK)
                    } else {
                        self.complete = true;
                        0
                    }
                }
            };
            if self.frontier.restart(total).is_err() {
                self.failed = true;
                error = Some(CurveError::SizeOverflow);
            }
        }
        Ok(Published {
            resources: completed.resources,
            error,
            outcome: completed.outcome,
            result: self.result(),
        })
    }

    /// Final result, including identity for an empty input; absent on failure.
    pub fn result(&self) -> Option<ProjectivePoint<C>> {
        (self.complete && !self.failed).then_some(self.result)
    }

    /// Whether a failed kernel or invalid resource bundle poisoned this run.
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Number of detached tasks still requiring publication or draining.
    pub fn inflight(&self) -> usize {
        self.frontier.inflight()
    }

    /// Rebinds completed metadata to another planned invocation, preserving
    /// epochs so old task keys remain stale. Consumers must release retained
    /// buffers before the application reuses their storage. Returns
    /// [`TaskError::Busy`] before completion, [`TaskError::Failed`] on failure,
    /// or [`TaskError::Storage`] for a mismatched input length.
    pub fn rebind(&mut self, plan: MsmPlan<C>, input: Input<'i, C>) -> Result<(), TaskError> {
        if input.len() != plan.terms {
            return Err(TaskError::Storage);
        }
        self.rebind_range(plan, input, 0..input.len())
    }

    fn rebind_range(
        &mut self,
        plan: MsmPlan<C>,
        input: Input<'i, C>,
        range: core::ops::Range<usize>,
    ) -> Result<(), TaskError> {
        if self.failed {
            return Err(TaskError::Failed);
        }
        if !self.complete {
            return Err(TaskError::Busy);
        }
        let complete = range.is_empty();
        self.frontier
            .restart(range.len().min(plan.cap).div_ceil(recode::CHUNK))?;
        self.input = input;
        self.plan = plan;
        self.geometry = plan.initial_geometry(input);
        self.shape = Shape { bits: 0, weight: 0 };
        self.offset = range.start;
        self.end = range.end;
        self.kind = WorkKind::Prepare;
        self.result = ProjectivePoint::IDENTITY;
        self.complete = complete;
        Ok(())
    }
}
