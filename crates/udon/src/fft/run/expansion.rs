use super::super::{Expansion, ExpansionOrder, ExpansionStorage};
use super::*;

/// Worker-independent expansion geometry and input liveness.
///
/// Use [`Self::execute`] for preserved input, [`Self::execute_disposable`] to
/// reuse an evaluation buffer for coefficients, or [`ExpansionRun`] for
/// incremental scheduling. Output layouts follow [`ExpansionOrder`].
#[derive(Clone, Copy, Debug)]
pub struct ExpansionPlan<'t, M: PrimeModulus> {
    pub(super) expansion: Expansion<'t, M>,
    pub(super) storage: ExpansionStorage,
    pub(super) order: ExpansionOrder,
    pub(super) support: InputSupport,
    pub(super) input_order: ElementOrder,
    tile: NonZeroUsize,
    codelet: Codelet,
    coefficient_scale: PastaField<M>,
    pub(super) budget: crate::exec::TaskBudget,
    execution: Option<(super::super::StorageLayout, crate::exec::ExecutionOptions)>,
    memory_limit: Option<usize>,
}

impl<'t, M: PrimeModulus> ExpansionPlan<'t, M> {
    /// Resolves an expansion from mathematical layout and resource constraints.
    ///
    /// The task budget covers both residue concurrency and each inner transform.
    /// The workspace ceiling includes separately retained coefficients and all
    /// concurrent transform scratch. Incremental binding checks its slot count
    /// against the same ceiling.
    ///
    /// Coefficients accept a natural prefix of length zero through the base size,
    /// or full input in either order. Other storage modes require full base
    /// evaluations in either order; a prefix returns
    /// [`FftError::InvalidExecution`]. Other request and fragment-layout errors
    /// follow [`FftPlan::new`]. Required coefficients and scratch that cannot fit
    /// the byte ceiling return [`FftError::MemoryLimit`]. Construction borrows
    /// the expansion's tables without binding or writing working storage.
    pub fn new(
        expansion: Expansion<'t, M>,
        storage: ExpansionStorage,
        order: ExpansionOrder,
        support: InputSupport,
        input_order: ElementOrder,
        layout: super::super::StorageLayout,
        options: crate::exec::ExecutionOptions,
    ) -> Result<Self, FftError> {
        let mut result = Self::with_strategy(
            expansion,
            storage,
            order,
            support,
            input_order,
            NonZeroUsize::MIN,
            Codelet::Radix2,
        )?;
        let coefficient_bytes = result.coefficient_fields() * core::mem::size_of::<PastaField<M>>();
        let mut budget = options.task_budget();
        loop {
            let (jobs, inner) = budget.partition(result.residues()).unwrap();
            let mut inner_options = options.with_task_budget(inner);
            if let Some(limit) = options.memory_limit() {
                let remaining =
                    limit
                        .checked_sub(coefficient_bytes)
                        .ok_or(FftError::MemoryLimit {
                            required: coefficient_bytes,
                            limit,
                        })?;
                inner_options = inner_options.with_memory_limit(remaining / jobs);
            }
            result.execution = Some((layout, inner_options));
            let validation = result
                .try_transform(0, storage == ExpansionStorage::ReuseOutput)
                .and_then(|transform| {
                    if storage != ExpansionStorage::Coefficients {
                        result.try_inverse()?;
                    }
                    Ok(transform)
                });
            match validation {
                Ok(transform) => {
                    result.tile = NonZeroUsize::new(transform.tile()).unwrap();
                    break;
                }
                Err(FftError::MemoryLimit { .. }) if budget.get() > 1 => {
                    budget = crate::exec::TaskBudget::new(budget.get().div_ceil(2)).unwrap();
                }
                Err(error) => return Err(error),
            }
        }
        result.budget = budget;
        result.memory_limit = options.memory_limit();
        Ok(result)
    }

    fn resolve(
        &self,
        transform: Transform<'t, M>,
        request: TransformRequest,
    ) -> Result<FftPlan<'t, M>, FftError> {
        match self.execution {
            Some((layout, options)) => FftPlan::new(transform, request, layout, options),
            None => FftPlan::with_strategy(transform, request, self.tile, self.codelet),
        }
    }

    /// Validates an expansion without binding working storage.
    ///
    /// Coefficients accept a natural prefix or full input in either order.
    /// Other storage modes accept full base evaluations in either order.
    /// Validation of support and tile geometry follows [`FftPlan::new`]; a
    /// prefix for evaluation input returns
    /// [`FftError::InvalidExecution`]. The expansion's tables remain borrowed.
    pub(crate) fn with_strategy(
        expansion: Expansion<'t, M>,
        storage: ExpansionStorage,
        order: ExpansionOrder,
        support: InputSupport,
        input_order: ElementOrder,
        tile: NonZeroUsize,
        codelet: Codelet,
    ) -> Result<Self, FftError> {
        if storage != ExpansionStorage::Coefficients && support != InputSupport::Full {
            return Err(FftError::InvalidExecution);
        }
        let result = Self {
            expansion,
            storage,
            order,
            support,
            input_order,
            tile,
            codelet,
            coefficient_scale: PastaField::ONE,
            execution: None,
            memory_limit: None,
            budget: crate::exec::TaskBudget::SERIAL,
        };
        result.try_transform(0, false)?;
        if storage != ExpansionStorage::Coefficients {
            result.try_inverse()?;
        }
        Ok(result)
    }

    /// Multiplies preserved coefficients by a common normalization factor.
    ///
    /// For a [`super::super::CoefficientView`], pass its `normalization_factor()` to
    /// recover ordinary polynomial evaluations. Evaluation storage modes derive this
    /// from their inverse scale. Panics unless the plan selects
    /// [`ExpansionStorage::Coefficients`].
    pub fn with_coefficient_scale(mut self, scale: PastaField<M>) -> Self {
        assert_eq!(
            self.storage,
            ExpansionStorage::Coefficients,
            "coefficient scale requires coefficient input"
        );
        self.coefficient_scale = scale;
        self
    }

    /// Number of independently completed output residue blocks.
    pub fn residues(&self) -> usize {
        self.expansion.layout.residues()
    }

    /// Fields in each residue and in a retained coefficient bank.
    pub fn base_size(&self) -> usize {
        self.expansion.base.domain().size()
    }

    /// Writable fragment size within every base transform.
    pub fn tile(&self) -> usize {
        self.tile.get().min(self.base_size())
    }

    /// Additional coefficient fields, excluding reused input or output banks.
    pub fn coefficient_fields(&self) -> usize {
        if matches!(self.storage, ExpansionStorage::CoefficientWorkspace { .. }) {
            self.base_size()
        } else {
            0
        }
    }

    /// Snapshot field count per retained transform slot.
    ///
    /// The inverse and successive residues reuse that bank. Charge all
    /// provisioned slots and metadata, even while idle; coefficients remain live
    /// through consumers.
    pub fn snapshot_fields(&self) -> usize {
        let residue = self
            .transform(0, self.storage == ExpansionStorage::ReuseOutput)
            .retained_fields();
        let inverse = if self.storage == ExpansionStorage::Coefficients {
            0
        } else {
            self.inverse().retained_fields()
        };
        residue.max(inverse)
    }

    fn coefficients(&self) -> ExpansionBank {
        match self.storage {
            ExpansionStorage::Coefficients | ExpansionStorage::DisposableInput { .. } => {
                ExpansionBank::Input
            }
            ExpansionStorage::CoefficientWorkspace { .. } => ExpansionBank::Coefficients,
            ExpansionStorage::ReuseOutput => ExpansionBank::Output(0),
        }
    }

    pub(super) fn inverse(&self) -> FftPlan<'t, M> {
        self.try_inverse().expect("validated inverse geometry")
    }

    pub(super) fn transform(&self, block: usize, in_place: bool) -> FftPlan<'t, M> {
        self.try_transform(block, in_place)
            .expect("validated residue geometry")
    }

    fn try_inverse(&self) -> Result<FftPlan<'t, M>, FftError> {
        let scale = match self.storage {
            ExpansionStorage::CoefficientWorkspace { scale }
            | ExpansionStorage::DisposableInput { scale } => scale,
            _ => InverseScale::Normalized,
        };
        self.resolve(
            self.expansion.base,
            TransformRequest {
                input_storage: if !matches!(self.storage, ExpansionStorage::DisposableInput { .. })
                {
                    crate::fft::InputStorage::Preserve
                } else {
                    crate::fft::InputStorage::InPlace
                },
                input_order: self.input_order,
                inverse_scale: scale,
                ..TransformRequest::new(Direction::Inverse)
            },
        )
    }

    fn try_transform(&self, block: usize, in_place: bool) -> Result<FftPlan<'t, M>, FftError> {
        let residue = if self.order == ExpansionOrder::BitReversed {
            reverse(block, self.residues().ilog2())
        } else {
            block
        };
        let base = self.expansion.residue_base(residue);
        let normalized = !matches!(
            self.storage,
            ExpansionStorage::CoefficientWorkspace {
                scale: InverseScale::Unscaled
            } | ExpansionStorage::DisposableInput {
                scale: InverseScale::Unscaled
            }
        );
        let extra = if normalized {
            self.coefficient_scale
        } else {
            self.expansion.base.domain().domain().size_inverse()
        };
        Ok(self
            .resolve(
                base,
                TransformRequest {
                    input_storage: if !in_place {
                        crate::fft::InputStorage::Preserve
                    } else {
                        crate::fft::InputStorage::InPlace
                    },
                    support: self.support,
                    input_order: if self.storage == ExpansionStorage::Coefficients {
                        self.input_order
                    } else {
                        ElementOrder::Natural
                    },
                    output_order: if self.order == ExpansionOrder::Residues {
                        ElementOrder::Natural
                    } else {
                        ElementOrder::BitReversed
                    },
                    ..TransformRequest::new(Direction::Forward)
                },
            )?
            .with_residue_scales(self.expansion, residue, extra))
    }
}

/// Application-owned physical banks used by an expansion task.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpansionBank {
    /// Original coefficients or evaluations; writable for disposable input.
    Input,
    /// Separate retained coefficient workspace.
    Coefficients,
    /// Physical output residue block in the selected expansion order.
    Output(usize),
}

/// One base-transform task with its physical bank mapping.
#[derive(Clone, Debug)]
pub struct ExpansionRequest<'a> {
    /// Metadata and snapshot slot to use.
    pub slot: usize,
    /// Base-transform accesses; ranges are relative to their mapped banks.
    pub task: Request<'a>,
    /// Physical bank corresponding to [`Bank::Input`].
    pub input: ExpansionBank,
    /// Physical bank corresponding to [`Bank::Values`].
    pub values: ExpansionBank,
}

/// Publication from an expansion, including immediately available residues.
pub struct ExpansionPublished<R> {
    /// Base-transform publication and returned resource envelope.
    pub task: Published<R>,
    /// Physical residue block whose final task just completed.
    pub residue: Option<usize>,
    /// Whether every residue completed successfully.
    pub complete: bool,
}

/// Incremental inverse and residue transforms using fixed retained slots.
///
/// `SLOTS` bounds retained snapshots and metadata, independently of workers.
/// Residues publish separately and refill a completed slot immediately. The
/// output-reuse policy transforms block zero last: its coefficient bank stays
/// shared until every other residue finishes reading it. A failed task stops
/// production while outstanding receipts remain drainable.
pub struct ExpansionRun<'a, 't, M: PrimeModulus, const SLOTS: usize> {
    plan: ExpansionPlan<'t, M>,
    runs: [FftRun<'a, 't, M>; SLOTS],
    blocks: [Option<usize>; SLOTS],
    inverse: bool,
    next: usize,
    completed: usize,
    product: bool,
    failed: bool,
}

impl<'a, 't, M: PrimeModulus, const SLOTS: usize> ExpansionRun<'a, 't, M, SLOTS> {
    /// Binds fixed metadata for incremental expansion.
    ///
    /// `SLOTS` and `TASKS` must be nonzero. Returns [`TaskError::Storage`] when all
    /// slots' snapshots and the separate coefficient bank exceed the plan's workspace
    /// ceiling. Unrepresentable workspace returns [`TaskError::Overflow`]. These checks
    /// precede metadata writes. `product` requests factors in physical output residue
    /// order. All banks and queue capacity must be admitted together through the last
    /// coefficient and residue consumer before dispatch.
    pub fn new<const TASKS: usize>(
        plan: ExpansionPlan<'t, M>,
        product: bool,
        identities: &'a mut [Identity; SLOTS],
        storage: &'a mut [[TaskStorage; TASKS]; SLOTS],
    ) -> Result<Self, TaskError> {
        const {
            assert!(
                SLOTS > 0 && TASKS > 0,
                "slot and task capacities must be nonzero"
            )
        };
        let fields = plan
            .snapshot_fields()
            .checked_mul(SLOTS)
            .and_then(|n| n.checked_add(plan.coefficient_fields()))
            .ok_or(TaskError::Overflow)?;
        let bytes = fields
            .checked_mul(core::mem::size_of::<PastaField<M>>())
            .ok_or(TaskError::Overflow)?;
        if plan.memory_limit.is_some_and(|limit| bytes > limit) {
            return Err(TaskError::Storage);
        }
        let dummy = plan.transform(0, false);
        let mut metadata = identities.iter_mut().zip(storage);
        let runs = core::array::from_fn(|_| {
            let (id, slots) = metadata.next().unwrap();
            FftRun::empty(dummy, id, slots)
        });
        let mut run = Self {
            plan,
            runs,
            blocks: [None; SLOTS],
            inverse: plan.storage != ExpansionStorage::Coefficients,
            next: usize::from(plan.storage == ExpansionStorage::ReuseOutput),
            completed: 0,
            product,
            failed: false,
        };
        if run.inverse {
            run.runs[0].rebind(plan.inverse(), false)?;
        } else {
            run.refill()?;
        }
        Ok(run)
    }

    fn refill(&mut self) -> Result<(), TaskError> {
        for slot in 0..SLOTS {
            if self.blocks[slot].is_some() {
                continue;
            }
            let block = if self.next < self.plan.residues() {
                let block = self.next;
                self.next += 1;
                block
            } else if self.plan.storage == ExpansionStorage::ReuseOutput
                && self.completed + 1 == self.plan.residues()
                && !self.blocks.contains(&Some(0))
            {
                0
            } else {
                break;
            };
            let in_place = block == 0 && self.plan.storage == ExpansionStorage::ReuseOutput;
            self.runs[slot].rebind(self.plan.transform(block, in_place), self.product)?;
            self.blocks[slot] = Some(block);
        }
        Ok(())
    }

    /// Pages through one slot's ready tasks. Visit all slots and task indices
    /// before concluding that no compatible bundle is available; restart
    /// enumeration after publication. Snapshot banks are private to `slot`.
    pub fn ready_slot_from(
        &self,
        slot: usize,
        start: usize,
        output: &mut [Option<ExpansionRequest<'a>>],
    ) -> usize {
        if self.failed
            || slot >= SLOTS
            || (self.inverse && slot != 0)
            || (!self.inverse && self.blocks[slot].is_none())
        {
            return 0;
        }
        let mut cursor = start;
        let mut written = 0;
        for entry in output {
            let mut request = [None];
            if self.runs[slot].ready_from(cursor, &mut request) == 0 {
                break;
            }
            let task = request[0].take().unwrap();
            cursor = task.key.index() + 1;
            *entry = Some(ExpansionRequest {
                slot,
                task,
                input: if self.inverse {
                    ExpansionBank::Input
                } else {
                    self.plan.coefficients()
                },
                values: if self.inverse {
                    self.plan.coefficients()
                } else {
                    ExpansionBank::Output(self.blocks[slot].unwrap())
                },
            });
            written += 1;
        }
        written
    }

    /// Acquires mapped banks and scratch before detaching a bounded FFT task.
    pub fn try_claim<R: Resources<M>>(
        &mut self,
        request: ExpansionRequest<'a>,
        acquire: impl FnOnce() -> Option<R>,
    ) -> Result<Option<Task<'a, FftKernel<'t, M>, R>>, TaskError> {
        if self.failed {
            return Err(TaskError::Failed);
        }
        self.runs
            .get_mut(request.slot)
            .ok_or(TaskError::Stale)?
            .try_claim(request.task, acquire)
    }

    /// Publishes a task to its original slot. Returned guards must be released
    /// before reusing banks. Foreign receipts are returned intact.
    pub fn complete<R>(
        &mut self,
        slot: usize,
        receipt: Completion<'a, R, ()>,
    ) -> Result<ExpansionPublished<R>, crate::exec::run::PublishError<'a, R, ()>> {
        let Some(run) = self.runs.get_mut(slot) else {
            return Err((TaskError::Stale, receipt));
        };
        let mut task = run.complete(receipt)?;
        self.failed |= run.is_failed();
        let mut residue = None;
        if task.complete && !self.failed {
            if self.inverse {
                self.inverse = false;
            } else {
                residue = self.blocks[slot].take();
                self.completed += 1;
            }
            if let Err(error) = self.refill() {
                self.failed = true;
                task.error = Some(error);
            }
        }
        Ok(ExpansionPublished {
            task,
            residue,
            complete: self.is_complete(),
        })
    }

    /// Whether every output residue is ready for consumption.
    pub fn is_complete(&self) -> bool {
        !self.failed && self.completed == self.plan.residues()
    }

    /// Whether cancellation, unwind, or a phase transition failed a task.
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Detached tasks still requiring publication, including on failure.
    pub fn inflight(&self) -> usize {
        self.runs.iter().map(FftRun::inflight).sum()
    }
}
