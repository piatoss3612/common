use super::super::{Expansion, ExpansionOrder, ExpansionScaleNormalization, ExpansionStorage};
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
}

impl<'t, M: PrimeModulus> ExpansionPlan<'t, M> {
    /// Validates an expansion without binding working storage.
    ///
    /// Coefficients accept a natural prefix or full input in either order.
    /// Other storage modes accept full base evaluations in either order.
    /// Validation of support and tile geometry follows [`FftPlan::new`]; a
    /// prefix for evaluation input returns
    /// [`FftError::InvalidExecution`]. The expansion's tables remain borrowed.
    pub fn new(
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
        };
        result.transform(0, false)?;
        if storage != ExpansionStorage::Coefficients {
            result.inverse()?;
        }
        Ok(result)
    }

    /// Multiplies preserved coefficients by a common normalization factor.
    ///
    /// For a [`super::super::CoefficientView`], pass its `normalization_factor()`
    /// to recover ordinary polynomial evaluations. Evaluation storage modes
    /// derive this from their inverse scale and reject this override with
    /// [`FftError::InvalidExecution`].
    pub fn with_coefficient_scale(mut self, scale: PastaField<M>) -> Result<Self, FftError> {
        if self.storage != ExpansionStorage::Coefficients {
            return Err(FftError::InvalidExecution);
        }
        self.coefficient_scale = scale;
        Ok(self)
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
            .expect("validated expansion geometry")
            .retained_fields();
        let inverse = if self.storage == ExpansionStorage::Coefficients {
            0
        } else {
            self.inverse().expect("validated inverse").retained_fields()
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

    pub(super) fn inverse(&self) -> Result<FftPlan<'t, M>, FftError> {
        let scale = match self.storage {
            ExpansionStorage::CoefficientWorkspace { scale }
            | ExpansionStorage::DisposableInput { scale } => scale,
            _ => InverseScale::Normalized,
        };
        FftPlan::new(
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
            self.tile,
            self.codelet,
        )
    }

    pub(super) fn transform(
        &self,
        block: usize,
        in_place: bool,
    ) -> Result<FftPlan<'t, M>, FftError> {
        let residue = if self.order == ExpansionOrder::BitReversed {
            reverse(block, self.residues().ilog2())
        } else {
            block
        };
        let shift = self.expansion.extended.shift().mul(
            &self
                .expansion
                .extended
                .domain()
                .root()
                .pow_u64(residue as u64),
        );
        let domain = self.expansion.base.domain().domain().coset(shift)?;
        // Subgroup twiddles are independent of the coset. Finish scales are
        // tied to the original domain and are not reused for a forward FFT.
        let mut base = self.expansion.base;
        base.domain = domain;
        base.tables.inverse_scales = None;
        base.tables.inverse_finish = None;
        let normalized = !matches!(
            self.storage,
            ExpansionStorage::CoefficientWorkspace {
                scale: InverseScale::Unscaled
            } | ExpansionStorage::DisposableInput {
                scale: InverseScale::Unscaled
            }
        );
        let mut extra = if normalized {
            self.coefficient_scale
        } else {
            self.expansion.base.domain().domain().size_inverse()
        };
        let scales = self.expansion.scales.filter(|_| {
            !normalized || self.expansion.normalization == ExpansionScaleNormalization::Coefficients
        });
        if scales.is_some()
            && self.expansion.normalization == ExpansionScaleNormalization::UnscaledInverse
        {
            extra = PastaField::ONE;
        }
        let mut plan = FftPlan::new(
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
            self.tile,
            self.codelet,
        )?
        .with_input_scale(extra)?;
        plan.forward_scales = scales
            .map(|scales| &scales[residue * self.base_size()..(residue + 1) * self.base_size()]);
        Ok(plan)
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
    /// Returns [`TaskError::Storage`] for zero slots or frontier capacity.
    /// `product` requests factors in physical output
    /// residue order. All banks and queue capacity must be admitted together
    /// through the last coefficient and residue consumer before dispatch.
    pub fn new<const TASKS: usize>(
        plan: ExpansionPlan<'t, M>,
        product: bool,
        identities: &'a mut [Identity; SLOTS],
        storage: &'a mut [[TaskStorage; TASKS]; SLOTS],
    ) -> Result<Self, TaskError> {
        if SLOTS == 0 || TASKS == 0 {
            return Err(TaskError::Storage);
        }
        let dummy = plan.transform(0, false).expect("validated expansion");
        let mut metadata = identities.iter_mut().zip(storage);
        let runs = core::array::from_fn(|_| {
            let (id, slots) = metadata.next().unwrap();
            FftRun::empty(dummy, id, slots).expect("nonempty metadata")
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
            run.runs[0].rebind(plan.inverse().expect("validated inverse"), false)?;
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
            self.runs[slot].rebind(
                self.plan
                    .transform(block, in_place)
                    .expect("validated residue"),
                self.product,
            )?;
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
        receipt: Completion<'a, R, Result<(), FftError>>,
    ) -> Result<ExpansionPublished<R>, crate::exec::run::PublishError<'a, R, Result<(), FftError>>>
    {
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
            if self.refill().is_err() {
                self.failed = true;
                task.error = Some(FftError::SizeOverflow);
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

    /// Whether arithmetic validation, cancellation, or unwind failed a task.
    pub fn is_failed(&self) -> bool {
        self.failed
    }

    /// Detached tasks still requiring publication, including on failure.
    pub fn inflight(&self) -> usize {
        self.runs.iter().map(FftRun::inflight).sum()
    }
}
