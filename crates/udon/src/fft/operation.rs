use super::finish::InverseFinish;
use super::stages::StageKernel;
use super::transform::Run;
use super::{
    CoefficientView, ElementOrder, ExecutionOptions, Executor, FftError, InverseScale, PastaField,
    Plan, PowerTable, PrimeModulus, ScratchRequirements, SerialExecutor, TwiddleTable,
    check_domain_size, check_field_count, check_length, min, reverse,
};
use crate::exec::{TaskBudget, for_each_chunk_mut};

/// Mathematical transform direction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    /// Evaluate coefficients on the coset.
    Forward,
    /// Interpolate evaluations to coefficients.
    Inverse,
}

/// Declared nonzero support in natural logical input order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputSupport {
    /// Every domain position may be nonzero.
    Full,
    /// Only positions `0..length` may be nonzero.
    ///
    /// A separate input contains exactly this prefix; in-place execution ignores
    /// and overwrites its tail.
    /// For an inverse these positions are evaluations, not coefficients.
    Prefix(usize),
}

/// Liveness of input storage during prepared execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputPolicy {
    /// Use [`PreparedOperation::execute_into`] with a distinct output.
    Preserve,
    /// Use either an in-place buffer or a separate output.
    Disposable,
}

/// Mathematical and storage semantics fixed before executing an operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformRequest {
    /// Forward evaluation or inverse interpolation.
    pub direction: Direction,
    /// Declared input support; a prefix requires natural input order.
    pub support: InputSupport,
    /// Order of coefficients or evaluations in the input.
    pub input_order: ElementOrder,
    /// Desired order of coefficients or evaluations in the output.
    pub output_order: ElementOrder,
    /// Inverse-size factor; forward requests must use [`InverseScale::Normalized`].
    pub inverse_scale: InverseScale,
    /// Whether input storage must be preserved.
    pub input_policy: InputPolicy,
}

impl TransformRequest {
    /// A full natural-order transform, permitting in-place execution.
    pub const fn new(direction: Direction) -> Self {
        Self {
            direction,
            support: InputSupport::Full,
            input_order: ElementOrder::Natural,
            output_order: ElementOrder::Natural,
            inverse_scale: InverseScale::Normalized,
            input_policy: InputPolicy::Disposable,
        }
    }
    pub(super) const fn input_len(self, size: usize) -> usize {
        match self.support {
            InputSupport::Full => size,
            InputSupport::Prefix(len) => len,
        }
    }
}

/// Execution families selected through [`Strategy`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Backend {
    /// Select blocked execution when the supplied geometry and budget permit;
    /// otherwise select the zero-scratch stage schedule.
    Auto,
    /// Stage execution with zero transform scratch and optional parallel work.
    InPlace,
    /// Local transforms and column jobs using the queried scratch partitions.
    Blocked,
}

/// Initialization of a separate destination during execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Initialization {
    /// Visit the input consecutively and scatter to working order.
    Scatter,
    /// Partition consecutive destination regions and gather from the input.
    Gather,
    /// Gather within bounded tiles, initializing scales independently per tile.
    Blocked,
}

/// Small straight-line radix schedules; larger radices are opt-in candidates.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Codelet {
    /// Individual radix-2 rounds.
    Radix2,
    /// Four-value local schedules.
    Radix4,
    /// Eight-value local schedules.
    Radix8,
}

/// Ceilings for caller-owned temporary storage, tables, and concurrent work.
///
/// Counts exclude input/output storage, fixed stack frames, and executor resources.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResourceBudget {
    /// Maximum initialized temporary fields, including expansion workspaces.
    pub scratch_fields: usize,
    /// Maximum retained table bytes, conservatively summing borrowed slices.
    pub table_bytes: usize,
    /// Nonzero concurrent work-partition budget, including nested transforms.
    pub max_tasks: usize,
}

impl ResourceBudget {
    /// No memory ceiling and the specified total task budget.
    pub const fn for_tasks(max_tasks: usize) -> Self {
        Self {
            scratch_fields: usize::MAX,
            table_bytes: usize::MAX,
            max_tasks,
        }
    }
}

/// Reusable scheduling description for execution with caller-owned storage.
///
/// Configuration uses deterministic resource heuristics without timing-based
/// tuning. Downstream tuning can select and retain a description for later use.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Strategy {
    /// Backend selection or constraint.
    pub backend: Backend,
    /// Separate-output initialization policy.
    pub initialization: Initialization,
    /// Local radix choice for the stage backend. Other backends require radix 2.
    pub codelet: Codelet,
    /// Low-level geometry, with task count clamped to the total budget.
    pub execution: ExecutionOptions,
    /// Resource ceilings checked before execution.
    pub budget: ResourceBudget,
}

impl Strategy {
    /// Zero-scratch serial execution, using radix 2 and gather initialization.
    pub const fn serial() -> Self {
        Self {
            backend: Backend::InPlace,
            initialization: Initialization::Gather,
            codelet: Codelet::Radix2,
            execution: ExecutionOptions::serial(),
            budget: ResourceBudget::for_tasks(1),
        }
    }
    /// Select deterministic geometry under caller resource ceilings.
    pub const fn budgeted(budget: ResourceBudget) -> Self {
        Self {
            backend: Backend::Auto,
            initialization: Initialization::Blocked,
            codelet: Codelet::Radix2,
            execution: ExecutionOptions {
                tile_len: 1024,
                columns_per_task: 32,
                max_tasks: budget.max_tasks,
            },
            budget,
        }
    }
}

impl Default for Strategy {
    fn default() -> Self {
        Self::serial()
    }
}

/// Exact dynamic storage requirements of a prepared operation.
///
/// Fixed-size stack frames and executor resources are excluded. Borrowed table
/// slices are counted separately even if a caller aliases their backing storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OperationRequirements {
    /// Input length for separate-output execution.
    pub input_fields: usize,
    /// Output length (also the full in-place working length).
    pub output_fields: usize,
    /// Total bytes in retained immutable table slices.
    pub retained_table_bytes: usize,
    /// Total number of initialized mutable scratch field elements.
    pub scratch_fields: usize,
    /// Field elements per concurrent blocked column job, or zero without scratch.
    pub per_worker_scratch_fields: usize,
    /// Number of simultaneous blocked column jobs with scratch partitions.
    ///
    /// Zero when no scratch is needed. Multiplying this count by
    /// [`Self::per_worker_scratch_fields`] gives [`Self::scratch_fields`].
    pub scratch_partitions: usize,
    /// Backend selected by the description and ceilings.
    pub backend: Backend,
    /// Total work-partition budget after clamping.
    pub max_tasks: usize,
}

/// Field-independent description driving both const sizing and runtime binding.
///
/// The same description sizes caller storage and configures a reusable operation:
///
/// ```
/// use zakura_udon::{
///     exec::SerialExecutor,
///     field::Fp,
///     fft::{Direction, Domain, OperationDescription, Plan, ResourceBudget,
///         Strategy, TransformRequest},
/// };
///
/// const DESCRIPTION: OperationDescription = OperationDescription {
///     size: 2048,
///     request: TransformRequest::new(Direction::Forward),
///     strategy: Strategy::budgeted(ResourceBudget {
///         scratch_fields: 0,
///         table_bytes: 0,
///         max_tasks: 4,
///     }),
/// };
/// const SCRATCH: usize = match DESCRIPTION.requirements(0) {
///     Ok(required) => required.scratch_fields,
///     Err(_) => panic!("unsupported operation"),
/// };
/// let domain = Domain::for_size(DESCRIPTION.size).unwrap().subgroup();
/// let plan = Plan::without_tables(domain);
/// let operation = plan
///     .configure(DESCRIPTION.request, DESCRIPTION.strategy)
///     .unwrap();
/// let mut values = [Fp::ONE; 2048];
/// let mut scratch = [Fp::ZERO; SCRATCH];
/// operation.execute(&mut values, &SerialExecutor, &mut scratch).unwrap();
/// assert_eq!(values[0], Fp::from_u64(2048));
/// assert!(values[1..].iter().all(|value| *value == Fp::ZERO));
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct OperationDescription {
    /// Domain size.
    pub size: usize,
    /// Mathematical and liveness contract.
    pub request: TransformRequest,
    /// Execution constraints.
    pub strategy: Strategy,
}

impl OperationDescription {
    /// Checks the request and computes storage for a retained-table byte count.
    ///
    /// Count each retained table slice separately, even when slices share
    /// backing storage or the selected backend leaves them unused. This query
    /// uses the supplied count unchanged; [`Plan::configure`] determines which
    /// inverse-normalization tables to retain before supplying its count.
    /// No domain construction or executor is needed.
    ///
    /// Returns [`FftError::InvalidPrefix`] for a prefix longer than `size`, or
    /// [`FftError::InvalidExecution`] for a prefix in bit-reversed input order,
    /// an unscaled forward request, a non-radix-2 codelet outside [`Backend::InPlace`],
    /// or a zero task budget. Exceeded memory ceilings return
    /// [`FftError::ResourceLimit`]. Domain size, execution geometry, and storage
    /// overflow errors follow [`ExecutionOptions::requirements`].
    pub const fn requirements(
        self,
        retained_table_bytes: usize,
    ) -> Result<OperationRequirements, FftError> {
        if let Err(error) = check_domain_size(self.size) {
            return Err(error);
        }
        if let Err(error) = self.strategy.execution.validate() {
            return Err(error);
        }
        if self.strategy.budget.max_tasks == 0 {
            return Err(FftError::InvalidExecution);
        }
        let input_fields = self.request.input_len(self.size);
        if input_fields > self.size {
            return Err(FftError::InvalidPrefix {
                min: 0,
                max: self.size,
                actual: input_fields,
            });
        }
        if matches!(self.request.support, InputSupport::Prefix(_))
            && matches!(self.request.input_order, ElementOrder::BitReversed)
            || matches!(self.request.direction, Direction::Forward)
                && matches!(self.request.inverse_scale, InverseScale::Unscaled)
        {
            return Err(FftError::InvalidExecution);
        }
        if retained_table_bytes > self.strategy.budget.table_bytes {
            return Err(FftError::ResourceLimit);
        }
        let mut options = self.strategy.execution;
        options.max_tasks = min(options.max_tasks, self.strategy.budget.max_tasks);
        let blocked = match options.requirements(self.size) {
            Ok(required) => required.field_elements,
            Err(error) => return Err(error),
        };
        let backend = match self.strategy.backend {
            Backend::Auto => {
                if blocked > 0
                    && blocked <= self.strategy.budget.scratch_fields
                    && matches!(self.strategy.codelet, Codelet::Radix2)
                    && matches!(self.request.output_order, ElementOrder::Natural)
                {
                    Backend::Blocked
                } else {
                    Backend::InPlace
                }
            }
            backend => backend,
        };
        if !matches!(backend, Backend::InPlace) && !matches!(self.strategy.codelet, Codelet::Radix2)
        {
            return Err(FftError::InvalidExecution);
        }
        let scratch_fields = match backend {
            Backend::Blocked => blocked,
            _ => 0,
        };
        if scratch_fields > self.strategy.budget.scratch_fields {
            return Err(FftError::ResourceLimit);
        }
        let geometry = options.geometry(self.size);
        let partitions = if matches!(backend, Backend::Blocked) && blocked > 0 {
            geometry.jobs
        } else {
            0
        };
        Ok(OperationRequirements {
            input_fields,
            output_fields: self.size,
            retained_table_bytes,
            scratch_fields,
            per_worker_scratch_fields: if partitions > 0 {
                blocked / partitions
            } else {
                0
            },
            scratch_partitions: partitions,
            backend,
            max_tasks: options.max_tasks,
        })
    }
}

/// A borrowed plan with validated semantics and fixed resource requirements.
///
/// Ordering, support, and inverse scaling follow the configured
/// [`TransformRequest`] and [`Plan`]'s transform formula. Separate-input methods
/// require exactly [`OperationRequirements::input_fields`] elements; outputs
/// and in-place buffers require exactly [`OperationRequirements::output_fields`].
/// Scratch must contain at least [`OperationRequirements::scratch_fields`]
/// initialized fields. Obtain these counts from [`Self::requirements`]; batches
/// instead use [`Self::batch_requirements`].
///
/// Incorrect lengths return [`FftError::LengthMismatch`]; insufficient scratch
/// returns [`FftError::ScratchTooSmall`]. Execution methods document additional
/// restrictions. The module's [validation and working-storage rules](super)
/// apply, including unchanged buffers on returned errors and partial results
/// on panic. Configuration checks table compatibility without rescanning entries.
#[derive(Clone, Copy)]
pub struct PreparedOperation<'a, M: PrimeModulus> {
    pub(super) plan: Plan<'a, M>,
    description: OperationDescription,
    required: OperationRequirements,
    pub(super) twiddles: Option<TwiddleTable<'a, M>>,
    forward_scales: Option<PowerTable<'a, M>>,
}

impl<M: PrimeModulus> core::fmt::Debug for PreparedOperation<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PreparedOperation")
            .field("plan", &self.plan)
            .field("description", &self.description)
            .field("requirements", &self.required)
            .finish()
    }
}

impl<'a, M: PrimeModulus> Plan<'a, M> {
    /// Prepares repeated execution, checking semantic combinations and ceilings.
    ///
    /// Errors follow [`OperationDescription::requirements`], with
    /// [`FftError::SizeOverflow`] if the plan's retained table byte count overflows.
    /// Tables follow the module's [validation contract](super); configuration
    /// does not rescan their entries.
    ///
    /// Before checking the table-byte ceiling, the returned operation omits
    /// [`Tables::inverse_finish`](super::Tables::inverse_finish) and
    /// [`Tables::inverse_scales`](super::Tables::inverse_scales) for forward
    /// requests, unscaled inverses, and domains with shift one. Other normalized
    /// inverses retain `inverse_finish` only if a permitted execution path ends
    /// in a radix-2 butterfly. For a shift of order three, `inverse_scales` is
    /// retained only alongside `inverse_finish`; other shifts retain it for
    /// coefficient scaling. All remaining table slices count toward the ceiling,
    /// including unused forward or inverse twiddles. The original plan can be
    /// reused with all its tables.
    pub fn configure(
        self,
        request: TransformRequest,
        strategy: Strategy,
    ) -> Result<PreparedOperation<'a, M>, FftError> {
        let description = OperationDescription {
            size: self.domain.size(),
            request,
            strategy,
        };
        let backend = description.requirements(0)?.backend;
        let mut plan = self;
        let radix = if backend == Backend::Blocked {
            2
        } else {
            match strategy.codelet {
                Codelet::Radix2 => 2,
                Codelet::Radix4 => 4,
                Codelet::Radix8 => 8,
            }
        };
        // A radix-4/8 codelet can finish the whole transform. Starting at a
        // later stage bypasses that codelet; skipping every stage leaves only
        // coefficient scaling, which cannot use combined butterfly factors.
        let terminal_butterfly = |first| {
            first <= description.size && (first != 2 || description.size > radix || radix == 2)
        };
        let first = match request.support {
            InputSupport::Full => 2,
            InputSupport::Prefix(len) => 2 * (description.size / len.max(1).next_power_of_two()),
        };
        // Separate-output prefixes skip identity stages. Disposable execution
        // also permits the in-place path, which starts at the first stage.
        let uses_combined = terminal_butterfly(first)
            || request.input_policy == InputPolicy::Disposable && terminal_butterfly(2);
        let normalized = request.direction == Direction::Inverse
            && request.inverse_scale == InverseScale::Normalized;
        if !normalized || !uses_combined {
            plan.tables.inverse_finish = None;
        }
        if !normalized
            || !matches!(
                InverseFinish::select(plan.domain, plan.tables.inverse_finish.is_some()),
                InverseFinish::ScaledInputs
            )
        {
            plan.tables.inverse_finish = None;
            plan.tables.inverse_scales = None;
        }
        let required = description.requirements(plan.tables.retained_bytes()?)?;
        Ok(PreparedOperation {
            plan,
            description,
            required,
            twiddles: None,
            forward_scales: None,
        })
    }
    /// Natural-order serial evaluation without scratch or an executor argument.
    ///
    /// Input ordering, lengths, and errors follow [`Self::forward`].
    pub fn forward_serial(self, values: &mut [PastaField<M>]) -> Result<(), FftError> {
        self.forward(values, ExecutionOptions::serial(), &SerialExecutor, &mut [])
    }
    /// Normalized natural-order serial interpolation without scratch.
    ///
    /// Input ordering, lengths, and errors follow [`Self::inverse`].
    pub fn inverse_serial(self, values: &mut [PastaField<M>]) -> Result<(), FftError> {
        self.inverse(values, ExecutionOptions::serial(), &SerialExecutor, &mut [])
    }
    /// Replaces natural coefficients with bit-reversed evaluations without scratch.
    ///
    /// Output uses [`ElementOrder::BitReversed`]. `values` must have the domain size
    /// or this returns [`FftError::LengthMismatch`]. A zero `max_tasks` returns
    /// [`FftError::InvalidExecution`]; table accounting errors follow
    /// [`Self::configure`].
    pub fn forward_bit_reversed<E: Executor>(
        self,
        values: &mut [PastaField<M>],
        max_tasks: usize,
        executor: &E,
    ) -> Result<(), FftError> {
        let mut request = TransformRequest::new(Direction::Forward);
        request.output_order = ElementOrder::BitReversed;
        self.configure(request, stage_strategy(max_tasks))?
            .execute(values, executor, &mut [])
    }
    /// Inverse with an explicit size factor; untwisting is always included.
    ///
    /// [`InverseScale`] defines the returned coefficient scaling. Ordering,
    /// lengths, and scratch requirements follow [`Self::inverse`]; configuration
    /// errors follow [`Self::configure`].
    pub fn inverse_scaled<E: Executor>(
        self,
        values: &mut [PastaField<M>],
        scale: InverseScale,
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let mut request = TransformRequest::new(Direction::Inverse);
        request.inverse_scale = scale;
        let strategy = Strategy {
            backend: Backend::Blocked,
            execution: options,
            budget: ResourceBudget::for_tasks(options.max_tasks),
            ..Strategy::serial()
        };
        self.configure(request, strategy)?
            .execute(values, executor, scratch)
    }
    /// Interpolates a natural evaluation prefix with a zero evaluation suffix.
    ///
    /// For domain size `n`, the prefix may contain `0..=n` evaluations;
    /// longer prefixes return [`FftError::InvalidPrefix`]. Its length describes
    /// evaluation support, independently of the polynomial's degree. An empty
    /// prefix represents the zero polynomial. Output contains normalized,
    /// natural-order coefficients. Output length and scratch follow
    /// [`Self::inverse`]; configuration errors follow [`Self::configure`].
    pub fn inverse_prefix<E: Executor>(
        self,
        prefix: &[PastaField<M>],
        output: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let mut request = TransformRequest::new(Direction::Inverse);
        request.support = InputSupport::Prefix(prefix.len());
        self.configure(
            request,
            Strategy {
                backend: Backend::Blocked,
                execution: options,
                budget: ResourceBudget::for_tasks(options.max_tasks),
                ..Strategy::serial()
            },
        )?
        .execute_into(prefix, output, executor, scratch)
    }
}

pub(super) const fn stage_strategy(max_tasks: usize) -> Strategy {
    Strategy {
        execution: ExecutionOptions {
            max_tasks,
            ..ExecutionOptions::serial()
        },
        budget: ResourceBudget::for_tasks(max_tasks),
        ..Strategy::serial()
    }
}

// Computed starting powers make scaling independent across regions. Incrementing
// a storage index clears its trailing one bits and sets the next zero bit. In
// bit-reversed order, this changes the coefficient degree by an amount determined
// by that trailing-one count. The ratios array stores the corresponding powers
// of the coset shift, so each region can advance without repeated exponentiation.
struct CoefficientPowers<M: PrimeModulus> {
    shift: PastaField<M>,
    extra: PastaField<M>,
    ratios: [PastaField<M>; 32],
    order: ElementOrder,
    log_size: u32,
}

impl<M: PrimeModulus> CoefficientPowers<M> {
    fn new(domain: super::CosetDomain<M>, order: ElementOrder) -> Self {
        let mut ratios = [PastaField::ONE; 32];
        let log_size = domain.domain().log_size();
        if order == ElementOrder::BitReversed && domain.shift() != PastaField::ONE {
            let mut reciprocal = [PastaField::ONE; 32];
            let mut power = domain.shift();
            let mut inverse = domain.inverse_shift();
            for index in 0..log_size as usize {
                ratios[index] = power;
                reciprocal[index] = inverse;
                power = power.square();
                inverse = inverse.square();
            }
            ratios[..log_size as usize].reverse();
            let mut prefix = PastaField::ONE;
            for index in 0..log_size as usize {
                ratios[index] = ratios[index].mul(&prefix);
                prefix = prefix.mul(&reciprocal[log_size as usize - 1 - index]);
            }
        }
        Self {
            shift: domain.shift(),
            extra: PastaField::ONE,
            ratios,
            order,
            log_size,
        }
    }
    fn at(&self, index: usize) -> PastaField<M> {
        if self.shift == PastaField::ONE {
            return PastaField::ONE;
        }
        let degree = if self.order == ElementOrder::Natural {
            index
        } else {
            reverse(index, self.log_size)
        };
        self.shift.pow_u64(degree as u64)
    }
    fn next(&self, index: usize, power: PastaField<M>) -> PastaField<M> {
        if self.shift == PastaField::ONE {
            return power;
        }
        power.mul(if self.order == ElementOrder::Natural {
            &self.shift
        } else {
            &self.ratios[index.trailing_ones() as usize]
        })
    }
}

// Dispatch once per initialization: a supplied table must eliminate power
// generation, and an inverse must not generate or apply forward coset scales.
trait CoefficientScaling<M: PrimeModulus>: Sync {
    type Seed;
    fn seed(&self, index: usize) -> Self::Seed;
    fn scale(&self, value: PastaField<M>, degree: usize, seed: &Self::Seed) -> PastaField<M>;
    fn advance(&self, index: usize, seed: &mut Self::Seed);
}

impl<M: PrimeModulus> CoefficientScaling<M> for CoefficientPowers<M> {
    type Seed = PastaField<M>;
    fn seed(&self, index: usize) -> Self::Seed {
        if self.extra == PastaField::ONE {
            self.at(index)
        } else {
            self.at(index).mul(&self.extra)
        }
    }
    fn scale(&self, value: PastaField<M>, _: usize, seed: &Self::Seed) -> PastaField<M> {
        value.mul(seed)
    }
    fn advance(&self, index: usize, seed: &mut Self::Seed) {
        *seed = self.next(index, *seed);
    }
}

impl<M: PrimeModulus> CoefficientScaling<M> for &[PastaField<M>] {
    type Seed = ();
    fn seed(&self, _: usize) {}
    fn scale(&self, value: PastaField<M>, degree: usize, _: &()) -> PastaField<M> {
        value.mul(&self[degree])
    }
    fn advance(&self, _: usize, _: &mut ()) {}
}

struct IdentityScaling;

impl<M: PrimeModulus> CoefficientScaling<M> for PastaField<M> {
    type Seed = ();
    fn seed(&self, _: usize) {}
    fn scale(&self, value: PastaField<M>, _: usize, _: &()) -> PastaField<M> {
        value.mul(self)
    }
    fn advance(&self, _: usize, _: &mut ()) {}
}

struct ScaledTable<'a, M: PrimeModulus> {
    values: &'a [PastaField<M>],
    extra: PastaField<M>,
}

impl<M: PrimeModulus> CoefficientScaling<M> for ScaledTable<'_, M> {
    type Seed = ();
    fn seed(&self, _: usize) {}
    fn scale(&self, value: PastaField<M>, degree: usize, _: &()) -> PastaField<M> {
        value.mul(&self.values[degree]).mul(&self.extra)
    }
    fn advance(&self, _: usize, _: &mut ()) {}
}

impl<M: PrimeModulus> CoefficientScaling<M> for IdentityScaling {
    type Seed = ();
    fn seed(&self, _: usize) {}
    fn scale(&self, value: PastaField<M>, _: usize, _: &()) -> PastaField<M> {
        value
    }
    fn advance(&self, _: usize, _: &mut ()) {}
}

impl<'a, M: PrimeModulus> PreparedOperation<'a, M> {
    /// Fixed resource requirements, including all currently borrowed tables.
    pub const fn requirements(self) -> OperationRequirements {
        self.required
    }
    /// Description suitable for downstream tuning and const sizing.
    pub const fn description(self) -> OperationDescription {
        self.description
    }

    /// Selects a twiddle provider for stage execution.
    ///
    /// [`TwiddleTable`] defines how its size and direction serve each transform.
    /// Attachment does not rescan the table's entries.
    /// Explicit [`Backend::Blocked`] returns [`FftError::InvalidExecution`];
    /// [`Backend::Auto`] switches to [`Backend::InPlace`] if it selected blocked
    /// execution. Requirements are recomputed with the additional retained bytes,
    /// following [`OperationDescription::requirements`] and rejecting byte-count
    /// overflow with [`FftError::SizeOverflow`].
    pub fn with_twiddles(mut self, table: TwiddleTable<'a, M>) -> Result<Self, FftError> {
        if self.required.backend == Backend::Blocked {
            if self.description.strategy.backend != Backend::Auto {
                return Err(FftError::InvalidExecution);
            }
            self.description.strategy.backend = Backend::InPlace;
        }
        self.twiddles = Some(table);
        self.refresh()?;
        Ok(self)
    }
    /// Borrows forward scales for every natural coefficient index.
    ///
    /// Requires a forward request, `first = 1`, and `step` equal to the plan's
    /// coset shift, otherwise returning [`FftError::InvalidTables`]. The table
    /// must have the full domain length or this returns
    /// [`FftError::LengthMismatch`]. Attachment does not rescan entries; their
    /// validity follows [`PowerTable`]'s preparation and binding contract.
    /// Additional retained bytes are checked as in [`Self::with_twiddles`].
    pub fn with_forward_scales(mut self, table: PowerTable<'a, M>) -> Result<Self, FftError> {
        if self.description.request.direction != Direction::Forward
            || table.first() != PastaField::ONE
            || table.step() != self.plan.domain.shift()
        {
            return Err(FftError::InvalidTables);
        }
        check_length(
            "forward_scales",
            self.plan.domain.size(),
            table.as_slice().len(),
        )?;
        self.forward_scales = Some(table);
        self.refresh()?;
        Ok(self)
    }
    fn refresh(&mut self) -> Result<(), FftError> {
        let mut bytes = self.plan.tables.retained_bytes()?;
        for len in [
            self.twiddles.map_or(0, |t| t.as_slice().len()),
            self.forward_scales.map_or(0, |t| t.as_slice().len()),
        ] {
            bytes = bytes
                .checked_add(len.checked_mul(32).ok_or(FftError::SizeOverflow)?)
                .ok_or(FftError::SizeOverflow)?;
        }
        self.required = self.description.requirements(bytes)?;
        Ok(())
    }
    fn check(self, output: usize, scratch: usize) -> Result<(), FftError> {
        check_length("output", self.required.output_fields, output)?;
        ScratchRequirements {
            field_elements: self.required.scratch_fields,
        }
        .check(scratch)
    }
    fn working_order(self) -> ElementOrder {
        if self.description.request.direction == Direction::Forward
            && self.description.request.output_order == ElementOrder::BitReversed
            && self.required.backend == Backend::InPlace
        {
            ElementOrder::Natural
        } else {
            ElementOrder::BitReversed
        }
    }
    /// Executes in place, consuming the input.
    ///
    /// [`InputPolicy::Preserve`] returns [`FftError::InvalidExecution`]. Prefix
    /// support ignores and overwrites the full buffer's unused tail. Lengths,
    /// scratch, and other error behavior follow [`PreparedOperation`].
    pub fn execute<E: Executor>(
        self,
        values: &mut [PastaField<M>],
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        self.check(values.len(), scratch.len())?;
        if self.description.request.input_policy == InputPolicy::Preserve {
            return Err(FftError::InvalidExecution);
        }
        let request = self.description.request;
        if let InputSupport::Prefix(len) = request.support {
            values[len..].fill(PastaField::ZERO);
        }
        if request.direction == Direction::Forward {
            self.scale_input(values, request.input_order, executor);
        }
        if request.input_order != self.working_order() {
            self.plan.permute(values);
        }
        self.run(values, 2, executor, scratch, None);
        Ok(())
    }

    /// Writes a transform to `output`, preserving the separate input.
    ///
    /// Both input policies are accepted. Full or prefix input lengths, scratch,
    /// and errors follow [`PreparedOperation`].
    /// Forward input contains normalized coefficients. Use
    /// [`Self::execute_coefficients`] to preserve a [`CoefficientView`]'s scale.
    pub fn execute_into<E: Executor>(
        self,
        input: &[PastaField<M>],
        output: &mut [PastaField<M>],
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        check_length("input", self.required.input_fields, input.len())?;
        self.check(output.len(), scratch.len())?;
        let first = self.initialize(input, output, executor, PastaField::ONE);
        self.run(output, first, executor, scratch, None);
        Ok(())
    }

    /// Evaluates a coefficient view, folding its scale into initialization.
    ///
    /// Requires a forward request with natural input order, otherwise returning
    /// [`FftError::InvalidExecution`]. Input must have exactly the configured
    /// full or prefix length; a mismatch returns [`FftError::LengthMismatch`].
    /// The view can come from a smaller expansion when the request declares
    /// that prefix length. Output order, scratch, input preservation, and other
    /// errors follow [`Self::execute_into`].
    pub fn execute_coefficients<E: Executor>(
        self,
        input: CoefficientView<'_, M>,
        output: &mut [PastaField<M>],
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        self.check_coefficients(input, output.len(), scratch.len())?;
        let first = self.initialize(
            input.as_slice(),
            output,
            executor,
            input.normalization_factor(),
        );
        self.run(output, first, executor, scratch, None);
        Ok(())
    }

    fn check_coefficients(
        self,
        input: CoefficientView<'_, M>,
        output_len: usize,
        scratch_len: usize,
    ) -> Result<(), FftError> {
        let request = self.description.request;
        if request.direction != Direction::Forward || request.input_order != ElementOrder::Natural {
            return Err(FftError::InvalidExecution);
        }
        check_length("input", self.required.input_fields, input.as_slice().len())?;
        self.check(output_len, scratch_len)
    }

    /// Writes the pointwise product of a forward transform and `factor`.
    ///
    /// Returns [`FftError::InvalidLayout`] unless the request is forward and the
    /// factor matches the plan's coset domain and requested output order.
    /// Input preservation, lengths, scratch, and other errors follow
    /// [`Self::execute_into`].
    pub fn execute_product_into<E: Executor>(
        self,
        input: &[PastaField<M>],
        factor: super::EvaluationView<'_, M>,
        output: &mut [PastaField<M>],
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let layout = match self.description.request.output_order {
            ElementOrder::Natural => super::EvaluationLayout::Natural,
            ElementOrder::BitReversed => super::EvaluationLayout::BitReversed,
        };
        if self.description.request.direction != Direction::Forward
            || !factor.domain().same_domain(self.plan.domain)
            || factor.layout() != layout
        {
            return Err(FftError::InvalidLayout);
        }
        check_length("input", self.required.input_fields, input.len())?;
        self.check(output.len(), scratch.len())?;
        let first = self.initialize(input, output, executor, PastaField::ONE);
        self.run(output, first, executor, scratch, Some(factor.as_slice()));
        Ok(())
    }

    /// Evaluates a coefficient view and multiplies the evaluations by `factor`.
    ///
    /// Coefficient scaling, request restrictions, lengths, scratch, and input
    /// preservation follow [`Self::execute_coefficients`]. Returns
    /// [`FftError::InvalidLayout`] if `factor` differs from the plan's coset
    /// domain or requested output order.
    pub fn execute_coefficient_product<E: Executor>(
        self,
        input: CoefficientView<'_, M>,
        factor: super::EvaluationView<'_, M>,
        output: &mut [PastaField<M>],
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        self.check_coefficients(input, output.len(), scratch.len())?;
        let layout = match self.description.request.output_order {
            ElementOrder::Natural => super::EvaluationLayout::Natural,
            ElementOrder::BitReversed => super::EvaluationLayout::BitReversed,
        };
        if !factor.domain().same_domain(self.plan.domain) || factor.layout() != layout {
            return Err(FftError::InvalidLayout);
        }
        let first = self.initialize(
            input.as_slice(),
            output,
            executor,
            input.normalization_factor(),
        );
        self.run(output, first, executor, scratch, Some(factor.as_slice()));
        Ok(())
    }

    fn scale_input<E: Executor>(
        self,
        values: &mut [PastaField<M>],
        order: ElementOrder,
        executor: &E,
    ) {
        if self.plan.domain.shift() == PastaField::ONE && self.forward_scales.is_none() {
            return;
        }
        if let Some(table) = self.forward_scales {
            self.scale_input_with(values, order, executor, table.as_slice());
        } else {
            self.scale_input_with(
                values,
                order,
                executor,
                CoefficientPowers::new(self.plan.domain, order),
            );
        }
    }

    fn scale_input_with<E: Executor, S: CoefficientScaling<M>>(
        self,
        values: &mut [PastaField<M>],
        order: ElementOrder,
        executor: &E,
        scales: S,
    ) {
        let chunk = values.len().div_ceil(self.required.max_tasks);
        for_each_chunk_mut(
            values,
            chunk,
            TaskBudget::new(self.required.max_tasks).unwrap(),
            executor,
            |job, values, _| {
                let start = job * chunk;
                let mut seed = scales.seed(start);
                let count = values.len();
                for (offset, value) in values.iter_mut().enumerate() {
                    let index = start + offset;
                    let degree = if order == ElementOrder::BitReversed {
                        reverse(index, self.plan.domain.domain().log_size())
                    } else {
                        index
                    };
                    *value = scales.scale(*value, degree, &seed);
                    if offset + 1 < count {
                        scales.advance(index, &mut seed);
                    }
                }
            },
        );
    }

    fn initialize<E: Executor>(
        self,
        input: &[PastaField<M>],
        output: &mut [PastaField<M>],
        executor: &E,
        extra: PastaField<M>,
    ) -> usize {
        let request = self.description.request;
        // Sparse DIT initialization broadcasts the nonzero support through
        // identity-only stages, equally for forward and inverse roots.
        if matches!(request.support, InputSupport::Prefix(_))
            && self.working_order() == ElementOrder::BitReversed
        {
            return self.plan.fill_prefix(
                input,
                output,
                if request.direction == Direction::Forward {
                    self.plan.domain.shift()
                } else {
                    PastaField::ONE
                },
                self.forward_scales.map(|t| t.as_slice()),
                extra,
            );
        }
        if request.direction == Direction::Inverse {
            self.initialize_with(input, output, executor, IdentityScaling);
        } else if let Some(table) = self.forward_scales {
            if extra == PastaField::ONE {
                self.initialize_with(input, output, executor, table.as_slice());
            } else {
                self.initialize_with(
                    input,
                    output,
                    executor,
                    ScaledTable {
                        values: table.as_slice(),
                        extra,
                    },
                );
            }
        } else if self.plan.domain.shift() == PastaField::ONE {
            if extra == PastaField::ONE {
                self.initialize_with(input, output, executor, IdentityScaling);
            } else {
                self.initialize_with(input, output, executor, extra);
            }
        } else {
            let order = if self.description.strategy.initialization == Initialization::Scatter {
                request.input_order
            } else {
                self.working_order()
            };
            let mut powers = CoefficientPowers::new(self.plan.domain, order);
            powers.extra = extra;
            self.initialize_with(input, output, executor, powers);
        }
        2
    }

    fn initialize_with<E: Executor, S: CoefficientScaling<M>>(
        self,
        input: &[PastaField<M>],
        output: &mut [PastaField<M>],
        executor: &E,
        scales: S,
    ) {
        let request = self.description.request;
        let size = output.len();
        let log_size = size.ilog2();
        if self.description.strategy.initialization == Initialization::Scatter {
            output.fill(PastaField::ZERO);
            let mut seed = scales.seed(0);
            for (index, value) in input.iter().enumerate() {
                let degree = if request.input_order == ElementOrder::Natural {
                    index
                } else {
                    reverse(index, log_size)
                };
                let destination = if self.working_order() == ElementOrder::Natural {
                    degree
                } else {
                    reverse(degree, log_size)
                };
                output[destination] = scales.scale(*value, degree, &seed);
                if index + 1 < input.len() {
                    scales.advance(index, &mut seed);
                }
            }
        } else {
            let chunk = if self.description.strategy.initialization == Initialization::Blocked {
                64
            } else {
                size.div_ceil(self.required.max_tasks)
            };
            for_each_chunk_mut(
                output,
                chunk,
                TaskBudget::new(self.required.max_tasks).unwrap(),
                executor,
                |job, output, _| {
                    let start = job * chunk;
                    let mut seed = scales.seed(start);
                    let count = output.len();
                    for (offset, output) in output.iter_mut().enumerate() {
                        let index = start + offset;
                        let degree = if self.working_order() == ElementOrder::Natural {
                            index
                        } else {
                            reverse(index, log_size)
                        };
                        let source = if request.input_order == ElementOrder::Natural {
                            degree
                        } else {
                            reverse(degree, log_size)
                        };
                        let value = input.get(source).copied().unwrap_or(PastaField::ZERO);
                        *output = scales.scale(value, degree, &seed);
                        if offset + 1 < count {
                            scales.advance(index, &mut seed);
                        }
                    }
                },
            );
        }
    }

    fn run<E: Executor>(
        self,
        values: &mut [PastaField<M>],
        first: usize,
        executor: &E,
        scratch: &mut [PastaField<M>],
        factor: Option<&[PastaField<M>]>,
    ) {
        let request = self.description.request;
        let mut options = self.description.strategy.execution;
        options.max_tasks = self.required.max_tasks;
        let scratch = &mut scratch[..self.required.scratch_fields];
        let kernel = StageKernel {
            plan: self.plan,
            inverse: request.direction == Direction::Inverse,
            dif: self.working_order() == ElementOrder::Natural,
            scale: request.inverse_scale,
            codelet: self.description.strategy.codelet,
            twiddles: self.twiddles,
            output_order: request.output_order,
            factor,
        };
        match self.required.backend {
            Backend::InPlace => kernel.run(values, first, self.required.max_tasks, executor),
            Backend::Blocked => {
                let fused_factor = factor.filter(|_| request.output_order == ElementOrder::Natural);
                let mut run = if request.direction == Direction::Forward {
                    fused_factor.map_or_else(
                        || Run::forward(first),
                        |factor| Run::forward_product(first, factor),
                    )
                } else if request.inverse_scale == InverseScale::Normalized {
                    Run::inverse(&[])
                } else {
                    Run::inverse_unscaled()
                };
                run.set_first(first);
                self.plan.run(values, options, executor, scratch, run);
                if request.direction == Direction::Inverse
                    && request.inverse_scale == InverseScale::Unscaled
                {
                    kernel.untwist(values, self.required.max_tasks, executor);
                }
                if request.output_order == ElementOrder::BitReversed {
                    self.plan.permute(values);
                }
                if let Some(factor) = factor.filter(|_| fused_factor.is_none()) {
                    for (value, factor) in values.iter_mut().zip(factor) {
                        *value = value.mul(factor);
                    }
                }
            }
            Backend::Auto => unreachable!(),
        }
    }

    /// Exact scratch for a contiguous batch of full in-place transforms.
    ///
    /// Work partitions are divided across polynomials and within each transform.
    /// Requires [`InputSupport::Full`] and [`InputPolicy::Disposable`], otherwise
    /// returning [`FftError::InvalidExecution`], including for an empty batch.
    /// Exceeding the scratch budget returns [`FftError::ResourceLimit`]; storage
    /// overflow returns [`FftError::SizeOverflow`]. A zero `count` needs no scratch.
    pub const fn batch_requirements(self, count: usize) -> Result<ScratchRequirements, FftError> {
        let (operation, jobs) = match self.batch_configuration(count) {
            Ok(o) => o,
            Err(e) => return Err(e),
        };
        let count = match operation.required.scratch_fields.checked_mul(jobs) {
            Some(n) => n,
            None => return Err(FftError::SizeOverflow),
        };
        if count > self.description.strategy.budget.scratch_fields {
            return Err(FftError::ResourceLimit);
        }
        match check_field_count(count) {
            Ok(field_elements) => Ok(ScratchRequirements { field_elements }),
            Err(e) => Err(e),
        }
    }

    const fn batch_configuration(mut self, count: usize) -> Result<(Self, usize), FftError> {
        if !matches!(self.description.request.support, InputSupport::Full)
            || matches!(self.description.request.input_policy, InputPolicy::Preserve)
        {
            return Err(FftError::InvalidExecution);
        }
        let budget = TaskBudget::new(self.required.max_tasks).unwrap();
        let (jobs, inner) = match budget.partition(count) {
            Some(partition) => partition,
            None => return Ok((self, 0)),
        };
        self.description.strategy.backend = self.required.backend;
        self.description.strategy.execution.max_tasks = inner.get();
        self.required = match self
            .description
            .requirements(self.required.retained_table_bytes)
        {
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        Ok((self, jobs))
    }

    /// Transforms consecutive full-domain polynomials in place.
    ///
    /// `values` stores each polynomial's entire input before the next one's.
    /// Its length must be a multiple of the domain size, otherwise this returns
    /// [`FftError::InvalidLayout`]. Empty batches are accepted. Request constraints
    /// and scratch requirements follow [`Self::batch_requirements`]; insufficient
    /// scratch returns [`FftError::ScratchTooSmall`]. The validation and panic
    /// contract of [`PreparedOperation`] applies.
    pub fn execute_batch<E: Executor>(
        self,
        values: &mut [PastaField<M>],
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let size = self.required.output_fields;
        if !values.len().is_multiple_of(size) {
            return Err(FftError::InvalidLayout);
        }
        let count = values.len() / size;
        let required = self.batch_requirements(count)?;
        required.check(scratch.len())?;
        if count == 0 {
            return Ok(());
        }
        let (operation, jobs) = self.batch_configuration(count)?;
        fn visit<M: PrimeModulus, E: Executor>(
            operation: PreparedOperation<'_, M>,
            values: &mut [PastaField<M>],
            jobs: usize,
            executor: &E,
            scratch: &mut [PastaField<M>],
        ) {
            let size = operation.required.output_fields;
            if jobs <= 1 {
                for values in values.chunks_exact_mut(size) {
                    operation.execute(values, executor, scratch).unwrap();
                }
            } else {
                let left_jobs = jobs / 2;
                let (left, right) = values.split_at_mut((values.len() / size / 2) * size);
                let (left_scratch, right_scratch) =
                    scratch.split_at_mut(operation.required.scratch_fields * left_jobs);
                executor.join(
                    || visit(operation, left, left_jobs, executor, left_scratch),
                    || visit(operation, right, jobs - left_jobs, executor, right_scratch),
                );
            }
        }
        visit(
            operation,
            values,
            jobs,
            executor,
            &mut scratch[..required.field_elements],
        );
        Ok(())
    }
}
