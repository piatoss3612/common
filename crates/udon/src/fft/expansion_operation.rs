use super::expansion::ResidueJobs;
use super::transform::Run;
use super::{
    Codelet, CosetDomain, EvaluationLayout, EvaluationView, ExecutionOptions, Executor, Expansion,
    ExpansionOptions, FftError, InputOrder, InverseScale, PastaField, PrimeModulus, ResourceBudget,
    ScratchRequirements, check_domain_size, check_field_count, check_len, check_prefix, min,
};

/// Persistent order of a complete expansion result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpansionOrder {
    /// Naturally numbered residues and rows, as defined by [`super::ResidueLayout`].
    Residues,
    /// Extended rows in [`InputOrder::BitReversed`] order.
    ///
    /// Both residue blocks and rows within each block are bit-reversed. A full
    /// inverse accepting bit-reversed input can consume this layout directly.
    BitReversed,
}

/// Input liveness and coefficient storage for a prepared expansion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpansionStorage {
    /// Preserve a natural coefficient prefix, treating its missing suffix as zero.
    Coefficients,
    /// Preserve natural base evaluations, using output storage for coefficients.
    ReuseOutput,
    /// Preserve natural base evaluations using a separate coefficient workspace.
    CoefficientWorkspace,
    /// Consume natural base evaluations in their own buffer as coefficient storage.
    DisposableInput,
}

/// Geometry and total resource ceilings for expansion.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpansionStrategy {
    /// Geometry within base transforms; its task count is capped by the budget.
    pub transform: ExecutionOptions,
    /// Ceilings for the whole operation, across and within residues.
    pub budget: ResourceBudget,
}

impl ExpansionStrategy {
    /// Serial expansion with zero transform scratch.
    pub const fn serial() -> Self {
        Self {
            transform: ExecutionOptions::serial(),
            budget: ResourceBudget::for_tasks(1),
        }
    }
}

/// Const description shared by expansion sizing and runtime preparation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpansionDescription {
    /// Base subgroup size.
    pub base_size: usize,
    /// Extended coset size, at least the base size.
    pub extended_size: usize,
    /// Persistent output order.
    pub order: ExpansionOrder,
    /// Source and storage liveness.
    pub storage: ExpansionStorage,
    /// Geometry and whole-operation budgets.
    pub strategy: ExpansionStrategy,
}

/// Storage used by a prepared expansion, excluding its input and output.
///
/// Fixed stack frames and executor resources are also excluded.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpansionRequirements {
    /// Retained bytes in the base tables and optional residue scales.
    pub retained_table_bytes: usize,
    /// Total initialized scratch fields, reused between inverse and residues.
    pub scratch_fields: usize,
    /// Scratch used by the base inverse, overlapping the residue partitions.
    pub inverse_scratch_fields: usize,
    /// Fields per concurrent residue's scratch partition.
    pub per_worker_scratch_fields: usize,
    /// Concurrent residue partitions; zero-sized partitions need no storage.
    pub scratch_partitions: usize,
    /// Separate coefficient fields, required only by the workspace policy.
    pub coefficient_fields: usize,
    /// Total task ceiling, including nested transform work.
    pub max_tasks: usize,
}

impl ExpansionDescription {
    const fn options(self) -> Result<(ExpansionOptions, ExecutionOptions), FftError> {
        if let Err(e) = check_domain_size(self.base_size) {
            return Err(e);
        }
        if let Err(e) = check_domain_size(self.extended_size) {
            return Err(e);
        }
        if self.base_size > self.extended_size {
            return Err(FftError::InvalidLayout);
        }
        let budget = self.strategy.budget.max_tasks;
        if budget == 0 {
            return Err(FftError::InvalidExecution);
        }
        let mut residues = self.extended_size / self.base_size;
        if matches!(self.storage, ExpansionStorage::ReuseOutput) && residues > 1 {
            residues -= 1;
        }
        let jobs = min(residues, budget);
        let mut inverse = self.strategy.transform;
        inverse.max_tasks = min(inverse.max_tasks, budget);
        let mut transform = inverse;
        transform.max_tasks = min(transform.max_tasks, budget / jobs);
        Ok((
            ExpansionOptions {
                max_residue_tasks: jobs,
                transform,
            },
            inverse,
        ))
    }

    /// Exact resources for these sizes, including optional coefficient storage.
    ///
    /// The scratch ceiling covers `scratch_fields + coefficient_fields`; these
    /// buffers are reported separately because they have different lifetimes.
    /// Count all borrowed tables in `retained_table_bytes`, even when their
    /// backing storage overlaps.
    ///
    /// Returns [`FftError::InvalidLayout`] if the base size exceeds the extended
    /// size, [`FftError::InvalidExecution`] for a zero task budget, or
    /// [`FftError::ResourceLimit`] if a memory ceiling is exceeded. Size, geometry,
    /// and storage overflow errors follow [`ExecutionOptions::requirements`].
    pub const fn requirements(
        self,
        retained_table_bytes: usize,
    ) -> Result<ExpansionRequirements, FftError> {
        let (options, inverse) = match self.options() {
            Ok(o) => o,
            Err(e) => return Err(e),
        };
        let transform_fields = match options.transform.requirements(self.base_size) {
            Ok(r) => r.field_elements,
            Err(e) => return Err(e),
        };
        let inverse_fields = match inverse.requirements(self.base_size) {
            Ok(r) => r.field_elements,
            Err(e) => return Err(e),
        };
        let per_worker = if matches!(self.order, ExpansionOrder::BitReversed) {
            0
        } else {
            transform_fields
        };
        let inverse_scratch_fields = if matches!(self.storage, ExpansionStorage::Coefficients) {
            0
        } else {
            inverse_fields
        };
        let residues = match per_worker.checked_mul(options.max_residue_tasks) {
            Some(n) => n,
            None => return Err(FftError::SizeOverflow),
        };
        let scratch_fields = if residues > inverse_scratch_fields {
            residues
        } else {
            inverse_scratch_fields
        };
        if let Err(e) = check_field_count(scratch_fields) {
            return Err(e);
        }
        let coefficient_fields = if matches!(self.storage, ExpansionStorage::CoefficientWorkspace) {
            self.base_size
        } else {
            0
        };
        let temporary_fields = match scratch_fields.checked_add(coefficient_fields) {
            Some(n) => n,
            None => return Err(FftError::SizeOverflow),
        };
        if temporary_fields > self.strategy.budget.scratch_fields
            || retained_table_bytes > self.strategy.budget.table_bytes
        {
            return Err(FftError::ResourceLimit);
        }
        Ok(ExpansionRequirements {
            retained_table_bytes,
            scratch_fields,
            inverse_scratch_fields,
            per_worker_scratch_fields: per_worker,
            scratch_partitions: options.max_residue_tasks,
            coefficient_fields,
            max_tasks: self.strategy.budget.max_tasks,
        })
    }
}

/// Expansion with fixed ordering, liveness, and total resource requirements.
///
/// For base size `n` and extended size `N` from [`Self::description`], coefficient
/// inputs are natural-order prefixes of length `0..=n`; omitted coefficients are
/// zero. Evaluation inputs contain exactly `n` natural-order subgroup evaluations.
/// Every output contains exactly `N` fields in [`Self::layout`] order. Scratch
/// must contain at least [`ExpansionRequirements::scratch_fields`] initialized
/// fields, as reported by [`Self::requirements`].
///
/// A coefficient prefix longer than `n` returns [`FftError::InvalidPrefix`];
/// other length mismatches return [`FftError::LengthMismatch`]. Insufficient
/// scratch returns [`FftError::ScratchTooSmall`], and using an execution method
/// with an unsupported [`ExpansionStorage`] policy returns
/// [`FftError::InvalidExecution`]. The module's [validation and working-storage
/// rules](super) apply, including unchanged buffers on returned errors and
/// partial results on panic. Configuration does not validate table contents.
#[derive(Clone, Copy)]
pub struct PreparedExpansion<'a, M: PrimeModulus> {
    expansion: Expansion<'a, M>,
    description: ExpansionDescription,
    required: ExpansionRequirements,
    options: ExpansionOptions,
    inverse: ExecutionOptions,
}

impl<M: PrimeModulus> core::fmt::Debug for PreparedExpansion<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PreparedExpansion")
            .field("expansion", &self.expansion)
            .field("description", &self.description)
            .field("requirements", &self.required)
            .finish()
    }
}

impl<'a, M: PrimeModulus> Expansion<'a, M> {
    /// Fixes output order, input liveness, and budgets for repeated expansion.
    ///
    /// Errors follow [`ExpansionDescription::requirements`], with
    /// [`FftError::SizeOverflow`] if the retained table byte count overflows.
    /// Table contents retain [`Expansion`]'s explicit validation contract.
    pub fn configure(
        self,
        order: ExpansionOrder,
        storage: ExpansionStorage,
        strategy: ExpansionStrategy,
    ) -> Result<PreparedExpansion<'a, M>, FftError> {
        let description = ExpansionDescription {
            base_size: self.base.domain().size(),
            extended_size: self.extended.size(),
            order,
            storage,
            strategy,
        };
        let retained = self
            .base
            .tables
            .retained_bytes()?
            .checked_add(self.scales.map_or(0, |scales| scales.len() * 32))
            .ok_or(FftError::SizeOverflow)?;
        let required = description.requirements(retained)?;
        let (options, inverse) = description.options()?;
        Ok(PreparedExpansion {
            expansion: self,
            description,
            required,
            options,
            inverse,
        })
    }

    /// Selects one naturally numbered residue for a reusable base-sized output.
    ///
    /// `order` describes positions inside that residue only. Returns
    /// [`FftError::InvalidLayout`] if `residue` is outside [`Self::layout`]'s
    /// residue count.
    pub fn residue(self, residue: usize, order: InputOrder) -> Result<Residue<'a, M>, FftError> {
        if residue >= self.layout.residues() {
            return Err(FftError::InvalidLayout);
        }
        Ok(Residue {
            expansion: self,
            residue,
            order,
        })
    }
}

impl<M: PrimeModulus> PreparedExpansion<'_, M> {
    /// Description used by the const sizing query.
    pub const fn description(self) -> ExpansionDescription {
        self.description
    }
    /// Fixed storage and total task requirements.
    pub const fn requirements(self) -> ExpansionRequirements {
        self.required
    }
    /// Complete output layout selected by [`ExpansionOrder`].
    pub fn layout(self) -> EvaluationLayout {
        match self.description.order {
            ExpansionOrder::Residues => EvaluationLayout::Residues(self.expansion.layout),
            ExpansionOrder::BitReversed => EvaluationLayout::BitReversed,
        }
    }
    /// Binds output storage to this operation's mathematical domain and order.
    ///
    /// Returns [`FftError::LengthMismatch`] unless `values` has the extended
    /// domain size. Contents are not checked; see [`EvaluationView::bind`].
    pub fn view(self, values: &[PastaField<M>]) -> Result<EvaluationView<'_, M>, FftError> {
        EvaluationView::bind(values, self.expansion.extended, self.layout())
    }
    fn check(self, input: usize, output: usize, scratch: usize) -> Result<(), FftError> {
        if self.description.storage == ExpansionStorage::Coefficients {
            check_prefix(input, 0, self.description.base_size)?;
        } else {
            check_len("input", input, self.description.base_size)?;
        }
        check_len("output", output, self.description.extended_size)?;
        ScratchRequirements {
            field_elements: self.required.scratch_fields,
        }
        .check(scratch)
    }

    /// Preserves coefficients or base evaluations while writing the expansion.
    ///
    /// Accepts [`ExpansionStorage::Coefficients`] or [`ExpansionStorage::ReuseOutput`].
    /// Lengths, scratch, and errors follow [`PreparedExpansion`].
    pub fn execute_into<E: Executor>(
        self,
        input: &[PastaField<M>],
        output: &mut [PastaField<M>],
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        self.check(input.len(), output.len(), scratch.len())?;
        match self.description.storage {
            ExpansionStorage::Coefficients => {
                self.residues(input, output, 0, true, None, executor, scratch)
            }
            ExpansionStorage::ReuseOutput => {
                let (first, rest) = output.split_at_mut(self.description.base_size);
                self.expansion.base.scatter(input, first);
                self.inverse_coefficients(first, executor, scratch);
                self.residues(first, rest, 1, false, None, executor, scratch);
                // All readers of this coefficient buffer have completed.
                let (_, extra) = self.expansion.inverse_policy();
                if let Some(scales) = self.expansion.scales {
                    for (value, scale) in first.iter_mut().zip(scales) {
                        *value = value.mul(scale);
                    }
                } else {
                    let mut power = extra;
                    for value in first.iter_mut() {
                        *value = value.mul(&power);
                        power = power.mul(&self.expansion.extended.shift());
                    }
                }
                if self.description.order == ExpansionOrder::BitReversed {
                    super::stages::StageKernel {
                        plan: self.expansion.base,
                        inverse: false,
                        dif: true,
                        scale: InverseScale::Normalized,
                        codelet: Codelet::Radix2,
                        twiddles: None,
                        output_order: InputOrder::BitReversed,
                        factor: None,
                    }
                    .run(first, 2, self.options.transform.max_tasks, executor);
                } else {
                    self.expansion.base.permute(first);
                    self.expansion.base.run(
                        first,
                        self.options.transform,
                        executor,
                        &mut scratch[..self.required.per_worker_scratch_fields],
                        Run::forward(2),
                    );
                }
            }
            _ => return Err(FftError::InvalidExecution),
        }
        Ok(())
    }

    /// Expands preserved base evaluations using a separate coefficient workspace.
    ///
    /// Requires [`ExpansionStorage::CoefficientWorkspace`] and exactly
    /// [`ExpansionRequirements::coefficient_fields`] entries in `coefficients`.
    /// Other lengths, scratch, and errors follow [`PreparedExpansion`].
    ///
    /// On success the workspace uses natural coefficient order. With
    /// [`super::ExpansionScaleNormalization::Coefficients`] tables it contains
    /// the polynomial's coefficients. With pre-normalized tables or no table,
    /// each coefficient is multiplied by the base size; the output evaluations
    /// include the omitted inverse-size factor.
    pub fn execute_with_workspace<E: Executor>(
        self,
        input: &[PastaField<M>],
        output: &mut [PastaField<M>],
        coefficients: &mut [PastaField<M>],
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        self.check(input.len(), output.len(), scratch.len())?;
        if self.description.storage != ExpansionStorage::CoefficientWorkspace {
            return Err(FftError::InvalidExecution);
        }
        check_len(
            "coefficients",
            coefficients.len(),
            self.required.coefficient_fields,
        )?;
        self.expansion.base.scatter(input, coefficients);
        self.inverse_coefficients(coefficients, executor, scratch);
        self.residues(coefficients, output, 0, false, None, executor, scratch);
        Ok(())
    }

    /// Consumes base evaluations as coefficient storage.
    ///
    /// Requires [`ExpansionStorage::DisposableInput`]. On success the input
    /// follows the coefficient ordering and scaling of
    /// [`Self::execute_with_workspace`]. Lengths, scratch, and errors follow
    /// [`PreparedExpansion`].
    pub fn execute_disposable<E: Executor>(
        self,
        input: &mut [PastaField<M>],
        output: &mut [PastaField<M>],
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        self.check(input.len(), output.len(), scratch.len())?;
        if self.description.storage != ExpansionStorage::DisposableInput {
            return Err(FftError::InvalidExecution);
        }
        self.expansion.base.permute(input);
        self.inverse_coefficients(input, executor, scratch);
        self.residues(input, output, 0, false, None, executor, scratch);
        Ok(())
    }

    /// Writes the pointwise product of a coefficient expansion and `factor`.
    ///
    /// Requires [`ExpansionStorage::Coefficients`]. The factor must match the
    /// extended coset domain and [`Self::layout`], otherwise this returns
    /// [`FftError::InvalidLayout`]. Empty prefixes represent the zero polynomial.
    /// Input is preserved; lengths, scratch, and other errors follow
    /// [`PreparedExpansion`].
    pub fn execute_product_into<E: Executor>(
        self,
        input: &[PastaField<M>],
        factor: EvaluationView<'_, M>,
        output: &mut [PastaField<M>],
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        self.check(input.len(), output.len(), scratch.len())?;
        if self.description.storage != ExpansionStorage::Coefficients {
            return Err(FftError::InvalidExecution);
        }
        if !factor.domain().same_domain(self.expansion.extended) || factor.layout() != self.layout()
        {
            return Err(FftError::InvalidLayout);
        }
        self.residues(
            input,
            output,
            0,
            true,
            Some(factor.as_slice()),
            executor,
            scratch,
        );
        Ok(())
    }

    fn inverse_coefficients<E: Executor>(
        self,
        values: &mut [PastaField<M>],
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) {
        self.expansion.base.run(
            values,
            self.inverse,
            executor,
            &mut scratch[..self.required.inverse_scratch_fields],
            self.expansion.inverse_policy().0,
        );
    }

    #[allow(clippy::too_many_arguments)]
    fn residues<E: Executor>(
        self,
        coefficients: &[PastaField<M>],
        output: &mut [PastaField<M>],
        first: usize,
        normalized: bool,
        factor: Option<&[PastaField<M>]>,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) {
        let tasks = self
            .options
            .max_residue_tasks
            .min(output.len() / self.description.base_size);
        ResidueJobs {
            expansion: self.expansion,
            coefficients,
            factor,
            normalized_coefficients: normalized,
            order: self.description.order,
            extra: if normalized {
                PastaField::ONE
            } else {
                self.expansion.inverse_policy().1
            },
            options: self.options.transform,
            executor,
        }
        .run(
            output,
            first,
            tasks,
            &mut scratch[..self.required.per_worker_scratch_fields * tasks],
        );
    }
}

/// A selected residue whose output needs only the base domain's storage.
#[derive(Clone, Copy)]
pub struct Residue<'a, M: PrimeModulus> {
    expansion: Expansion<'a, M>,
    residue: usize,
    order: InputOrder,
}

impl<M: PrimeModulus> core::fmt::Debug for Residue<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Residue")
            .field("expansion", &self.expansion)
            .field("residue", &self.residue)
            .field("order", &self.order)
            .finish()
    }
}

impl<M: PrimeModulus> Residue<'_, M> {
    /// Base-sized coset containing the selected residue's extended rows.
    ///
    /// Its natural row `k` is extended row `s + r * k`, where `s` is the selected
    /// residue number, `r` is the expansion's residue count, and `0 <= k < n`
    /// for base size `n`.
    pub fn domain(self) -> CosetDomain<M> {
        self.expansion
            .base
            .domain()
            .domain()
            .coset(
                self.expansion.extended.shift().mul(
                    &self
                        .expansion
                        .extended
                        .domain()
                        .root()
                        .pow_u64(self.residue as u64),
                ),
            )
            .unwrap()
    }
    /// Scratch for this residue's output order.
    ///
    /// Validates options through [`ExecutionOptions::requirements`] for the base
    /// size, with the same errors. Natural output uses that scratch count;
    /// bit-reversed output needs no scratch.
    pub const fn scratch_requirements(
        self,
        options: ExecutionOptions,
    ) -> Result<ScratchRequirements, FftError> {
        let required = match options.requirements(self.expansion.base.domain().size()) {
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        Ok(if matches!(self.order, InputOrder::BitReversed) {
            ScratchRequirements { field_elements: 0 }
        } else {
            required
        })
    }
    /// Evaluates a coefficient prefix into a reusable base-sized output.
    ///
    /// For base size `n`, input contains `0..=n` natural-order coefficients,
    /// with omitted coefficients treated as zero. Output has exactly `n` fields
    /// and uses the order selected by [`Expansion::residue`].
    /// Scratch must meet [`Self::scratch_requirements`], including for empty input.
    ///
    /// Returns [`FftError::InvalidPrefix`] for an oversized input,
    /// [`FftError::LengthMismatch`] for an incorrect output length, or
    /// [`FftError::ScratchTooSmall`] for insufficient scratch. Option errors follow
    /// [`Self::scratch_requirements`]. The module's [validation and working-storage
    /// rules](super) apply.
    pub fn coefficients<E: Executor>(
        self,
        input: &[PastaField<M>],
        output: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        check_prefix(input.len(), 0, self.expansion.base.domain().size())?;
        check_len("output", output.len(), self.expansion.base.domain().size())?;
        let required = self.scratch_requirements(options)?;
        required.check(scratch.len())?;
        ResidueJobs {
            expansion: self.expansion,
            coefficients: input,
            factor: None,
            normalized_coefficients: true,
            order: if self.order == InputOrder::Natural {
                ExpansionOrder::Residues
            } else {
                ExpansionOrder::BitReversed
            },
            extra: PastaField::ONE,
            options,
            executor,
        }
        .residue(
            output,
            self.residue,
            self.residue,
            &mut scratch[..required.field_elements],
        );
        Ok(())
    }
}
