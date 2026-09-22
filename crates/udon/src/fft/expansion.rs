#[cfg(test)]
use super::transform::Run;
#[cfg(test)]
use super::{Codelet, InverseScale, Strategy, check_domain_size, check_field_count, min, reverse};
use super::{
    CosetDomain, EvaluationLayout, EvaluationView, Executor, ExpansionOrder,
    ExpansionScaleNormalization, ExpansionScales, FftError, PastaField, PrimeModulus,
    ResidueLayout, ScratchRequirements, Transform, check_length, check_prefix,
};
use super::{ElementOrder, ExpansionStorage, InputSupport, StorageLayout, run::ExpansionPlan};
use crate::exec::ExecutionOptions;

/// Caller-selected concurrency across residues and within each base transform.
///
/// At most `max_residue_tasks` residues run concurrently, each using `transform`
/// and the caller's executor. Thus the total concurrency is bounded by
/// `min(max_residue_tasks, residue_count) * transform.max_tasks`. Set either
/// task count to one to concentrate parallel work at the other level.
/// Executors must support nested joins, as required by [`Executor`].
///
/// Scratch is partitioned between concurrent residues. The const queries
/// [`Self::coefficient_requirements`] and [`Self::evaluation_requirements`]
/// size storage without constructing domains or an expansion.
/// A zero residue task count or invalid transform settings
/// return [`FftError::InvalidExecution`], even for singleton domains.
/// The default is [`Self::serial`].
#[cfg(test)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) struct ExpansionStrategy {
    /// Maximum number of concurrently executing residue transforms.
    pub max_residue_tasks: usize,
    /// Scheduling and scratch bounds within every base-size transform.
    pub transform: Strategy,
}

#[cfg(test)]
impl ExpansionStrategy {
    /// Executes whole residue transforms sequentially, with no scratch or joins.
    pub const fn serial() -> Self {
        Self {
            max_residue_tasks: 1,
            transform: Strategy::serial(),
        }
    }

    /// Scratch for coefficient expansion or a short product, given domain sizes.
    ///
    /// Applies to both Pasta fields, with or without prepared tables and residue
    /// scales. Whole-transform tiles need no scratch. Other counts follow the
    /// current execution implementation.
    ///
    /// Size limits and errors are those of [`super::Domain::for_size`]. Returns
    /// [`FftError::InvalidLayout`] if `base_size > extended_size`,
    /// [`FftError::InvalidExecution`] for invalid options, or
    /// [`FftError::SizeOverflow`] if the combined scratch count overflows `usize`
    /// or its field slice would exceed `isize::MAX` bytes.
    pub const fn coefficient_requirements(
        self,
        base_size: usize,
        extended_size: usize,
    ) -> Result<ScratchRequirements, FftError> {
        self.requirements(base_size, extended_size, false)
    }

    /// Scratch for expansion from base evaluations, given domain sizes.
    ///
    /// Output also stores the coefficients, so no separate coefficient buffer
    /// is needed. Accepted sizes and errors are those of
    /// [`Self::coefficient_requirements`].
    pub const fn evaluation_requirements(
        self,
        base_size: usize,
        extended_size: usize,
    ) -> Result<ScratchRequirements, FftError> {
        self.requirements(base_size, extended_size, true)
    }

    const fn requirements(
        self,
        base_size: usize,
        extended_size: usize,
        from_evaluations: bool,
    ) -> Result<ScratchRequirements, FftError> {
        if self.max_residue_tasks == 0 {
            return Err(FftError::InvalidExecution);
        }
        let fields = match self.transform.requirements(base_size) {
            Ok(required) => required.field_elements,
            Err(error) => return Err(error),
        };
        if let Err(error) = check_domain_size(extended_size) {
            return Err(error);
        }
        if base_size > extended_size {
            return Err(FftError::InvalidLayout);
        }
        let mut residues = extended_size / base_size;
        // The inverse and first residue reuse one partition. The other
        // residues run together while reading coefficients from residue zero.
        if from_evaluations && residues > 1 {
            residues -= 1;
        }
        let fields = match fields.checked_mul(min(self.max_residue_tasks, residues)) {
            Some(fields) => fields,
            None => return Err(FftError::SizeOverflow),
        };
        match check_field_count(fields) {
            Ok(field_elements) => Ok(ScratchRequirements { field_elements }),
            Err(error) => Err(error),
        }
    }
}

#[cfg(test)]
impl Default for ExpansionStrategy {
    fn default() -> Self {
        Self::serial()
    }
}

/// Evaluates a base polynomial on a coset of equal or larger size.
///
/// For base size `n` and extended size `r*n`, residue `s` contains the extended
/// domain's natural rows `s + r*k`, for `0 <= s < r` and `0 <= k < n`. Direct
/// methods store each residue contiguously, as described by [`Self::layout`];
/// use [`EvaluationView`] for lookup by natural row.
/// [`super::run::ExpansionPlan`] also supports bit-reversed output. The ratio
/// `r` can be any supported power of two, including one.
///
/// Residues use their output as working storage, avoiding a full zero-padded
/// FFT. [`ExecutionOptions`] supplies one resource allowance shared across and
/// within residues; every transform uses the caller's executor. All output
/// buffers must have the extended domain's size. The module's
/// [validation and working-storage rules](super) apply, including table validity
/// and buffer state after errors or panics.
///
/// ```
/// use zakura_udon::{
///     exec::{ExecutionOptions, SerialExecutor},
///     field::Fp,
///     fft::{Domain, Expansion, Transform, EvaluationLayout, EvaluationView},
/// };
///
/// let base = Transform::new(Domain::new(1).unwrap().subgroup());
/// let extended = Domain::new(3).unwrap().coset(Fp::from_u64(7)).unwrap();
/// let expansion = Expansion::new(base, extended, None).unwrap();
/// let coefficients = [Fp::from_u64(3), Fp::from_u64(2)]; // 3 + 2*x
/// let mut output = [Fp::ZERO; 8];
/// expansion.coefficients(
///     &coefficients, &mut output, ExecutionOptions::default(),
///     &SerialExecutor, &mut [],
/// ).unwrap();
/// let view = EvaluationView::bind(
///     &output, extended, EvaluationLayout::Residues(expansion.layout()),
/// ).unwrap();
/// for row in 0..extended.size() {
///     let root_power = extended.domain().root().pow_u64(row as u64);
///     let point = extended.shift().mul(&root_power);
///     let expected = coefficients[0].add(&coefficients[1].mul(&point));
///     assert_eq!(view.get(row), Some(&expected));
/// }
/// ```
#[derive(Clone, Copy)]
pub struct Expansion<'a, M: PrimeModulus> {
    pub(super) base: Transform<'a, M>,
    pub(super) extended: CosetDomain<M>,
    pub(super) scales: Option<&'a [PastaField<M>]>,
    pub(super) normalization: ExpansionScaleNormalization,
    pub(super) layout: ResidueLayout,
}

impl<M: PrimeModulus> core::fmt::Debug for Expansion<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Expansion")
            .field("base", &self.base)
            .field("extended", &self.extended)
            .field("scales", &self.scales)
            .field("layout", &self.layout)
            .finish()
    }
}

impl<'a, M: PrimeModulus> Expansion<'a, M> {
    /// Constructs an expansion after checking the base and optional scale domains.
    ///
    /// `base` must describe a subgroup (shift one) and fit in `extended`, or
    /// this returns [`FftError::InvalidLayout`].
    ///
    /// Optional scales retain their domain and normalization. Compatibility
    /// checks and scale usage follow [`Self::with_scales`]; construction does
    /// not rescan table entries.
    pub fn new(
        base: Transform<'a, M>,
        extended: CosetDomain<M>,
        scales: Option<ExpansionScales<'a, M>>,
    ) -> Result<Self, FftError> {
        if base.domain().shift() != PastaField::ONE || base.domain().size() > extended.size() {
            return Err(FftError::InvalidLayout);
        }
        let expansion = Self {
            base,
            extended,
            scales: None,
            normalization: ExpansionScaleNormalization::Coefficients,
            layout: ResidueLayout::new(extended.size(), extended.size() / base.domain().size())?,
        };
        match scales {
            Some(scales) => expansion.with_scales(scales),
            None => Ok(expansion),
        }
    }

    /// Replaces the scaling table after checking its domain and base size.
    ///
    /// A domain or base-size mismatch returns [`FftError::InvalidTables`].
    /// Coefficient input, including an unscaled
    /// [`CoefficientView`](super::CoefficientView), can use either convention.
    /// Initialization accounts for the view's source base size and any factor
    /// already present in the table. [`Self::evaluations`] accepts either
    /// convention. [`super::run::ExpansionPlan`] also accounts for the retained
    /// coefficient scale during residue initialization.
    ///
    /// Table contents follow [`ExpansionScales`]' preparation and binding
    /// contract. Attachment does not rescan entries.
    pub fn with_scales(mut self, scales: ExpansionScales<'a, M>) -> Result<Self, FftError> {
        if scales.base_size != self.base.domain().size()
            || !scales.extended.same_domain(self.extended)
        {
            return Err(FftError::InvalidTables);
        }
        self.scales = Some(scales.values);
        self.normalization = scales.normalization;
        Ok(self)
    }

    /// Layout used by direct expansion methods and their factor inputs.
    ///
    /// [`super::run::ExpansionPlan`] can instead select
    /// [`ExpansionOrder::BitReversed`]; bind those results with
    /// [`EvaluationLayout::BitReversed`].
    pub const fn layout(self) -> ResidueLayout {
        self.layout
    }

    /// Binds values to the extended domain and direct methods' residue layout.
    ///
    /// Returns [`FftError::LengthMismatch`] unless `values` has the extended
    /// size. Contents are not checked; see [`EvaluationView::bind`].
    pub fn view(self, values: &[PastaField<M>]) -> Result<EvaluationView<'_, M>, FftError> {
        EvaluationView::bind(
            values,
            self.extended,
            EvaluationLayout::Residues(self.layout),
        )
    }
    /// Number of field elements in the optional residue-scaling table.
    pub const fn scale_count(self) -> usize {
        self.extended.size()
    }

    /// Prepares residue scales into caller storage after checking its exact length.
    ///
    /// Entries use [`ExpansionScaleNormalization::Coefficients`], regardless
    /// of any already borrowed scales. Use [`ExpansionScales::prepare`] to
    /// select another normalization. Returns [`FftError::LengthMismatch`]
    /// without writing if `output.len()` differs from [`Self::scale_count`].
    pub fn prepare_scales(
        self,
        output: &mut [PastaField<M>],
    ) -> Result<ExpansionScales<'_, M>, FftError> {
        check_length("output", self.scale_count(), output.len())?;
        ExpansionScales::prepare(
            self.base.domain().size(),
            self.extended,
            ExpansionScaleNormalization::Coefficients,
            output,
        )
    }

    /// Checks every supplied residue scale without allocating or mutating it.
    ///
    /// Checks the declared [`ExpansionScaleNormalization`], returning
    /// [`FftError::InvalidTables`] for an incorrect or unreduced entry.
    /// Succeeds immediately if scales were omitted. This does not validate the
    /// base plan's tables; their validity follows [`super::Tables`]' contract.
    pub fn validate_scales(self) -> Result<(), FftError> {
        if let Some(scales) = self.scales {
            ExpansionScales::bind(
                self.base.domain().size(),
                self.extended,
                self.normalization,
                scales,
            )?;
        }
        Ok(())
    }

    fn check(self, coefficients: usize, min: usize, output: usize) -> Result<(), FftError> {
        check_length("output", self.extended.size(), output)?;
        check_prefix(coefficients, min, self.base.domain().size())
    }

    /// Scratch selected for coefficient expansion and short products.
    ///
    /// Resolves full natural-order coefficients through
    /// [`ExpansionPlan::new`] and [`ExpansionPlan::scratch_fields`], with their
    /// sizing errors. Direct execution can adapt to smaller or empty scratch.
    pub fn coefficient_scratch(
        self,
        options: ExecutionOptions,
    ) -> Result<ScratchRequirements, FftError> {
        self.scratch(ExpansionStorage::Coefficients, options)
    }

    /// Scratch selected for expansion from base evaluations.
    ///
    /// Resolves [`ExpansionStorage::ReuseOutput`] through [`ExpansionPlan::new`]
    /// and [`ExpansionPlan::scratch_fields`], with their sizing errors. Direct
    /// execution can adapt to smaller or empty scratch.
    pub fn evaluation_scratch(
        self,
        options: ExecutionOptions,
    ) -> Result<ScratchRequirements, FftError> {
        self.scratch(ExpansionStorage::ReuseOutput, options)
    }

    fn scratch(
        self,
        storage: ExpansionStorage,
        options: ExecutionOptions,
    ) -> Result<ScratchRequirements, FftError> {
        Ok(ScratchRequirements {
            field_elements: self
                .plan(storage, InputSupport::Full, options)?
                .scratch_fields()?,
        })
    }

    fn plan(
        self,
        storage: ExpansionStorage,
        support: InputSupport,
        options: ExecutionOptions,
    ) -> Result<ExpansionPlan<'a, M>, FftError> {
        ExpansionPlan::new(
            self,
            storage,
            ExpansionOrder::Residues,
            support,
            ElementOrder::Natural,
            StorageLayout::Contiguous,
            options,
        )
    }

    /// Expands a preserved natural coefficient prefix into residue-major evaluations.
    ///
    /// Input contains zero through the base size's natural-order coefficients;
    /// missing coefficients are zero. A [`super::CoefficientView`] supplies a
    /// normalization factor that is applied during initialization. Output must
    /// have the extended domain's size and uses [`Self::layout`]. Udon divides
    /// the task budget across residues and their transforms, adapting to
    /// scratch capacity, including empty scratch. Surplus scratch is untouched.
    ///
    /// Returns [`FftError::InvalidPrefix`] for oversized input or
    /// [`FftError::LengthMismatch`] for an incorrect output length. Planning and
    /// sizing errors follow [`Self::coefficient_scratch`]. All returned errors
    /// precede writes; panic behavior follows the module's
    /// [working-storage rules](super).
    pub fn coefficients<'input, E: Executor>(
        self,
        input: impl Into<super::CoefficientView<'input, M>>,
        output: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let input = input.into();
        self.plan(
            ExpansionStorage::Coefficients,
            InputSupport::Prefix(input.as_slice().len()),
            options.for_scratch::<PastaField<M>>(scratch.len()),
        )?
        .with_coefficient_scale(input.normalization_factor())?
        .execute(input.as_slice(), output, &mut [], None, scratch, executor)?;
        Ok(())
    }

    /// Preserves base evaluations and writes residue-major extended evaluations.
    ///
    /// Input must contain exactly the base size's evaluations in natural order;
    /// output must have the extended domain's size. Incorrect lengths return
    /// [`FftError::LengthMismatch`] before writes. Output also holds intermediate
    /// coefficients. Scratch adapts as in [`Self::coefficients`], with sizing
    /// errors from [`Self::evaluation_scratch`]. Its error and panic guarantees
    /// apply here too.
    pub fn evaluations<E: Executor>(
        self,
        input: &[PastaField<M>],
        output: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        self.plan(
            ExpansionStorage::ReuseOutput,
            InputSupport::Full,
            options.for_scratch::<PastaField<M>>(scratch.len()),
        )?
        .execute(input, output, &mut [], None, scratch, executor)?;
        Ok(())
    }

    /// Fuses coefficient expansion with multiplication by extended-domain factors.
    ///
    /// The coefficient prefix contains one through the base size's entries and
    /// is preserved. Scaling, output length, scratch, and panic behavior follow
    /// [`Self::coefficients`]. Factors must use this expansion's extended domain
    /// and [`Self::layout`], or this returns [`FftError::InvalidLayout`]. An empty
    /// or oversized prefix returns [`FftError::InvalidPrefix`]. All returned
    /// errors precede writes. Interpolating the full product requires its degree
    /// to be below the extended domain size; that degree bound is not checked.
    pub fn short_product<'input, E: Executor>(
        self,
        input: impl Into<super::CoefficientView<'input, M>>,
        factor: EvaluationView<'_, M>,
        output: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let input = input.into();
        self.check(input.as_slice().len(), 1, output.len())?;
        if factor.layout() != EvaluationLayout::Residues(self.layout)
            || !factor.domain().same_domain(self.extended)
        {
            return Err(FftError::InvalidLayout);
        }
        self.plan(
            ExpansionStorage::Coefficients,
            InputSupport::Prefix(input.as_slice().len()),
            options.for_scratch::<PastaField<M>>(scratch.len()),
        )?
        .with_coefficient_scale(input.normalization_factor())?
        .execute(
            input.as_slice(),
            output,
            &mut [],
            Some(factor.as_slice()),
            scratch,
            executor,
        )?;
        Ok(())
    }

    pub(super) fn residue_base(self, residue: usize) -> Transform<'a, M> {
        let mut base = self.base;
        base.domain = self
            .residue(residue, ElementOrder::Natural)
            .unwrap()
            .domain();
        // Root powers and permutations still apply to this shifted domain;
        // inverse scales and finishes encode the original domain's shift.
        base.tables.inverse_scales = None;
        base.tables.inverse_finish = None;
        base
    }

    /// Scratch needed by [`Self::coefficients`] and [`Self::short_product`].
    ///
    /// Delegates to [`ExpansionStrategy::coefficient_requirements`] with the
    /// base and extended sizes. That const query also sizes arrays without
    /// constructing an expansion.
    ///
    /// Returns [`FftError::InvalidExecution`] for invalid [`ExpansionStrategy`],
    /// or [`FftError::SizeOverflow`] if the combined scratch slice would exceed
    /// `isize::MAX` bytes or its element count overflows `usize`.
    #[cfg(test)]
    pub(crate) const fn coefficient_scratch_with(
        self,
        options: ExpansionStrategy,
    ) -> Result<ScratchRequirements, FftError> {
        options.coefficient_requirements(self.base.domain().size(), self.extended.size())
    }

    /// Expands a coefficient prefix into residue-major coset evaluations.
    ///
    /// Accepts ordinary coefficients or a
    /// [`CoefficientView`](super::CoefficientView) in increasing degree order;
    /// an empty prefix represents zero. A view's normalization is applied
    /// during initialization, with table usage as
    /// described by [`Self::with_scales`]. Input is preserved. Execution uses
    /// [`ExpansionStrategy`] across and within residues; scratch must meet
    /// [`Self::coefficient_scratch`].
    ///
    /// Returns [`FftError::InvalidPrefix`] if the prefix exceeds the base size,
    /// [`FftError::LengthMismatch`] if the output length differs from the
    /// extended size, or [`FftError::ScratchTooSmall`] for insufficient scratch.
    /// Options and storage limits have the errors of [`Self::coefficient_scratch`].
    #[cfg(test)]
    pub(crate) fn coefficients_with<'input, E: Executor>(
        self,
        coefficients: impl Into<super::CoefficientView<'input, M>>,
        output: &mut [PastaField<M>],
        options: ExpansionStrategy,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let coefficients = coefficients.into();
        let (normalized_coefficients, extra) = self.coefficient_input(coefficients);
        let coefficients = coefficients.as_slice();
        self.check(coefficients.len(), 0, output.len())?;
        let required = self.coefficient_scratch_with(options)?;
        required.check(scratch.len())?;
        if coefficients.len() <= 1 {
            output.fill(coefficients.first().copied().unwrap_or(PastaField::ZERO));
            return Ok(());
        }
        ResidueJobs {
            expansion: self,
            coefficients,
            factor: None,
            normalized_coefficients,
            order: ExpansionOrder::Residues,
            extra,
            options: options.transform,
            executor,
        }
        .run(
            output,
            0,
            options.max_residue_tasks.min(self.layout.residues()),
            &mut scratch[..required.field_elements],
        );
        Ok(())
    }

    /// Scratch needed by [`Self::evaluations`].
    ///
    /// Delegates to [`ExpansionStrategy::evaluation_requirements`] with the
    /// base and extended sizes. Errors are those of [`Self::coefficient_scratch`].
    #[cfg(test)]
    pub(crate) const fn evaluation_scratch_with(
        self,
        options: ExpansionStrategy,
    ) -> Result<ScratchRequirements, FftError> {
        options.evaluation_requirements(self.base.domain().size(), self.extended.size())
    }

    /// Preserves base-subgroup evaluations and expands them into the coset.
    ///
    /// Input must contain exactly the base size in natural evaluation order;
    /// output must contain exactly the extended size and uses residue order.
    /// A wrong length returns [`FftError::LengthMismatch`]. Insufficient scratch
    /// returns [`FftError::ScratchTooSmall`]; options and storage limits have
    /// the errors described by [`Self::evaluation_scratch`].
    ///
    /// `options.transform` controls the inverse base transform and every
    /// residue transform. After the inverse, at most `options.max_residue_tasks`
    /// of the remaining residues run concurrently. The first residue holds
    /// their coefficient input and is completed after they finish reading it.
    #[cfg(test)]
    pub(crate) fn evaluations_with<E: Executor>(
        self,
        evaluations: &[PastaField<M>],
        output: &mut [PastaField<M>],
        options: ExpansionStrategy,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        check_length("evaluations", self.base.domain().size(), evaluations.len())?;
        check_length("output", self.extended.size(), output.len())?;
        let required = self.evaluation_scratch_with(options)?;
        required.check(scratch.len())?;
        if self.extended.shift() == PastaField::ONE && output.len() == evaluations.len() {
            output.copy_from_slice(evaluations);
            return Ok(());
        }
        let scratch = &mut scratch[..required.field_elements];
        let transform_scratch = self
            .base
            .scratch_requirements_with(options.transform)?
            .field_elements;
        // Reuse the first residue as a coefficient buffer. Ordinary tables need
        // a normalized inverse; pre-normalized tables already include division
        // by the base size. Without tables, absorb division into the coefficient
        // progression. Readers finish before reuse below.
        let (first, rest) = output.split_at_mut(evaluations.len());
        self.base.scatter(evaluations, first);
        let (inverse, extra) = self.inverse_policy();
        self.base.run(
            first,
            options.transform,
            executor,
            &mut scratch[..transform_scratch],
            inverse,
        );
        ResidueJobs {
            expansion: self,
            coefficients: first,
            factor: None,
            normalized_coefficients: false,
            order: ExpansionOrder::Residues,
            extra,
            options: options.transform,
            executor,
        }
        .run(
            rest,
            1,
            options.max_residue_tasks.min(self.layout.residues() - 1),
            scratch,
        );
        if self.extended.shift() == PastaField::ONE {
            first.copy_from_slice(evaluations);
            return Ok(());
        }
        let first_plan = Transform {
            domain: CosetDomain::with_inverse(
                self.base.domain().domain(),
                self.extended.shift(),
                self.extended.inverse_shift(),
            ),
            tables: self.base.tables,
        };
        if let Some(scales) = self.scales {
            for (value, scale) in first.iter_mut().zip(scales) {
                *value = value.mul(scale);
            }
        } else {
            first_plan.scale_coefficients(first, extra);
        }
        first_plan.permute(first);
        first_plan.run(
            first,
            options.transform,
            executor,
            &mut scratch[..transform_scratch],
            Run::forward(2),
        );
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn inverse_policy(self) -> (Run<'a, 'a, M>, PastaField<M>) {
        if self.scales.is_some() {
            if self.normalization == ExpansionScaleNormalization::Coefficients {
                (Run::inverse(&[]), PastaField::ONE)
            } else {
                (Run::inverse_unscaled(), PastaField::ONE)
            }
        } else {
            (
                Run::inverse_unscaled(),
                self.base.domain().domain().size_inverse(),
            )
        }
    }

    /// Evaluates a short polynomial times a supplied factor.
    ///
    /// `short` accepts ordinary coefficients or a
    /// [`CoefficientView`](super::CoefficientView), with ordering and scaling as
    /// in [`Self::coefficients`]. It must have between one and the base size
    /// entries. Its expanded evaluations are multiplied pointwise by `factor`.
    /// The factor must use this expansion's layout and exact ordered evaluation
    /// domain; both are checked.
    ///
    /// Scheduling, scratch requirements, and output-length errors are those of
    /// [`Self::coefficients`].
    /// An empty or oversized prefix returns [`FftError::InvalidPrefix`], and a
    /// different factor layout or domain returns [`FftError::InvalidLayout`].
    /// Recovering the full product by interpolation additionally requires its
    /// degree to be below the extended domain size.
    #[cfg(test)]
    pub(crate) fn short_product_with<'input, E: Executor>(
        self,
        short: impl Into<super::CoefficientView<'input, M>>,
        factor: EvaluationView<'_, M>,
        output: &mut [PastaField<M>],
        options: ExpansionStrategy,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let short = short.into();
        let (normalized_coefficients, extra) = self.coefficient_input(short);
        let short = short.as_slice();
        self.check(short.len(), 1, output.len())?;
        if factor.layout() != EvaluationLayout::Residues(self.layout)
            || !factor.domain().same_domain(self.extended)
        {
            return Err(FftError::InvalidLayout);
        }
        let required = self.coefficient_scratch_with(options)?;
        required.check(scratch.len())?;
        ResidueJobs {
            expansion: self,
            coefficients: short,
            factor: Some(factor.as_slice()),
            normalized_coefficients,
            order: ExpansionOrder::Residues,
            extra,
            options: options.transform,
            executor,
        }
        .run(
            output,
            0,
            options.max_residue_tasks.min(self.layout.residues()),
            &mut scratch[..required.field_elements],
        );
        Ok(())
    }

    #[cfg(test)]
    pub(super) fn coefficient_input(
        self,
        coefficients: super::CoefficientView<'_, M>,
    ) -> (bool, PastaField<M>) {
        let normalized = coefficients.scale() == InverseScale::Normalized;
        let extra = if !normalized
            && self.scales.is_some()
            && self.normalization == ExpansionScaleNormalization::UnscaledInverse
        {
            // The table divides by this expansion's base size; the view needs
            // division by its source size. Their ratio is a power of two.
            // Oversized inputs are rejected before execution by every caller.
            if self.base.domain().size() == coefficients.as_slice().len() {
                PastaField::ONE
            } else {
                PastaField::from_u64(
                    (self.base.domain().size() / coefficients.as_slice().len()) as u64,
                )
            }
        } else {
            coefficients.normalization_factor()
        };
        (normalized, extra)
    }
}

#[cfg(test)]
pub(super) struct ResidueJobs<'a, 'b, M: PrimeModulus, E> {
    pub expansion: Expansion<'a, M>,
    pub coefficients: &'b [PastaField<M>],
    pub factor: Option<&'b [PastaField<M>]>,
    pub normalized_coefficients: bool,
    pub order: ExpansionOrder,
    pub extra: PastaField<M>,
    pub options: Strategy,
    pub executor: &'b E,
}

#[cfg(test)]
impl<M: PrimeModulus, E: Executor> ResidueJobs<'_, '_, M, E> {
    pub fn run(
        &self,
        output: &mut [PastaField<M>],
        first: usize,
        tasks: usize,
        scratch: &mut [PastaField<M>],
    ) {
        let size = self.expansion.base.domain().size();
        if tasks <= 1 {
            for (residue, output) in output.chunks_exact_mut(size).enumerate() {
                let block = first + residue;
                let residue = if self.order == ExpansionOrder::BitReversed {
                    reverse(block, self.expansion.layout.residues().ilog2())
                } else {
                    block
                };
                self.residue(output, residue, block, scratch);
            }
        } else {
            // Split scratch ownership with the residue budget. Each leaf owns
            // one transform's scratch and reuses it for its assigned residues;
            // nested transform joins use only that leaf's partition.
            let left_residues = output.len() / size / 2;
            let left_tasks = tasks / 2;
            let (left, right) = output.split_at_mut(left_residues * size);
            let (left_scratch, right_scratch) =
                scratch.split_at_mut(scratch.len() / tasks * left_tasks);
            self.executor.join(
                || self.run(left, first, left_tasks, left_scratch),
                || {
                    self.run(
                        right,
                        first + left_residues,
                        tasks - left_tasks,
                        right_scratch,
                    )
                },
            );
        }
    }

    pub fn residue(
        &self,
        output: &mut [PastaField<M>],
        residue: usize,
        block: usize,
        scratch: &mut [PastaField<M>],
    ) {
        // The extended root raised to the residue count is the base root.
        // Thus a base FFT with coefficients scaled by (shift * root^s)^j
        // evaluates exactly the extended rows s + r*k of residue s.
        let expansion = self.expansion;
        let size = expansion.base.domain().size();
        let scales = expansion
            .scales
            .filter(|_| {
                !self.normalized_coefficients
                    || expansion.normalization == ExpansionScaleNormalization::Coefficients
            })
            .map(|scales| &scales[residue * size..(residue + 1) * size]);
        let shift = if scales.is_some() {
            PastaField::ONE
        } else {
            expansion
                .extended
                .shift()
                .mul(&expansion.extended.domain().root().pow_u64(residue as u64))
        };
        let factor = self
            .factor
            .map(|factor| &factor[block * size..(block + 1) * size]);
        if self.order == ExpansionOrder::BitReversed {
            // With at least four zero-only stages, broadcasting a short
            // prefix and permuting the DIT result saves more work than a full
            // DIF. The terminal store reads the factor in its declared order.
            if self.coefficients.len() <= size / 16 {
                let first = expansion.base.fill_prefix(
                    self.coefficients,
                    output,
                    shift,
                    scales,
                    self.extra,
                );
                super::stages::StageKernel {
                    plan: expansion.base,
                    inverse: false,
                    dif: false,
                    scale: InverseScale::Normalized,
                    codelet: Codelet::Radix2,
                    twiddles: None,
                    output_order: ElementOrder::BitReversed,
                    factor,
                }
                .drive(output, first, self.options, self.executor);
                return;
            }
            let (prefix, tail) = output.split_at_mut(self.coefficients.len());
            tail.fill(PastaField::ZERO);
            if let Some(scales) = scales {
                if self.extra == PastaField::ONE {
                    for ((output, value), scale) in
                        prefix.iter_mut().zip(self.coefficients).zip(scales)
                    {
                        *output = value.mul(scale);
                    }
                } else {
                    for ((output, value), scale) in
                        prefix.iter_mut().zip(self.coefficients).zip(scales)
                    {
                        *output = value.mul(&scale.mul(&self.extra));
                    }
                }
            } else {
                let mut power = self.extra;
                for (output, value) in prefix.iter_mut().zip(self.coefficients) {
                    *output = value.mul(&power);
                    power = power.mul(&shift);
                }
            }
            super::stages::StageKernel {
                plan: expansion.base,
                inverse: false,
                dif: true,
                scale: InverseScale::Normalized,
                codelet: Codelet::Radix2,
                twiddles: None,
                output_order: ElementOrder::BitReversed,
                factor,
            }
            .drive(output, 2, self.options, self.executor);
            return;
        }
        let first =
            expansion
                .base
                .fill_prefix(self.coefficients, output, shift, scales, self.extra);
        expansion.base.run(
            output,
            self.options,
            self.executor,
            scratch,
            factor.map_or_else(
                || Run::forward(first),
                |factor| Run::forward_product(first, factor),
            ),
        );
    }
}
