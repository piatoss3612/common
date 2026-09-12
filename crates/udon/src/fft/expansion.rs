use super::transform::Run;
use super::{
    Codelet, CosetDomain, ExecutionOptions, Executor, ExpansionOrder, ExpansionScaleNormalization,
    ExpansionScales, FftError, InputOrder, InverseScale, PastaField, Plan, PrimeModulus,
    ResidueLayout, ResidueView, ScratchRequirements, check_domain_size, check_field_count,
    check_len, check_prefix, min, reverse,
};

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
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpansionOptions {
    /// Maximum number of concurrently executing residue transforms.
    pub max_residue_tasks: usize,
    /// Scheduling and scratch bounds within every base-size transform.
    pub transform: ExecutionOptions,
}

impl ExpansionOptions {
    /// Executes whole residue transforms sequentially, with no scratch or joins.
    pub const fn serial() -> Self {
        Self {
            max_residue_tasks: 1,
            transform: ExecutionOptions::serial(),
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

impl Default for ExpansionOptions {
    fn default() -> Self {
        Self::serial()
    }
}

/// Evaluates a base polynomial on a coset of equal or larger size.
///
/// For base size `n` and extended size `r*n`, residue `s` contains the extended
/// domain's natural rows `s + r*k`, for `0 <= s < r` and `0 <= k < n`. Direct
/// methods store each residue contiguously, as described by [`Self::layout`];
/// use [`ResidueView`] for lookup by natural row. [`Self::configure`] also
/// supports bit-reversed output. The ratio `r` can be any supported power of
/// two, including one.
///
/// Residues use their output as working storage, avoiding a full zero-padded
/// FFT. [`ExpansionOptions`] controls execution across and within residues;
/// every transform uses the caller's executor. All output buffers must have
/// the extended domain's size. The module's [validation and working-storage
/// rules](super) apply, including table validity and buffer state after errors
/// or panics.
///
/// ```
/// use zakura_udon::{
///     field::Fp,
///     fft::{Domain, Expansion, ExpansionOptions, Plan, ResidueView, SerialExecutor},
/// };
///
/// let base = Plan::without_tables(Domain::new(1).unwrap().subgroup());
/// let extended = Domain::new(3).unwrap().coset(Fp::from_u64(7)).unwrap();
/// let expansion = Expansion::new(base, extended, None).unwrap();
/// let coefficients = [Fp::from_u64(3), Fp::from_u64(2)]; // 3 + 2*x
/// let mut output = [Fp::ZERO; 8];
/// expansion.coefficients(
///     &coefficients, &mut output, ExpansionOptions::serial(),
///     &SerialExecutor, &mut [],
/// ).unwrap();
/// let view = ResidueView::new(&output, expansion.layout()).unwrap();
/// for row in 0..extended.size() {
///     let root_power = extended.domain().root().pow_u64(row as u64);
///     let point = extended.shift().mul(&root_power);
///     let expected = coefficients[0].add(&coefficients[1].mul(&point));
///     assert_eq!(view.get(row), Some(&expected));
/// }
/// ```
#[derive(Clone, Copy)]
pub struct Expansion<'a, M: PrimeModulus> {
    pub(super) base: Plan<'a, M>,
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
    /// Checks domain compatibility and the optional residue-scaling table length.
    ///
    /// `base` must describe a subgroup (shift one) and fit in `extended`, or
    /// this returns [`FftError::InvalidLayout`].
    ///
    /// Optional scales have `extended.size()` entries, in residue-major order:
    /// entry `s*n + j` is `(shift * root^s)^j`, where `n` is the base size and
    /// `shift` and `root` come from `extended`. A wrong length returns
    /// [`FftError::LengthMismatch`]. Contents are trusted as with
    /// [`super::Tables`]; use [`Self::validate_scales`] for a full check.
    pub fn new(
        base: Plan<'a, M>,
        extended: CosetDomain<M>,
        scales: Option<&'a [PastaField<M>]>,
    ) -> Result<Self, FftError> {
        if base.domain().shift() != PastaField::ONE || base.domain().size() > extended.size() {
            return Err(FftError::InvalidLayout);
        }
        if let Some(scales) = scales {
            check_len("scales", scales.len(), extended.size())?;
        }
        Ok(Self {
            base,
            extended,
            scales,
            normalization: ExpansionScaleNormalization::Coefficients,
            layout: ResidueLayout::new(extended.size(), extended.size() / base.domain().size())?,
        })
    }

    /// Replaces the scaling table after checking its domain and base size.
    ///
    /// A domain or base-size mismatch returns [`FftError::InvalidTables`].
    /// Coefficient expansion uses ordinary powers and ignores tables declared
    /// for an unscaled inverse. Evaluation expansion accepts either
    /// [`ExpansionScaleNormalization`]. Call [`ExpansionScales::validate`] to
    /// check imported entries.
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
    /// [`Self::configure`] can select a different order, reported by
    /// [`super::PreparedExpansion::layout`].
    pub const fn layout(self) -> ResidueLayout {
        self.layout
    }
    /// Number of field elements in the optional residue-scaling table.
    pub const fn scale_count(self) -> usize {
        self.extended.size()
    }

    /// Prepares residue scales into caller storage after checking its exact length.
    ///
    /// Entries follow [`Self::new`]'s ordinary coefficient formula, regardless
    /// of any already borrowed scales. Use [`ExpansionScales::prepare`] to
    /// select another normalization. Returns [`FftError::LengthMismatch`]
    /// without writing if `output.len()` differs from [`Self::scale_count`].
    pub fn prepare_scales(self, output: &mut [PastaField<M>]) -> Result<(), FftError> {
        check_len("output", output.len(), self.scale_count())?;
        let mut step = self.extended.shift();
        for (index, residue) in output
            .chunks_exact_mut(self.base.domain().size())
            .enumerate()
        {
            let mut power = PastaField::ONE;
            for (column, value) in residue.iter_mut().enumerate() {
                *value = power;
                if column + 1 < self.base.domain().size() {
                    power = power.mul(&step);
                }
            }
            if index + 1 < self.layout.residues() {
                step = step.mul(&self.extended.domain().root());
            }
        }
        Ok(())
    }

    /// Checks every supplied residue scale without allocating or mutating it.
    ///
    /// Checks the declared [`ExpansionScaleNormalization`], returning
    /// [`FftError::InvalidTables`] for an incorrect or unreduced entry.
    /// Succeeds immediately if scales were omitted. This does not validate the
    /// base plan's tables; those are checked with [`super::Tables::validate`].
    pub fn validate_scales(self) -> Result<(), FftError> {
        if let Some(scales) = self.scales {
            ExpansionScales::bind(
                self.base.domain().size(),
                self.extended,
                self.normalization,
                scales,
            )?
            .validate()?;
        }
        Ok(())
    }

    fn check(self, coefficients: usize, min: usize, output: usize) -> Result<(), FftError> {
        check_len("output", output, self.extended.size())?;
        check_prefix(coefficients, min, self.base.domain().size())
    }

    /// Scratch needed by [`Self::coefficients`] and [`Self::short_product`].
    ///
    /// Delegates to [`ExpansionOptions::coefficient_requirements`] with the
    /// base and extended sizes. That const query also sizes arrays without
    /// constructing an expansion.
    ///
    /// Returns [`FftError::InvalidExecution`] for invalid [`ExpansionOptions`],
    /// or [`FftError::SizeOverflow`] if the combined scratch slice would exceed
    /// `isize::MAX` bytes or its element count overflows `usize`.
    pub const fn coefficient_scratch(
        self,
        options: ExpansionOptions,
    ) -> Result<ScratchRequirements, FftError> {
        options.coefficient_requirements(self.base.domain().size(), self.extended.size())
    }

    /// Expands a coefficient prefix into residue-major coset evaluations.
    ///
    /// Coefficients are in increasing degree order; an empty prefix represents
    /// zero. Execution uses [`ExpansionOptions`] across and within residues;
    /// scratch must meet [`Self::coefficient_scratch`].
    ///
    /// Returns [`FftError::InvalidPrefix`] if the prefix exceeds the base size,
    /// [`FftError::LengthMismatch`] if the output length differs from the
    /// extended size, or [`FftError::ScratchTooSmall`] for insufficient scratch.
    /// Options and storage limits have the errors of [`Self::coefficient_scratch`].
    pub fn coefficients<E: Executor>(
        self,
        coefficients: &[PastaField<M>],
        output: &mut [PastaField<M>],
        options: ExpansionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        self.check(coefficients.len(), 0, output.len())?;
        let required = self.coefficient_scratch(options)?;
        required.check(scratch.len())?;
        if coefficients.len() <= 1 {
            output.fill(coefficients.first().copied().unwrap_or(PastaField::ZERO));
            return Ok(());
        }
        ResidueJobs {
            expansion: self,
            coefficients,
            factor: None,
            normalized_coefficients: true,
            order: ExpansionOrder::Residues,
            extra: PastaField::ONE,
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
    /// Delegates to [`ExpansionOptions::evaluation_requirements`] with the
    /// base and extended sizes. Errors are those of [`Self::coefficient_scratch`].
    pub const fn evaluation_scratch(
        self,
        options: ExpansionOptions,
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
    pub fn evaluations<E: Executor>(
        self,
        evaluations: &[PastaField<M>],
        output: &mut [PastaField<M>],
        options: ExpansionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        check_len("evaluations", evaluations.len(), self.base.domain().size())?;
        check_len("output", output.len(), self.extended.size())?;
        let required = self.evaluation_scratch(options)?;
        required.check(scratch.len())?;
        if self.extended.shift() == PastaField::ONE && output.len() == evaluations.len() {
            output.copy_from_slice(evaluations);
            return Ok(());
        }
        let scratch = &mut scratch[..required.field_elements];
        let transform_scratch = self
            .base
            .scratch_requirements(options.transform)?
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
        let first_plan = Plan {
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
    /// `short` contains coefficients in increasing degree order and must have
    /// between one and the base size entries. Its expanded evaluations are
    /// multiplied pointwise by `factor`. The factor must use this expansion's
    /// layout and evaluation domain; only the layout is checked. A different
    /// domain gives incorrect polynomial products without risking memory safety.
    ///
    /// Scheduling, scratch requirements, and output-length errors are those of
    /// [`Self::coefficients`].
    /// An empty or oversized prefix returns [`FftError::InvalidPrefix`], and a
    /// different factor layout returns [`FftError::InvalidLayout`]. The result
    /// is an evaluation vector; recovering the full product by interpolation
    /// additionally requires its degree to be below the extended domain size.
    pub fn short_product<E: Executor>(
        self,
        short: &[PastaField<M>],
        factor: ResidueView<'_, M>,
        output: &mut [PastaField<M>],
        options: ExpansionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        self.check(short.len(), 1, output.len())?;
        if factor.layout() != self.layout {
            return Err(FftError::InvalidLayout);
        }
        let required = self.coefficient_scratch(options)?;
        required.check(scratch.len())?;
        ResidueJobs {
            expansion: self,
            coefficients: short,
            factor: Some(factor.as_slice()),
            normalized_coefficients: true,
            order: ExpansionOrder::Residues,
            extra: PastaField::ONE,
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
}

pub(super) struct ResidueJobs<'a, 'b, M: PrimeModulus, E> {
    pub expansion: Expansion<'a, M>,
    pub coefficients: &'b [PastaField<M>],
    pub factor: Option<&'b [PastaField<M>]>,
    pub normalized_coefficients: bool,
    pub order: ExpansionOrder,
    pub extra: PastaField<M>,
    pub options: ExecutionOptions,
    pub executor: &'b E,
}

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
        let shift = expansion
            .extended
            .shift()
            .mul(&expansion.extended.domain().root().pow_u64(residue as u64));
        let factor = self
            .factor
            .map(|factor| &factor[block * size..(block + 1) * size]);
        if self.order == ExpansionOrder::BitReversed {
            let mut power = self.extra;
            for (index, output) in output.iter_mut().enumerate() {
                *output = self
                    .coefficients
                    .get(index)
                    .map_or(PastaField::ZERO, |value| {
                        value.mul(&scales.map_or(power, |scales| scales[index].mul(&self.extra)))
                    });
                power = power.mul(&shift);
            }
            super::stages::StageKernel {
                plan: expansion.base,
                inverse: false,
                dif: true,
                scale: InverseScale::Normalized,
                codelet: Codelet::Radix2,
                twiddles: None,
                output_order: InputOrder::BitReversed,
                factor,
            }
            .run(output, 2, self.options.max_tasks, self.executor);
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
