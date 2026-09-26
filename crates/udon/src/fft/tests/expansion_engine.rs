//! Test-only expansion driver over explicit transform schedules.
//!
//! The production [`ExpansionPlan`](crate::fft::execution::ExpansionPlan) resolves
//! its geometry from [`ExecutionOptions`](crate::exec::ExecutionOptions).
//! This driver instead takes caller-selected tile, column, and task counts so
//! tests can exercise every schedule directly, and it shares the transform
//! kernels with the production paths.

use crate::exec::Executor;
use crate::fft::{
    Codelet, CoefficientView, ElementOrder, EvaluationLayout, EvaluationView, Expansion,
    ExpansionOrder, ExpansionScaleNormalization, FftError, InverseScale, PastaField, PrimeModulus,
    Residue, Strategy, Transform, assert_length, bit_reverse, check_domain_size, check_field_count,
    check_prefix, check_scratch, min, stages::StageKernel, transform::Run,
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
/// The default is [`Self::SERIAL`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) struct ExpansionStrategy {
    /// Maximum number of concurrently executing residue transforms.
    pub(super) max_residue_tasks: usize,
    /// Scheduling and scratch bounds within every base-size transform.
    pub(super) transform: Strategy,
}

impl ExpansionStrategy {
    /// A sequential schedule with whole-transform tiles and no scratch or joins.
    pub(super) const SERIAL: Self = Self {
        max_residue_tasks: 1,
        transform: Strategy::SERIAL,
    };

    /// Scratch field count for coefficient expansion or a short product.
    ///
    /// Applies to both Pasta fields, with or without prepared tables and residue
    /// scales. Whole-transform tiles need no scratch. Other counts follow the
    /// current execution implementation.
    ///
    /// Size limits and errors are those of [`Domain::for_size`](crate::fft::Domain::for_size). Returns
    /// [`FftError::InvalidLayout`] if `base_size > extended_size`,
    /// [`FftError::InvalidExecution`] for invalid options, or
    /// [`FftError::SizeOverflow`] if the combined scratch count overflows `usize`
    /// or its field slice would exceed `isize::MAX` bytes.
    pub(super) const fn coefficient_requirements(
        self,
        base_size: usize,
        extended_size: usize,
    ) -> Result<usize, FftError> {
        self.requirements(base_size, extended_size, false)
    }

    /// Scratch field count for expansion from base evaluations.
    ///
    /// Output also stores the coefficients, so no separate coefficient buffer
    /// is needed. Accepted sizes and errors are those of
    /// [`Self::coefficient_requirements`].
    pub(super) const fn evaluation_requirements(
        self,
        base_size: usize,
        extended_size: usize,
    ) -> Result<usize, FftError> {
        self.requirements(base_size, extended_size, true)
    }

    const fn requirements(
        self,
        base_size: usize,
        extended_size: usize,
        from_evaluations: bool,
    ) -> Result<usize, FftError> {
        if self.max_residue_tasks == 0 {
            return Err(FftError::InvalidExecution);
        }
        let fields = match self.transform.requirements(base_size) {
            Ok(required) => required,
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
        check_field_count(fields)
    }
}

impl Default for ExpansionStrategy {
    fn default() -> Self {
        Self::SERIAL
    }
}

impl<'a, M: PrimeModulus> Expansion<'a, M> {
    /// Required scratch field count for [`Self::coefficients_with`].
    ///
    /// Delegates to [`ExpansionStrategy::coefficient_requirements`] with the
    /// base and extended sizes. That const query also sizes arrays without
    /// constructing an expansion.
    ///
    /// Returns [`FftError::InvalidExecution`] for invalid [`ExpansionStrategy`],
    /// or [`FftError::SizeOverflow`] if the combined scratch slice would exceed
    /// `isize::MAX` bytes or its element count overflows `usize`.
    pub(super) const fn coefficient_scratch_with(
        self,
        options: ExpansionStrategy,
    ) -> Result<usize, FftError> {
        options.coefficient_requirements(self.base.domain().size(), self.extended.size())
    }

    /// Expands a coefficient prefix into residue-major coset evaluations.
    ///
    /// Accepts ordinary coefficients or a
    /// [`CoefficientView`](CoefficientView) in increasing degree order;
    /// an empty prefix represents zero. A view's normalization is applied
    /// during initialization, with table usage as
    /// described by [`Self::with_scales`]. Input is preserved. Execution uses
    /// [`ExpansionStrategy`] across and within residues; scratch must meet
    /// [`Self::coefficient_scratch_with`].
    ///
    /// Returns [`FftError::InvalidPrefix`] if the prefix exceeds the base size. Panics
    /// before writes unless output has the extended size and scratch meets the
    /// requirement. Options and storage limits have the errors of
    /// [`Self::coefficient_scratch_with`].
    pub(super) fn coefficients_with<'input, E: Executor>(
        self,
        coefficients: impl Into<CoefficientView<'input, M>>,
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
        check_scratch(required, scratch.len());
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
            options.max_residue_tasks.min(self.layout().residues()),
            &mut scratch[..required],
        );
        Ok(())
    }

    /// Required scratch field count for [`Self::evaluations_with`].
    ///
    /// Delegates to [`ExpansionStrategy::evaluation_requirements`] with the
    /// base and extended sizes. Errors follow [`Self::coefficient_scratch_with`].
    pub(super) const fn evaluation_scratch_with(
        self,
        options: ExpansionStrategy,
    ) -> Result<usize, FftError> {
        options.evaluation_requirements(self.base.domain().size(), self.extended.size())
    }

    /// Preserves base-subgroup evaluations and expands them into the coset.
    ///
    /// Input must contain exactly the base size in natural evaluation order;
    /// output must contain exactly the extended size and uses residue order.
    /// Scratch must meet [`Self::evaluation_scratch_with`]. Incorrect buffer
    /// lengths panic before writes; options and storage limits have the errors
    /// described by that query.
    ///
    /// `options.transform` controls the inverse base transform and every
    /// residue transform. After the inverse, at most `options.max_residue_tasks`
    /// of the remaining residues run concurrently. The first residue holds
    /// their coefficient input and is completed after they finish reading it.
    pub(super) fn evaluations_with<E: Executor>(
        self,
        evaluations: &[PastaField<M>],
        output: &mut [PastaField<M>],
        options: ExpansionStrategy,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        assert_length("evaluations", self.base.domain().size(), evaluations.len());
        assert_length("output", self.extended.size(), output.len());
        let required = self.evaluation_scratch_with(options)?;
        check_scratch(required, scratch.len());
        if self.extended.is_subgroup() && output.len() == evaluations.len() {
            output.copy_from_slice(evaluations);
            return Ok(());
        }
        let scratch = &mut scratch[..required];
        let transform_scratch = self.base.scratch_requirements_with(options.transform)?;
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
            options.max_residue_tasks.min(self.layout().residues() - 1),
            scratch,
        );
        if self.extended.is_subgroup() {
            first.copy_from_slice(evaluations);
            return Ok(());
        }
        let first_plan = Transform {
            domain: self.base.domain().domain().coset(),
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

    fn inverse_policy(self) -> (Run<'a, 'a, M>, PastaField<M>) {
        if self.scales.is_some() {
            if self.normalization == ExpansionScaleNormalization::Coefficients {
                (Run::inverse(&[]), PastaField::ONE)
            } else {
                (Run::INVERSE_UNSCALED, PastaField::ONE)
            }
        } else {
            (
                Run::INVERSE_UNSCALED,
                self.base.domain().domain().size_inverse(),
            )
        }
    }

    /// Evaluates a short polynomial times a supplied factor.
    ///
    /// `short` accepts ordinary coefficients or a
    /// [`CoefficientView`](CoefficientView), with ordering and scaling as
    /// in [`Self::coefficients`]. It must have between one and the base size
    /// entries. Its expanded evaluations are multiplied pointwise by `factor`.
    /// The factor must use this expansion's layout and exact ordered evaluation
    /// domain; both are checked.
    ///
    /// Scheduling and buffer contracts follow [`Self::coefficients`]. An empty or
    /// oversized prefix returns [`FftError::InvalidPrefix`]. An incompatible factor
    /// panics before writes. Recovering the full product by interpolation additionally
    /// requires its degree to be below the extended domain size.
    pub(super) fn short_product_with<'input, E: Executor>(
        self,
        short: impl Into<CoefficientView<'input, M>>,
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
        assert!(
            factor.layout() == EvaluationLayout::Residues(self.layout())
                && factor.domain().same_domain(self.extended),
            "factors must match the expansion domain and layout"
        );
        let required = self.coefficient_scratch_with(options)?;
        check_scratch(required, scratch.len());
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
            options.max_residue_tasks.min(self.layout().residues()),
            &mut scratch[..required],
        );
        Ok(())
    }

    fn coefficient_input(self, coefficients: CoefficientView<'_, M>) -> (bool, PastaField<M>) {
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

struct ResidueJobs<'a, 'b, M: PrimeModulus, E> {
    expansion: Expansion<'a, M>,
    coefficients: &'b [PastaField<M>],
    factor: Option<&'b [PastaField<M>]>,
    normalized_coefficients: bool,
    order: ExpansionOrder,
    extra: PastaField<M>,
    options: Strategy,
    executor: &'b E,
}

impl<M: PrimeModulus, E: Executor> ResidueJobs<'_, '_, M, E> {
    fn run(
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
                    bit_reverse(block, self.expansion.layout().residues().ilog2())
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

    fn residue(
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
        let shift = expansion.residue_shift(residue);
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
                StageKernel {
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
                if self.extra.reduce() == PastaField::<M>::ONE.reduce() {
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
                    power = power.mul(&shift.shift());
                }
            }
            StageKernel {
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

impl<'a, M: PrimeModulus> Residue<'a, M> {
    /// Required scratch field count for this residue's output order.
    ///
    /// Validates options through [`Strategy::requirements`] for the base
    /// size, with the same errors. Natural output uses that scratch count;
    /// bit-reversed output needs no scratch.
    pub(super) const fn scratch_requirements_with(
        self,
        options: Strategy,
    ) -> Result<usize, FftError> {
        let required = match options.requirements(self.expansion.base.domain().size()) {
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        Ok(if matches!(self.order, ElementOrder::BitReversed) {
            0
        } else {
            required
        })
    }
    /// Evaluates a coefficient prefix into a reusable base-sized output.
    ///
    /// For base size `n`, input contains `0..=n` natural-order coefficients,
    /// with omitted coefficients treated as zero. Accepts ordinary coefficients
    /// or a [`CoefficientView`], with scaling and table usage as in
    /// [`Expansion::coefficients`]. Input is preserved. Output has exactly `n`
    /// fields and uses the order selected by [`Expansion::residue`].
    /// Scratch must meet [`Self::scratch_requirements_with`], even for empty input.
    ///
    /// Returns [`FftError::InvalidPrefix`] for an oversized input. Option errors
    /// follow [`Self::scratch_requirements_with`]. Incorrect output or scratch
    /// lengths panic before writes. The module's [working-storage rules](super)
    /// apply.
    pub(super) fn coefficients_with<'input, E: Executor>(
        self,
        input: impl Into<CoefficientView<'input, M>>,
        output: &mut [PastaField<M>],
        options: Strategy,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let input = input.into();
        let (normalized_coefficients, extra) = self.expansion.coefficient_input(input);
        let input = input.as_slice();
        check_prefix(input.len(), 0, self.expansion.base.domain().size())?;
        assert_length("output", self.expansion.base.domain().size(), output.len());
        let required = self.scratch_requirements_with(options)?;
        check_scratch(required, scratch.len());
        ResidueJobs {
            expansion: self.expansion,
            coefficients: input,
            factor: None,
            normalized_coefficients,
            order: if self.order == ElementOrder::Natural {
                ExpansionOrder::Residues
            } else {
                ExpansionOrder::BitReversed
            },
            extra,
            options,
            executor,
        }
        .residue(output, self.residue, self.residue, &mut scratch[..required]);
        Ok(())
    }
}
