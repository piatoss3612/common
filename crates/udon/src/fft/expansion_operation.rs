use super::{
    CoefficientView, CosetDomain, ElementOrder, Executor, Expansion, FftError, InverseScale,
    PastaField, PrimeModulus, ScratchRequirements, check_length,
};
#[cfg(test)]
use super::{Strategy, check_prefix, expansion::ResidueJobs};
use crate::exec::ExecutionOptions;

/// Persistent order of a complete expansion result.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpansionOrder {
    /// Naturally numbered residues and rows, as defined by [`super::ResidueLayout`].
    Residues,
    /// Extended rows in [`ElementOrder::BitReversed`] order.
    ///
    /// Both residue blocks and rows within each block are bit-reversed. A full
    /// inverse accepting bit-reversed input can consume this layout directly.
    BitReversed,
}

/// Input liveness and coefficient storage for [`super::run::ExpansionPlan`].
///
/// Workspace and disposable-input policies retain coefficients in their
/// selected buffer. Its mathematical scale uses the base subgroup size, as
/// defined by [`InverseScale`]; output evaluations always have their ordinary
/// values. Scale handling follows [`super::run::ExpansionPlan`].
/// Input order and support are selected when constructing that plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpansionStorage {
    /// Preserve full coefficients or a natural prefix with a zero suffix.
    ///
    /// A coefficient scale can be supplied to [`super::run::ExpansionPlan`].
    Coefficients,
    /// Preserve base evaluations, using output storage for coefficients.
    ReuseOutput,
    /// Preserve base evaluations using a separate coefficient workspace.
    CoefficientWorkspace {
        /// Mathematical scale of the retained natural-order coefficients.
        scale: InverseScale,
    },
    /// Consume base evaluations in their own buffer as coefficient storage.
    DisposableInput {
        /// Mathematical scale of the retained natural-order coefficients.
        scale: InverseScale,
    },
}

impl<'a, M: PrimeModulus> Expansion<'a, M> {
    /// Selects one naturally numbered residue for a reusable base-sized output.
    ///
    /// `order` describes positions inside that residue only. Returns
    /// [`FftError::InvalidLayout`] if `residue` is outside [`Self::layout`]'s
    /// residue count.
    pub fn residue(self, residue: usize, order: ElementOrder) -> Result<Residue<'a, M>, FftError> {
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

/// A selected residue whose output needs only the base domain's storage.
#[derive(Clone, Copy)]
pub struct Residue<'a, M: PrimeModulus> {
    expansion: Expansion<'a, M>,
    residue: usize,
    order: ElementOrder,
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

impl<'a, M: PrimeModulus> Residue<'a, M> {
    /// Scratch selected for this residue under the shared resource constraints.
    ///
    /// Resolves a full preserved coefficient input through
    /// [`super::run::FftPlan::new`]. Direct execution adapts to smaller or empty
    /// scratch without changing the output's size or order.
    pub fn scratch_requirements(
        self,
        options: ExecutionOptions,
    ) -> Result<ScratchRequirements, FftError> {
        Ok(ScratchRequirements {
            field_elements: self
                .plan(super::InputSupport::Full, options)?
                .retained_fields(),
        })
    }

    fn plan(
        self,
        support: super::InputSupport,
        options: ExecutionOptions,
    ) -> Result<super::run::FftPlan<'a, M>, FftError> {
        super::run::FftPlan::new(
            self.expansion.residue_base(self.residue),
            super::TransformRequest {
                input_storage: super::InputStorage::Preserve,
                output_order: self.order,
                support,
                ..super::TransformRequest::new(super::Direction::Forward)
            },
            super::StorageLayout::Contiguous,
            options,
        )
    }

    /// Evaluates a preserved natural coefficient prefix into this residue.
    ///
    /// Input contains zero through the base size's coefficients; omitted
    /// coefficients are zero. Scaling follows [`Expansion::coefficients`].
    /// Output has exactly the base-domain size and the selected element order.
    /// Execution adapts to scratch capacity, including empty scratch; unused
    /// scratch tails remain untouched.
    ///
    /// Returns [`FftError::InvalidPrefix`] for oversized input or
    /// [`FftError::LengthMismatch`] for an incorrect output length. Planning
    /// errors follow [`Self::scratch_requirements`]. All returned errors
    /// precede writes; panic behavior follows the module's
    /// [working-storage rules](super).
    pub fn coefficients<'input, E: Executor>(
        self,
        input: impl Into<CoefficientView<'input, M>>,
        output: &mut [PastaField<M>],
        options: crate::exec::ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let input = input.into();
        check_length("output", self.expansion.base.domain().size(), output.len())?;
        self.plan(
            super::InputSupport::Prefix(input.as_slice().len()),
            options.for_scratch::<PastaField<M>>(scratch.len()),
        )?
        .with_residue_scales(self.expansion, self.residue, input.normalization_factor())?
        .execute(Some(input.as_slice()), output, None, scratch, executor)
    }

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
    /// Validates options through [`Strategy::requirements`] for the base
    /// size, with the same errors. Natural output uses that scratch count;
    /// bit-reversed output needs no scratch.
    #[cfg(test)]
    pub(crate) const fn scratch_requirements_with(
        self,
        options: Strategy,
    ) -> Result<ScratchRequirements, FftError> {
        let required = match options.requirements(self.expansion.base.domain().size()) {
            Ok(r) => r,
            Err(e) => return Err(e),
        };
        Ok(if matches!(self.order, ElementOrder::BitReversed) {
            ScratchRequirements { field_elements: 0 }
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
    /// Scratch must meet [`Self::scratch_requirements`], including for empty input.
    ///
    /// Returns [`FftError::InvalidPrefix`] for an oversized input,
    /// [`FftError::LengthMismatch`] for an incorrect output length, or
    /// [`FftError::ScratchTooSmall`] for insufficient scratch. Option errors follow
    /// [`Self::scratch_requirements`]. The module's [validation and working-storage
    /// rules](super) apply.
    #[cfg(test)]
    pub(crate) fn coefficients_with<'input, E: Executor>(
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
        check_length("output", self.expansion.base.domain().size(), output.len())?;
        let required = self.scratch_requirements_with(options)?;
        required.check(scratch.len())?;
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
        .residue(
            output,
            self.residue,
            self.residue,
            &mut scratch[..required.field_elements],
        );
        Ok(())
    }
}
