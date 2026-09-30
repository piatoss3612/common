use super::{
    CoefficientView, ElementOrder, Executor, Expansion, FftError, InverseScale, PastaField,
    PrimeModulus, assert_length,
};
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

/// Input liveness and coefficient storage for [`super::execution::ExpansionPlan`].
///
/// Workspace and disposable-input policies retain coefficients in their
/// selected buffer. Its mathematical scale uses the base subgroup size, as
/// defined by [`InverseScale`]; output evaluations always have their ordinary
/// values. Scale handling follows [`super::execution::ExpansionPlan`].
/// Input order and support are selected when constructing that plan.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpansionStorage {
    /// Preserve full coefficients or a natural prefix with a zero suffix.
    ///
    /// A coefficient scale can be supplied to [`super::execution::ExpansionPlan`].
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
        if residue >= self.layout().residues() {
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
///
/// For residue `s` and residue count `r`, natural output row `k` evaluates the
/// extended domain's row `s + r*k`. The order selected by [`Expansion::residue`]
/// describes its physical positions.
#[derive(Clone, Copy)]
pub struct Residue<'a, M: PrimeModulus> {
    pub(super) expansion: Expansion<'a, M>,
    pub(super) residue: usize,
    pub(super) order: ElementOrder,
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
    /// Preferred scratch field count for this residue under the resource limits.
    ///
    /// Counts initialized field elements. Resolves a full preserved coefficient
    /// input through [`super::execution::FftPlan::new`], with its planning errors. Direct
    /// execution adapts to smaller or empty scratch without changing the output's
    /// size or order.
    pub fn scratch_requirements(self, options: ExecutionOptions) -> Result<usize, FftError> {
        Ok(self
            .plan(super::InputSupport::Full, options)?
            .retained_fields())
    }

    fn plan(
        self,
        support: super::InputSupport,
        options: ExecutionOptions,
    ) -> Result<super::execution::FftPlan<'a, M>, FftError> {
        super::execution::FftPlan::new(
            self.expansion.base,
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
    /// Returns [`FftError::InvalidPrefix`] for oversized input. Planning errors follow
    /// [`Self::scratch_requirements`]. An incorrect output length panics before writes.
    /// Returned errors also precede writes; executor panics follow the module's
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
        assert_length("output", self.expansion.base.domain().size(), output.len());
        self.plan(
            super::InputSupport::Prefix(input.as_slice().len()),
            options.for_scratch::<PastaField<M>>(scratch.len()),
        )?
        .with_residue_scales(self.expansion, self.residue, input.normalization_factor())
        .execute(Some(input.as_slice()), output, None, scratch, executor);
        Ok(())
    }
}
