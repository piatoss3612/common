use super::{
    CosetDomain, EvaluationLayout, EvaluationView, Executor, ExpansionOrder,
    ExpansionScaleNormalization, ExpansionScales, FftError, PastaField, PrimeModulus,
    ResidueLayout, Transform, assert_length, check_prefix,
};
use super::{ElementOrder, ExpansionStorage, InputSupport, StorageLayout, run::ExpansionPlan};
use crate::exec::ExecutionOptions;

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
/// [table and working-storage contracts](super) apply, including trusted tables
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
/// let extended = Domain::new(3).unwrap().coset();
/// let expansion = Expansion::new(base, extended, None).unwrap();
/// let coefficients = [Fp::from_u64(3), Fp::from_u64(2)]; // 3 + 2*x
/// let mut output = [Fp::ZERO; 8];
/// expansion.coefficients(
///     &coefficients, &mut output, ExecutionOptions::default(),
///     &SerialExecutor, &mut [],
/// ).unwrap();
/// let view = EvaluationView::bind(
///     &output, extended, EvaluationLayout::Residues(expansion.layout()),
/// );
/// for row in 0..extended.size() {
///     let root_power = extended.domain().root().pow_u64(row as u64);
///     let point = extended.shift().mul(&root_power);
///     let expected = coefficients[0].add(&coefficients[1].mul(&point));
///     assert_eq!(view.get(row).map(|value| value.reduce()), Some(expected.reduce()));
/// }
/// ```
#[derive(Clone, Copy)]
pub struct Expansion<'a, M: PrimeModulus> {
    pub(super) base: Transform<'a, M>,
    pub(super) extended: CosetDomain<M>,
    pub(super) scales: Option<&'a [PastaField<M>]>,
    pub(super) normalization: ExpansionScaleNormalization,
    layout: ResidueLayout,
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
        if !base.domain().is_subgroup() || base.domain().size() > extended.size() {
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
            Some(scales) => Ok(expansion.with_scales(scales)),
            None => Ok(expansion),
        }
    }

    /// Replaces the scaling table after checking its domain and base size.
    ///
    /// Panics unless the scales describe this expansion's base and extended domains.
    /// Coefficient input, including an unscaled
    /// [`CoefficientView`](super::CoefficientView), can use either convention.
    /// Initialization accounts for the view's source base size and any factor already
    /// present in the table. [`Self::evaluations`] accepts either convention.
    /// [`super::run::ExpansionPlan`] also accounts for the retained coefficient scale
    /// during residue initialization.
    ///
    /// Table contents follow [`ExpansionScales`]' preparation and binding
    /// contract. Attachment does not rescan entries.
    pub fn with_scales(mut self, scales: ExpansionScales<'a, M>) -> Self {
        assert!(
            scales.base_size() == self.base.domain().size()
                && scales.domain().same_domain(self.extended),
            "scales must match the expansion"
        );
        self.scales = Some(scales.as_slice());
        self.normalization = scales.normalization();
        self
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
    /// Panics unless `values` has the extended domain's size. Contents are not checked;
    /// see [`EvaluationView::bind`].
    pub fn view(self, values: &[PastaField<M>]) -> EvaluationView<'_, M> {
        EvaluationView::bind(
            values,
            self.extended,
            EvaluationLayout::Residues(self.layout),
        )
    }
    pub(super) fn check(
        self,
        coefficients: usize,
        min: usize,
        output: usize,
    ) -> Result<(), FftError> {
        assert_length("output", self.extended.size(), output);
        check_prefix(coefficients, min, self.base.domain().size())
    }

    /// Preferred scratch field count for coefficient expansion and short products.
    ///
    /// Counts initialized field elements. Builds [`ExpansionPlan::new`] for full
    /// natural-order coefficients and returns [`ExpansionPlan::scratch_fields`],
    /// with planning errors from construction. Direct execution can adapt to
    /// smaller or empty scratch.
    pub fn coefficient_scratch(self, options: ExecutionOptions) -> Result<usize, FftError> {
        self.scratch(ExpansionStorage::Coefficients, options)
    }

    /// Preferred scratch field count for expansion from base evaluations.
    ///
    /// Counts initialized field elements. Builds [`ExpansionPlan::new`] with
    /// [`ExpansionStorage::ReuseOutput`] and returns the plan's
    /// [`ExpansionPlan::scratch_fields`], with planning errors from construction.
    /// Direct execution can adapt to smaller or empty scratch.
    pub fn evaluation_scratch(self, options: ExecutionOptions) -> Result<usize, FftError> {
        self.scratch(ExpansionStorage::ReuseOutput, options)
    }

    fn scratch(
        self,
        storage: ExpansionStorage,
        options: ExecutionOptions,
    ) -> Result<usize, FftError> {
        Ok(self
            .plan(storage, InputSupport::Full, options)?
            .scratch_fields())
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
    /// Returns [`FftError::InvalidPrefix`] for oversized input. Planning errors follow
    /// [`Self::coefficient_scratch`]. Panics for an incorrect output length. These
    /// checks precede writes; executor panic behavior follows the module's
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
        .with_coefficient_scale(input.normalization_factor())
        .execute(input.as_slice(), output, &mut [], None, scratch, executor);
        Ok(())
    }

    /// Preserves base evaluations and writes residue-major extended evaluations.
    ///
    /// Input must contain exactly the base size's evaluations in natural order; output
    /// must have the extended domain's size. Incorrect lengths panic before writes.
    /// Output also holds intermediate coefficients. Scratch adapts as in
    /// [`Self::coefficients`], with planning errors from [`Self::evaluation_scratch`].
    /// Its error and panic guarantees apply here too.
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
        .execute(input, output, &mut [], None, scratch, executor);
        Ok(())
    }

    /// Fuses coefficient expansion with multiplication by extended-domain factors.
    ///
    /// The coefficient prefix contains one through the base size's entries and is
    /// preserved. Scaling, output length, scratch, and panic behavior follow
    /// [`Self::coefficients`]. Factors must use this expansion's extended domain and
    /// [`Self::layout`], or this panics before writes. An empty or oversized prefix
    /// returns [`FftError::InvalidPrefix`]. All returned errors precede writes.
    /// Interpolating the full product requires its degree to be below the extended
    /// domain size; that degree bound is not checked.
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
        assert!(
            factor.layout() == EvaluationLayout::Residues(self.layout)
                && factor.domain().same_domain(self.extended),
            "factors must match the expansion domain and layout"
        );
        self.plan(
            ExpansionStorage::Coefficients,
            InputSupport::Prefix(input.as_slice().len()),
            options.for_scratch::<PastaField<M>>(scratch.len()),
        )?
        .with_coefficient_scale(input.normalization_factor())
        .execute(
            input.as_slice(),
            output,
            &mut [],
            Some(factor.as_slice()),
            scratch,
            executor,
        );
        Ok(())
    }

    pub(super) fn residue_shift(self, residue: usize) -> super::factors::ForwardShift<M> {
        use super::factors::ForwardShift;
        if residue == 0 {
            ForwardShift::for_domain(self.extended)
        } else {
            ForwardShift::Residue {
                shift: self
                    .extended
                    .shift()
                    .mul(&self.extended.domain().root().pow_u64(residue as u64)),
                inverse: self.extended.inverse_shift().mul(
                    &self
                        .extended
                        .domain()
                        .inverse_root()
                        .pow_u64(residue as u64),
                ),
            }
        }
    }
}
