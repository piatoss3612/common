use super::transform::Run;
use super::{
    ElementOrder, Executor, FftError, PastaField, PrimeModulus, ScratchRequirements, Strategy,
    Transform, assert_length,
};

/// Meaning of an interpolation class's current working storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClassState {
    /// Initialized evaluations in the class's declared order.
    Evaluations,
    /// Natural-order polynomial coefficients from a completed inverse.
    Coefficients,
    /// Storage consumed by sum-only interpolation or an interrupted transform.
    ///
    /// No individual polynomial is promised. Reduced representations follow
    /// the module's [working-storage rules](super).
    Consumed,
}

pub(super) struct Class<'a, M: PrimeModulus> {
    pub(super) plan: Transform<'a, M>,
    pub(super) values: &'a mut [PastaField<M>],
    pub(super) order: ElementOrder,
    pub(super) state: ClassState,
}

impl<M: PrimeModulus> core::fmt::Debug for Class<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Class")
            .field("plan", &self.plan)
            .field("values", &self.values)
            .field("order", &self.order)
            .field("state", &self.state)
            .finish()
    }
}

impl<'a, M: PrimeModulus> Class<'a, M> {
    pub fn new(
        plan: Transform<'a, M>,
        values: &'a mut [PastaField<M>],
        order: ElementOrder,
    ) -> Self {
        assert_length("values", plan.domain().size(), values.len());
        Self {
            plan,
            values,
            order,
            state: ClassState::Evaluations,
        }
    }
    pub(super) const fn check_evaluations(&self) -> Result<(), FftError> {
        if matches!(self.state, ClassState::Evaluations) {
            Ok(())
        } else {
            Err(FftError::InvalidClassState)
        }
    }

    // Entry points validate before mutation. Sum interpolation can then merge
    // evaluations into consumed storage before calling this inverse.
    pub(super) fn inverse<E: Executor>(
        &mut self,
        lifts: &[Class<'_, M>],
        options: Strategy,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        self.state = ClassState::Consumed;
        if self.order == ElementOrder::Natural {
            self.plan.permute(self.values);
        }
        let required = self.plan.scratch_requirements_with(options)?.field_elements;
        self.plan.run(
            self.values,
            options,
            executor,
            &mut scratch[..required],
            Run::inverse(lifts),
        );
        self.order = ElementOrder::Natural;
        self.state = ClassState::Coefficients;
        Ok(())
    }
}

const fn include_lift(
    options: Strategy,
    required: ScratchRequirements,
    output_size: usize,
    lift_size: usize,
) -> Result<ScratchRequirements, FftError> {
    if lift_size > output_size {
        return Err(FftError::InvalidClass);
    }
    match options.requirements(lift_size) {
        Ok(lift) if lift.field_elements > required.field_elements => Ok(lift),
        Ok(_) => Ok(required),
        Err(error) => Err(error),
    }
}

pub const fn interpolation_scratch<M: PrimeModulus>(
    output: &Class<'_, M>,
    lifts: &[Class<'_, M>],
    options: Strategy,
) -> Result<ScratchRequirements, FftError> {
    if let Err(error) = output.check_evaluations() {
        return Err(error);
    }
    let mut required = match output.plan.scratch_requirements_with(options) {
        Ok(required) => required,
        Err(error) => return Err(error),
    };
    let mut index = 0;
    while index < lifts.len() {
        if let Err(error) = lifts[index].check_evaluations() {
            return Err(error);
        }
        required = match include_lift(
            options,
            required,
            output.values.len(),
            lifts[index].values.len(),
        ) {
            Ok(required) => required,
            Err(error) => return Err(error),
        };
        index += 1;
    }
    Ok(required)
}

pub fn interpolate_classes<M: PrimeModulus, E: Executor>(
    output: &mut Class<'_, M>,
    lifts: &mut [Class<'_, M>],
    options: Strategy,
    executor: &E,
    scratch: &mut [PastaField<M>],
) -> Result<(), FftError> {
    interpolation_scratch(output, lifts, options)?.check(scratch.len());
    for lift in lifts.iter_mut() {
        lift.inverse(&[], options, executor, scratch)?;
    }
    output.inverse(lifts, options, executor, scratch)
}
