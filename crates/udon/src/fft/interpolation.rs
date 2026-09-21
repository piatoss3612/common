use super::transform::Run;
use super::{
    ElementOrder, ExecutionOptions, Executor, FftError, PastaField, Plan, PrimeModulus,
    ScratchRequirements, check_length,
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
    pub(super) plan: Plan<'a, M>,
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
        plan: Plan<'a, M>,
        values: &'a mut [PastaField<M>],
        order: ElementOrder,
    ) -> Result<Self, FftError> {
        check_length("values", plan.domain().size(), values.len())?;
        Ok(Self {
            plan,
            values,
            order,
            state: ClassState::Evaluations,
        })
    }
    #[cfg(test)]
    pub fn values(&self) -> &[PastaField<M>] {
        self.values
    }
    #[cfg(test)]
    pub const fn order(&self) -> ElementOrder {
        self.order
    }

    #[cfg(test)]
    pub const fn state(&self) -> ClassState {
        self.state
    }

    pub(super) const fn check_evaluations(&self) -> Result<(), FftError> {
        if matches!(self.state, ClassState::Evaluations) {
            Ok(())
        } else {
            Err(FftError::InvalidClassState)
        }
    }

    #[cfg(test)]
    pub fn scatter(&mut self, start: usize, values: &[PastaField<M>]) -> Result<(), FftError> {
        self.scatter_strided(start, 1, values)
    }

    #[cfg(test)]
    pub fn scatter_strided(
        &mut self,
        start: usize,
        stride: usize,
        values: &[PastaField<M>],
    ) -> Result<(), FftError> {
        self.check_evaluations()?;
        if stride == 0 {
            return Err(FftError::InvalidLayout);
        }
        if let Some(last) = values.len().checked_sub(1) {
            let end = last
                .checked_mul(stride)
                .and_then(|offset| start.checked_add(offset))
                .ok_or(FftError::SizeOverflow)?;
            if end >= self.values.len() {
                return Err(FftError::InvalidLayout);
            }
        } else if start > self.values.len() {
            return Err(FftError::InvalidLayout);
        }
        for (index, value) in values.iter().enumerate() {
            let row = start + index * stride;
            let destination = match self.order {
                ElementOrder::Natural => row,
                ElementOrder::BitReversed => self.plan.reversed(row),
            };
            self.values[destination] = *value;
        }
        Ok(())
    }
}

#[cfg(test)]
impl ExecutionOptions {
    pub(crate) const fn interpolation_requirements(
        self,
        output_size: usize,
        lift_sizes: &[usize],
    ) -> Result<ScratchRequirements, FftError> {
        let mut required = match self.requirements(output_size) {
            Ok(required) => required,
            Err(error) => return Err(error),
        };
        let mut index = 0;
        while index < lift_sizes.len() {
            required = match include_lift(self, required, output_size, lift_sizes[index]) {
                Ok(required) => required,
                Err(error) => return Err(error),
            };
            index += 1;
        }
        Ok(required)
    }
}

const fn include_lift(
    options: ExecutionOptions,
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
    options: ExecutionOptions,
) -> Result<ScratchRequirements, FftError> {
    if let Err(error) = output.check_evaluations() {
        return Err(error);
    }
    let mut required = match output.plan.scratch_requirements(options) {
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
    options: ExecutionOptions,
    executor: &E,
    scratch: &mut [PastaField<M>],
) -> Result<(), FftError> {
    interpolation_scratch(output, lifts, options)?.check(scratch.len())?;
    for lift in lifts.iter_mut() {
        lift.state = ClassState::Consumed;
        if lift.order == ElementOrder::Natural {
            lift.plan.permute(lift.values);
        }
        let required = lift.plan.scratch_requirements(options)?.field_elements;
        lift.plan.run(
            lift.values,
            options,
            executor,
            &mut scratch[..required],
            Run::inverse(&[]),
        );
        lift.order = ElementOrder::Natural;
        lift.state = ClassState::Coefficients;
    }
    output.state = ClassState::Consumed;
    if output.order == ElementOrder::Natural {
        output.plan.permute(output.values);
    }
    let required = output.plan.scratch_requirements(options)?.field_elements;
    output.plan.run(
        output.values,
        options,
        executor,
        &mut scratch[..required],
        Run::inverse(lifts),
    );
    output.order = ElementOrder::Natural;
    output.state = ClassState::Coefficients;
    Ok(())
}
