use super::{
    Class, ClassState, Executor, FftError, PastaField, PrimeModulus, Strategy,
    interpolation_scratch,
};

fn add_coefficients<M: PrimeModulus>(output: &mut Class<'_, M>, lifts: &[Class<'_, M>]) {
    for lift in lifts
        .iter()
        .filter(|lift| lift.state == ClassState::Coefficients)
    {
        for (value, coefficient) in output.values.iter_mut().zip(lift.values.iter()) {
            *value = value.add(coefficient);
        }
    }
}

fn merge_evaluations<M: PrimeModulus>(output: &mut Class<'_, M>, lift: &mut Class<'_, M>) {
    output.state = ClassState::Consumed;
    lift.state = ClassState::Consumed;
    for (index, value) in output.values.iter_mut().enumerate() {
        let source = if output.order == lift.order {
            index
        } else {
            lift.plan.reversed(index)
        };
        *value = value.add(&lift.values[source]);
    }
}

pub fn interpolate_sum<M: PrimeModulus, E: Executor>(
    output: &mut Class<'_, M>,
    lifts: &mut [Class<'_, M>],
    options: Strategy,
    executor: &E,
    scratch: &mut [PastaField<M>],
) -> Result<(), FftError> {
    interpolation_scratch(output, lifts, options)?.check(scratch.len());
    output.state = ClassState::Consumed;
    for lift in lifts.iter_mut() {
        if lift.plan.domain().same_domain(output.plan.domain()) {
            merge_evaluations(output, lift);
        }
    }
    for index in 0..lifts.len() {
        let (before, after) = lifts.split_at_mut(index + 1);
        let class = &mut before[index];
        if class.state != ClassState::Evaluations {
            continue;
        }
        for lift in after.iter_mut() {
            if lift.state == ClassState::Evaluations
                && class.plan.domain().same_domain(lift.plan.domain())
            {
                merge_evaluations(class, lift);
            }
        }
        class.inverse(&[], options, executor, scratch)?;
    }
    output.inverse(&[], options, executor, scratch)?;
    add_coefficients(output, lifts);
    for lift in lifts {
        lift.state = ClassState::Consumed;
    }
    Ok(())
}
