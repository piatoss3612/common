use super::transform::Run;
use super::{
    ExecutionOptions, Executor, FftError, PastaField, Plan, PrimeModulus, ScratchRequirements,
    check_len,
};

/// Storage order of logical input or output positions.
///
/// For coefficients, logical position `j` is degree `j`. For evaluations it is
/// the point `shift * root^j` from the plan, with `0 <= j < size`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputOrder {
    /// Logical position `j` is at index `j`.
    Natural,
    /// Logical position `j` is at the reversal of its low `log2(size)` bits.
    BitReversed,
}

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

/// A borrowed interpolation domain and its initialized working buffer.
///
/// Scatter overwrites selected evaluations; unwritten entries retain their
/// existing values. The caller is responsible for filling the intended complete
/// evaluation vector. [`interpolate_classes`] and
/// [`super::interpolate_classes_parallel`] replace each lift with its own
/// natural-order coefficients; [`super::interpolate_sum`] consumes lift storage
/// without promising its polynomial contents. The output class contains the
/// coefficient sum on success. [`Self::state`] identifies the buffer's meaning.
/// Once interpolation begins, further scatter or interpolation calls return
/// [`FftError::InvalidClassState`], including after an execution panic. To reuse
/// storage, release the class, refill its buffer, and bind a new class.
pub struct Class<'a, M: PrimeModulus> {
    pub(super) plan: Plan<'a, M>,
    pub(super) values: &'a mut [PastaField<M>],
    pub(super) order: InputOrder,
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
    /// Binds a fully initialized input buffer after checking its length.
    ///
    /// Returns [`FftError::LengthMismatch`] unless the buffer length equals
    /// the plan's domain size. `order` describes the input's storage order;
    /// construction does not permute or validate evaluation contents.
    pub fn new(
        plan: Plan<'a, M>,
        values: &'a mut [PastaField<M>],
        order: InputOrder,
    ) -> Result<Self, FftError> {
        check_len("values", values.len(), plan.domain().size())?;
        Ok(Self {
            plan,
            values,
            order,
            state: ClassState::Evaluations,
        })
    }
    /// Borrows the current values; [`Self::state`] identifies their meaning.
    pub fn values(&self) -> &[PastaField<M>] {
        self.values
    }
    /// Storage order for the current evaluation or coefficient phase.
    ///
    /// Completed coefficients use [`InputOrder::Natural`]. The value has no
    /// result-order meaning in [`ClassState::Consumed`].
    pub const fn order(&self) -> InputOrder {
        self.order
    }

    /// Evaluation, coefficient, or consumed phase of this working buffer.
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

    /// Writes evaluations at natural positions `start..start + values.len()`.
    ///
    /// Maps positions according to [`Self::order`]. This is
    /// [`Self::scatter_strided`] with stride one, including its range checks
    /// and errors; repeated writes overwrite rather than add.
    pub fn scatter(&mut self, start: usize, values: &[PastaField<M>]) -> Result<(), FftError> {
        self.scatter_strided(start, 1, values)
    }

    /// Writes natural positions `start + i*stride`, mapping directly to storage.
    ///
    /// Returns [`FftError::InvalidLayout`] for zero stride or an out-of-range
    /// position, or [`FftError::SizeOverflow`] for index arithmetic overflow,
    /// before writing. Empty writes accept `start <= size` with nonzero stride.
    /// Repeated writes overwrite values. A bit-reversal table, if present, must
    /// satisfy [`Tables`](super::Tables)' content contract.
    /// Returns [`FftError::InvalidClassState`] if interpolation already began.
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
                InputOrder::Natural => row,
                InputOrder::BitReversed => self.plan.reversed(row),
            };
            self.values[destination] = *value;
        }
        Ok(())
    }
}

impl ExecutionOptions {
    /// Scratch for fused interpolation, given the output and lift domain sizes.
    ///
    /// This const query sizes storage without constructing plans or classes.
    /// Classes reuse scratch in sequence, so the result is the maximum of the
    /// individual transform requirements. An empty lift slice is accepted.
    ///
    /// Returns [`FftError::InvalidClass`] if a lift exceeds the output size.
    /// Other size limits and errors are those of [`Self::requirements`].
    pub const fn interpolation_requirements(
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

/// Scratch needed to interpolate `output` and add the interpolated `lifts`.
///
/// Each lift must be at most the output size. Coset shifts may differ:
/// each class describes its own polynomial. Classes reuse the same scratch in
/// sequence, with parallel work within each transform, so the requirement is
/// the maximum of their individual requirements, independent of class count.
/// Use [`ExecutionOptions::interpolation_requirements`] to query the same
/// requirement from sizes in a const context, without constructing classes.
///
/// Returns [`FftError::InvalidClass`] for an oversized lift. Other errors are
/// those of [`Plan::scratch_requirements`]. An empty lift slice is accepted.
/// Returns [`FftError::InvalidClassState`] if interpolation already began on
/// any class.
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

/// Interpolates every class, adding the lift coefficient vectors to `output`.
///
/// If `output` initially evaluates a polynomial `p` and each lift evaluates
/// `q_i`, the output becomes the coefficients of `p + sum(q_i)`. Each lift
/// retains its own interpolated coefficients. Smaller coefficient vectors are
/// implicitly zero-padded, with no degree shift or other multiplication.
///
/// Every lift must fit in the output domain. Their coset shifts may
/// differ, and an empty lift slice is accepted. All buffers finish in increasing
/// degree order, and their [`Class::order`] becomes [`InputOrder::Natural`].
/// Interpolation consumes each class's evaluation phase: another interpolation
/// or scatter returns [`FftError::InvalidClassState`]. This also applies to any
/// class whose transform began before an execution panic.
///
/// Size and execution errors are those of [`interpolation_scratch`]; a shorter
/// scratch slice returns [`FftError::ScratchTooSmall`]. Validation precedes any
/// mutation. The module's [working-storage rules](super) cover table validity,
/// scratch reuse, and buffer state after panics.
///
/// Residue expansion can scatter directly into bit-reversed interpolation
/// storage, avoiding a separate conversion to natural evaluation order:
///
/// ```
/// use zakura_udon::{
///     field::Fp,
///     fft::{
///         Class, Domain, ExecutionOptions, Expansion, ExpansionOptions, InputOrder, Plan,
///         SerialExecutor, interpolate_classes,
///     },
/// };
///
/// let base = Plan::without_tables(Domain::new(1).unwrap().subgroup());
/// let extended = Domain::new(2).unwrap().coset(Fp::from_u64(7)).unwrap();
/// let expansion = Expansion::new(base, extended, None).unwrap();
/// let coefficients = [Fp::from_u64(3), Fp::from_u64(2)];
/// let mut evaluations = [Fp::ZERO; 4];
/// expansion.coefficients(
///     &coefficients, &mut evaluations, ExpansionOptions::serial(), &SerialExecutor, &mut [],
/// ).unwrap();
/// let mut buffer = [Fp::ZERO; 4];
/// let mut output = Class::new(
///     Plan::without_tables(extended), &mut buffer, InputOrder::BitReversed,
/// ).unwrap();
/// let layout = expansion.layout();
/// for (residue, values) in evaluations.chunks_exact(layout.rows()).enumerate() {
///     output.scatter_strided(residue, layout.residues(), values).unwrap();
/// }
/// // The lift evaluates the constant polynomial 5 on a singleton subgroup.
/// let mut constant = [Fp::from_u64(5)];
/// let mut lifts = [Class::new(
///     Plan::without_tables(Domain::new(0).unwrap().subgroup()),
///     &mut constant, InputOrder::Natural,
/// ).unwrap()];
/// interpolate_classes(
///     &mut output, &mut lifts, ExecutionOptions::serial(),
///     &SerialExecutor, &mut [],
/// ).unwrap();
/// let expected = [Fp::from_u64(8), Fp::from_u64(2), Fp::ZERO, Fp::ZERO];
/// assert_eq!(output.values(), &expected);
/// assert_eq!(lifts[0].values(), &[Fp::from_u64(5)]);
/// ```
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
        if lift.order == InputOrder::Natural {
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
        lift.order = InputOrder::Natural;
        lift.state = ClassState::Coefficients;
    }
    output.state = ClassState::Consumed;
    if output.order == InputOrder::Natural {
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
    output.order = InputOrder::Natural;
    output.state = ClassState::Coefficients;
    Ok(())
}
