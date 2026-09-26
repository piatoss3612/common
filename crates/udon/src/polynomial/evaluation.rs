use crate::field::{PastaField, PrimeModulus, ReductionState, fill_powers};

#[cfg(test)]
#[path = "tests/evaluation.rs"]
mod tests;

/// Insufficient evaluation storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationError {
    /// The plan cannot evaluate the full coefficient slice.
    PowersTooShort {
        /// Required number of retained powers, excluding the constant power.
        required: usize,
        /// Supplied number of retained powers.
        actual: usize,
    },
    /// The output cannot hold one value per input polynomial.
    OutputTooShort {
        /// Required number of output values.
        required: usize,
        /// Supplied number of output values.
        actual: usize,
    },
}

impl core::fmt::Display for EvaluationError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::PowersTooShort { required, actual } => write!(
                f,
                "evaluation requires {required} retained powers, received {actual}"
            ),
            Self::OutputTooShort { required, actual } => write!(
                f,
                "evaluation requires {required} output values, received {actual}"
            ),
        }
    }
}

impl core::error::Error for EvaluationError {}

/// Evaluates ascending coefficients at `point` using Horner's rule.
///
/// Empty input evaluates to zero. This serial operation takes linear work,
/// constant auxiliary space, and no preparation or caller scratch. For repeated
/// evaluations at the same point, [`EvaluationPlan`] can share retained powers.
pub fn evaluate<M: PrimeModulus, S: ReductionState>(
    coefficients: &[PastaField<M, S>],
    point: &PastaField<M, impl ReductionState>,
) -> PastaField<M> {
    horner(
        coefficients.iter(),
        PastaField::ZERO,
        |value| value.into_loose(),
        |value, coefficient| value.mul_add(point, coefficient),
    )
}

// Both APIs traverse the same ascending coefficients from the leading term.
fn horner<C, F>(
    mut coefficients: impl DoubleEndedIterator<Item = C>,
    zero: F,
    initial: impl FnOnce(C) -> F,
    step: impl FnMut(F, C) -> F,
) -> F {
    let Some(last) = coefficients.next_back() else {
        return zero;
    };
    coefficients.rev().fold(initial(last), step)
}

/// Evaluates an iterator of ascending coefficients at `point` by Horner's rule.
///
/// This consumer API requires `traits`. Like [`evaluate`], empty input evaluates
/// to zero and no allocation or scratch is needed. Each step dispatches through
/// `Field::mul_add`, including the native Pasta implementation. The iterator must
/// support reading from its highest coefficient back to the constant term.
#[cfg(feature = "traits")]
pub fn evaluate_iter<'a, F: crate::field::Field, I>(coefficients: I, point: F) -> F
where
    I: IntoIterator<Item = &'a F>,
    I::IntoIter: DoubleEndedIterator,
{
    horner(
        coefficients.into_iter(),
        F::ZERO,
        |value| *value,
        |value, coefficient| value.mul_add(&point, coefficient),
    )
}

/// Borrows powers for repeated polynomial evaluation at one field point.
///
/// The retained entries are `point^(i+1)` in ascending exponent order: the
/// constant power is omitted. A table of `k` entries supports any coefficient
/// slice of length at most `k + 1`, including empty slices. Trailing zero
/// coefficients count toward this limit. Points zero and one are supported.
/// The field is fixed by `M`; both reduced and loose coefficients are accepted.
///
/// [`Self::prepare`] constructs the powers; [`Self::bind`] borrows existing
/// entries whose mathematical contents are the caller's responsibility.
/// Preparation, binding, and evaluation allocate nothing, use constant auxiliary
/// space, and need no caller scratch or executor. Storage consists of the supplied
/// field slice and this handle's point and slice descriptor. Evaluation shares
/// deferred product reductions and adds the constant coefficient directly. All
/// arithmetic is variable-time.
///
/// Retaining powers amortizes their preparation across multiple polynomials.
/// [`evaluate`] provides Horner evaluation without retained storage; choosing
/// points, grouping queries, and deciding when to prepare belong to the caller.
///
/// ```
/// use zakura_udon::{field::Fp, polynomial::{evaluate, EvaluationPlan}};
///
/// let point = <Fp>::from_u64(2);
/// let a = [3, 4, 5].map(<Fp>::from_u64);
/// let b = [<Fp>::from_u64(7)];
/// let mut powers = [Fp::ZERO; 2];
/// let plan = EvaluationPlan::prepare(&point, &mut powers);
/// let mut output = [<Fp>::ZERO; 2];
/// plan.evaluate_many(&[&a, &b], &mut output)?;
/// assert_eq!(output.map(Fp::reduce), [31, 7].map(|n| <Fp>::from_u64(n).reduce()));
/// assert_eq!(plan.evaluate(&a)?.reduce(), evaluate(&a, &point).reduce());
/// # Ok::<(), zakura_udon::polynomial::EvaluationError>(())
/// ```
#[derive(Clone, Copy)]
pub struct EvaluationPlan<'a, M: PrimeModulus> {
    point: PastaField<M>,
    powers: &'a [PastaField<M>],
}

impl<M: PrimeModulus> core::fmt::Debug for EvaluationPlan<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EvaluationPlan")
            .field("point", &self.point)
            .field("powers", &self.powers)
            .finish()
    }
}

impl<'a, M: PrimeModulus> EvaluationPlan<'a, M> {
    /// Number of retained field elements for a coefficient extent.
    ///
    /// Empty and constant polynomials need no entries. This element count is
    /// `coefficient_count.saturating_sub(1)` and cannot overflow. Conversion to
    /// bytes and allocation remain the caller's responsibility.
    pub const fn power_count(coefficient_count: usize) -> usize {
        coefficient_count.saturating_sub(1)
    }

    /// Fills all supplied storage with successive positive powers of `point`.
    ///
    /// Initial storage values are ignored. Supply exactly
    /// [`Self::power_count`] entries for the largest planned coefficient extent,
    /// or a larger table to serve later queries. Preparation takes linear work.
    pub fn prepare(
        point: &PastaField<M, impl ReductionState>,
        powers: &'a mut [PastaField<M>],
    ) -> Self {
        let point = point.into_loose();
        fill_powers(point, point, powers);
        Self { point, powers }
    }

    /// Borrows existing positive powers for `point` with constant work.
    ///
    /// The caller must supply `point^(i+1)` at each index `i`; contents are not
    /// checked, and incorrect entries give incorrect results. Loose
    /// representations are accepted. Empty tables serve empty or constant
    /// polynomials. Binding performs no arithmetic and needs no scratch.
    pub const fn bind(
        point: &PastaField<M, impl ReductionState>,
        powers: &'a [PastaField<M>],
    ) -> Self {
        Self {
            point: point.into_loose(),
            powers,
        }
    }

    /// Point shared by every evaluation using this plan.
    pub const fn point(&self) -> PastaField<M> {
        self.point
    }

    /// Retained positive powers in ascending exponent order.
    pub const fn powers(&self) -> &'a [PastaField<M>] {
        self.powers
    }

    /// Evaluates one ascending coefficient slice, returning zero for empty input.
    ///
    /// Returns [`EvaluationError::PowersTooShort`] if the coefficient extent
    /// requires more entries than the plan retains. Work is linear in the
    /// coefficient extent, independent of unused retained entries.
    pub fn evaluate(
        &self,
        coefficients: &[PastaField<M, impl ReductionState>],
    ) -> Result<PastaField<M>, EvaluationError> {
        self.check_length(coefficients.len())?;
        Ok(self.evaluate_unchecked(coefficients))
    }

    /// Writes one evaluation per input slice, in the same order as `inputs`.
    ///
    /// Input slices may differ in length and share storage with each other or
    /// the retained powers. Each must fit this plan. Output must hold at least
    /// `inputs.len()` values; only that prefix is overwritten. All dimensions
    /// are checked before writing, so errors leave output unchanged. Rust's
    /// exclusive output borrow prevents aliasing with inputs or powers.
    ///
    /// Execution is serial. Work is proportional to the number of inputs plus
    /// their total coefficient extent; input lengths need not sum to a `usize`.
    pub fn evaluate_many<S: ReductionState>(
        &self,
        inputs: &[&[PastaField<M, S>]],
        output: &mut [PastaField<M>],
    ) -> Result<(), EvaluationError> {
        if output.len() < inputs.len() {
            return Err(EvaluationError::OutputTooShort {
                required: inputs.len(),
                actual: output.len(),
            });
        }
        for input in inputs {
            self.check_length(input.len())?;
        }
        for (input, out) in inputs.iter().zip(output) {
            *out = self.evaluate_unchecked(input);
        }
        Ok(())
    }

    fn check_length(&self, coefficient_count: usize) -> Result<(), EvaluationError> {
        let required = Self::power_count(coefficient_count);
        if required > self.powers.len() {
            return Err(EvaluationError::PowersTooShort {
                required,
                actual: self.powers.len(),
            });
        }
        Ok(())
    }

    fn evaluate_unchecked(
        &self,
        coefficients: &[PastaField<M, impl ReductionState>],
    ) -> PastaField<M> {
        match coefficients {
            [] => PastaField::ZERO,
            [constant] => constant.into_loose(),
            [constant, linear] => linear.mul_add(&self.point, constant),
            [constant, rest @ ..] => {
                PastaField::sum_of_products_slice(rest, &self.powers[..rest.len()]).add(constant)
            }
        }
    }
}

/// Evaluates `1 + X + ... + X^(terms - 1)` at `ratio`.
///
/// Doubling the covered block uses `O(log terms)` multiplications. Zero terms
/// give zero.
#[cfg(feature = "traits")]
pub fn geometric_sum<F: crate::field::Field>(mut ratio: F, mut terms: usize) -> F {
    let mut block = F::ONE;
    let mut sum = F::ZERO;
    let mut step = F::ONE;
    while terms > 0 {
        if terms & 1 == 1 {
            sum += step * block;
            step *= ratio;
        }
        block += ratio * block;
        ratio = ratio.square();
        terms >>= 1;
    }
    sum
}
