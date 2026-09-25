use super::{PastaField, PrimeModulus, ReductionState};

#[cfg(test)]
#[path = "tests/fractions.rs"]
mod tests;

/// Invalid buffer lengths for running products of fractions.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FractionPrefixError {
    /// Numerator and denominator counts differ.
    LengthMismatch,
    /// The fraction count plus its initial value does not fit in `usize`.
    LengthOverflow,
    /// Output cannot hold the initial value and all running products.
    OutputTooShort {
        /// Required number of field elements.
        required: usize,
        /// Supplied number of field elements.
        actual: usize,
    },
    /// Denominator scratch cannot hold one field element per fraction.
    ScratchTooShort {
        /// Required number of field elements.
        required: usize,
        /// Supplied number of field elements.
        actual: usize,
    },
}

impl core::fmt::Display for FractionPrefixError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::LengthMismatch => f.write_str("numerator and denominator counts differ"),
            Self::LengthOverflow => f.write_str("fraction output length overflows usize"),
            Self::OutputTooShort { required, actual } => write!(
                f,
                "fraction prefixes require {required} output elements, received {actual}"
            ),
            Self::ScratchTooShort { required, actual } => write!(
                f,
                "fraction prefixes require {required} scratch elements, received {actual}"
            ),
        }
    }
}

impl core::error::Error for FractionPrefixError {}

fn validate(count: usize, output: usize, scratch: usize) -> Result<usize, FractionPrefixError> {
    let extent = count
        .checked_add(1)
        .ok_or(FractionPrefixError::LengthOverflow)?;
    if output < extent {
        return Err(FractionPrefixError::OutputTooShort {
            required: extent,
            actual: output,
        });
    }
    if scratch < count {
        return Err(FractionPrefixError::ScratchTooShort {
            required: count,
            actual: scratch,
        });
    }
    Ok(extent)
}

/// Writes running products of fractions, including the supplied initial value.
///
/// For `n` fractions, writes `output[0] = initial` and
/// `output[i + 1] = output[i] * numerators[i] * inverse_or_zero(denominators[i])`,
/// returning `n + 1`. A zero denominator preserves all earlier results and
/// makes every subsequent result zero, even if its numerator is also zero.
/// A zero numerator likewise starts a zero suffix. Empty input writes only
/// the initial value.
///
/// Numerator and denominator lengths must match. Output needs at least `n + 1`
/// elements and scratch at least `n`, including when the initial value is zero.
/// All validation precedes writes; errors leave output and scratch unchanged.
/// Output beyond `n + 1` and scratch beyond `n` are untouched. Initial scratch
/// contents do not matter; its used contents are unspecified after success.
/// Inputs may share storage. Rust's mutable borrowing rules require output and
/// scratch to be disjoint from each other and from all inputs.
///
/// Execution is serial and allocation-free, with constant auxiliary stack space
/// and work linear in the input length. A nonzero initial value and nonempty,
/// nonzero denominators need one field inversion. Zero denominators can require
/// a further scan and product over the prefix before the first zero. Arithmetic
/// is variable-time, including zero positions.
///
/// ```
/// use zakura_udon::field::{Fp, fraction_prefixes};
///
/// let numerators = [2, 3, 5].map(<Fp>::from_u64);
/// let denominators = [1, 2, 0].map(<Fp>::from_u64);
/// let mut output = [<Fp>::from_u64(99); 5];
/// let mut scratch = [<Fp>::ZERO; 3];
/// let extent = fraction_prefixes(
///     &numerators, &denominators, &<Fp>::from_u64(7), &mut output, &mut scratch,
/// ).unwrap();
/// assert_eq!(extent, 4);
/// assert_eq!(output.map(Fp::reduce), [7, 14, 21, 0, 99].map(|n| <Fp>::from_u64(n).reduce()));
/// ```
pub fn fraction_prefixes<
    M: PrimeModulus,
    N: ReductionState,
    D: ReductionState,
    I: ReductionState,
>(
    numerators: &[PastaField<M, N>],
    denominators: &[PastaField<M, D>],
    initial: &PastaField<M, I>,
    output: &mut [PastaField<M>],
    scratch: &mut [PastaField<M>],
) -> Result<usize, FractionPrefixError> {
    if numerators.len() != denominators.len() {
        return Err(FractionPrefixError::LengthMismatch);
    }
    let extent = validate(denominators.len(), output.len(), scratch.len())?;
    if !initial.is_zero() {
        // Reuse the in-place recovery without another numerator buffer.
        for (out, numerator) in output.iter_mut().zip(numerators) {
            *out = numerator.into_loose();
        }
    }
    prefixes(output, denominators, &initial.into_loose(), scratch);
    Ok(extent)
}

/// Replaces numerators with running fraction products and prepends `initial`.
///
/// If `n = denominators.len()`, `values[..n]` supplies the numerators, and
/// `values[..n + 1]` receives the result defined by [`fraction_prefixes`]. The
/// initial content of `values[n]` is ignored. Returns `n + 1`; values beyond
/// that extent remain unchanged. Even empty input requires one output slot.
///
/// Scratch needs at least `n` fields. Size checks precede all writes, including
/// for a zero initial value; errors leave both mutable buffers unchanged.
/// Scratch tails, borrowing, zero behavior, and costs follow
/// [`fraction_prefixes`].
///
/// ```
/// use zakura_udon::field::{Fp, fraction_prefixes_in_place};
///
/// let mut values = [2, 3, 99].map(<Fp>::from_u64);
/// let denominators = [1, 2].map(<Fp>::from_u64);
/// fraction_prefixes_in_place(
///     &mut values, &denominators, &<Fp>::ONE, &mut [<Fp>::ZERO; 2],
/// ).unwrap();
/// assert_eq!(values.map(Fp::reduce), [1, 2, 3].map(|n| <Fp>::from_u64(n).reduce()));
/// ```
pub fn fraction_prefixes_in_place<M: PrimeModulus, D: ReductionState, I: ReductionState>(
    values: &mut [PastaField<M>],
    denominators: &[PastaField<M, D>],
    initial: &PastaField<M, I>,
    scratch: &mut [PastaField<M>],
) -> Result<usize, FractionPrefixError> {
    let extent = validate(denominators.len(), values.len(), scratch.len())?;
    prefixes(values, denominators, &initial.into_loose(), scratch);
    Ok(extent)
}

fn prefixes<M: PrimeModulus, D: ReductionState>(
    values: &mut [PastaField<M>],
    denominators: &[PastaField<M, D>],
    initial: &PastaField<M>,
    scratch: &mut [PastaField<M>],
) {
    let n = denominators.len();
    if initial.is_zero() || n == 0 {
        values[..=n].fill(PastaField::ZERO);
        values[0] = *initial;
        return;
    }

    // Numerator prefixes and paired denominator products let reverse recovery
    // write running results directly, without retaining individual fractions.
    let mut numerator = *initial;
    let mut denominator = PastaField::ONE;
    for i in (0..n - 1).step_by(2) {
        let low = numerator.mul(&values[i]);
        let pair = denominators[i].mul(&denominators[i + 1]);
        numerator = low.mul(&values[i + 1]);
        denominator = if i == 0 { pair } else { denominator.mul(&pair) };
        values[i] = low;
        values[i + 1] = numerator;
        // Retaining the high member recovers the intermediate prefix inverse
        // without separately recovering either denominator's inverse.
        scratch[i] = pair;
        scratch[i + 1] = denominators[i + 1].into_loose();
    }
    if n & 1 != 0 {
        values[n - 1] = numerator.mul(&values[n - 1]);
        scratch[n - 1] = denominators[n - 1].into_loose();
        denominator = if n == 1 {
            scratch[n - 1]
        } else {
            denominator.mul(&scratch[n - 1])
        };
    }

    let (count, mut inverse) = match denominator.invert() {
        Some(inverse) => (n, inverse),
        None => {
            // Pair products lose the low member when the high member is zero.
            // Locate the first zero in the original inputs, then restore that
            // low member if the surviving prefix ends halfway through a pair.
            let count = denominators.iter().position(PastaField::is_zero).unwrap();
            values[count + 1..=n].fill(PastaField::ZERO);
            if count == 0 {
                values[0] = *initial;
                return;
            }
            if count & 1 != 0 {
                scratch[count - 1] = denominators[count - 1].into_loose();
            }
            let mut product = scratch[0];
            for i in (2..count).step_by(2) {
                product = product.mul(&scratch[i]);
            }
            (count, product.invert().expect("nonzero denominator prefix"))
        }
    };

    if count & 1 != 0 {
        values[count] = values[count - 1].mul(&inverse);
        if count > 1 {
            inverse = inverse.mul(&scratch[count - 1]);
        }
    }
    for i in (0..count / 2).rev().map(|pair| pair * 2) {
        // inverse = 1 / product(denominators[..i + 2]). Multiplying by the
        // high denominator gives the preceding prefix inverse. Read both
        // numerator prefixes before the shifted output overwrites either.
        let low_inverse = inverse.mul(&scratch[i + 1]);
        let high = values[i + 1].mul(&inverse);
        let low = values[i].mul(&low_inverse);
        values[i + 2] = high;
        values[i + 1] = low;
        if i != 0 {
            inverse = inverse.mul(&scratch[i]);
        }
    }
    values[0] = *initial;
}
