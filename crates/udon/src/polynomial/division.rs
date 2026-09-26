use crate::field::{PastaField, PrimeModulus, ReductionState};

#[cfg(test)]
#[path = "tests/division.rs"]
mod tests;

/// Divides by `X - point`, storing the remainder before the quotient.
///
/// Coefficients are in ascending order. Returns the split between remainder and
/// quotient: zero for empty input, one otherwise. The remainder is the constant
/// `f(point)`, and the quotient satisfies `f(X) = (X - point) * q(X) + f(point)`.
/// Neither extent trims zero coefficients. A constant input is unchanged and
/// entirely remainder. Every point is accepted, including zero.
///
/// As with [`divide_monic_in_place`], pass only the active dividend slice to
/// preserve any storage beyond it. Repeated division can advance through the
/// quotient suffix without moving coefficients. Coefficient storage is loose;
/// the point may be loose or reduced. Execution is serial, with linear work,
/// constant auxiliary space, and no allocation, scratch, or inversion.
///
/// ```
/// use zakura_udon::{field::Fp, polynomial::divide_linear_in_place};
///
/// // 3 + 4*X + 5*X^2 = (X - 2) * (14 + 5*X) + 31.
/// let mut coefficients = [3, 4, 5].map(<Fp>::from_u64);
/// let split = divide_linear_in_place(&mut coefficients, &<Fp>::from_u64(2));
/// let (remainder, quotient) = coefficients.split_at(split);
/// assert_eq!(split, 1);
/// assert_eq!(remainder[0].reduce(), <Fp>::from_u64(31).reduce());
/// assert_eq!(quotient[0].reduce(), <Fp>::from_u64(14).reduce());
/// assert_eq!(quotient[1].reduce(), <Fp>::from_u64(5).reduce());
/// ```
pub fn divide_linear_in_place<M: PrimeModulus>(
    coefficients: &mut [PastaField<M>],
    point: &PastaField<M, impl ReductionState>,
) -> usize {
    if coefficients.len() > 1 {
        divide_rolling(coefficients, [point.into_loose()]);
    }
    coefficients.len().min(1)
}

/// Streams the quotient by `X - point` in descending coefficient order.
///
/// Input coefficients are ascending; reading them from the back lets synthetic
/// division produce the highest quotient coefficient first without allocation
/// or scratch. Empty and constant inputs yield no quotient coefficients.
/// The remainder `p(point)` is discarded, whether or not `point` is a root.
/// Use [`divide_linear_in_place`] to retain both the remainder and an ascending
/// quotient in a mutable Pasta slice.
///
/// This consumer API requires `traits`. Its recurrence uses `Field::mul_add`,
/// including Pasta's native method. Work is linear and arithmetic variable-time.
#[cfg(feature = "traits")]
pub fn divide_linear_rev<F: crate::field::Field, I>(
    coefficients: I,
    point: F,
) -> impl Iterator<Item = F>
where
    I: IntoIterator<Item = F>,
    I::IntoIter: DoubleEndedIterator,
{
    let mut coefficients = coefficients.into_iter().rev().peekable();
    let mut carry = coefficients.next().unwrap_or(F::ZERO);
    core::iter::from_fn(move || {
        let coefficient = coefficients.next()?;
        let quotient = carry;
        // The constant coefficient only affects the discarded remainder.
        if coefficients.peek().is_some() {
            carry = carry.mul_add(&point, &coefficient);
        }
        Some(quotient)
    })
}

/// A divisor that does not explicitly end in a unit leading coefficient.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum MonicDivisionError {
    /// The divisor has no coefficients.
    EmptyDivisor,
    /// The final supplied coefficient is not one.
    NonMonicDivisor,
}

impl core::fmt::Display for MonicDivisionError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::EmptyDivisor => "monic division requires a nonempty divisor",
            Self::NonMonicDivisor => "the divisor's final coefficient must be one",
        })
    }
}

impl core::error::Error for MonicDivisionError {}

/// Divides by a monic polynomial, storing the remainder before the quotient.
///
/// Both slices use ascending coefficient order. The divisor must be nonempty
/// and its final supplied coefficient must equal one; trailing zeros are not
/// trimmed. These checks precede all writes, even for an empty dividend. Errors
/// leave `coefficients` unchanged.
///
/// For dividend length `n` and divisor degree `d = divisor.len() - 1`, returns
/// `remainder_len = min(n, d)`. The prefix `coefficients[..remainder_len]` holds
/// the remainder, and the suffix holds the quotient, satisfying
/// `original(X) = divisor(X) * quotient(X) + remainder(X)`. Neither extent trims
/// zero coefficients. A short dividend (`n <= d`) is entirely remainder and is
/// unchanged. Dividing by the constant one returns zero and leaves the entire
/// dividend as quotient. Empty input returns zero.
///
/// Pass only the active dividend slice to preserve any storage beyond it.
/// Coefficient storage is loose; the divisor may be loose or reduced. Rust's
/// borrows keep writable coefficients disjoint from the divisor.
///
/// Any divisor degree fitting the supplied slices is accepted. Execution is
/// serial, takes `O(d * (n - d))` arithmetic when `0 < d < n`, and needs constant
/// auxiliary space with no allocation, caller scratch, preparation, or inversion.
///
/// ```
/// use zakura_udon::{field::Fp, polynomial::divide_monic_in_place};
///
/// // 1 + 2*X + 3*X^2 + 4*X^3 = (1 + X^2)*(3 + 4*X) - 2 - 2*X.
/// let mut coefficients = [1, 2, 3, 4].map(<Fp>::from_u64);
/// let divisor = [1, 0, 1].map(<Fp>::from_u64);
/// let split = divide_monic_in_place(&mut coefficients, &divisor).unwrap();
/// let (remainder, quotient) = coefficients.split_at(split);
/// assert_eq!(split, 2);
/// assert!(remainder.iter().all(|r| r.reduce() == <Fp>::from_i64(-2).reduce()));
/// assert_eq!(quotient[0].reduce(), <Fp>::from_u64(3).reduce());
/// assert_eq!(quotient[1].reduce(), <Fp>::from_u64(4).reduce());
/// ```
pub fn divide_monic_in_place<M: PrimeModulus, S: ReductionState>(
    coefficients: &mut [PastaField<M>],
    divisor: &[PastaField<M, S>],
) -> Result<usize, MonicDivisionError> {
    let Some((leading, lower)) = divisor.split_last() else {
        return Err(MonicDivisionError::EmptyDivisor);
    };
    if !leading.is_one() {
        return Err(MonicDivisionError::NonMonicDivisor);
    }
    let degree = lower.len();
    let remainder_len = coefficients.len().min(degree);
    if degree != 0 && coefficients.len() > degree {
        match lower {
            [a] => divide_rolling(coefficients, [a.neg()]),
            [a, b] => divide_rolling(coefficients, [a.neg(), b.neg()]),
            [a, b, c] => divide_rolling(coefficients, [a.neg(), b.neg(), c.neg()]),
            _ => divide_general(coefficients, lower),
        }
    }
    Ok(remainder_len)
}

fn divide_general<M: PrimeModulus, S: ReductionState>(
    coefficients: &mut [PastaField<M>],
    lower: &[PastaField<M, S>],
) {
    let degree = lower.len();
    for index in (0..coefficients.len() - degree).rev() {
        let negative_quotient = coefficients[index + degree].neg();
        for (coefficient, divisor) in coefficients[index..index + degree].iter_mut().zip(lower) {
            *coefficient = negative_quotient.mul_add(divisor, coefficient);
        }
    }
}

fn divide_rolling<M: PrimeModulus, const D: usize>(
    coefficients: &mut [PastaField<M>],
    negative_divisor: [PastaField<M>; D],
) {
    let quotient_len = coefficients.len() - D;
    let mut state: [PastaField<M>; D] = coefficients[quotient_len..].try_into().unwrap();
    // The state carries the next D high coefficients of the remaining dividend.
    // Each step consumes one untouched input and writes one final quotient term;
    // only the final low remainder needs to be written back from the state.
    for index in (0..quotient_len).rev() {
        let quotient = state[D - 1];
        for j in (1..D).rev() {
            state[j] = quotient.mul_add(&negative_divisor[j], &state[j - 1]);
        }
        state[0] = quotient.mul_add(&negative_divisor[0], &coefficients[index]);
        coefficients[index + D] = quotient;
    }
    coefficients[..D].copy_from_slice(&state);
}
