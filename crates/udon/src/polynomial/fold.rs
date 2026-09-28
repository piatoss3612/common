use crate::field::{PastaField, PrimeModulus, ProductSum, ReductionState};

#[cfg(test)]
#[path = "tests/fold.rs"]
mod tests;

/// Invalid dimensions for a weighted coefficient fold.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FoldError {
    /// The number of weights differs from the number of input slices.
    WeightCount,
    /// The output cannot hold the longest input slice.
    OutputTooShort {
        /// Required number of output coefficients.
        required: usize,
        /// Supplied number of output coefficients.
        actual: usize,
    },
}

impl core::fmt::Display for FoldError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::WeightCount => f.write_str("one weight is required per input slice"),
            Self::OutputTooShort { required, actual } => write!(
                f,
                "coefficient fold requires {required} output elements, received {actual}"
            ),
        }
    }
}

impl core::error::Error for FoldError {}

/// Writes a weighted sum of coefficient slices and returns its extent.
///
/// For each written position `j`, the result is
/// `output[j] = sum(weights[i] * inputs[i][j])`. Missing input coefficients
/// contribute zero. The extent is the longest input length, including inputs
/// with zero weight and trailing zero coefficients; no inputs give extent zero.
/// Only `output[..extent]` is overwritten. Any remaining output is unchanged.
///
/// There must be exactly one weight per input, including empty inputs, and
/// output must hold the entire extent. Both checks precede all writes; errors
/// leave output unchanged. Inputs may share storage with each other or with
/// weights. The mutable output borrow must be disjoint from all inputs and
/// weights, as enforced by Rust's borrowing rules.
///
/// Execution is serial, requires no preparation or caller scratch, and uses
/// constant auxiliary stack space. Work is at most proportional to the number
/// of inputs times the output extent, plus input validation. Products across
/// inputs share deferred reductions; zero and unit weights avoid multiplication.
///
/// ```
/// use zakura_udon::{field::Fp, polynomial::fold_weighted};
///
/// let a = [<Fp>::from_u64(2), <Fp>::from_u64(3)];
/// let b = [<Fp>::from_u64(5)];
/// let weights = [<Fp>::ONE, <Fp>::from_u64(4)];
/// let mut output = [<Fp>::from_u64(99); 3];
/// let extent = fold_weighted(&[&a, &b], &weights, &mut output).unwrap();
/// assert_eq!(extent, 2);
/// assert_eq!(output.map(Fp::reduce), [22, 3, 99].map(|n| <Fp>::from_u64(n).reduce()));
/// ```
pub fn fold_weighted<M: PrimeModulus, S: ReductionState, W: ReductionState>(
    inputs: &[&[PastaField<M, S>]],
    weights: &[PastaField<M, W>],
    output: &mut [PastaField<M>],
) -> Result<usize, FoldError> {
    if inputs.len() != weights.len() {
        return Err(FoldError::WeightCount);
    }
    let extent = inputs.iter().map(|input| input.len()).max().unwrap_or(0);
    if output.len() < extent {
        return Err(FoldError::OutputTooShort {
            required: extent,
            actual: output.len(),
        });
    }
    let output = &mut output[..extent];
    match inputs {
        [] => {}
        [input] => scale(input, &weights[0], output),
        [a, b] => fold_two(a, b, &weights[0], &weights[1], output),
        _ => fold_columns::<M, S, W, 4>(inputs, weights, output),
    }
    Ok(extent)
}

fn scale<M: PrimeModulus, S: ReductionState, W: ReductionState>(
    input: &[PastaField<M, S>],
    weight: &PastaField<M, W>,
    output: &mut [PastaField<M>],
) {
    if weight.is_zero() {
        output.fill(PastaField::ZERO);
    } else if weight.is_one() {
        for (out, value) in output.iter_mut().zip(input) {
            *out = value.into_loose();
        }
    } else {
        for (out, value) in output.iter_mut().zip(input) {
            *out = value.mul(weight);
        }
    }
}

fn fold_two<M: PrimeModulus, S: ReductionState, W: ReductionState>(
    a: &[PastaField<M, S>],
    b: &[PastaField<M, S>],
    wa: &PastaField<M, W>,
    wb: &PastaField<M, W>,
    output: &mut [PastaField<M>],
) {
    if wa.is_zero() || wb.is_zero() {
        let (input, weight) = if wa.is_zero() { (b, wb) } else { (a, wa) };
        scale(input, weight, &mut output[..input.len()]);
        output[input.len()..].fill(PastaField::ZERO);
        return;
    }
    let common = a.len().min(b.len());
    let pairs = output[..common].iter_mut().zip(a.iter().zip(b));
    match (wa.is_one(), wb.is_one()) {
        (true, true) => {
            for (out, (a, b)) in pairs {
                *out = a.add(b);
            }
        }
        (true, false) => {
            for (out, (a, b)) in pairs {
                *out = b.mul_add(wb, a);
            }
        }
        (false, true) => {
            for (out, (a, b)) in pairs {
                *out = a.mul_add(wa, b);
            }
        }
        (false, false) => {
            for (out, (a, b)) in pairs {
                *out = PastaField::sum_of_products(&[*a, *b], &[*wa, *wb]);
            }
        }
    }
    let (longer, weight) = if a.len() > b.len() { (a, wa) } else { (b, wb) };
    scale(&longer[common..], weight, &mut output[common..]);
}

fn fold_columns<M: PrimeModulus, S: ReductionState, W: ReductionState, const LANES: usize>(
    inputs: &[&[PastaField<M, S>]],
    weights: &[PastaField<M, W>],
    output: &mut [PastaField<M>],
) {
    let mut offset = 0;
    for chunk in output.chunks_mut(LANES) {
        // Each neighboring coefficient has its own carry chain. Keep the
        // accumulator's unlimited overflow folding even for repeated slices.
        let mut sums = [const { ProductSum::<M>::new() }; LANES];
        for (input, weight) in inputs.iter().zip(weights) {
            let Some(tail) = input.get(offset..) else {
                continue;
            };
            if weight.is_zero() {
                continue;
            }
            if weight.is_one() {
                for (sum, value) in sums.iter_mut().zip(tail) {
                    sum.add_term(value);
                }
            } else {
                for (sum, value) in sums.iter_mut().zip(tail) {
                    sum.add_product(value, weight);
                }
            }
        }
        for (out, sum) in chunk.iter_mut().zip(sums) {
            *out = sum.finish();
        }
        // This is bounded by output.len(); neither total input lengths nor
        // a rows-by-columns element count need to fit in usize.
        offset += chunk.len();
    }
}
