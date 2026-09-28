use super::{Loose, PastaField, PrimeModulus, ReductionState};

/// Invalid constant-prefix input or insufficient materialization storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ConstantPrefixError {
    /// The explicit tail exceeds the total sequence length.
    TailTooLong {
        /// Total sequence length.
        length: usize,
        /// Number of supplied tail values.
        tail: usize,
    },
    /// The output cannot hold the complete sequence.
    OutputTooShort {
        /// Required number of field elements.
        required: usize,
        /// Supplied number of field elements.
        actual: usize,
    },
}

impl core::fmt::Display for ConstantPrefixError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::TailTooLong { length, tail } => {
                write!(f, "tail of length {tail} exceeds sequence length {length}")
            }
            Self::OutputTooShort { required, actual } => {
                write!(f, "sequence requires {required} fields, received {actual}")
            }
        }
    }
}

impl core::error::Error for ConstantPrefixError {}

/// A repeated field value followed by a borrowed explicit tail.
///
/// The sequence has `len - tail.len()` copies of `constant`, followed by the
/// actual values in `tail`, in order. Tail entries are values, not differences
/// from the constant. An empty tail represents a constant sequence; a full
/// tail represents arbitrary input and makes the constant immaterial. Empty
/// sequences are accepted. No domain, coefficient order, or sparsity claim is
/// implicit in this descriptor.
///
/// Construction only checks the tail bound. Storage consists of one field,
/// a slice borrow and a length, with no allocation or arithmetic preparation.
/// Both field reduction states are accepted. [`Self::write_values`] materializes
/// the sequence for ordinary slice operations. FFT consumers include
/// [`CosetDomain::interpolate_constant_prefix`](crate::fft::CosetDomain::interpolate_constant_prefix)
/// and [`ConstantPrefixExpansion`](crate::fft::ConstantPrefixExpansion).
#[derive(Clone, Copy)]
pub struct ConstantPrefix<'a, M: PrimeModulus, S: ReductionState = Loose> {
    length: usize,
    constant: PastaField<M>,
    tail: &'a [PastaField<M, S>],
}

impl<M: PrimeModulus, S: ReductionState> core::fmt::Debug for ConstantPrefix<'_, M, S> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ConstantPrefix")
            .field("length", &self.length)
            .field("constant", &self.constant)
            .field("tail", &self.tail)
            .finish()
    }
}

impl<'a, M: PrimeModulus, S: ReductionState> ConstantPrefix<'a, M, S> {
    /// Borrows `tail`, rejecting a tail longer than `length`.
    pub fn new(
        length: usize,
        constant: &PastaField<M, impl ReductionState>,
        tail: &'a [PastaField<M, S>],
    ) -> Result<Self, ConstantPrefixError> {
        if tail.len() > length {
            return Err(ConstantPrefixError::TailTooLong {
                length,
                tail: tail.len(),
            });
        }
        Ok(Self {
            length,
            constant: constant.into_loose(),
            tail,
        })
    }

    /// Total number of represented values.
    pub const fn len(self) -> usize {
        self.length
    }

    /// Whether the sequence contains no values.
    pub const fn is_empty(self) -> bool {
        self.length == 0
    }

    /// Number of implicit copies of [`Self::constant`].
    pub const fn prefix_len(self) -> usize {
        self.length - self.tail.len()
    }

    /// The value repeated before the explicit tail.
    pub const fn constant(self) -> PastaField<M> {
        self.constant
    }

    /// Actual tail values in sequence order.
    pub const fn tail(self) -> &'a [PastaField<M, S>] {
        self.tail
    }

    /// Writes the complete sequence to the output prefix, preserving its tail.
    ///
    /// Insufficient output returns an error before mutation. Input is preserved;
    /// Rust borrows require disjoint writable storage. This takes linear work
    /// in [`Self::len`], with no scratch or allocation.
    pub fn write_values(self, output: &mut [PastaField<M>]) -> Result<(), ConstantPrefixError> {
        if output.len() < self.length {
            return Err(ConstantPrefixError::OutputTooShort {
                required: self.length,
                actual: output.len(),
            });
        }
        let (prefix, tail) = output[..self.length].split_at_mut(self.prefix_len());
        prefix.fill(self.constant);
        for (out, value) in tail.iter_mut().zip(self.tail) {
            *out = value.into_loose();
        }
        Ok(())
    }
}
