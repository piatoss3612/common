use core::ops::Range;

use super::CosetDomain;
use crate::field::{PastaField, PrimeModulus, ReductionState, batch_invert_scaled, fill_powers};

/// An invalid basis-index range or insufficient Lagrange evaluation storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum LagrangeError {
    /// Natural indices must satisfy `start <= end <= size`; ranges do not wrap.
    InvalidRange {
        /// First requested natural index.
        start: usize,
        /// Exclusive end of the request.
        end: usize,
        /// Number of domain nodes.
        size: usize,
    },
    /// The supplied buffer cannot hold the requested range.
    BufferTooShort {
        /// Required number of field elements.
        required: usize,
        /// Supplied number of field elements.
        actual: usize,
    },
}

impl core::fmt::Display for LagrangeError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::InvalidRange { start, end, size } => {
                write!(f, "Lagrange range {start}..{end} is outside 0..{size}")
            }
            Self::BufferTooShort { required, actual } => write!(
                f,
                "Lagrange evaluation requires {required} fields, received {actual}"
            ),
        }
    }
}

impl core::error::Error for LagrangeError {}

#[derive(Clone, Copy)]
enum Finish<M: PrimeModulus> {
    Scale(PastaField<M>),
    Delta(Option<usize>),
}

/// Completes a prepared range after caller-managed denominator inversion.
///
/// [`CosetDomain::prepare_lagrange`] creates this descriptor and writes its
/// denominators into caller storage. Invert those entries with
/// [`batch_invert_groups`](crate::field::batch_invert_groups), optionally sharing
/// the batch with other queries or arithmetic, then call [`Self::complete`].
/// The descriptor retains only a field scale or node position and an element
/// count; it borrows no buffer and allocates nothing. Keep it paired with its
/// prepared entries. Completion checks length, not mathematical contents.
#[derive(Clone, Copy)]
pub struct LagrangeCompletion<M: PrimeModulus> {
    count: usize,
    finish: Finish<M>,
}

impl<M: PrimeModulus> core::fmt::Debug for LagrangeCompletion<M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        let mut debug = f.debug_struct("LagrangeCompletion");
        debug.field("count", &self.count);
        match self.finish {
            Finish::Scale(scale) => debug.field("scale", &scale),
            Finish::Delta(index) => debug.field("node_offset", &index),
        };
        debug.finish()
    }
}

impl<M: PrimeModulus> LagrangeCompletion<M> {
    /// Number of prepared denominators and resulting basis values.
    pub const fn value_count(self) -> usize {
        self.count
    }

    /// Replaces unscaled denominator inverses with the requested basis values.
    ///
    /// The first [`Self::value_count`] entries must be the inverses of this
    /// descriptor's prepared denominators, in their original order, with zero
    /// entries preserved. No scale may be applied during inversion. This content
    /// requirement is the caller's responsibility; incorrect entries give
    /// incorrect results. Empty ranges, singleton domains, and exact node hits need
    /// no inverses; their prepared zeros may be passed through a shared batch.
    ///
    /// A short buffer returns [`LagrangeError::BufferTooShort`] before mutation.
    /// Only the required prefix is overwritten; its tail is untouched. Work is
    /// linear in the requested count, with constant auxiliary space, no scratch,
    /// allocation, or inversion. Arithmetic is variable-time.
    pub fn complete(self, inverses: &mut [PastaField<M>]) -> Result<(), LagrangeError> {
        check_buffer(self.count, inverses.len())?;
        let values = &mut inverses[..self.count];
        match self.finish {
            Finish::Scale(scale) => {
                for value in values {
                    *value = value.mul(&scale);
                }
            }
            Finish::Delta(index) => {
                values.fill(PastaField::ZERO);
                if let Some(index) = index {
                    values[index] = PastaField::ONE;
                }
            }
        }
        Ok(())
    }
}

impl<M: PrimeModulus> CosetDomain<M> {
    /// Evaluates a natural-index range of this domain's Lagrange basis at `point`.
    ///
    /// With nodes `x_i = shift * root^i`, basis polynomial `L_i` has degree below
    /// `size` and satisfies `L_i(x_j) = 1` for `i = j` and zero otherwise. Output
    /// entry `j` receives `L_(range.start + j)(point)`. Both the subgroup and
    /// ZETA coset are supported, and the point may be reduced or loose.
    ///
    /// Ranges must satisfy `start <= end <= size`, including empty ranges at
    /// either boundary. No wrapping or bit reversal is implicit; split a request
    /// that crosses the last node into two calls. Output needs at least
    /// `end - start` elements, and its unused tail is untouched. All validation
    /// precedes writes to output or scratch; errors leave both unchanged. Rust's
    /// exclusive borrows keep writable buffers disjoint from each other and the
    /// point.
    ///
    /// Exact node hits produce a Kronecker delta, or all zeros if the hit is
    /// outside the range, without inversion or scratch writes. A singleton
    /// domain's only basis value is one at every point. Empty ranges do no work.
    /// Otherwise [`batch_invert_scaled`] uses one inversion with at least one
    /// scratch element per requested value; smaller scratch bounds each batch,
    /// and empty scratch inverts each denominator separately. Scratch contents
    /// are ignored and entries beyond the requested count remain untouched.
    ///
    /// Work outside inversion is `O(log(size) + range.len())`. Execution is
    /// serial, variable-time, allocation-free, and uses constant auxiliary space
    /// plus the supplied buffers. No tables are retained. To combine denominator
    /// batches with other operations, use [`Self::prepare_lagrange`].
    ///
    /// ```
    /// use zakura_udon::{fft::Domain, field::Fp};
    /// let domain = Domain::new(3)?.coset();
    /// let node = domain.shift().mul(&domain.domain().root());
    /// let mut values = [Fp::ZERO; 3];
    /// domain.evaluate_lagrange(&node, 0..3, &mut values, &mut [])?;
    /// assert_eq!(values.map(Fp::reduce), [Fp::ZERO, Fp::ONE, Fp::ZERO]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn evaluate_lagrange(
        self,
        point: &PastaField<M, impl ReductionState>,
        range: Range<usize>,
        output: &mut [PastaField<M>],
        scratch: &mut [PastaField<M>],
    ) -> Result<(), LagrangeError> {
        let completion = self.prepare_lagrange(point, range, output)?;
        match completion.finish {
            Finish::Scale(scale) => {
                batch_invert_scaled(&mut output[..completion.count], &scale, scratch);
            }
            Finish::Delta(_) => completion.complete(output)?,
        }
        Ok(())
    }

    /// Prepares denominators for a requested Lagrange basis range.
    ///
    /// Range, point, output order, validation, and untouched-tail rules are those
    /// of [`Self::evaluate_lagrange`]. The required prefix receives
    /// `(point / shift) * root^(-i) - 1` for each requested natural index `i`.
    /// Empty ranges, singleton domains, and exact domain-node hits instead write
    /// zeros, contributing no nonzero factors to a shared inversion batch.
    ///
    /// Invert the prepared prefix without scaling, preserving zeros, then pass
    /// it to [`LagrangeCompletion::complete`]. The descriptor supplies the scale
    /// or exact node result. Preparation performs no inversion and needs no
    /// scratch; work is `O(log(size) + range.len())`. All arithmetic is
    /// variable-time, with no allocation or retained tables.
    ///
    /// ```
    /// use zakura_udon::{fft::Domain, field::{Fp, batch_invert_groups}};
    /// let domain = Domain::new(2)?.subgroup();
    /// let mut first = [Fp::ZERO; 2];
    /// let mut second = [Fp::ZERO; 2];
    /// let a = domain.prepare_lagrange(&<Fp>::ZERO, 0..2, &mut first)?;
    /// let b = domain.prepare_lagrange(&<Fp>::ONE, 0..2, &mut second)?;
    /// batch_invert_groups(&mut [&mut first[..], &mut second[..]],
    ///                     &mut [Fp::ZERO; 4]);
    /// a.complete(&mut first)?;
    /// b.complete(&mut second)?;
    /// assert_eq!(first.map(Fp::reduce), [domain.domain().size_inverse().reduce(); 2]);
    /// assert_eq!(second.map(Fp::reduce), [Fp::ONE, Fp::ZERO]);
    /// # Ok::<(), Box<dyn std::error::Error>>(())
    /// ```
    pub fn prepare_lagrange(
        self,
        point: &PastaField<M, impl ReductionState>,
        range: Range<usize>,
        denominators: &mut [PastaField<M>],
    ) -> Result<LagrangeCompletion<M>, LagrangeError> {
        if range.start > range.end || range.end > self.size() {
            return Err(LagrangeError::InvalidRange {
                start: range.start,
                end: range.end,
                size: self.size(),
            });
        }
        let count = range.end - range.start;
        check_buffer(count, denominators.len())?;
        let denominators = &mut denominators[..count];
        if count == 0 || self.size() == 1 {
            denominators.fill(PastaField::ZERO);
            return Ok(LagrangeCompletion {
                count,
                finish: Finish::Delta((count != 0).then_some(0)),
            });
        }

        let relative = if self.is_subgroup() {
            point.into_loose()
        } else {
            point.mul(&self.inverse_shift())
        };
        let mut power = relative;
        for _ in 0..self.domain().log_size() {
            power = power.square();
        }
        let vanishing = power.sub(&PastaField::<M>::ONE);
        let step = self.domain().inverse_root();
        // Domain construction bounds indices by 2^32, so the exponent fits u64.
        let first = relative.mul(&step.pow_u64(range.start as u64));
        fill_powers(first, step, denominators);
        let finish = if vanishing.is_zero() {
            let index = denominators.iter().position(PastaField::is_one);
            denominators.fill(PastaField::ZERO);
            Finish::Delta(index)
        } else {
            for value in denominators {
                *value = value.sub(&PastaField::<M>::ONE);
            }
            // For t = point/shift, factoring the node out of the usual
            // (t^n - 1)*root^i / (n*(t - root^i)) formula leaves one shared
            // numerator and the denominators t*root^(-i) - 1.
            Finish::Scale(vanishing.mul(&self.domain().size_inverse()))
        };
        Ok(LagrangeCompletion { count, finish })
    }
}

fn check_buffer(required: usize, actual: usize) -> Result<(), LagrangeError> {
    if actual < required {
        Err(LagrangeError::BufferTooShort { required, actual })
    } else {
        Ok(())
    }
}
