//! Digit storage shared by preparation tasks and arithmetic kernels.
//!
//! Small inputs reserve `MAX_DIGITS` bytes per term, holding either joint digits
//! or a short scalar's 16 canonical little-endian bytes, followed by zeros.
//! Booth inputs use chunks of up to `CHUNK` terms. Each chunk stores rows in
//! `(window, GLV half)` order, with one signed byte per term, including the final
//! carry window. The last chunk's rows use its actual term count as their length.
//! This layout gives each preparation task an exclusive chunk while keeping
//! each window's digits contiguous for the arithmetic kernels.

use super::Input;
use crate::{
    curve::{
        EisensteinScalar, PastaCurve, eisenstein::MAX_DIGITS, glv_decompose, scalar::centered_digit,
    },
    exec::{Executor, TaskBudget, for_each_chunk_mut},
};

pub(super) const CHUNK: usize = 256;

pub(super) const fn stride(terms: usize) -> usize {
    if terms < super::BOOTH_MIN {
        MAX_DIGITS
    } else {
        2 * super::WINDOWS
    }
}

pub(super) fn short_bits<C: PastaCurve>(input: &Input<'_, C>) -> Option<usize> {
    let mut bits = 0;
    for scalar in input.scalars {
        bits = bits.max(
            scalar
                .to_canonical_uint()
                .highest_set_bit()
                .map_or(0, |b| b + 1),
        );
    }
    if bits <= 128 { Some(bits) } else { None }
}

pub(super) fn prepare<C: PastaCurve, X: Executor>(
    input: &Input<'_, C>,
    digits: &mut [u8],
    budget: TaskBudget,
    executor: &X,
) {
    let n = input.len();
    let stride = stride(n);
    let short = n < super::BOOTH_MIN && short_bits(input).is_some();
    for_each_chunk_mut(
        digits,
        CHUNK * stride,
        budget,
        executor,
        |chunk, rows, _| {
            let len = rows.len() / stride;
            // Zero padding prevents a shorter scalar from reusing stale high digits.
            rows.fill(0);
            for (i, scalar) in input.scalars[chunk * CHUNK..chunk * CHUNK + len]
                .iter()
                .enumerate()
            {
                if n < super::BOOTH_MIN {
                    if short {
                        let bytes = scalar.to_canonical_uint().to_le_bytes();
                        rows[i * stride..i * stride + 16].copy_from_slice(&bytes[..16]);
                    } else {
                        let scalar = EisensteinScalar::<C>::new(scalar);
                        rows[i * stride..i * stride + scalar.digits().len()]
                            .copy_from_slice(scalar.digits());
                    }
                } else {
                    let (a, b) = glv_decompose::<C>(scalar);
                    for (half, value) in [a, b].into_iter().enumerate() {
                        let mut magnitude = value.unsigned_abs();
                        let mut carry = 0;
                        for window in 0..super::WINDOWS - 1 {
                            rows[(window * 2 + half) * len + i] = centered_digit(
                                (magnitude & ((1 << super::WINDOW_BITS) - 1)) as u16,
                                value < 0,
                                &mut carry,
                                super::WINDOW_BITS as u32,
                            )
                                as u8;
                            magnitude >>= super::WINDOW_BITS;
                        }
                        rows[(2 * (super::WINDOWS - 1) + half) * len + i] =
                            if value < 0 { -carry } else { carry } as u8;
                    }
                }
            }
        },
    );
}

/// Reads a signed Booth digit from a chunked row.
#[inline]
pub(super) fn digit(digits: &[u8], terms: usize, term: usize, row: usize) -> i8 {
    let start = term / CHUNK * CHUNK;
    let len = (terms - start).min(CHUNK);
    digits[start * stride(terms) + row * len + term - start] as i8
}
