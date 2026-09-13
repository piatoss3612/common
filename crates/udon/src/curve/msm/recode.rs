//! Digit storage shared by preparation tasks and arithmetic kernels.
//!
//! Inputs below [`super::BOOTH_MIN`] reserve [`MAX_DIGITS`] bytes per term,
//! holding either joint Eisenstein digits or a short scalar's 16 canonical
//! little-endian bytes, followed by zeros. A leading byte records the maximum
//! scalar bit length on the short path or 255 for joint digits, avoiding scalar
//! scans in arithmetic partitions. Empty and Booth inputs have no header.
//! Booth inputs use chunks of up to [`CHUNK`] terms. Each chunk stores rows in
//! `(window, GLV half)` order, with one signed byte per term, including the final
//! carry window. The last chunk's rows use its actual term count as their length.
//! This layout gives each preparation task an exclusive chunk while keeping
//! each window's digits contiguous for the arithmetic kernels.

use super::{CurveError, checked_count};
use crate::{
    curve::{
        EisensteinScalar, PastaCurve, eisenstein::MAX_DIGITS, glv_decompose, scalar::centered_digit,
    },
    exec::{Executor, TaskBudget, for_each_chunk_mut},
    field::PastaField,
};

pub(super) const CHUNK: usize = 256;

pub(super) const fn stride(terms: usize) -> usize {
    if terms < super::BOOTH_MIN {
        MAX_DIGITS
    } else {
        2 * super::WINDOWS
    }
}

pub(super) const fn storage_len(terms: usize) -> Result<usize, CurveError> {
    let digits = match checked_count::<u8>(terms, stride(terms)) {
        Ok(n) => n,
        Err(error) => return Err(error),
    };
    let header = (terms != 0 && terms < super::BOOTH_MIN) as usize;
    match digits.checked_add(header) {
        Some(n) => checked_count::<u8>(n, 1),
        None => Err(CurveError::SizeOverflow),
    }
}

fn short_bits<C: PastaCurve>(scalars: &[PastaField<C::Scalar>]) -> Option<usize> {
    let mut bits = 0;
    let mut long_weight = 0;
    let dense_candidate = (8..=32).contains(&scalars.len());
    for scalar in scalars {
        let integer = scalar.to_canonical_uint();
        let scalar_bits = integer.highest_set_bit().map_or(0, |b| b + 1);
        bits = bits.max(scalar_bits);
        if bits > 128 {
            return None;
        }
        if dense_candidate && scalar_bits > 64 {
            let limbs = integer.limbs();
            long_weight += (limbs[0].count_ones() + limbs[1].count_ones()) as usize;
        }
    }
    // Dense 128-bit inputs amortize joint tables from eight terms, where their
    // preparation shares inversions. Keep the change through 32 terms: gains
    // near the larger crossover were not stable in the measurements recorded in
    // docs/MSM_REVIEW_PERFORMANCE.md.
    // The length/weight guards preserve short ladders for smaller coefficients
    // and sparse high bits, even when the maximum bit length is near 128.
    if dense_candidate && bits > 120 && long_weight > 32 * scalars.len() {
        None
    } else {
        Some(bits)
    }
}

pub(super) fn prepare<C: PastaCurve, X: Executor>(
    scalars: &[PastaField<C::Scalar>],
    digits: &mut [u8],
    budget: TaskBudget,
    executor: &X,
) {
    let n = scalars.len();
    if n == 0 {
        return;
    }
    let stride = stride(n);
    let short = if n < super::BOOTH_MIN {
        short_bits::<C>(scalars)
    } else {
        None
    };
    let digits = if n < super::BOOTH_MIN {
        digits[0] = short.map_or(255, |bits| bits as u8);
        &mut digits[1..]
    } else {
        digits
    };
    for_each_chunk_mut(
        digits,
        CHUNK * stride,
        budget,
        executor,
        |chunk, rows, _| {
            let len = rows.len() / stride;
            // Zero padding prevents a shorter scalar from reusing stale high digits.
            rows.fill(0);
            for (i, scalar) in scalars[chunk * CHUNK..chunk * CHUNK + len]
                .iter()
                .enumerate()
            {
                if n < super::BOOTH_MIN {
                    if short.is_some() {
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
