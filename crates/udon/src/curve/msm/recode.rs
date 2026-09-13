//! Opaque recoding geometry shared by sizing, cached preparation, and kernels.
//!
//! Joint digits retain their actual length. Booth storage uses chunked rows of
//! up to 256 terms; widths through eight use one byte and wider widths use two.
//! Every Booth byte is overwritten, including partial final chunks.

use super::{CurveError, ExecutionOptions, PastaCurve, ScalarStorage, checked_count};
use crate::curve::{eisenstein, parameters::GlvParameters, scalar::centered_digit};

pub(super) const CHUNK: usize = 256;
pub(super) const JOINT_STRIDE: usize = eisenstein::MAX_DIGITS + 1;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum Geometry {
    Short(u8),
    Joint,
    Booth(u8),
}

impl Geometry {
    pub const fn for_len(n: usize, options: ExecutionOptions) -> Self {
        if options.joint_tables {
            Self::Joint
        } else if let Some(width) = options.window_bits {
            Self::Booth(width)
        } else if n < super::BOOTH_MIN {
            Self::Joint
        } else {
            // Native comparisons and controls are recorded in
            // docs/MSM_REVIEW_PERFORMANCE.md. Ties prefer fewer bytes.
            Self::Booth(if n < 192 {
                6
            } else if n < 512 {
                7
            } else if n < 4096 {
                8
            } else if n < 32768 {
                10
            } else {
                11
            })
        }
    }

    pub fn for_prepared<C: PastaCurve>(
        records: &[ScalarStorage<C>],
        options: ExecutionOptions,
    ) -> Self {
        if options.window_bits.is_some() || options.joint_tables {
            return Self::for_len(records.len(), options);
        }
        Self::for_shape(records.len(), Shape::of(records), options)
    }

    pub const fn for_shape(n: usize, shape: Shape, options: ExecutionOptions) -> Self {
        let Shape { bits, weight } = shape;
        if options.window_bits.is_some() || options.joint_tables || bits == 255 {
            return Self::for_len(n, options);
        }
        // Dense bounded rows eventually favor Booth buckets too. Keep sparse
        // rows on the short kernel; their high bit alone does not imply work.
        if n >= 512 && bits >= 32 && weight > 8 * n {
            Self::for_len(n, options)
        // Preserve the measured dense 128-bit crossover at 8--32 terms.
        } else if n >= 8 && n <= 32 && bits > 120 && weight > 32 * n {
            Self::Joint
        } else {
            Self::Short(bits)
        }
    }

    pub const fn windows(self) -> usize {
        match self {
            Self::Booth(width) => 128_usize.div_ceil(width as usize),
            _ => 1,
        }
    }
    pub const fn width(self) -> usize {
        match self {
            Self::Booth(w) => w as usize,
            _ => 0,
        }
    }
    pub const fn buckets(self) -> usize {
        1 << (self.width() - 1)
    }
    pub const fn stride(self) -> usize {
        match self {
            Self::Short(_) => 0,
            Self::Joint => JOINT_STRIDE,
            Self::Booth(w) => 2 * self.windows() * if w > 8 { 2 } else { 1 },
        }
    }
    pub const fn storage_len(self, terms: usize) -> Result<usize, CurveError> {
        checked_count::<u8>(terms, self.stride())
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Shape {
    // Maximum magnitude width, or 255 if any record needs full-width arithmetic.
    pub bits: u8,
    // Sum of magnitude population counts, meaningful only when bits != 255.
    // Retained chunks inherit the whole row's weight as a conservative bound.
    pub weight: usize,
}
impl Shape {
    pub fn of<C: PastaCurve>(records: &[ScalarStorage<C>]) -> Self {
        let mut shape = Self { bits: 0, weight: 0 };
        for record in records {
            shape.bits = shape.bits.max(record.bits);
            if shape.bits == 255 {
                break;
            }
            shape.weight += record.magnitude.count_ones() as usize;
        }
        shape
    }
}

#[derive(Clone, Copy, Debug)]
pub(super) struct Cache<'a> {
    pub geometry: Geometry,
    pub digits: &'a [u8],
}

pub(super) fn write<C: PastaCurve>(
    records: &[ScalarStorage<C>],
    geometry: Geometry,
    digits: &mut [u8],
) {
    let stride = geometry.stride();
    match geometry {
        Geometry::Short(_) => (),
        Geometry::Joint => {
            for (record, row) in records.iter().zip(digits.chunks_exact_mut(stride)) {
                let (codes, len) = eisenstein::recode(record.halves[0], record.halves[1]);
                row[..eisenstein::MAX_DIGITS].copy_from_slice(&codes);
                row[eisenstein::MAX_DIGITS] = len as u8;
            }
        }
        Geometry::Booth(width) => {
            // Both curve bounds leave room for an incoming carry in the last
            // window for every supported width. Width eight has 16 data rows.
            const {
                assert!(GlvParameters::<C>::BOOTH_WINDOWS == 16);
            }
            let shift = (geometry.windows() - 1) * usize::from(width);
            for bound in GlvParameters::<C>::BOUNDS {
                assert!((bound >> shift) + 1 < 1 << (width - 1));
            }
            let bytes = if width > 8 { 2 } else { 1 };
            for (chunk, rows) in records.chunks(CHUNK).zip(digits.chunks_mut(CHUNK * stride)) {
                for (i, record) in chunk.iter().enumerate() {
                    for (half, value) in record.halves.into_iter().enumerate() {
                        let mut magnitude = value.unsigned_abs();
                        let mut carry = 0;
                        for window in 0..geometry.windows() {
                            let digit = centered_digit(
                                (magnitude & ((1 << width) - 1)) as u16,
                                value < 0,
                                &mut carry,
                                u32::from(width),
                            );
                            let offset = ((window * 2 + half) * chunk.len() + i) * bytes;
                            rows[offset] = digit as u8;
                            if bytes == 2 {
                                rows[offset + 1] = (digit >> 8) as u8;
                            }
                            magnitude >>= width;
                        }
                        debug_assert_eq!(magnitude, 0);
                        debug_assert_eq!(carry, 0);
                    }
                }
            }
        }
    }
}

/// Visits contiguous rows once per intersecting chunk, amortizing row lookup.
pub(super) fn rows(
    digits: &[u8],
    terms: usize,
    range: core::ops::Range<usize>,
    geometry: Geometry,
    window: usize,
    mut visit: impl FnMut(usize, i16, i16),
) {
    let bytes = if geometry.width() > 8 { 2 } else { 1 };
    let stride = geometry.stride();
    let mut first = range.start;
    while first < range.end {
        let start = first / CHUNK * CHUNK;
        let len = (terms - start).min(CHUNK);
        let end = range.end.min(start + len);
        let offset = start * stride + 2 * window * len * bytes;
        let a = &digits[offset..offset + len * bytes];
        let b = &digits[offset + len * bytes..offset + 2 * len * bytes];
        for term in first..end {
            let i = (term - start) * bytes;
            let read = |row: &[u8]| {
                if bytes == 1 {
                    i16::from(row[i] as i8)
                } else {
                    i16::from_le_bytes([row[i], row[i + 1]])
                }
            };
            visit(term, read(a), read(b));
        }
        first = end;
    }
}
