//! Canonical compressed encodings, independent of stored Montgomery layout.

use super::{AffinePoint, PastaCurve, Point, curve_rhs};
use crate::field::PastaField;

impl<C: PastaCurve> AffinePoint<C> {
    /// Encodes this point in 32 canonical compressed bytes.
    ///
    /// The low 255 bits hold canonical little-endian `x`; the high bit of the
    /// last byte is the parity of canonical `y` (set for odd). This encoding is
    /// independent of the Montgomery layout used by [`bento::Pod`].
    pub fn to_bytes(&self) -> [u8; 32] {
        let mut bytes = self.x.to_bytes();
        bytes[31] |= u8::from(self.y.is_odd()) << 7;
        bytes
    }

    /// Decodes a canonical compressed nonidentity point.
    ///
    /// Uses the format from [`Self::to_bytes`]. Returns `None` for the all-zero
    /// identity encoding or any encoding rejected by [`Point::from_bytes`].
    pub fn from_bytes(bytes: [u8; 32]) -> Option<Self> {
        Point::<C>::from_bytes(bytes)?.0
    }
}

impl<C: PastaCurve> Point<C> {
    /// Encodes identity as 32 zero bytes or a nonidentity point in compressed form.
    ///
    /// Nonidentity points use [`AffinePoint::to_bytes`].
    pub fn to_bytes(&self) -> [u8; 32] {
        self.as_affine().map_or([0; 32], AffinePoint::to_bytes)
    }

    /// Decodes identity or a canonical compressed point.
    ///
    /// Uses the format from [`Self::to_bytes`], accepting all-zero bytes as
    /// identity. For other inputs, returns `None` if `x` is at least the
    /// base-field modulus, `x³ + 5` has no square root, or neither root has the
    /// encoded parity. There is no affine point at `x = 0` on either Pasta
    /// curve, so identity's encoding is unambiguous; setting its sign bit is
    /// rejected.
    ///
    /// Both Pasta groups have prime order, so curve membership also establishes
    /// subgroup membership. No additional subgroup check is needed.
    pub fn from_bytes(mut bytes: [u8; 32]) -> Option<Self> {
        if bytes == [0; 32] {
            return Some(Self::IDENTITY);
        }
        let odd = bytes[31] >> 7 != 0;
        bytes[31] &= 0x7f;
        let x = PastaField::from_bytes(bytes)?;
        let mut y = curve_rhs(&x).sqrt()?;
        // Zero is the only root whose negation does not change parity.
        if y.is_zero() && odd {
            return None;
        }
        if y.is_odd() != odd {
            y = y.neg();
        }
        // Canonical decoding and the square root already establish reduced,
        // on-curve coordinates; the public coordinate constructor rechecks both.
        debug_assert_eq!(y.square(), curve_rhs(&x));
        Some(
            AffinePoint {
                x,
                y,
                marker: core::marker::PhantomData,
            }
            .to_point(),
        )
    }
}
