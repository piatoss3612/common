//! Signed-62 Bernstein–Yang divsteps for moduli below 2^255.
//!
//! Field implementations provide the modular Bézout row update.

/// Maximum number of 62-step batches needed for a 255-bit modulus.
///
/// Twelve batches provide 744 divsteps, sufficient by the Bernstein–Yang
/// bound; see Section 12.1 and Theorem 11.2 of
/// <https://gcd.cr.yp.to/safegcd-20190413.pdf>.
pub(super) const SAFEGCD_BATCHES: usize = 12;

/// Mask for one 62-bit safegcd limb.
pub(super) const SIGNED62_MASK: i64 = (1 << 62) - 1;

/// Repacks a little-endian integer below `2^255` into five signed 62-bit limbs.
///
/// Every limb is nonnegative here; later [`update_fg`] results keep limbs
/// `0..4` in `[0, 2^62)` while the top limb carries the sign.
#[inline]
pub(super) const fn to_signed62(limbs: &[u64; 4]) -> [i64; 5] {
    let mut result = [0i64; 5];
    let mut index = 0;
    while index < 5 {
        let bit = 62 * index;
        let word = bit / 64;
        let shift = bit % 64;
        let mut value = limbs[word] >> shift;
        if shift != 0 && word + 1 < 4 {
            value |= limbs[word + 1] << (64 - shift);
        }
        result[index] = (value & SIGNED62_MASK as u64) as i64;
        index += 1;
    }
    result
}

/// Runs 62 Bernstein–Yang divsteps on the low bits of `f` (odd) and `g`.
///
/// Returns the updated `delta` and the integer transition matrix `[u, v, q, r]`,
/// whose action is divided by `2^62` in [`update_fg`]. Each row's sum of absolute
/// values is at most `2^62`, so its entries fit in `i64`.
#[inline]
pub(super) fn divsteps_62(mut delta: i64, mut f: u64, mut g: u64) -> (i64, [i64; 4]) {
    let (mut u, mut v, mut q, mut r) = (1i64, 0i64, 0i64, 1i64);
    for _ in 0..62 {
        debug_assert_eq!(f & 1, 1);
        if delta > 0 && (g & 1) == 1 {
            delta = 1 - delta;
            let (next_f, next_g) = (g, g.wrapping_sub(f));
            let (next_u, next_v) = (q, r);
            q -= u;
            r -= v;
            u = next_u << 1;
            v = next_v << 1;
            f = next_f;
            g = next_g >> 1;
        } else if (g & 1) == 1 {
            delta += 1;
            g = g.wrapping_add(f) >> 1;
            q += u;
            r += v;
            u <<= 1;
            v <<= 1;
        } else {
            delta += 1;
            g >>= 1;
            u <<= 1;
            v <<= 1;
        }
    }
    (delta, [u, v, q, r])
}

/// Applies the divstep matrix to the full-width `f, g`, returning
/// `((u·f + v·g) >> 62, (q·f + r·g) >> 62)` exactly over signed 62-bit limbs.
#[inline]
pub(super) fn update_fg(f: &[i64; 5], g: &[i64; 5], matrix: [i64; 4]) -> ([i64; 5], [i64; 5]) {
    let [u, v, q, r] = matrix.map(i128::from);
    let (mut out_f, mut out_g) = ([0i64; 5], [0i64; 5]);
    let mut carry_f = u * i128::from(f[0]) + v * i128::from(g[0]);
    let mut carry_g = q * i128::from(f[0]) + r * i128::from(g[0]);
    debug_assert_eq!(carry_f & i128::from(SIGNED62_MASK), 0);
    debug_assert_eq!(carry_g & i128::from(SIGNED62_MASK), 0);
    carry_f >>= 62;
    carry_g >>= 62;
    for index in 1..5 {
        carry_f += u * i128::from(f[index]) + v * i128::from(g[index]);
        carry_g += q * i128::from(f[index]) + r * i128::from(g[index]);
        out_f[index - 1] = (carry_f & i128::from(SIGNED62_MASK)) as i64;
        out_g[index - 1] = (carry_g & i128::from(SIGNED62_MASK)) as i64;
        carry_f >>= 62;
        carry_g >>= 62;
    }
    out_f[4] = carry_f as i64;
    out_g[4] = carry_g as i64;
    (out_f, out_g)
}

/// Little-endian limbs of `modulus << 63`, the `≡ 0 (mod p)` offset that
/// keeps the fused signed Bézout row update nonnegative.
#[inline(always)]
pub(super) const fn bezout_offset(modulus: &[u64; 4]) -> [u64; 5] {
    [
        modulus[0] << 63,
        (modulus[1] << 63) | (modulus[0] >> 1),
        (modulus[2] << 63) | (modulus[1] >> 1),
        (modulus[3] << 63) | (modulus[2] >> 1),
        modulus[3] >> 1,
    ]
}
