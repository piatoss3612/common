//! Batched safegcd inversion: signed-62 integer rows and Montgomery coefficients.
//!
//! Keep the divstep scale, fused row update, and final correction together;
//! their shared invariant is documented on [`PastaField::invert_safegcd`].

use super::montgomery::reduce_once;
use super::word::{adc, mac};
use super::{PastaField, PrimeModulus, Reduced, ReductionState};

use crate::field::pasta::safegcd::{
    SAFEGCD_BATCHES, SIGNED62_MASK, divsteps_62, to_signed62, update_fg,
};

#[cfg(test)]
mod tests;

#[cfg(test)]
std::thread_local! {
    static INVERSION_COUNT: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn count_inversions(f: impl FnOnce()) -> usize {
    INVERSION_COUNT.with(|count| {
        let before = count.get();
        f();
        count.get() - before
    })
}

impl<M: PrimeModulus, S: ReductionState> PastaField<M, S> {
    /// Returns the multiplicative inverse, or `None` for zero.
    ///
    /// The safegcd loop terminates according to the input.
    pub fn invert(&self) -> Option<PastaField<M>> {
        if self.is_zero() {
            None
        } else {
            #[cfg(test)]
            INVERSION_COUNT.with(|count| count.set(count.get() + 1));
            Some(self.invert_safegcd())
        }
    }

    /// Modular inverse via batched Bernstein–Yang safegcd.
    ///
    /// Requires nonzero `self`. The loop stops once the full-width `g` row is
    /// zero, within the bound documented on [`SAFEGCD_BATCHES`], and uses the
    /// correction for the number of completed batches.
    ///
    /// The core inverts the canonical integer `x`; the Bézout coefficients `d`
    /// and `e` are tracked as field elements (initialized `0` and `1`) whose
    /// per-batch matrix update is one fused signed row pass ending in a single
    /// Montgomery reduction round. Each batch therefore scales the rows by
    /// `2^62` (the matrix scale) and `2^-64` (the reduction round). After all
    /// divsteps `g = 0` and `f = ±1`, with the invariant
    /// `d ≡ f · x^-1 · 2^-(2 · batches)`, so
    /// `x^-1 = sign(f) · d · 2^(2 · batches)`. The final multiplication uses
    /// the prederived Montgomery correction for the completed batch count.
    fn invert_safegcd(&self) -> PastaField<M> {
        let mut f = M::MODULUS_SIGNED62;
        let mut g = to_signed62(&self.canonical_limbs());
        let mut d = PastaField::<M, Reduced>::ZERO;
        let mut e = PastaField::<M, Reduced>::ONE;
        let mut delta = 1i64;
        let mut completed_batches = 0;

        for batch in 0..SAFEGCD_BATCHES {
            let f_low = f[0] as u64 | ((f[1] as u64) << 62);
            let g_low = g[0] as u64 | ((g[1] as u64) << 62);
            let (next_delta, matrix) = divsteps_62(delta, f_low, g_low);
            delta = next_delta;
            (f, g) = update_fg(&f, &g, matrix);
            let terminal = g == [0; 5];

            let [u, v, q, r] = matrix;
            let prev_d = d;
            d = PastaField::bezout_row_update(u, &prev_d, v, &e);
            if !terminal {
                // Once g is zero, only d contributes to the inverse; the
                // second coefficient row has no remaining consumer.
                e = PastaField::bezout_row_update(q, &prev_d, r, &e);
            }
            completed_batches = batch + 1;
            if terminal {
                break;
            }
        }

        debug_assert_eq!(g, [0i64; 5], "safegcd did not converge (g != 0)");
        debug_assert!(
            f == [1, 0, 0, 0, 0]
                || f == [
                    SIGNED62_MASK,
                    SIGNED62_MASK,
                    SIGNED62_MASK,
                    SIGNED62_MASK,
                    -1
                ],
            "safegcd final f is not ±1"
        );

        let correction = PastaField::<M, Reduced>::from_montgomery(
            M::SAFEGCD_CORRECTIONS[completed_batches - 1],
        );
        let corrected = d.mul(&correction);
        if f[4] < 0 { corrected.neg() } else { corrected }
    }
}

impl<M: PrimeModulus> PastaField<M, Reduced> {
    /// Computes `(u * lhs + v * rhs) * 2^-64` in Montgomery form with one
    /// fused signed limb pass and a single Montgomery reduction round.
    ///
    /// Each divstep row has `|u| + |v| <= 2^62`. This keeps signed limb
    /// products and their sum inside `i128`. The operands are reduced, so
    /// the signed sum has magnitude below `2^62 * p`; adding
    /// the `p * 2^63` offset (`≡ 0 mod p`) keeps the five-limb accumulator
    /// nonnegative and below `2^64 * p`, and the reduction round returns a
    /// value below `2p`.
    #[inline]
    fn bezout_row_update(u: i64, lhs: &Self, v: i64, rhs: &Self) -> Self {
        debug_assert!(u.unsigned_abs() + v.unsigned_abs() <= 1 << 62);
        let offset = M::SAFEGCD_OFFSET;
        let (u, v) = (i128::from(u), i128::from(v));
        let mut limbs = [0u64; 5];
        let mut carry = 0i128;
        for index in 0..4 {
            let acc = carry
                + u * (lhs.limbs[index] as i128)
                + v * (rhs.limbs[index] as i128)
                + (offset[index] as i128);
            limbs[index] = acc as u64;
            carry = acc >> 64;
        }
        let top = carry + (offset[4] as i128);
        debug_assert!(0 <= top && top <= u64::MAX as i128);
        limbs[4] = top as u64;

        let multiplier = limbs[0].wrapping_mul(M::MONTGOMERY_INV);
        let (cancelled, carry) = mac(limbs[0], multiplier, M::MODULUS[0], 0);
        debug_assert_eq!(cancelled, 0);
        let (r0, carry) = mac(limbs[1], multiplier, M::MODULUS[1], carry);
        let (r1, carry) = adc(limbs[2], 0, carry);
        let (r2, carry) = adc(limbs[3], multiplier << 62, carry);
        let (r3, overflow) = adc(limbs[4], multiplier >> 2, carry);
        debug_assert_eq!(overflow, 0);
        Self::from_montgomery(reduce_once::<M>([r0, r1, r2, r3]))
    }
}
