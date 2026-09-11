//! Wide product accumulation and signed product differences.
//!
//! The range arguments beside each kernel explain how much reduction its
//! accumulator needs before returning a canonical field element.

use core::marker::PhantomData;

use super::montgomery::{montgomery_reduce, montgomery_reduce_unreduced, reduce_once};
use super::word::{adc, mac, multiply_wide, sbb};
use super::{PastaField, PrimeModulus};

#[cfg(test)]
#[path = "tests/products.rs"]
mod tests;

/// A sum of field products with deferred Montgomery reduction.
///
/// Use [`add_product`](Self::add_product) for products,
/// [`add_term`](Self::add_term) for individual values, and
/// [`merge`](Self::merge) to combine partial sums. [`finish`](Self::finish)
/// returns the accumulated field value; an empty sum returns zero.
/// Accumulation and merging have no term limit. Overflow is folded modulo
/// the field modulus before accumulation continues.
pub struct ProductSum<M: PrimeModulus> {
    wide: [u64; 8],
    carry: u64,
    marker: PhantomData<M>,
}

impl<M: PrimeModulus> Default for ProductSum<M> {
    #[inline(always)]
    fn default() -> Self {
        Self::new()
    }
}

impl<M: PrimeModulus> ProductSum<M> {
    /// Constructs an empty product sum.
    #[inline(always)]
    pub const fn new() -> Self {
        Self {
            wide: [0; 8],
            carry: 0,
            marker: PhantomData,
        }
    }

    /// Adds `lhs * rhs` to the sum.
    #[inline(always)]
    pub fn add_product(&mut self, lhs: &PastaField<M>, rhs: &PastaField<M>) {
        let (d0, carry) = mac(self.wide[0], lhs.limbs[0], rhs.limbs[0], 0);
        let (d1, carry) = mac(self.wide[1], lhs.limbs[0], rhs.limbs[1], carry);
        let (d2, carry) = mac(self.wide[2], lhs.limbs[0], rhs.limbs[2], carry);
        let (d3, carry) = mac(self.wide[3], lhs.limbs[0], rhs.limbs[3], carry);
        let (d4, overflow) = adc(self.wide[4], carry, 0);

        let (d1, carry) = mac(d1, lhs.limbs[1], rhs.limbs[0], 0);
        let (d2, carry) = mac(d2, lhs.limbs[1], rhs.limbs[1], carry);
        let (d3, carry) = mac(d3, lhs.limbs[1], rhs.limbs[2], carry);
        let (d4, carry) = mac(d4, lhs.limbs[1], rhs.limbs[3], carry);
        let (d5, overflow) = adc(self.wide[5], carry, overflow);

        let (d2, carry) = mac(d2, lhs.limbs[2], rhs.limbs[0], 0);
        let (d3, carry) = mac(d3, lhs.limbs[2], rhs.limbs[1], carry);
        let (d4, carry) = mac(d4, lhs.limbs[2], rhs.limbs[2], carry);
        let (d5, carry) = mac(d5, lhs.limbs[2], rhs.limbs[3], carry);
        let (d6, overflow) = adc(self.wide[6], carry, overflow);

        let (d3, carry) = mac(d3, lhs.limbs[3], rhs.limbs[0], 0);
        let (d4, carry) = mac(d4, lhs.limbs[3], rhs.limbs[1], carry);
        let (d5, carry) = mac(d5, lhs.limbs[3], rhs.limbs[2], carry);
        let (d6, carry) = mac(d6, lhs.limbs[3], rhs.limbs[3], carry);
        let (d7, overflow) = adc(self.wide[7], carry, overflow);

        self.wide = [d0, d1, d2, d3, d4, d5, d6, d7];
        let (carry, carry_overflow) = adc(self.carry, overflow, 0);
        self.carry = carry;
        self.fold_overflow(carry_overflow);
    }

    /// Adds one field value to the sum.
    #[inline(always)]
    pub fn add_term(&mut self, term: &PastaField<M>) {
        // Insert term * R so the final REDC returns the original stored value.
        let (d4, carry) = adc(self.wide[4], term.limbs[0], 0);
        let (d5, carry) = adc(self.wide[5], term.limbs[1], carry);
        let (d6, carry) = adc(self.wide[6], term.limbs[2], carry);
        let (d7, carry) = adc(self.wide[7], term.limbs[3], carry);
        let (accumulator_carry, carry_overflow) = adc(self.carry, 0, carry);
        self.wide[4..].copy_from_slice(&[d4, d5, d6, d7]);
        self.carry = accumulator_carry;
        self.fold_overflow(carry_overflow);
    }

    /// Adds another partial sum, preserving the sum of their field values.
    #[inline(always)]
    pub fn merge(&mut self, other: &Self) {
        let mut carry = 0;
        for (limb, rhs) in self.wide.iter_mut().zip(&other.wide) {
            let (sum, next) = adc(*limb, *rhs, carry);
            *limb = sum;
            carry = next;
        }
        let (carry, carry_overflow) = adc(self.carry, other.carry, carry);
        self.carry = carry;
        self.fold_overflow(carry_overflow);
    }

    /// Returns the accumulated field value with one Montgomery reduction.
    #[inline(always)]
    pub fn finish(self) -> PastaField<M> {
        PastaField::from_montgomery(montgomery_reduce::<M>(self.partial_reduce()))
    }

    // Restore the bit lost when the 576-bit accumulator overflows. First
    // fold its wrapped value below 2^449, then add
    // 2^576 mod p ≡ (2^512 mod p) * 2^64 = R2 * 2^64 mod p.
    // The result is below 2^449 + 2^319, so this addition cannot overflow.
    #[inline]
    fn fold_overflow(&mut self, overflow: u64) {
        if overflow == 0 {
            return;
        }
        debug_assert_eq!(overflow, 1);
        self.wide = self.partial_reduce();
        self.carry = 0;
        let mut carry = 0;
        for (limb, correction) in self.wide[1..5].iter_mut().zip(M::R2) {
            (*limb, carry) = adc(*limb, correction, carry);
        }
        for limb in &mut self.wide[5..] {
            (*limb, carry) = adc(*limb, 0, carry);
        }
        debug_assert_eq!(carry, 0);
    }

    // Fold the top two limbs using B448 ≡ 2^448 and R2 ≡ 2^512 (mod p).
    // The result is below 2^448 + 2^65 * p < 2^449 < p * R, the input
    // bound for Montgomery reduction with one conditional subtraction.
    #[inline(always)]
    fn partial_reduce(&self) -> [u64; 8] {
        let upper = self.wide[7];
        let (t0, carry) = mac(0, upper, M::B448[0], 0);
        let (t1, carry) = mac(0, upper, M::B448[1], carry);
        let (t2, carry) = mac(0, upper, M::B448[2], carry);
        let (t3, carry) = mac(0, upper, M::B448[3], carry);
        let t4 = carry;

        let (t0, carry) = mac(t0, self.carry, M::R2[0], 0);
        let (t1, carry) = mac(t1, self.carry, M::R2[1], carry);
        let (t2, carry) = mac(t2, self.carry, M::R2[2], carry);
        let (t3, carry) = mac(t3, self.carry, M::R2[3], carry);
        let (t4, overflow) = adc(t4, 0, carry);
        debug_assert_eq!(overflow, 0);

        let (d0, carry) = adc(self.wide[0], t0, 0);
        let (d1, carry) = adc(self.wide[1], t1, carry);
        let (d2, carry) = adc(self.wide[2], t2, carry);
        let (d3, carry) = adc(self.wide[3], t3, carry);
        let (d4, carry) = adc(self.wide[4], t4, carry);
        let (d5, carry) = adc(self.wide[5], 0, carry);
        let (d6, carry) = adc(self.wide[6], 0, carry);
        let d7 = carry;
        debug_assert!(d7 <= 1);

        [d0, d1, d2, d3, d4, d5, d6, d7]
    }
}

impl<M: PrimeModulus> PastaField<M> {
    /// Computes `self * multiplier - 2 * doubled_lhs * doubled_rhs`.
    pub fn mul_sub_double_product(
        &self,
        multiplier: &Self,
        doubled_lhs: &Self,
        doubled_rhs: &Self,
    ) -> Self {
        self.product_difference::<true>(multiplier, doubled_lhs, doubled_rhs)
    }

    /// Computes `self * multiplier - lhs * rhs` with one Montgomery reduction.
    pub fn mul_sub_product(&self, multiplier: &Self, lhs: &Self, rhs: &Self) -> Self {
        self.product_difference::<false>(multiplier, lhs, rhs)
    }

    fn product_difference<const DOUBLE: bool>(
        &self,
        multiplier: &Self,
        lhs: &Self,
        rhs: &Self,
    ) -> Self {
        // Represent the difference as a*b + pR - c*d (or -2c*d).
        // Since p < R/3, even 2c*d < 2p² < pR, so it is nonnegative.
        // The input is below p² + pR. Unreduced REDC returns less than 3p,
        // requiring two conditional subtractions for a reduced residue.
        let mut wide = multiply_wide(&self.limbs, &multiplier.limbs);
        let mut carry = 0;
        for (upper, modulus) in wide[4..].iter_mut().zip(M::MODULUS) {
            (*upper, carry) = adc(*upper, modulus, carry);
        }
        debug_assert_eq!(carry, 0);

        let mut product = multiply_wide(&lhs.limbs, &rhs.limbs);
        if DOUBLE {
            let mut carry = 0;
            for limb in &mut product {
                let next = *limb >> 63;
                *limb = (*limb << 1) | carry;
                carry = next;
            }
            debug_assert_eq!(carry, 0);
        }
        let mut borrow = 0;
        for (accumulator, product) in wide.iter_mut().zip(product) {
            (*accumulator, borrow) = sbb(*accumulator, product, borrow);
        }
        debug_assert_eq!(borrow, 0);

        let reduced = reduce_once::<M>(montgomery_reduce_unreduced::<M>(wide));
        Self::from_montgomery(reduce_once::<M>(reduced))
    }

    /// Returns the inner product of two arrays, or zero for empty arrays.
    ///
    /// The lengths agree by type. Products share one Montgomery reduction.
    pub fn sum_of_products<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> Self {
        Self::sum_of_product_pairs(lhs.iter().zip(rhs))
    }

    /// Returns the inner product of equal-length slices.
    ///
    /// Returns `None` when the lengths differ and `Some(Self::ZERO)` for two
    /// empty slices. Products share one Montgomery reduction.
    pub fn checked_sum_of_products(lhs: &[Self], rhs: &[Self]) -> Option<Self> {
        if lhs.len() != rhs.len() {
            return None;
        }
        // Long inner products run four independent accumulator lanes, each
        // a carry chain of its own, merged by limb addition before the one
        // reduction; merging preserves the sum modulo p.
        const LANES: usize = 4;
        const LANE_THRESHOLD: usize = 64;
        if lhs.len() < LANE_THRESHOLD {
            return Some(Self::sum_of_product_pairs(lhs.iter().zip(rhs)));
        }
        let mut lanes: [ProductSum<M>; LANES] = core::array::from_fn(|_| ProductSum::new());
        let mut lhs_chunks = lhs.chunks_exact(LANES);
        let mut rhs_chunks = rhs.chunks_exact(LANES);
        for (lhs, rhs) in lhs_chunks.by_ref().zip(rhs_chunks.by_ref()) {
            for (lane, (lhs, rhs)) in lanes.iter_mut().zip(lhs.iter().zip(rhs)) {
                lane.add_product(lhs, rhs);
            }
        }
        for (lhs, rhs) in lhs_chunks.remainder().iter().zip(rhs_chunks.remainder()) {
            lanes[0].add_product(lhs, rhs);
        }
        let mut sum = ProductSum::new();
        for lane in &lanes {
            sum.merge(lane);
        }
        Some(sum.finish())
    }

    /// Returns the sum of pairwise products, or zero for an empty iterator.
    ///
    /// Accepts pairs from noncontiguous sources, such as strided columns.
    /// Products share one Montgomery reduction.
    pub fn sum_of_product_pairs<'a>(pairs: impl IntoIterator<Item = (&'a Self, &'a Self)>) -> Self {
        let mut sum = ProductSum::new();
        for (lhs, rhs) in pairs {
            sum.add_product(lhs, rhs);
        }
        sum.finish()
    }
}
