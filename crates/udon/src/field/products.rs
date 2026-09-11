//! Wide product accumulation and signed product differences.
//!
//! The range arguments beside each kernel explain how much reduction its
//! accumulator needs before returning a canonical field element.

use core::marker::PhantomData;

use super::montgomery::montgomery_reduce;
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
        self.add_product_inner::<false>(lhs, rhs);
    }

    // Bounded callers start from zero and feed at most one physical slice.
    // On 32/64-bit targets its byte-size bound implies fewer than 2^59 terms;
    // with each product below 2^510, the 576-bit accumulator cannot overflow.
    #[inline(always)]
    fn add_product_inner<const BOUNDED: bool>(&mut self, lhs: &PastaField<M>, rhs: &PastaField<M>) {
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
        if BOUNDED {
            debug_assert_eq!(carry_overflow, 0);
        } else {
            self.fold_overflow(carry_overflow);
        }
    }

    // Column accumulation adapted from common's deferred.rs at revision
    // 812e867748943ba3830f0f16cd627da456cc58cd. A block shares carry handoffs
    // across terms. Only fresh, physically bounded slice sums call this path.
    #[cfg(target_arch = "aarch64")]
    fn add_product_block(&mut self, lhs: &[PastaField<M>], rhs: &[PastaField<M>]) {
        assert_eq!(lhs.len(), rhs.len());

        macro_rules! add_block {
            ($columns:ident, $lhs_limb:literal, $rhs_limb:literal) => {{
                let (lhs_quads, lhs_remainder) = lhs.as_chunks::<4>();
                let (rhs_quads, rhs_remainder) = rhs.as_chunks::<4>();
                for (lhs, rhs) in lhs_quads.iter().zip(rhs_quads) {
                    add_product(
                        &mut $columns[0],
                        lhs[0].limbs[$lhs_limb],
                        rhs[0].limbs[$rhs_limb],
                    );
                    add_product(
                        &mut $columns[1],
                        lhs[1].limbs[$lhs_limb],
                        rhs[1].limbs[$rhs_limb],
                    );
                    add_product(
                        &mut $columns[2],
                        lhs[2].limbs[$lhs_limb],
                        rhs[2].limbs[$rhs_limb],
                    );
                    add_product(
                        &mut $columns[3],
                        lhs[3].limbs[$lhs_limb],
                        rhs[3].limbs[$rhs_limb],
                    );
                }
                for (lhs, rhs) in lhs_remainder.iter().zip(rhs_remainder) {
                    let lhs = &lhs.limbs;
                    let rhs = &rhs.limbs;
                    add_product(&mut $columns[0], lhs[$lhs_limb], rhs[$rhs_limb]);
                }
            }};
        }

        // Comba columns let every product in the block share each carry
        // handoff. Three limbs suffice for every realizable input slice: a
        // 64-bit target can hold fewer than 2^59 four-limb values, so the
        // widest column sums fewer than 2^61 128-bit products.
        let mut columns = [[self.wide[0], 0, 0], [0; 3], [0; 3], [0; 3]];
        add_block!(columns, 0, 0);
        let mut column = merge_columns(columns);
        self.wide[0] = column[0];

        columns = start_columns(self.wide[1], column[1], column[2]);
        add_block!(columns, 0, 1);
        add_block!(columns, 1, 0);
        column = merge_columns(columns);
        self.wide[1] = column[0];

        columns = start_columns(self.wide[2], column[1], column[2]);
        add_block!(columns, 0, 2);
        add_block!(columns, 1, 1);
        add_block!(columns, 2, 0);
        column = merge_columns(columns);
        self.wide[2] = column[0];

        columns = start_columns(self.wide[3], column[1], column[2]);
        add_block!(columns, 0, 3);
        add_block!(columns, 1, 2);
        add_block!(columns, 2, 1);
        add_block!(columns, 3, 0);
        column = merge_columns(columns);
        self.wide[3] = column[0];

        columns = start_columns(self.wide[4], column[1], column[2]);
        add_block!(columns, 1, 3);
        add_block!(columns, 2, 2);
        add_block!(columns, 3, 1);
        column = merge_columns(columns);
        self.wide[4] = column[0];

        columns = start_columns(self.wide[5], column[1], column[2]);
        add_block!(columns, 2, 3);
        add_block!(columns, 3, 2);
        column = merge_columns(columns);
        self.wide[5] = column[0];

        columns = start_columns(self.wide[6], column[1], column[2]);
        add_block!(columns, 3, 3);
        column = merge_columns(columns);
        self.wide[6] = column[0];

        column = start_column(self.wide[7], column[1], column[2]);
        self.wide[7] = column[0];
        debug_assert_eq!(column[2], 0);
        let (carry, overflow) = self.carry.overflowing_add(column[1]);
        debug_assert!(!overflow, "carry overflow: too many accumulated products");
        self.carry = carry;
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
        // Subtract first; only a negative difference needs pR. Since
        // -2p² < delta < p² and p < R/3, either result lies in [0,pR).
        let mut wide = multiply_wide(&self.limbs, &multiplier.limbs);
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
        let mask = 0u64.wrapping_sub(borrow);
        let mut carry = 0;
        for (upper, modulus) in wide[4..].iter_mut().zip(M::MODULUS) {
            (*upper, carry) = adc(*upper, modulus & mask, carry);
        }
        // A negative subtraction wrapped modulo R²; restoration wraps once.
        debug_assert_eq!(carry, borrow);
        Self::from_montgomery(montgomery_reduce::<M>(wide))
    }

    /// Returns the inner product of two arrays, or zero for empty arrays.
    ///
    /// The lengths agree by type. Products share one Montgomery reduction.
    pub fn sum_of_products<const N: usize>(lhs: &[Self; N], rhs: &[Self; N]) -> Self {
        Self::sum_of_products_slice(lhs, rhs)
    }

    /// Returns the inner product of equal-length slices.
    ///
    /// Returns `None` when the lengths differ and `Some(Self::ZERO)` for two
    /// empty slices. Products share one Montgomery reduction.
    pub fn checked_sum_of_products(lhs: &[Self], rhs: &[Self]) -> Option<Self> {
        if lhs.len() != rhs.len() {
            return None;
        }
        Some(Self::sum_of_products_slice(lhs, rhs))
    }

    #[inline]
    fn sum_of_products_slice(lhs: &[Self], rhs: &[Self]) -> Self {
        const {
            assert!(
                usize::BITS <= 64,
                "bounded sums require at most 64-bit pointers"
            );
        }
        if lhs.is_empty() {
            return Self::ZERO;
        }
        if lhs.len() == 1 {
            return lhs[0].mul(&rhs[0]);
        }
        if lhs.len() <= 3 {
            // 3(p-1)² < pR for both Pasta primes. Four terms exceed this
            // bound even though their sum still fits in eight limbs.
            let mut wide = [0; 8];
            for (lhs, rhs) in lhs.iter().zip(rhs) {
                let product = multiply_wide(&lhs.limbs, &rhs.limbs);
                let mut carry = 0;
                for (limb, term) in wide.iter_mut().zip(product) {
                    (*limb, carry) = adc(*limb, term, carry);
                }
                debug_assert_eq!(carry, 0);
            }
            return Self::from_montgomery(montgomery_reduce::<M>(wide));
        }
        #[cfg(target_arch = "aarch64")]
        if lhs.len() >= 32 {
            let mut sum = ProductSum::new();
            for (lhs, rhs) in lhs.chunks(32).zip(rhs.chunks(32)) {
                sum.add_product_block(lhs, rhs);
            }
            return sum.finish();
        }
        // Long inner products run four independent accumulator lanes, each
        // a carry chain of its own, merged by limb addition before the one
        // reduction; merging preserves the sum modulo p.
        const LANES: usize = 4;
        const LANE_THRESHOLD: usize = 64;
        if lhs.len() < LANE_THRESHOLD {
            let mut sum = ProductSum::new();
            for (lhs, rhs) in lhs.iter().zip(rhs) {
                sum.add_product_inner::<true>(lhs, rhs);
            }
            return sum.finish();
        }
        let mut lanes: [ProductSum<M>; LANES] = core::array::from_fn(|_| ProductSum::new());
        let mut lhs_chunks = lhs.chunks_exact(LANES);
        let mut rhs_chunks = rhs.chunks_exact(LANES);
        for (lhs, rhs) in lhs_chunks.by_ref().zip(rhs_chunks.by_ref()) {
            for (lane, (lhs, rhs)) in lanes.iter_mut().zip(lhs.iter().zip(rhs)) {
                lane.add_product_inner::<true>(lhs, rhs);
            }
        }
        for (lhs, rhs) in lhs_chunks.remainder().iter().zip(rhs_chunks.remainder()) {
            lanes[0].add_product_inner::<true>(lhs, rhs);
        }
        let mut sum = ProductSum::new();
        for lane in &lanes {
            sum.merge(lane);
        }
        sum.finish()
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

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn add_product(column: &mut [u64; 3], lhs: u64, rhs: u64) {
    let product = (lhs as u128) * (rhs as u128);
    let (low, low_carry) = column[0].overflowing_add(product as u64);
    let (middle, high_carry) = column[1].overflowing_add((product >> 64) as u64);
    let (middle, middle_carry) = middle.overflowing_add(low_carry as u64);
    let carry = high_carry as u64 + middle_carry as u64;
    let (high, overflow) = column[2].overflowing_add(carry);
    debug_assert!(!overflow);
    *column = [low, middle, high];
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn start_column(accumulator: u64, carry_low: u64, carry_high: u64) -> [u64; 3] {
    let (low, carry) = accumulator.overflowing_add(carry_low);
    let (middle, high) = carry_high.overflowing_add(carry as u64);
    [low, middle, high as u64]
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn start_columns(accumulator: u64, carry_low: u64, carry_high: u64) -> [[u64; 3]; 4] {
    [
        start_column(accumulator, carry_low, carry_high),
        [0; 3],
        [0; 3],
        [0; 3],
    ]
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
fn merge_columns(columns: [[u64; 3]; 4]) -> [u64; 3] {
    let [mut result, column1, column2, column3] = columns;
    for column in [column1, column2, column3] {
        let (low, carry) = adc(result[0], column[0], 0);
        let (middle, carry) = adc(result[1], column[1], carry);
        let (high, overflow) = adc(result[2], column[2], carry);
        debug_assert_eq!(overflow, 0);
        result = [low, middle, high];
    }
    result
}
