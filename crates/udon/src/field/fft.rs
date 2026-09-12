//! Temporarily unreduced Montgomery arithmetic for FFT butterflies.
//!
//! For modulus `p`, working limbs stay in `[0, 2p)` and twiddles in `[0, p)`.
//! Only the kernels here may operate on loose values; ordinary field methods
//! require reduced inputs. A [`Guard`] restores `[0, p)` before a working
//! region is returned to other code, including during unwinding. These are
//! arithmetic invariants, not memory-safety requirements.

use super::{
    PastaField, PrimeModulus,
    montgomery::reduce_once,
    word::{adc, mac, subtract_limbs},
};

/// Reduces a borrowed working region when its kernel finishes or unwinds.
///
/// Entries must remain below `2p` throughout the borrow so one subtraction
/// suffices on drop. Only FFT internals may observe the loose storage; do not
/// pass this guard or its working slice to user callbacks.
pub(crate) struct Guard<'a, M: PrimeModulus> {
    pub(crate) values: &'a mut [PastaField<M>],
}

impl<M: PrimeModulus> Drop for Guard<'_, M> {
    fn drop(&mut self) {
        for value in self.values.iter_mut() {
            *value = normalize(*value);
        }
    }
}

// The loose bound makes a single conditional subtraction sufficient.
#[inline]
pub(crate) fn normalize<M: PrimeModulus>(value: PastaField<M>) -> PastaField<M> {
    PastaField::from_montgomery(reduce_once::<M>(value.limbs))
}

#[inline(always)]
fn double_modulus<M: PrimeModulus>() -> [u64; 4] {
    [
        M::MODULUS[0] << 1,
        (M::MODULUS[1] << 1) | (M::MODULUS[0] >> 63),
        (M::MODULUS[2] << 1) | (M::MODULUS[1] >> 63),
        (M::MODULUS[3] << 1) | (M::MODULUS[2] >> 63),
    ]
}

// Coarsely integrated operand scanning (CIOS) combines limb multiplication
// with Montgomery reduction: lhs < 2p, rhs < p, result < 2p.
// With R = 2^256 and 2p < R, (lhs*rhs + m*p)/R < 2p for m < R.
// The sealed Pasta moduli have limbs [p0, p1, 0, 1 << 62], which lets
// reduction replace two multiplication steps with shifts and addition.
// Keep this loop separate from canonical multiplication so FFT inlining
// decisions do not change the field's ordinary multiplication kernel.
#[inline]
fn multiply<M: PrimeModulus>(lhs: &[u64; 4], rhs: &[u64; 4]) -> [u64; 4] {
    let mut accumulator = [0; 5];
    for rhs_limb in rhs {
        let mut carry = 0;
        for index in 0..4 {
            (accumulator[index], carry) = mac(accumulator[index], lhs[index], *rhs_limb, carry);
        }
        let (upper, product_overflow) = adc(accumulator[4], carry, 0);
        accumulator[4] = upper;
        let multiplier = accumulator[0].wrapping_mul(M::MONTGOMERY_INV);
        let (cancelled, carry) = mac(accumulator[0], multiplier, M::MODULUS[0], 0);
        debug_assert_eq!(cancelled, 0);
        let (r0, carry) = mac(accumulator[1], multiplier, M::MODULUS[1], carry);
        let (r1, carry) = adc(accumulator[2], 0, carry);
        let (r2, carry) = adc(accumulator[3], multiplier << 62, carry);
        let (r3, reduction_overflow) = adc(accumulator[4], multiplier >> 2, carry);
        accumulator = [r0, r1, r2, r3, product_overflow + reduction_overflow];
    }
    debug_assert_eq!(accumulator[4], 0);
    accumulator[..4].try_into().unwrap()
}

// Both inputs and outputs are loose; a supplied twiddle must be reduced.
// None represents twiddle one, avoiding an identity multiplication.
#[inline]
pub(crate) fn butterfly<M: PrimeModulus>(
    left: &mut PastaField<M>,
    right: &mut PastaField<M>,
    twiddle: Option<&PastaField<M>>,
) {
    let product = match twiddle {
        Some(twiddle) => multiply::<M>(&right.limbs, &twiddle.limbs),
        None => right.limbs,
    };
    let modulus = double_modulus::<M>();
    debug_assert!(super::word::compare_limbs(&left.limbs, &modulus).is_lt());
    debug_assert!(super::word::compare_limbs(&product, &modulus).is_lt());
    let mut sum = [0; 4];
    let mut carry = 0;
    for (index, limb) in sum.iter_mut().enumerate() {
        (*limb, carry) = adc(left.limbs[index], product[index], carry);
    }
    let (reduced, borrow) = subtract_limbs(&sum, &modulus);
    if carry != 0 || borrow == 0 {
        sum = reduced;
    }
    let (mut difference, borrow) = subtract_limbs(&left.limbs, &product);
    if borrow != 0 {
        let mut carry = 0;
        for (limb, modulus) in difference.iter_mut().zip(modulus) {
            (*limb, carry) = adc(*limb, modulus, carry);
        }
        debug_assert_eq!(carry, 1);
    }
    left.limbs = sum;
    right.limbs = difference;
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{PallasBase, PallasScalar};
    use num_bigint::BigUint;

    fn integer(limbs: [u64; 4]) -> BigUint {
        limbs
            .iter()
            .rev()
            .fold(BigUint::from(0u32), |value, limb| (value << 64) + limb)
    }

    fn field<M: PrimeModulus>(value: &BigUint) -> PastaField<M> {
        let digits = value.to_u64_digits();
        // Bypass checked construction to exercise the kernel's loose range.
        let mut field = PastaField::ZERO;
        field.limbs = core::array::from_fn(|i| digits.get(i).copied().unwrap_or(0));
        field
    }

    fn boundaries<M: PrimeModulus>() {
        let p = integer(M::MODULUS);
        let twice = &p * 2u32;
        let inverse_r = (BigUint::from(1u32) << 256usize).modpow(&(&p - 2u32), &p);
        let values = [
            BigUint::from(0u32),
            BigUint::from(1u32),
            &p - 1u32,
            p.clone(),
            &p + 1u32,
            &twice - 1u32,
        ];
        for left in &values {
            for right in &values {
                for twiddle in [
                    None,
                    Some(PastaField::ZERO),
                    Some(PastaField::ONE),
                    Some(PastaField::ONE.neg()),
                    Some(PastaField::from_u64(7)),
                    Some(field::<M>(&BigUint::from(1u32))),
                    Some(field::<M>(&(&p / 2u32))),
                    Some(field::<M>(&(&p - 2u32))),
                    Some(field::<M>(&(&p - 1u32))),
                ] {
                    let product = twiddle.map_or_else(
                        || right % &p,
                        |twiddle| (right * integer(twiddle.montgomery_limbs()) * &inverse_r) % &p,
                    );
                    let mut low = field::<M>(left);
                    let mut high = field::<M>(right);
                    butterfly(&mut low, &mut high, twiddle.as_ref());
                    assert!(integer(low.limbs) < twice);
                    assert!(integer(high.limbs) < twice);
                    assert_eq!(integer(normalize(low).limbs), (left + &product) % &p);
                    assert_eq!(
                        integer(normalize(high).limbs),
                        (left + &twice - &product) % &p
                    );
                }
            }
        }
    }

    #[test]
    fn loose_butterflies_match_integer_arithmetic_at_modulus_boundaries() {
        boundaries::<PallasBase>();
        boundaries::<PallasScalar>();
    }
}
