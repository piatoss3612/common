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

#[cfg(test)]
mod experiments;

/// Reduces a borrowed working region when its kernel finishes or unwinds.
///
/// Entries must remain below `2p` throughout the borrow so one subtraction
/// suffices on drop. Only FFT internals may observe the loose storage; do not
/// pass this guard or its working slice to user callbacks.
pub(crate) struct Guard<'a, M: PrimeModulus> {
    pub(crate) values: &'a mut [PastaField<M>],
    armed: bool,
}

impl<'a, M: PrimeModulus> Guard<'a, M> {
    pub(crate) fn new(values: &'a mut [PastaField<M>]) -> Self {
        Self {
            values,
            armed: true,
        }
    }

    /// Ends the borrow without reducing again after a canonical terminal store.
    ///
    /// Every entry must already be below `p`. Call only after the entire region
    /// finishes successfully, so a panic during a partial finish still reduces
    /// the mixture of canonical and loose values.
    pub(crate) fn disarm(mut self) {
        debug_assert!(
            self.values
                .iter()
                .all(|value| { super::word::compare_limbs(&value.limbs, &M::MODULUS).is_lt() })
        );
        self.armed = false;
    }
}

impl<M: PrimeModulus> Drop for Guard<'_, M> {
    fn drop(&mut self) {
        if self.armed {
            for value in self.values.iter_mut() {
                *value = normalize(*value);
            }
        }
    }
}

// The loose bound makes a single conditional subtraction sufficient.
#[inline]
pub(crate) fn normalize<M: PrimeModulus>(value: PastaField<M>) -> PastaField<M> {
    PastaField::from_montgomery(reduce_once::<M>(value.limbs))
}

/// Divides a loose Montgomery value by `2^log_size`, returning a reduced value.
///
/// Requires `log_size <= 32` and input limbs `x < 2p`. Both Pasta primes have
/// `p = 1 mod 2^32`, so `q = -x mod 2^log_size` makes `x + q*p` divisible by
/// `2^log_size`. This preserves Montgomery scale. For a canonical input the
/// quotient is below `p`; for a loose input and `log_size >= 1`, it is below
/// `(1 + 2^-log_size)*p < 2p`, so one subtraction suffices. At zero, reduce the
/// input directly instead of shifting a limb by 64.
#[inline]
pub(crate) fn divide_by_power_of_two<M: PrimeModulus>(
    value: PastaField<M>,
    log_size: u32,
) -> PastaField<M> {
    debug_assert!(log_size <= 32);
    debug_assert!(super::word::compare_limbs(&value.limbs, &double_modulus::<M>()).is_lt());
    if log_size == 0 {
        return normalize(value);
    }
    let [x0, x1, x2, x3] = value.limbs;
    let mask = (1u64 << log_size) - 1;
    let q = x0.wrapping_neg() & mask;
    let (r0, carry) = mac(x0, q, M::MODULUS[0], 0);
    let (r1, carry) = mac(x1, q, M::MODULUS[1], carry);
    let (r2, carry) = adc(x2, 0, carry);
    let (r3, carry) = adc(x3, q << 62, carry);
    // The numerator may exceed four limbs. Its high limb is below 2^log_size
    // because the quotient is below 2p < 2^256; preserve it in the final shift.
    let r4 = (q >> 2) + carry;
    debug_assert_eq!(r0 & mask, 0);
    debug_assert!(r4 <= mask);
    PastaField::from_montgomery(reduce_once::<M>([
        (r0 >> log_size) | (r1 << (64 - log_size)),
        (r1 >> log_size) | (r2 << (64 - log_size)),
        (r2 >> log_size) | (r3 << (64 - log_size)),
        (r3 >> log_size) | (r4 << (64 - log_size)),
    ]))
}

/// Multiplies a loose value by a reduced scale and returns a reduced value.
#[inline]
pub(crate) fn scale<M: PrimeModulus>(
    value: PastaField<M>,
    factor: &PastaField<M>,
) -> PastaField<M> {
    debug_assert!(super::word::compare_limbs(&value.limbs, &double_modulus::<M>()).is_lt());
    debug_assert!(super::word::compare_limbs(&factor.limbs, &M::MODULUS).is_lt());
    PastaField::from_montgomery(reduce_once::<M>(multiply::<M>(&value.limbs, &factor.limbs)))
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

/// Decimation-in-frequency butterfly.
///
/// Adds and subtracts before multiplying the difference. The same loose range
/// bound holds as for [`butterfly`].
#[inline]
pub(crate) fn butterfly_dif<M: PrimeModulus>(
    left: &mut PastaField<M>,
    right: &mut PastaField<M>,
    twiddle: Option<&PastaField<M>>,
) {
    butterfly(left, right, None);
    if let Some(twiddle) = twiddle {
        right.limbs = multiply::<M>(&right.limbs, &twiddle.limbs);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{PallasBase, PallasScalar};
    use crate::test_support::{CORPUS_SEED, xorshift64};
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
                    if let Some(twiddle) = twiddle {
                        let scaled = scale(field::<M>(right), &twiddle);
                        assert_eq!(integer(scaled.limbs), product);
                    }
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
                    let mut low = field::<M>(left);
                    let mut high = field::<M>(right);
                    butterfly_dif(&mut low, &mut high, twiddle.as_ref());
                    assert!(integer(low.limbs) < twice);
                    assert!(integer(high.limbs) < twice);
                    assert_eq!(integer(normalize(low).limbs), (left + right) % &p);
                    let difference = (left + &twice - right) % &p;
                    let expected = twiddle.map_or_else(
                        || difference.clone(),
                        |twiddle| (&difference * integer(twiddle.limbs) * &inverse_r) % &p,
                    );
                    assert_eq!(integer(normalize(high).limbs), expected);
                }
            }
        }
    }

    #[test]
    fn loose_butterflies_match_integer_arithmetic_at_modulus_boundaries() {
        boundaries::<PallasBase>();
        boundaries::<PallasScalar>();
    }

    fn division<M: PrimeModulus>() {
        let p = integer(M::MODULUS);
        let twice = &p * 2u32;
        let mut values = std::vec![
            BigUint::from(0u32),
            BigUint::from(1u32),
            &p - 1u32,
            p.clone(),
            &p + 1u32,
            &twice - 1u32,
        ];
        // Exercise correction bits, carries into each limb, and the fifth
        // numerator limb, with both canonical and loose representatives.
        for bit in [
            1usize, 2, 31, 32, 33, 63, 64, 65, 127, 128, 129, 191, 192, 193, 253, 254, 255,
        ] {
            let power = BigUint::from(1u32) << bit;
            for value in [&power - 1u32, power.clone(), &power + 1u32] {
                values.push(value.clone());
                if value < p {
                    values.extend([&p - &value, &p + &value, &twice - &value]);
                }
            }
        }
        let mut seed = CORPUS_SEED;
        for _ in 0..128 {
            let mut limbs = core::array::from_fn(|_| xorshift64(&mut seed));
            limbs[3] &= (1 << 63) - 1;
            values.push(integer(limbs));
        }
        for log_size in 0..=32 {
            let inverse = (BigUint::from(1u32) << log_size as usize).modpow(&(&p - 2u32), &p);
            for value in &values {
                assert!(value < &twice);
                let result = divide_by_power_of_two(field::<M>(value), log_size);
                // Compare raw Montgomery integers: the operation must divide
                // without changing the representation's Montgomery scale.
                assert_eq!(
                    integer(result.limbs),
                    (value * &inverse) % &p,
                    "k={log_size}"
                );
            }
        }
    }

    #[test]
    fn inverse_power_scaling_matches_integers_for_canonical_and_loose_inputs() {
        division::<PallasBase>();
        division::<PallasScalar>();
    }

    fn partial_finish<M: PrimeModulus>() {
        use std::panic::{AssertUnwindSafe, catch_unwind};

        let p = integer(M::MODULUS);
        let original: [PastaField<M>; 4] = core::array::from_fn(|i| field(&(&p + i as u32)));
        for completed in 0..=original.len() {
            let mut values = original;
            assert!(
                catch_unwind(AssertUnwindSafe(|| {
                    let guard = Guard::new(&mut values);
                    for value in &mut guard.values[..completed] {
                        *value = divide_by_power_of_two(*value, 1);
                    }
                    panic!("interrupt terminal normalization");
                }))
                .is_err()
            );
            for (i, value) in values.iter().enumerate() {
                let expected = if i < completed {
                    (BigUint::from(i as u32) * (&p + 1u32) / 2u32) % &p
                } else {
                    BigUint::from(i as u32)
                };
                assert_eq!(integer(value.limbs), expected);
            }
        }
    }

    #[test]
    fn partial_terminal_normalization_keeps_the_guard_armed_on_unwind() {
        partial_finish::<PallasBase>();
        partial_finish::<PallasScalar>();
    }
}
