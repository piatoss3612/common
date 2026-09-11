//! Square roots and power-of-two roots of unity from compile-time field tables.

use super::{PastaField, PrimeModulus, parameters::TWO_ADICITY};

#[cfg(feature = "sqrt-table-large")]
mod large;
#[cfg(feature = "sqrt-table-large")]
pub(super) use large::LargeSqrtTable;

/// Validates and converts Montgomery limbs into field entries.
///
/// Table initializers call this during constant evaluation, establishing the
/// reduced-residue invariant before runtime.
///
/// # Panics
///
/// Panics if any entry is at least [`M::MODULUS`](PrimeModulus::MODULUS).
pub(super) const fn field_elements<M: PrimeModulus, const N: usize>(
    limbs: [[u64; 4]; N],
) -> [PastaField<M>; N] {
    let mut fields = [PastaField::ZERO; N];
    let mut k = 0;
    while k < N {
        fields[k] = PastaField::from_montgomery_limbs(limbs[k]);
        k += 1;
    }
    fields
}

impl<M: PrimeModulus> PastaField<M> {
    /// Returns a square root, or `None` for a nonsquare.
    ///
    /// Zero returns `Some(Self::ZERO)`. Either root may be returned, and the
    /// choice may differ between [table configurations](crate#features).
    /// Tables require no allocation or runtime initialization. Branches and
    /// table accesses depend on the input; there is no constant-time guarantee
    /// for secret inputs.
    ///
    /// ```
    /// use zakura_udon::field::Fp;
    ///
    /// let square = Fp::from_u64(7).square();
    /// assert_eq!(square.sqrt().unwrap().square(), square);
    /// assert_eq!(Fp::ZERO.sqrt(), Some(Fp::ZERO));
    /// assert_eq!(Fp::from_u64(5).sqrt(), None);
    /// ```
    pub fn sqrt(&self) -> Option<Self> {
        if self.is_zero() {
            return Some(Self::ZERO);
        }

        // With p - 1 = t * 2^32, one fixed exponentiation initializes both
        // self^((t + 1)/2) and self^t. Both algorithms share this schedule.
        let w = M::pow_sqrt_exponent(self);
        #[cfg(feature = "sqrt-table-large")]
        {
            M::sqrt_large(self, w)
        }
        #[cfg(not(feature = "sqrt-table-large"))]
        {
            super::algorithms::tonelli_shanks_with_roots(
                self,
                w,
                |k| M::ROOTS[k as usize],
                TWO_ADICITY,
            )
        }
    }

    /// Returns a primitive root of order `2^log_size`, or `None` above 32.
    ///
    /// The selected root is `5^((p - 1) / 2^log_size)`, where `p` is
    /// [`M::MODULUS`](PrimeModulus::MODULUS). `log_size = 0` returns one.
    pub const fn root_of_unity(log_size: u32) -> Option<Self> {
        if log_size > TWO_ADICITY {
            return None;
        }
        Some(M::ROOTS[log_size as usize])
    }

    /// Returns the inverse of [`Self::root_of_unity`], or `None` above 32.
    pub const fn root_of_unity_inverse(log_size: u32) -> Option<Self> {
        if log_size > TWO_ADICITY {
            return None;
        }
        Some(M::INVERSE_ROOTS[log_size as usize])
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field::{PallasBase, PallasScalar, algorithms};

    fn check<M: PrimeModulus>() {
        let root = PastaField::<M>::root_of_unity(32).unwrap();
        let compare = |value: PastaField<M>| {
            let w = M::pow_sqrt_exponent(&value);
            let reference = algorithms::tonelli_shanks(&value, w, root, TWO_ADICITY);
            let small = algorithms::tonelli_shanks_with_roots(
                &value,
                w,
                |k| M::ROOTS[k as usize],
                TWO_ADICITY,
            );
            assert_eq!(small, reference);
            let result = value.sqrt();
            assert_eq!(result.is_some(), reference.is_some());
            if let Some(result) = result {
                assert_eq!(result.square(), value);
                let reference = reference.unwrap();
                assert!(result == reference || result == reference.neg());
            }
        };
        compare(PastaField::ZERO);
        // Vary each byte of subgroup input exponents and straddle their carry
        // boundaries. Five's odd-cofactor power is root, so its inverse exponent
        // is u32::MAX; rounding that exponent must not overflow a 32-bit target.
        for shift in [0, 8, 16, 24] {
            for digit in 0..256u64 {
                compare(root.pow_u64(digit << shift));
            }
        }
        for exponent in [
            255, 256, 257, 65535, 65536, 65537, 0xffffff, 0x1000000, 0xfffffffe, 0xffffffff,
        ] {
            compare(root.pow_u64(exponent));
        }
        compare(PastaField::from_u64(5));
        assert!(PastaField::<M>::from_u64(5).sqrt().is_none());
    }

    #[test]
    fn square_root_algorithms_agree_on_subgroup_boundaries() {
        check::<PallasBase>();
        check::<PallasScalar>();
    }
}
