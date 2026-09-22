//! Generic exponentiation and square roots.
//!
//! Callers supply field operations and parameters. The generic form also
//! permits exhaustive testing over small fields with different two-adicities.

/// Field operations used by exponentiation and square roots.
///
/// Implementations must obey field laws and compare by field value.
pub(super) trait Field: Copy + Eq {
    /// The additive identity.
    #[cfg(any(test, not(feature = "sqrt-table-large")))]
    const ZERO: Self;
    /// The multiplicative identity.
    const ONE: Self;
    /// Whether this is the additive identity.
    #[cfg(any(test, not(feature = "sqrt-table-large")))]
    fn is_zero(&self) -> bool {
        *self == Self::ZERO
    }
    /// The field product.
    fn mul(&self, rhs: &Self) -> Self;
    /// The field square; implementations can override with a specialized kernel.
    fn square(&self) -> Self {
        self.mul(self)
    }
}

/// Raises this value to an unsigned exponent; exponent zero returns one.
///
/// The multiplication schedule depends on the exponent.
pub(super) fn pow_u64<F: Field>(value: &F, exponent: u64) -> F {
    if exponent == 0 {
        return F::ONE;
    }

    // The leading one initializes the accumulator. This avoids the extra
    // multiply and trailing square performed by a right-to-left ladder.
    let highest_bit = 63 - exponent.leading_zeros();
    let mut result = *value;
    for bit in (0..highest_bit).rev() {
        result = result.square();
        if exponent & (1u64 << bit) != 0 {
            result = result.mul(value);
        }
    }
    result
}

/// Tonelli–Shanks over an odd prime field with `p - 1 = t * 2^two_adicity`.
///
/// Requires odd `t` and `1 <= two_adicity <= 64`. The caller supplies
/// `w = value^((t - 1) / 2)`. For every `k` in `1..=two_adicity`, `root(k)`
/// must return a primitive root of order `2^k`. These roots must satisfy
/// `root(k + 1).square() == root(k)` for `1 <= k < two_adicity`.
///
/// Returns either square root, or `None` for a nonsquare; branches depend on
/// the input.
#[cfg(any(test, not(feature = "sqrt-table-large")))]
pub(super) fn tonelli_shanks_with_roots<F: Field>(
    value: &F,
    w: F,
    root: impl Fn(u32) -> F,
    two_adicity: u32,
) -> Option<F> {
    assert!(
        (1..=64).contains(&two_adicity),
        "two_adicity must be within 1..=64"
    );
    if value.is_zero() {
        return Some(F::ZERO);
    }
    let mut x = w.mul(value);
    let mut t = x.mul(&w);
    let mut m = two_adicity;

    while t != F::ONE {
        let mut i = 1u32;
        let mut t_squared = t.square();
        while i < m && t_squared != F::ONE {
            t_squared = t_squared.square();
            i += 1;
        }
        if i == m {
            return None;
        }

        // Matching root orientations ensure b_squared = b^2, preserving
        // x^2 = value * t when both accumulators are updated.
        let b = root(i + 1);
        let b_squared = root(i);
        x = x.mul(&b);
        t = t.mul(&b_squared);
        m = i;
    }

    Some(x)
}

// Simple reference retains the independently evolving c ladder.
#[cfg(test)]
pub(super) fn tonelli_shanks<F: Field>(value: &F, w: F, root: F, two_adicity: u32) -> Option<F> {
    assert!(
        (1..=64).contains(&two_adicity),
        "two_adicity must be within 1..=64"
    );
    if value.is_zero() {
        return Some(F::ZERO);
    }
    let mut x = w.mul(value);
    let mut t = x.mul(&w);
    let mut c = root;
    let mut m = two_adicity;

    while t != F::ONE {
        let mut i = 1u32;
        let mut t_squared = t.square();
        while i < m && t_squared != F::ONE {
            t_squared = t_squared.square();
            i += 1;
        }
        if i == m {
            return None;
        }

        let b = pow_u64(&c, 1u64 << (m - i - 1));
        let b_squared = b.square();
        x = x.mul(&b);
        t = t.mul(&b_squared);
        c = b_squared;
        m = i;
    }

    Some(x)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    struct SmallField<const P: u64>(u64);

    impl<const P: u64> Field for SmallField<P> {
        const ZERO: Self = Self(0);
        const ONE: Self = Self(1);
        fn mul(&self, rhs: &Self) -> Self {
            Self(self.0 * rhs.0 % P)
        }
    }

    fn check_field<const P: u64>(root: u64, two_adicity: u32) {
        for value in 0..P {
            let base = SmallField::<P>(value);
            let mut expected = SmallField::ONE;
            for exponent in 0..200 {
                assert_eq!(pow_u64(&base, exponent), expected);
                expected = expected.mul(&base);
            }
            assert_eq!(
                pow_u64(&base, u64::MAX),
                if value == 0 {
                    base
                } else {
                    pow_u64(&base, u64::MAX % (P - 1))
                }
            );
            let odd_cofactor = (P - 1) >> two_adicity;
            let w = pow_u64(&base, (odd_cofactor - 1) / 2);
            let result = tonelli_shanks(&base, w, SmallField(root), two_adicity);
            assert_eq!(
                result,
                tonelli_shanks_with_roots(
                    &base,
                    w,
                    |k| pow_u64(&SmallField(root), 1u64 << (two_adicity - k)),
                    two_adicity
                )
            );
            let has_root = (0..P).any(|candidate| candidate * candidate % P == value);
            assert_eq!(result.is_some(), has_root);
            if let Some(root) = result {
                assert_eq!(root.square(), base);
            }
        }
    }

    #[test]
    fn field_algorithms_cover_distinct_two_adicities() {
        check_field::<7>(6, 1);
        check_field::<13>(5, 2);
        check_field::<17>(3, 4);
        check_field::<97>(28, 5);
    }
}
