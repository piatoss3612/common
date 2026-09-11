//! Sealed Pasta moduli and compile-time field parameter derivation.

use bento::const_arithmetic::{m255, u256};

use super::PastaField;
use crate::field::safegcd::{SAFEGCD_BATCHES, to_signed62};

// Derive the remaining field constants from these inputs and each modulus.
// Tests compare independent integer derivations and fixed literal vectors.
pub(super) const TWO_ADICITY: u32 = 32;
const GENERATOR: u64 = 5;

/// Marker for the Pallas base field, which is also the Vesta scalar field.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum PallasBase {}

/// Marker for the Pallas scalar field, which is also the Vesta base field.
#[derive(Clone, Copy, Eq, PartialEq)]
pub enum PallasScalar {}

// The sparse Montgomery kernels and divstep bounds require the Pasta moduli.
mod sealed {
    use super::{INVERSE_POWER_TABLE_LEN, PastaField, SAFEGCD_BATCHES};

    pub trait Parameters: Sized {
        /// `-p^-1 mod 2^64`, used to cancel each low limb during reduction.
        const MONTGOMERY_INV: u64;
        /// `2^256 mod p`, also the Montgomery representation of one.
        const R: [u64; 4];
        /// `2^512 mod p`, used to enter Montgomery form.
        const R2: [u64; 4];
        /// `2^768 mod p`, used by wide decoding.
        const R3: [u64; 4];
        /// Forward roots indexed by logarithmic order.
        const ROOTS: [[u64; 4]; INVERSE_POWER_TABLE_LEN];
        /// Inverse roots indexed by logarithmic order.
        const INVERSE_ROOTS: [[u64; 4]; INVERSE_POWER_TABLE_LEN];
        /// `2^448 mod p`, used to fold the upper limb of a product sum.
        const B448: [u64; 4];
        /// The ordinary exponent `(t - 1) / 2`, where `p - 1 = t * 2^32`.
        const SQRT_EXPONENT: [u64; 4];
        /// The Montgomery representation of `2^-1`.
        const TWO_INVERSE: [u64; 4];
        /// The selected primitive `2^32`-th root of unity in Montgomery form.
        const ROOT_OF_UNITY: [u64; 4];
        /// The inverse of `ROOT_OF_UNITY`, in Montgomery form.
        const ROOT_OF_UNITY_INVERSE: [u64; 4];
        /// The Montgomery representation of `5^(2^32)`.
        const DELTA: [u64; 4];
        /// The protocol-selected primitive cube root of unity in Montgomery form.
        const ZETA: [u64; 4];
        /// The inverse of `ZETA`, in Montgomery form.
        const ZETA_INVERSE: [u64; 4];
        /// The modulus in the signed-62 representation the safegcd core runs on.
        const MODULUS_SIGNED62: [i64; 5];
        /// Per-batch safegcd corrections: Montgomery `2^(2·batches)`, indexed by
        /// `completed_batches - 1`.
        const SAFEGCD_CORRECTIONS: [[u64; 4]; SAFEGCD_BATCHES];
        /// Montgomery `2^-k` for `k <= TWO_ADICITY`: the inverse domain sizes of
        /// every supported transform.
        const POWER_OF_TWO_INVERSES: [[u64; 4]; INVERSE_POWER_TABLE_LEN];

        /// Raises `value` to [`Self::SQRT_EXPONENT`].
        ///
        /// `bento::addition_chain!` plans the multiplication schedule at
        /// compile time; only powers of the base are computed at runtime.
        fn pow_sqrt_exponent(value: &PastaField<Self>) -> PastaField<Self>
        where
            Self: super::PrimeModulus;
    }
}

/// Selects one of the two Pasta primes.
///
/// Only [`PallasBase`] and [`PallasScalar`] implement this sealed trait.
/// Field-valued parameters are available through [`PastaField`] methods.
///
/// ```compile_fail
/// #[derive(Clone, Copy, Eq, PartialEq)]
/// enum Foreign {}
/// impl zakura_udon::field::PrimeModulus for Foreign {
///     const MODULUS: [u64; 4] = [97, 0, 0, 0];
/// }
/// ```
pub trait PrimeModulus: sealed::Parameters + Copy + Eq + Send + Sync + 'static {
    /// The prime modulus as four ordinary little-endian 64-bit limbs.
    const MODULUS: [u64; 4];
}

// Derive each field's constants from its modulus and generator. The two cube
// roots have opposite orientations relative to 5^((p - 1)/3); verify their
// inverse pairing at compile time. The chain macro derives its exponent from
// the same modulus literal.
macro_rules! pasta_field_parameters {
    ($marker:ty, modulus: $modulus:literal,
     zeta: canonical_root $(,)?) => {
        pasta_field_parameters! {
            @impl $marker, $modulus,
            const ZETA: [u64; 4] = Self::PARAMETERS.cube_root;
            const ZETA_INVERSE: [u64; 4] =
                m255::mul(&Self::MODULUS, &Self::ZETA, &Self::ZETA);
        }
    };
    ($marker:ty, modulus: $modulus:literal,
     zeta: squared_canonical_root $(,)?) => {
        pasta_field_parameters! {
            @impl $marker, $modulus,
            const ZETA: [u64; 4] =
                m255::mul(&Self::MODULUS, &Self::ZETA_INVERSE, &Self::ZETA_INVERSE);
            const ZETA_INVERSE: [u64; 4] = Self::PARAMETERS.cube_root;
        }
    };
    (@impl $marker:ty, $modulus:literal, $($zeta:item)*) => {
        impl PrimeModulus for $marker {
            const MODULUS: [u64; 4] = u256::from_hex($modulus);
        }

        impl $marker {
            const PARAMETERS: FieldParameters = FieldParameters::new(Self::MODULUS);
        }

        impl sealed::Parameters for $marker {
            const MONTGOMERY_INV: u64 = Self::PARAMETERS.context.reduction_coefficient();
            const R: [u64; 4] = Self::PARAMETERS.context.one();
            const R2: [u64; 4] = Self::PARAMETERS.context.r2();
            const R3: [u64; 4] = Self::PARAMETERS.r3;
            const ROOTS: [[u64; 4]; INVERSE_POWER_TABLE_LEN] = Self::PARAMETERS.roots;
            const INVERSE_ROOTS: [[u64; 4]; INVERSE_POWER_TABLE_LEN] = Self::PARAMETERS.inverse_roots;
            const B448: [u64; 4] = Self::PARAMETERS.b448;
            const SQRT_EXPONENT: [u64; 4] =
                u256::tonelli_shanks_exponent(&Self::MODULUS, TWO_ADICITY);
            const TWO_INVERSE: [u64; 4] = Self::PARAMETERS.inverse_powers[1];
            const ROOT_OF_UNITY: [u64; 4] =
                Self::PARAMETERS.roots[TWO_ADICITY as usize];
            const ROOT_OF_UNITY_INVERSE: [u64; 4] =
                Self::PARAMETERS.inverse_roots[TWO_ADICITY as usize];
            const DELTA: [u64; 4] = Self::PARAMETERS.delta;
            const MODULUS_SIGNED62: [i64; 5] = to_signed62(&Self::MODULUS);
            const SAFEGCD_CORRECTIONS: [[u64; 4]; SAFEGCD_BATCHES] =
                Self::PARAMETERS.corrections;
            const POWER_OF_TWO_INVERSES: [[u64; 4]; INVERSE_POWER_TABLE_LEN] =
                Self::PARAMETERS.inverse_powers;
            $($zeta)*

            fn pow_sqrt_exponent(value: &PastaField<Self>) -> PastaField<Self> {
                bento::addition_chain!(Power(*value), tonelli_shanks($modulus, 32),
                    emission = batched).0
            }
        }

        const _: () = {
            // The derivations assume the validated modulus domain, the
            // orientation arms must pair the pinned zeta
            // with its actual inverse.
            m255::assert_modulus(&<$marker as PrimeModulus>::MODULUS);
            let modulus = <$marker as PrimeModulus>::MODULUS;
            assert!(
                modulus[2] == 0 && modulus[3] == 1 << 62,
                "Montgomery kernels require p = 2^254 plus a 128-bit integer"
            );
            let product = m255::mul(
                &<$marker as PrimeModulus>::MODULUS,
                &<$marker as sealed::Parameters>::ZETA,
                &<$marker as sealed::Parameters>::ZETA_INVERSE,
            );
            let one = <$marker as sealed::Parameters>::R;
            assert!(
                product[0] == one[0]
                    && product[1] == one[1]
                    && product[2] == one[2]
                    && product[3] == one[3],
                "the pinned zeta and its declared inverse must multiply to one"
            );
        };
    };
}

// Batched emission and the planner's smaller odd-power table matched the
// supplied common chains in local measurements; keep the generated schedules.
pasta_field_parameters! {
    PallasBase,
    modulus: "0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
    zeta: squared_canonical_root,
}

pasta_field_parameters! {
    PallasScalar,
    modulus: "0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
    zeta: canonical_root,
}

/// Number of inverse domain sizes from `2^0` through `2^TWO_ADICITY`.
const INVERSE_POWER_TABLE_LEN: usize = TWO_ADICITY as usize + 1;

impl<M: PrimeModulus> PastaField<M> {
    /// Returns a primitive root of order `2^log_size`, or `None` above 32.
    ///
    /// The selected root is `5^((p - 1) / 2^log_size)`; order one returns one.
    pub fn root_of_unity(log_size: u32) -> Option<Self> {
        if log_size > TWO_ADICITY {
            return None;
        }
        Some(Self::from_montgomery(M::ROOTS[log_size as usize]))
    }

    /// Returns the inverse of [`Self::root_of_unity`], or `None` above 32.
    pub fn root_of_unity_inverse(log_size: u32) -> Option<Self> {
        if log_size > TWO_ADICITY {
            return None;
        }
        Some(Self::from_montgomery(M::INVERSE_ROOTS[log_size as usize]))
    }

    /// Returns the inverse of two.
    pub const fn two_inverse() -> Self {
        Self::from_montgomery(M::TWO_INVERSE)
    }

    /// Returns `2^-log_size` for any `u32` exponent, including zero.
    pub fn power_of_two_inverse(log_size: u32) -> Self {
        if let Some(entry) = M::POWER_OF_TWO_INVERSES.get(log_size as usize) {
            return Self::from_montgomery(*entry);
        }
        Self::two_inverse().pow_u64(u64::from(log_size))
    }

    /// Returns `5^(2^32)`, a generator of the subgroup of odd order.
    pub const fn delta() -> Self {
        Self::from_montgomery(M::DELTA)
    }

    /// Returns the selected primitive cube root of unity.
    ///
    /// For [`PallasBase`] this is `5^(2 * (p - 1) / 3)`; for
    /// [`PallasScalar`] it is `5^((p - 1) / 3)`.
    pub const fn zeta() -> Self {
        Self::from_montgomery(M::ZETA)
    }

    /// Returns the inverse of [`Self::zeta`], the other primitive cube root.
    pub const fn zeta_inverse() -> Self {
        Self::from_montgomery(M::ZETA_INVERSE)
    }
}

// Multiplicative interpretation of an addition chain, kept distinct from the
// field's additive operations.
#[derive(Clone)]
struct Power<M: PrimeModulus>(PastaField<M>);

impl<M: PrimeModulus> bento::addchain::AdditionChain for Power<M> {
    // Forced inlining duplicates the kernels' debug stack slots throughout
    // the unrolled chain. Let optimized builds decide whether to inline.
    #[inline]
    fn double(&self) -> Self {
        Self(self.0.square())
    }

    #[inline]
    fn add(&self, rhs: &Self) -> Self {
        Self(self.0.mul(&rhs.0))
    }

    #[inline]
    fn double_n(&self, count: usize) -> Self {
        self.double_n_add_impl(count, None)
    }

    #[inline]
    fn double_n_add(&self, count: usize, rhs: &Self) -> Self {
        self.double_n_add_impl(count, Some(&rhs.0.limbs))
    }
}

impl<M: PrimeModulus> Power<M> {
    fn double_n_add_impl(&self, mut count: usize, factor: Option<&[u64; 4]>) -> Self {
        let mut limbs = self.0.limbs;
        while count > 256 {
            limbs = super::montgomery::square_run::<M>(&limbs, 256, None);
            count -= 256;
        }
        Self(PastaField::from_montgomery(
            super::montgomery::square_run::<M>(&limbs, count, factor),
        ))
    }
}

// One setup feeds every parameter and table. Raw table entries remain checked
// residues; field constructors preserve the representation boundary.
struct FieldParameters {
    context: m255::MontgomeryContext,
    r3: [u64; 4],
    b448: [u64; 4],
    roots: [[u64; 4]; INVERSE_POWER_TABLE_LEN],
    inverse_roots: [[u64; 4]; INVERSE_POWER_TABLE_LEN],
    inverse_powers: [[u64; 4]; INVERSE_POWER_TABLE_LEN],
    corrections: [[u64; 4]; SAFEGCD_BATCHES],
    cube_root: [u64; 4],
    delta: [u64; 4],
}

const fn add_wide(a: [u64; 8], b: [u64; 8]) -> [u64; 8] {
    let mut result = [0; 8];
    let mut carry = 0u128;
    let mut i = 0;
    while i < 8 {
        carry += a[i] as u128 + b[i] as u128;
        result[i] = carry as u64;
        carry >>= 64;
        i += 1;
    }
    assert!(carry == 0, "wide sum must fit");
    result
}

const fn assert_redc_bound(value: [u64; 8], modulus: &[u64; 4]) {
    assert!(
        !u256::ge(&[value[4], value[5], value[6], value[7]], modulus),
        "kernel input must be below pR"
    );
}

impl FieldParameters {
    const fn new(modulus: [u64; 4]) -> Self {
        let context = m255::MontgomeryContext::new(modulus);
        let r2 = context.r2();
        let r3 = context.mul(&r2, &r2);
        let (sum, carry) = u256::add_with_carry(&r2, &r3);
        assert!(
            carry == 0 && !u256::ge(&sum, &modulus),
            "wide decoder requires R2 + R3 < p"
        );
        // The Horner numerator is (V + D)*R2, V < p, D < R.
        assert_redc_bound(
            add_wide(
                u256::mul_wide(&u256::sub_u64(&modulus, 1), &r2),
                u256::mul_wide(&[u64::MAX; 4], &r2),
            ),
            &modulus,
        );
        let product = u256::mul_wide(&u256::sub_u64(&modulus, 1), &u256::sub_u64(&modulus, 1));
        let three_products = add_wide(add_wide(product, product), product);
        assert_redc_bound(three_products, &modulus);
        let four_products = add_wide(three_products, product);
        assert!(
            u256::ge(
                &[
                    four_products[4],
                    four_products[5],
                    four_products[6],
                    four_products[7]
                ],
                &modulus
            ),
            "four products require a wider reduction bound"
        );
        // Bound every lazy square by floor((B² + (R-1)p)/R), starting
        // from B=p-1. Checking all 256 steps avoids an asymptotic argument.
        let correction = u256::mul_wide(&[u64::MAX; 4], &modulus);
        let (twice_p, carry) = u256::add_with_carry(&modulus, &modulus);
        assert!(carry == 0);
        let mut bound = u256::sub_u64(&modulus, 1);
        let mut i = 0;
        while i < 256 {
            let square = u256::mul_wide(&bound, &bound);
            assert_redc_bound(square, &modulus);
            let next = add_wide(square, correction);
            let next_bound = [next[4], next[5], next[6], next[7]];
            assert!(
                u256::ge(&next_bound, &bound),
                "lazy bounds must be monotone"
            );
            bound = next_bound;
            assert!(!u256::ge(&bound, &twice_p));
            i += 1;
        }
        assert_redc_bound(
            u256::mul_wide(&bound, &u256::sub_u64(&modulus, 1)),
            &modulus,
        );
        let base = context.from_u64(GENERATOR);
        let mut roots = [[0; 4]; INVERSE_POWER_TABLE_LEN];
        let mut inverse_roots = roots;
        roots[TWO_ADICITY as usize] =
            context.pow(&base, &u256::odd_cofactor(&modulus, TWO_ADICITY));
        inverse_roots[TWO_ADICITY as usize] =
            context.pow(&roots[TWO_ADICITY as usize], &u256::sub_u64(&modulus, 2));
        let mut i = TWO_ADICITY as usize;
        while i > 0 {
            roots[i - 1] = context.mul(&roots[i], &roots[i]);
            inverse_roots[i - 1] = context.mul(&inverse_roots[i], &inverse_roots[i]);
            i -= 1;
        }
        let mut inverse_powers = [[0; 4]; INVERSE_POWER_TABLE_LEN];
        inverse_powers[0] = context.one();
        let mut i = 1;
        while i < INVERSE_POWER_TABLE_LEN {
            let prev = inverse_powers[i - 1];
            let even = if prev[0] & 1 == 0 {
                prev
            } else {
                u256::add_with_carry(&prev, &modulus).0
            };
            inverse_powers[i] = u256::shr(&even, 1);
            i += 1;
        }
        let mut corrections = [[0; 4]; SAFEGCD_BATCHES];
        let mut value = context.one();
        let mut i = 0;
        while i < SAFEGCD_BATCHES {
            value = m255::add(&modulus, &value, &value);
            value = m255::add(&modulus, &value, &value);
            corrections[i] = value;
            i += 1;
        }
        Self {
            context,
            r3,
            b448: context.from_u256(&[0, 0, 0, 1]),
            roots,
            inverse_roots,
            inverse_powers,
            corrections,
            cube_root: context.pow(&base, &u256::div_exact_u64(&u256::sub_u64(&modulus, 1), 3)),
            delta: context.pow(&base, &[1u64 << TWO_ADICITY, 0, 0, 0]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use bento::addchain::AdditionChain;

    fn check_runs<M: PrimeModulus>() {
        let value = Power(PastaField::<M>::from_u64(7));
        let factor = Power(PastaField::<M>::from_u64(11));
        let mut expected = value.0;
        for count in 0..=513 {
            if [0, 1, 255, 256, 257, 512, 513].contains(&count) {
                assert_eq!(value.double_n(count).0, expected);
                assert_eq!(
                    value.double_n_add(count, &factor).0,
                    expected.mul(&factor.0)
                );
            }
            expected = expected.square();
        }
    }

    #[test]
    fn power_hooks_normalize_between_bounded_runs() {
        check_runs::<PallasBase>();
        check_runs::<PallasScalar>();
    }
}
