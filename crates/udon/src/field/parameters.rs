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
// roots have opposite orientations relative to 5^((p - 1)/3); verify both the
// inverse pairing and the fixed square-root exponent at compile time.
macro_rules! pasta_field_parameters {
    ($marker:ty, modulus: $modulus:literal, sqrt_exponent: $sqrt:literal,
     zeta: canonical_root $(,)?) => {
        pasta_field_parameters! {
            @impl $marker, $modulus, $sqrt,
            const ZETA: [u64; 4] = m255::cube_root_of_unity(&Self::MODULUS, GENERATOR);
            const ZETA_INVERSE: [u64; 4] =
                m255::mul(&Self::MODULUS, &Self::ZETA, &Self::ZETA);
        }
    };
    ($marker:ty, modulus: $modulus:literal, sqrt_exponent: $sqrt:literal,
     zeta: squared_canonical_root $(,)?) => {
        pasta_field_parameters! {
            @impl $marker, $modulus, $sqrt,
            const ZETA: [u64; 4] =
                m255::mul(&Self::MODULUS, &Self::ZETA_INVERSE, &Self::ZETA_INVERSE);
            const ZETA_INVERSE: [u64; 4] = m255::cube_root_of_unity(&Self::MODULUS, GENERATOR);
        }
    };
    (@impl $marker:ty, $modulus:literal, $sqrt:literal, $($zeta:item)*) => {
        impl PrimeModulus for $marker {
            const MODULUS: [u64; 4] = u256::from_hex($modulus);
        }

        impl sealed::Parameters for $marker {
            const MONTGOMERY_INV: u64 = m255::reduction_coefficient(Self::MODULUS[0]);
            const R: [u64; 4] = m255::one(&Self::MODULUS);
            const R2: [u64; 4] = m255::r2(&Self::MODULUS);
            const B448: [u64; 4] = m255::pow2_mod(&Self::MODULUS, 448);
            const SQRT_EXPONENT: [u64; 4] =
                u256::tonelli_shanks_exponent(&Self::MODULUS, TWO_ADICITY);
            const TWO_INVERSE: [u64; 4] = m255::two_inverse(&Self::MODULUS);
            const ROOT_OF_UNITY: [u64; 4] =
                m255::two_adic_root_of_unity(&Self::MODULUS, GENERATOR, TWO_ADICITY);
            const ROOT_OF_UNITY_INVERSE: [u64; 4] =
                m255::invert_prime(&Self::MODULUS, &Self::ROOT_OF_UNITY);
            const DELTA: [u64; 4] = m255::odd_order_generator(&Self::MODULUS, GENERATOR, TWO_ADICITY);
            const MODULUS_SIGNED62: [i64; 5] = to_signed62(&Self::MODULUS);
            const SAFEGCD_CORRECTIONS: [[u64; 4]; SAFEGCD_BATCHES] =
                m255::safegcd_corrections_62_64(&Self::MODULUS);
            const POWER_OF_TWO_INVERSES: [[u64; 4]; INVERSE_POWER_TABLE_LEN] =
                m255::inverse_powers_of_two(&Self::MODULUS);
            $($zeta)*

            fn pow_sqrt_exponent(value: &PastaField<Self>) -> PastaField<Self> {
                bento::addition_chain!(Power(*value), $sqrt).0
            }
        }

        const _: () = {
            // The derivations assume the validated modulus domain, the
            // transcribed schedule exponent must be the derived square-root
            // exponent, and the orientation arms must pair the pinned zeta
            // with its actual inverse.
            m255::assert_modulus(&<$marker as PrimeModulus>::MODULUS);
            let modulus = <$marker as PrimeModulus>::MODULUS;
            assert!(
                modulus[2] == 0 && modulus[3] == 1 << 62,
                "Montgomery kernels require p = 2^254 plus a 128-bit integer"
            );
            let transcribed: [u64; 4] = u256::from_hex(stringify!($sqrt));
            let derived = <$marker as sealed::Parameters>::SQRT_EXPONENT;
            assert!(
                transcribed[0] == derived[0]
                    && transcribed[1] == derived[1]
                    && transcribed[2] == derived[2]
                    && transcribed[3] == derived[3],
                "the schedule exponent must be the derived square-root exponent"
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

pasta_field_parameters! {
    PallasBase,
    modulus: "0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001",
    sqrt_exponent: 0x000000002000000000000000000000000000000011234c7e04a67c8dcc969876,
    zeta: squared_canonical_root,
}

pasta_field_parameters! {
    PallasScalar,
    modulus: "0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001",
    sqrt_exponent: 0x000000002000000000000000000000000000000011234c7e04ca546ec6237590,
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
        let mut root = Self::from_montgomery(M::ROOT_OF_UNITY);
        for _ in log_size..TWO_ADICITY {
            root = root.square();
        }
        Some(root)
    }

    /// Returns the inverse of [`Self::root_of_unity`], or `None` above 32.
    pub fn root_of_unity_inverse(log_size: u32) -> Option<Self> {
        if log_size > TWO_ADICITY {
            return None;
        }
        let mut root = Self::from_montgomery(M::ROOT_OF_UNITY_INVERSE);
        for _ in log_size..TWO_ADICITY {
            root = root.square();
        }
        Some(root)
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
}
