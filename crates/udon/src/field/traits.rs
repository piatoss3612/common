//! Traits for code that is generic over a prime field.
//!
//! Generic consumers, such as evaluation domains, polynomial utilities, and
//! proof systems parameterized over a curve cycle, name a field through
//! [`Field`] instead of a concrete type. [`PastaField`](super::PastaField)
//! implements these traits for both Pasta fields by delegating to its inherent
//! methods and constants.
//!
//! The traits describe 256-bit prime fields with a 32-byte canonical
//! encoding and, for [`FftField`], the constants a radix-2 evaluation domain
//! and the Pasta curve endomorphism need. Randomness stays with the caller:
//! [`random`](super::random) reduces 64 caller-supplied bytes through
//! [`Field::from_uniform_bytes`] and depends on no particular random number
//! generator.
//!
//! Every operation is variable-time, like the arithmetic it delegates to.

use core::{
    fmt::Debug,
    iter::Sum,
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// A 256-bit prime field with operator arithmetic, identities, and a
/// canonical 32-byte encoding.
///
/// The operator supertraits cover by-value operands and right operands by
/// reference, which is what generic code written as `a * b`, `a * &b`, and
/// `a *= &b` needs. Equality compares field elements, not representations.
/// Implementations may hold redundant representations internally, as
/// [`PastaField`](super::PastaField) does with its loose residues.
pub trait Field:
    Copy
    + Eq
    + Default
    + Debug
    + Send
    + Sync
    + 'static
    + From<u64>
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Neg<Output = Self>
    + for<'a> Add<&'a Self, Output = Self>
    + for<'a> Sub<&'a Self, Output = Self>
    + for<'a> Mul<&'a Self, Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + for<'a> AddAssign<&'a Self>
    + for<'a> SubAssign<&'a Self>
    + for<'a> MulAssign<&'a Self>
    + Sum
    + for<'a> Sum<&'a Self>
{
    /// The additive identity.
    const ZERO: Self;

    /// The multiplicative identity.
    const ONE: Self;

    /// The modulus as ordinary little-endian 64-bit limbs.
    const MODULUS: [u64; 4];

    /// The bit length of the modulus.
    const NUM_BITS: u32;

    /// The largest bit length whose integers are all distinct field elements:
    /// one less than [`NUM_BITS`](Self::NUM_BITS).
    const CAPACITY: u32;

    /// Returns whether this is the additive identity.
    fn is_zero(&self) -> bool;

    /// Returns whether the canonical integer representative is odd.
    fn is_odd(&self) -> bool;

    /// Returns `self * self`.
    fn square(&self) -> Self;

    /// Returns `2 * self`.
    fn double(&self) -> Self;

    /// Returns the multiplicative inverse, or `None` for zero.
    fn invert(&self) -> Option<Self>;

    /// Replaces nonzero values by their inverses, preserving zeros.
    ///
    /// Implements [`super::batch_invert`]'s contract: scratch bounds the batch
    /// size, empty scratch uses individual inversions, and unused scratch is
    /// untouched. Empty and all-zero batches perform no inversion. No
    /// allocation is performed.
    fn batch_invert(values: &mut [Self], scratch: &mut [Self]);

    /// Returns a square root, or `None` for a nonsquare.
    ///
    /// Either root may be returned; zero returns `Some(ZERO)`.
    fn sqrt(&self) -> Option<Self>;

    /// Raises this value to an unsigned exponent; exponent zero returns one.
    ///
    /// The multiplication schedule depends on the exponent.
    fn pow_u64(&self, exponent: u64) -> Self;

    /// Encodes the canonical integer representative as 32 little-endian bytes.
    fn to_bytes(&self) -> [u8; 32];

    /// Decodes a canonical encoding, rejecting integers at or above the
    /// modulus.
    fn from_bytes(bytes: [u8; 32]) -> Option<Self>;

    /// Reduces 64 little-endian bytes into the field.
    ///
    /// Uniformly random input yields output whose distribution is
    /// statistically indistinguishable from uniform, which is how consumers
    /// derive elements from a random or hashed source.
    fn from_uniform_bytes(bytes: &[u8; 64]) -> Self;

    /// Sums products from paired operands; an empty iterator returns zero.
    ///
    /// The default multiplies each pair separately and sums the results.
    /// Implementations may defer reduction, as the Pasta fields do.
    /// [`super::dot`] uses this hook for its paired input sequences.
    fn sum_of_product_pairs<'a>(pairs: impl IntoIterator<Item = (&'a Self, &'a Self)>) -> Self {
        pairs.into_iter().map(|(lhs, rhs)| *lhs * rhs).sum()
    }

    /// Embeds an unsigned 128-bit integer.
    ///
    /// The default assembles the value from two 64-bit halves.
    fn from_u128(value: u128) -> Self {
        let two_to_the_64 = Self::from(1u64 << 63).double();
        Self::from((value >> 64) as u64) * two_to_the_64 + Self::from(value as u64)
    }

    /// Converts ordinary little-endian limbs, returning `None` if they are at
    /// least the modulus.
    ///
    /// The default encodes the limbs as bytes for [`Self::from_bytes`].
    fn from_limbs(limbs: [u64; 4]) -> Option<Self> {
        let mut bytes = [0u8; 32];
        for (chunk, limb) in bytes.chunks_exact_mut(8).zip(limbs) {
            chunk.copy_from_slice(&limb.to_le_bytes());
        }
        Self::from_bytes(bytes)
    }

    /// Returns the bits of the canonical representative, least significant
    /// first.
    fn to_le_bits(&self) -> [bool; 256] {
        let bytes = self.to_bytes();
        let mut bits = [false; 256];
        for (index, bit) in bits.iter_mut().enumerate() {
            *bit = (bytes[index / 8] >> (index % 8)) & 1 == 1;
        }
        bits
    }
}

/// A [`Field`] with radix-2 transforms and the constants an evaluation domain
/// and the Pasta endomorphism need.
///
/// With `g` the [`MULTIPLICATIVE_GENERATOR`](Self::MULTIPLICATIVE_GENERATOR)
/// and `s` the [`TWO_ADICITY`](Self::TWO_ADICITY) of `p - 1`:
/// [`ROOT_OF_UNITY`](Self::ROOT_OF_UNITY) is `g^((p - 1) / 2^s)`, of order
/// exactly `2^s`; [`DELTA`](Self::DELTA) is `g^(2^s)`, which generates the
/// odd-order part of the multiplicative group; and [`ZETA`](Self::ZETA) has
/// order three. The default root lookups square down from the maximal root;
/// implementations with tables override them.
pub trait FftField: Field {
    /// `s`, the largest `s` such that `2^s` divides `p - 1`: the logarithm of
    /// the largest supported power-of-two domain.
    const TWO_ADICITY: u32;

    /// The generator of the multiplicative group from which the other
    /// constants are derived.
    const MULTIPLICATIVE_GENERATOR: Self;

    /// A primitive root of unity of order `2^TWO_ADICITY`.
    const ROOT_OF_UNITY: Self;

    /// The inverse of [`ROOT_OF_UNITY`](Self::ROOT_OF_UNITY).
    const ROOT_OF_UNITY_INVERSE: Self;

    /// The inverse of two.
    const TWO_INVERSE: Self;

    /// The generator raised to `2^TWO_ADICITY`.
    const DELTA: Self;

    /// A primitive cube root of unity: the scalar by which the curve
    /// endomorphism `(x, y) -> (zeta * x, y)` multiplies.
    const ZETA: Self;

    /// Replaces coefficients with evaluations at `domain`'s elements.
    ///
    /// Both sides use natural order. Implements
    /// [`Domain::transform`](crate::fft::Domain::transform) without allocation.
    ///
    /// # Panics
    ///
    /// Panics before mutation if `values.len()` differs from the domain size.
    fn fft(domain: crate::fft::Domain<Self>, values: &mut [Self]);

    /// Replaces evaluations at `domain`'s elements with normalized coefficients.
    ///
    /// Both sides use natural order. Implements
    /// [`Domain::inverse_transform`](crate::fft::Domain::inverse_transform)
    /// without allocation, including division by the domain size.
    ///
    /// # Panics
    ///
    /// Panics before mutation if `values.len()` differs from the domain size.
    fn ifft(domain: crate::fft::Domain<Self>, values: &mut [Self]);

    /// Returns a primitive root of unity of order `2^log_size`, or `None`
    /// when `log_size` exceeds the two-adicity.
    fn root_of_unity(log_size: u32) -> Option<Self> {
        (log_size <= Self::TWO_ADICITY).then(|| {
            let mut root = Self::ROOT_OF_UNITY;
            for _ in log_size..Self::TWO_ADICITY {
                root = root.square();
            }
            root
        })
    }

    /// Returns the inverse of [`root_of_unity`](Self::root_of_unity) for the
    /// same size.
    fn root_of_unity_inverse(log_size: u32) -> Option<Self> {
        (log_size <= Self::TWO_ADICITY).then(|| {
            let mut root = Self::ROOT_OF_UNITY_INVERSE;
            for _ in log_size..Self::TWO_ADICITY {
                root = root.square();
            }
            root
        })
    }

    /// Returns `2^-log_size`, the normalization of a `2^log_size`-point
    /// inverse transform.
    fn power_of_two_inverse(log_size: u32) -> Self {
        Self::TWO_INVERSE.pow_u64(u64::from(log_size))
    }
}

/// A [`Field`] whose products can be accumulated before reduction.
///
/// A sum of products pays one reduction instead of one per term:
/// [`mul_accumulate`](Self::mul_accumulate) adds a product to the
/// accumulator and [`reduce`](Self::reduce) returns the field value once.
/// The accumulator starts from [`Default`].
pub trait DeferredField: Field {
    /// The unreduced accumulator.
    type Accumulator: Default;

    /// Adds `lhs * rhs` to the accumulator.
    fn mul_accumulate(accumulator: &mut Self::Accumulator, lhs: &Self, rhs: &Self);

    /// Reduces the accumulated sum to a field element.
    fn reduce(accumulator: Self::Accumulator) -> Self;
}
