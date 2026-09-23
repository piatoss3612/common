//! Traits for field arithmetic and optional representation capabilities.
//!
//! Generic consumers, such as evaluation domains, polynomial utilities, and
//! proof systems parameterized over a curve cycle, name a field through
//! [`Field`] instead of a concrete type. [`PastaField`](super::PastaField)
//! implements these traits for both Pasta fields by delegating to its inherent
//! methods and constants.
//!
//! [`PrimeField`] adds a modulus and field-specific encoding and bit widths.
//! [`FftField`] supplies radix-2 transforms, while [`CubeRootField`] supplies
//! the root used by the Pasta curve endomorphism. Randomness stays with the
//! caller: [`random`](super::random) reduces 64 caller-supplied bytes through
//! [`PrimeField::from_uniform_bytes`] and depends on no particular random number
//! generator.
//!
//! Every operation is variable-time, like the arithmetic it delegates to.

use super::pasta::TWO_ADICITY;
use super::{CanonicalUint, PastaField, PrimeModulus, ProductSum};
use crate::{
    exec::{ExecutionOptions, SerialExecutor},
    fft::{Domain, Transform},
};

use core::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// A field with operator arithmetic and additive and multiplicative identities.
///
/// The operator supertraits cover by-value operands and right operands by
/// reference, which is what generic code written as `a * b`, `a * &b`, and
/// `a *= &b` needs. Equality compares field elements, not representations.
/// Implementations may hold redundant representations internally, as
/// [`PastaField`](super::PastaField) does with its loose residues.
/// Iterator sums and products accept owned or borrowed elements; empty
/// iterators return [`ZERO`](Self::ZERO) and [`ONE`](Self::ONE), respectively.
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
    + Product
    + for<'a> Product<&'a Self>
{
    /// The additive identity.
    const ZERO: Self;

    /// The multiplicative identity.
    const ONE: Self;

    /// Returns whether this is the additive identity.
    fn is_zero(&self) -> bool;

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
}

/// A prime field with canonical little-endian representations.
///
/// Implementations choose the widths of their limbs, bytes, and bits.
pub trait PrimeField: Field {
    /// The canonical little-endian byte representation.
    ///
    /// Its length is fixed for the field and must hold every canonical
    /// representative. Zero encodes as all zero bytes.
    type Repr: AsRef<[u8]> + AsMut<[u8]> + Copy + Debug + Eq + Send + Sync + 'static;

    /// Ordinary little-endian 64-bit limbs, wide enough to hold the modulus.
    type Limbs: AsRef<[u64]> + Copy + Debug + Eq + Send + Sync + 'static;

    /// The canonical little-endian bits, including zero padding to the full
    /// width of [`Repr`](Self::Repr).
    type Bits: AsRef<[bool]> + Copy + Debug + Eq + Send + Sync + 'static;

    /// The modulus as ordinary little-endian 64-bit limbs.
    const MODULUS: Self::Limbs;

    /// The bit length of the modulus.
    const NUM_BITS: u32;

    /// The largest bit length whose integers are all distinct field elements:
    /// one less than [`NUM_BITS`](Self::NUM_BITS).
    const CAPACITY: u32;

    /// Returns whether the canonical integer representative is odd.
    fn is_odd(&self) -> bool;

    /// Encodes the canonical integer representative in little-endian bytes.
    fn to_bytes(&self) -> Self::Repr;

    /// Decodes a canonical encoding, rejecting integers at or above the
    /// modulus.
    fn from_bytes(bytes: Self::Repr) -> Option<Self>;

    /// Reduces 64 little-endian bytes into the field.
    ///
    /// For a modulus of at most 384 bits, uniformly random input yields a
    /// distribution within `2^-128` statistical distance of uniform. Larger
    /// fields require more input entropy for the same bound; this operation
    /// still performs reduction, but makes no such uniformity guarantee.
    fn from_uniform_bytes(bytes: &[u8; 64]) -> Self;

    /// Converts ordinary little-endian limbs, returning `None` if they are at
    /// least the modulus.
    ///
    /// The default encodes the limbs as bytes for [`Self::from_bytes`].
    fn from_limbs(limbs: Self::Limbs) -> Option<Self> {
        let mut bytes = Self::ZERO.to_bytes();
        for (index, byte) in limbs
            .as_ref()
            .iter()
            .flat_map(|limb| limb.to_le_bytes())
            .enumerate()
        {
            if let Some(slot) = bytes.as_mut().get_mut(index) {
                *slot = byte;
            } else if byte != 0 {
                return None;
            }
        }
        Self::from_bytes(bytes)
    }

    /// Returns the bits of the canonical representative, least significant
    /// first.
    fn to_le_bits(&self) -> Self::Bits;
}

/// A [`PrimeField`] with radix-2 transforms and evaluation-domain constants.
///
/// With `g` the [`MULTIPLICATIVE_GENERATOR`](Self::MULTIPLICATIVE_GENERATOR)
/// and `s` the [`TWO_ADICITY`](Self::TWO_ADICITY) of `p - 1`:
/// [`ROOT_OF_UNITY`](Self::ROOT_OF_UNITY) is `g^((p - 1) / 2^s)`, of order
/// exactly `2^s`; [`DELTA`](Self::DELTA) is `g^(2^s)`, which generates the
/// odd-order part of the multiplicative group. The default root lookups
/// square down from the maximal root; implementations with tables override them.
pub trait FftField: PrimeField {
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

/// A field with a chosen primitive cube root of unity.
///
/// This capability is independent of radix-2 FFT support. Curve
/// implementations must choose compatible roots for their endomorphisms.
pub trait CubeRootField: Field {
    /// The chosen element of multiplicative order three.
    const ZETA: Self;
}

impl<M: PrimeModulus> Field for PastaField<M> {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;
    fn is_zero(&self) -> bool {
        PastaField::is_zero(self)
    }

    fn square(&self) -> Self {
        PastaField::square(self)
    }

    fn double(&self) -> Self {
        PastaField::double(self)
    }

    fn invert(&self) -> Option<Self> {
        PastaField::invert(self)
    }

    fn batch_invert(values: &mut [Self], scratch: &mut [Self]) {
        super::pasta::batch_invert(values, scratch)
    }

    fn sqrt(&self) -> Option<Self> {
        self.reduce().sqrt().map(PastaField::into_loose)
    }

    fn pow_u64(&self, exponent: u64) -> Self {
        PastaField::pow_u64(self, exponent)
    }

    fn from_u128(value: u128) -> Self {
        PastaField::from_u128(value)
    }

    fn sum_of_product_pairs<'a>(pairs: impl IntoIterator<Item = (&'a Self, &'a Self)>) -> Self {
        PastaField::sum_of_product_pairs(pairs)
    }
}

impl<M: PrimeModulus> PrimeField for PastaField<M> {
    type Repr = [u8; 32];
    type Limbs = [u64; 4];
    type Bits = [bool; 256];

    const MODULUS: [u64; 4] = M::MODULUS;
    const NUM_BITS: u32 = 256 - M::MODULUS[3].leading_zeros();
    const CAPACITY: u32 = 255 - M::MODULUS[3].leading_zeros();

    fn is_odd(&self) -> bool {
        PastaField::is_odd(self)
    }

    fn to_bytes(&self) -> [u8; 32] {
        PastaField::to_bytes(*self)
    }

    fn from_bytes(bytes: [u8; 32]) -> Option<Self> {
        PastaField::from_bytes(bytes)
    }

    fn from_uniform_bytes(bytes: &[u8; 64]) -> Self {
        PastaField::from_wide_bytes_reduced(bytes)
    }

    fn from_limbs(limbs: [u64; 4]) -> Option<Self> {
        PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs))
    }

    fn to_le_bits(&self) -> Self::Bits {
        let bytes = PastaField::to_bytes(*self);
        core::array::from_fn(|index| (bytes[index / 8] >> (index % 8)) & 1 == 1)
    }
}

impl<M: PrimeModulus> FftField for PastaField<M> {
    const TWO_ADICITY: u32 = TWO_ADICITY;
    const MULTIPLICATIVE_GENERATOR: Self = Self::MULTIPLICATIVE_GENERATOR;
    const ROOT_OF_UNITY: Self = match Self::root_of_unity(TWO_ADICITY) {
        Some(root) => root,
        None => panic!("the two-adicity is a supported root order"),
    };
    const ROOT_OF_UNITY_INVERSE: Self = match Self::root_of_unity_inverse(TWO_ADICITY) {
        Some(root) => root,
        None => panic!("the two-adicity is a supported root order"),
    };
    const TWO_INVERSE: Self = Self::TWO_INVERSE;
    const DELTA: Self = Self::DELTA;

    fn fft(domain: Domain<Self>, values: &mut [Self]) {
        Transform::new(domain.subgroup())
            .forward(
                values,
                ExecutionOptions::default(),
                &SerialExecutor,
                &mut [],
            )
            .expect("a serial subgroup transform supports empty scratch");
    }

    fn ifft(domain: Domain<Self>, values: &mut [Self]) {
        Transform::new(domain.subgroup())
            .inverse(
                values,
                ExecutionOptions::default(),
                &SerialExecutor,
                &mut [],
            )
            .expect("a serial subgroup transform supports empty scratch");
    }

    fn root_of_unity(log_size: u32) -> Option<Self> {
        PastaField::root_of_unity(log_size)
    }

    fn root_of_unity_inverse(log_size: u32) -> Option<Self> {
        PastaField::root_of_unity_inverse(log_size)
    }

    fn power_of_two_inverse(log_size: u32) -> Self {
        PastaField::power_of_two_inverse(log_size)
    }
}

impl<M: PrimeModulus> DeferredField for PastaField<M> {
    type Accumulator = ProductSum<M>;

    fn mul_accumulate(accumulator: &mut ProductSum<M>, lhs: &Self, rhs: &Self) {
        accumulator.add_product(lhs, rhs);
    }

    fn reduce(accumulator: ProductSum<M>) -> Self {
        accumulator.finish()
    }
}

impl<M: PrimeModulus> CubeRootField for PastaField<M> {
    const ZETA: Self = Self::ZETA;
}
