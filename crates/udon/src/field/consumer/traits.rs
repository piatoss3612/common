//! The optional consumer interface for field arithmetic.
//!
//! Generic consumers, such as evaluation domains, polynomial utilities, and
//! proof systems parameterized over a curve cycle, name a field through
//! [`Field`] instead of a concrete type. [`FieldAdapter`](super::FieldAdapter)
//! implements this trait for both Pasta fields by delegating to the native
//! methods and constants. Every method is required, so each implementation
//! explicitly selects its arithmetic instead of inheriting a fallback.
//!
//! [`Field`] includes canonical representations, radix-2 transforms, and
//! product accumulation. Randomness stays with the caller: [`Field::random`]
//! accepts a callback that supplies uniform bytes, then reduces them through
//! [`Field::from_uniform_bytes`].
//!
//! Every operation is variable-time, like the arithmetic it delegates to.

use crate::fft::{Domain, FftError};

use core::{
    fmt::Debug,
    iter::{Product, Sum},
    ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign},
};

/// An odd prime field with canonical encodings, transforms, and product sums.
///
/// The operator supertraits cover by-value operands and right operands by
/// reference, which is what generic code written as `a * b`, `a * &b`, and
/// `a *= &b` needs. Equality compares field elements, not representations.
/// Implementations may hold redundant representations internally, as
/// [`PastaField`](crate::field::PastaField) does with its loose residues.
/// Iterator sums and products accept owned or borrowed elements; empty
/// iterators return [`ZERO`](Self::ZERO) and [`ONE`](Self::ONE), respectively.
/// Implementations choose their own limb, byte, and bit widths.
///
/// Transform methods and product accumulators let generic callers use the
/// implementation's arithmetic kernels through this single interface.
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

    /// A running sum of products, initialized to zero by [`Default`].
    ///
    /// [`mul_accumulate`](Self::mul_accumulate) adds each product, and
    /// [`reduce`](Self::reduce) returns the field value. Implementations can
    /// share reduction across products, as Pasta's
    /// [`ProductSum`](crate::field::ProductSum) does.
    type Accumulator: Default;

    /// The additive identity.
    const ZERO: Self;

    /// The multiplicative identity.
    const ONE: Self;

    /// The modulus as ordinary little-endian 64-bit limbs.
    const MODULUS: Self::Limbs;

    /// The bit length of the modulus.
    const NUM_BITS: u32;

    /// The largest bit length whose integers are all distinct field elements:
    /// one less than [`NUM_BITS`](Self::NUM_BITS).
    const CAPACITY: u32;

    /// `s`, the largest `s` such that `2^s` divides `p - 1`: the logarithm of
    /// the largest supported power-of-two domain.
    const TWO_ADICITY: u32;

    /// The generator of the multiplicative group from which the other
    /// constants are derived.
    const MULTIPLICATIVE_GENERATOR: Self;

    /// A primitive root of unity of order `2^TWO_ADICITY`.
    ///
    /// Equals `MULTIPLICATIVE_GENERATOR^((p - 1) / 2^TWO_ADICITY)`.
    const ROOT_OF_UNITY: Self;

    /// The inverse of [`ROOT_OF_UNITY`](Self::ROOT_OF_UNITY).
    const ROOT_OF_UNITY_INVERSE: Self;

    /// The inverse of two.
    const TWO_INVERSE: Self;

    /// The generator raised to `2^TWO_ADICITY`, generating the odd-order
    /// part of the multiplicative group.
    const DELTA: Self;

    /// A chosen cube root of unity, primitive when three divides `p - 1`.
    ///
    /// Equals [`ONE`](Self::ONE) when the field has no nontrivial cube root.
    /// [`EndomorphismAffine`](crate::curve::EndomorphismAffine) requires
    /// compatible primitive roots in the base and scalar fields.
    const ZETA: Self;

    /// Returns whether this is the additive identity.
    fn is_zero(&self) -> bool;

    /// Returns `self * self`.
    fn square(&self) -> Self;

    /// Returns `self * multiplier + addend`.
    fn mul_add(&self, multiplier: &Self, addend: &Self) -> Self;

    /// Returns `2 * self`.
    fn double(&self) -> Self;

    /// Returns the multiplicative inverse, or `None` for zero.
    fn invert(&self) -> Option<Self>;

    /// Replaces nonzero values by their inverses, preserving zeros.
    ///
    /// Implements [`crate::field::batch_invert`]'s contract: scratch bounds the batch
    /// size, empty scratch uses individual inversions, and unused scratch is
    /// untouched. Empty and all-zero batches perform no inversion. No
    /// allocation is performed.
    fn batch_invert(values: &mut [Self], scratch: &mut [Self]);

    /// Inverts nonzero entries across disjoint slices with shared scratch.
    ///
    /// Applies [`Self::batch_invert`]'s scratch and zero-preservation contract
    /// to the concatenation of `groups`, without copying or allocation. Empty
    /// groups are allowed. Each group's `AsMut::as_mut` must expose the same
    /// slice throughout the call.
    fn batch_invert_groups(groups: &mut [impl AsMut<[Self]>], scratch: &mut [Self]);

    /// Returns a square root, or `None` for a nonsquare.
    ///
    /// Either root may be returned; zero returns `Some(ZERO)`.
    fn sqrt(&self) -> Option<Self>;

    /// Raises this value to an unsigned exponent; exponent zero returns one.
    ///
    /// The multiplication schedule depends on the exponent.
    fn pow_u64(&self, exponent: u64) -> Self;

    /// Returns the inner product of two equal-length slices, or zero when empty.
    ///
    /// Pasta adapters use the native slice kernels, sharing reduction across
    /// products.
    ///
    /// # Panics
    ///
    /// Panics if the lengths differ.
    fn sum_of_products_slice(lhs: &[Self], rhs: &[Self]) -> Self;

    /// Sums products from paired operands; an empty iterator returns zero.
    ///
    /// Implementations may defer reduction, as the Pasta fields do.
    fn sum_of_product_pairs<'a>(pairs: impl IntoIterator<Item = (&'a Self, &'a Self)>) -> Self;

    /// Reduces an unsigned 128-bit integer modulo the field's modulus.
    fn from_u128(value: u128) -> Self;

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

    /// Samples an element by reducing 64 bytes from the caller's source.
    ///
    /// Calls `fill` exactly once with the entire buffer. The callback must fill
    /// it with uniformly random bytes; cryptographic use requires a
    /// cryptographically secure source. The result must equal
    /// [`Self::from_uniform_bytes`] on those bytes, without rejection sampling
    /// or additional draws. Pasta adapters delegate to the native sampler.
    ///
    /// The modulus must have at most 384 bits, giving statistical distance from
    /// uniform below `2^-128`. Implementations must reject calls for wider
    /// moduli before invoking `fill`.
    fn random(fill: impl FnOnce(&mut [u8; 64])) -> Self;

    /// Converts ordinary little-endian limbs, returning `None` if they are at
    /// least the modulus.
    fn from_limbs(limbs: Self::Limbs) -> Option<Self>;

    /// Returns the bits of the canonical representative, least significant
    /// first.
    fn to_le_bits(&self) -> Self::Bits;

    /// Constructs the canonical domain of `2^log_size` elements.
    ///
    /// Size one is supported. Returns [`FftError::InvalidSize`] above the
    /// field's two-adicity, or [`FftError::SizeOverflow`] if the element count
    /// does not fit `usize` or its slice would exceed `isize::MAX` bytes.
    /// Pasta delegates to the native [`Domain::new`] constructor. Other
    /// implementations can construct the descriptor with [`Domain::from_field`].
    fn domain(log_size: u32) -> Result<Domain<Self>, FftError>;

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

    /// Evaluates a prefix of the domain's Lagrange basis at `point`.
    ///
    /// Implements [`Domain::lagrange_evaluations`], including its node-index
    /// result, bounded scratch, and validation before mutation. Pasta delegates
    /// to its native range evaluator with scaled batch inversion.
    fn lagrange_evaluations(
        domain: Domain<Self>,
        point: Self,
        evaluations: &mut [Self],
        scratch: &mut [Self],
    ) -> Option<usize>;

    /// Returns a primitive root of unity of order `2^log_size`, or `None`
    /// when `log_size` exceeds the two-adicity.
    ///
    /// Equals `ROOT_OF_UNITY^(2^(TWO_ADICITY - log_size))`, so roots at
    /// different sizes are compatible. Size one returns [`ONE`](Self::ONE).
    fn root_of_unity(log_size: u32) -> Option<Self>;

    /// Returns the inverse of [`root_of_unity`](Self::root_of_unity) for the
    /// same size.
    fn root_of_unity_inverse(log_size: u32) -> Option<Self>;

    /// Returns `2^-log_size`, the normalization of a `2^log_size`-point
    /// inverse transform.
    fn power_of_two_inverse(log_size: u32) -> Self;

    /// Adds `lhs * rhs` to the accumulator.
    fn mul_accumulate(accumulator: &mut Self::Accumulator, lhs: &Self, rhs: &Self);

    /// Reduces the accumulated sum to a field element.
    fn reduce(accumulator: Self::Accumulator) -> Self;
}
