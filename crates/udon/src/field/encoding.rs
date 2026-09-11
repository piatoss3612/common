//! Conversion between canonical integers/bytes and stored Montgomery residues.
//!
//! Checked decoders reject noncanonical values; explicitly reducing constructors
//! accept wider integers. These are distinct from the raw stored-form accessors.

use core::marker::PhantomData;

use bento::const_arithmetic::{m255, u256};

use super::montgomery::{montgomery_multiply, montgomery_reduce};
use super::word::{adc, compare_limbs, multiply_wide};
use super::{CanonicalUint, ENCODED_SIZE, PastaField, PrimeModulus};

/// Constructs an [`Fp`](crate::field::Fp) constant from hexadecimal text.
///
/// Requires a canonical `0x`-prefixed, 64-digit integer, as accepted by
/// [`PastaField::from_hex`]. Malformed or noncanonical literals fail to build
/// even when the macro is used in a runtime expression.
///
/// ```compile_fail
/// // The modulus itself is not a canonical field element.
/// let _ = zakura_udon::fp_hex!(
///     "0x40000000000000000000000000000000224698fc094cf91b992d30ed00000001"
/// );
/// ```
///
/// ```compile_fail
/// // Hex strings must contain exactly 64 digits after the prefix.
/// let _ = zakura_udon::fp_hex!("0x01");
/// ```
#[macro_export]
macro_rules! fp_hex {
    ($value:literal $(,)?) => {
        const { $crate::field::Fp::from_hex($value) }
    };
}

/// Constructs an [`Fq`](crate::field::Fq) constant from hexadecimal text.
///
/// Requires a canonical `0x`-prefixed, 64-digit integer and compile-time
/// evaluation, exactly as [`fp_hex!`](crate::fp_hex) does for `Fp`.
///
/// ```compile_fail
/// let _ = zakura_udon::fq_hex!(
///     "0x40000000000000000000000000000000224698fc0994a8dd8c46eb2100000001"
/// );
/// ```
#[macro_export]
macro_rules! fq_hex {
    ($value:literal $(,)?) => {
        const { $crate::field::Fq::from_hex($value) }
    };
}

impl<M: PrimeModulus> PastaField<M> {
    pub(super) fn from_canonical_limbs(limbs: [u64; 4]) -> Self {
        debug_assert!(compare_limbs(&limbs, &M::MODULUS).is_lt());
        Self::from_montgomery(montgomery_multiply::<M>(&limbs, &M::R2))
    }

    pub(super) fn canonical_limbs(&self) -> [u64; 4] {
        let mut wide = [0; 8];
        wide[..4].copy_from_slice(&self.limbs);
        montgomery_reduce::<M>(wide)
    }

    /// Converts an ordinary integer to this field, returning `None` if it is
    /// at least [`M::MODULUS`](PrimeModulus::MODULUS).
    pub fn from_canonical_uint(value: CanonicalUint) -> Option<Self> {
        compare_limbs(&value.limbs(), &M::MODULUS)
            .is_lt()
            .then(|| Self::from_canonical_limbs(value.limbs()))
    }

    /// Reduces an arbitrary 256-bit integer into this field.
    pub fn from_uint_reduced(value: CanonicalUint) -> Self {
        Self::from_montgomery(montgomery_multiply::<M>(&value.limbs(), &M::R2))
    }

    /// Decodes a canonical 32-byte little-endian field representation.
    ///
    /// Returns `None` if the encoded integer is at least the modulus.
    pub fn from_bytes(bytes: [u8; ENCODED_SIZE]) -> Option<Self> {
        Self::from_canonical_uint(CanonicalUint::from_le_bytes(bytes))
    }

    /// Reduces a little-endian integer of arbitrary width into the field.
    ///
    /// An empty slice represents zero. This is modular reduction, so it
    /// accepts encodings rejected by [`Self::from_bytes`] and does not
    /// guarantee a uniform distribution from random input bytes.
    pub fn from_bytes_reduced(bytes: &[u8]) -> Self {
        if bytes.len() <= ENCODED_SIZE {
            let mut encoded = [0; ENCODED_SIZE];
            encoded[..bytes.len()].copy_from_slice(bytes);
            return Self::from_uint_reduced(CanonicalUint::from_le_bytes(encoded));
        }
        if bytes.len() <= 2 * ENCODED_SIZE {
            let mut wide = [0; 2 * ENCODED_SIZE];
            wide[..bytes.len()].copy_from_slice(bytes);
            return Self::from_wide_bytes_reduced(&wide);
        }

        let mut chunks = bytes.chunks(ENCODED_SIZE).rev();
        let high = chunks.next().unwrap();
        let mut high_bytes = [0; ENCODED_SIZE];
        high_bytes[..high.len()].copy_from_slice(high);
        let mut value = Self::from_uint_reduced(CanonicalUint::from_le_bytes(high_bytes));
        for chunk in chunks {
            let digit = CanonicalUint::from_le_bytes(chunk.try_into().unwrap());
            // Stored V=xR and ordinary D yield (V+D)R, representing xR+D.
            value = Self::from_montgomery(raw_product_sum::<M>(
                &value.limbs,
                &M::R2,
                &digit.limbs(),
                &M::R2,
            ));
        }
        value
    }

    /// Reduces a 64-byte little-endian integer into the field.
    ///
    /// This has the same result as [`Self::from_bytes_reduced`].
    pub fn from_wide_bytes_reduced(bytes: &[u8; 2 * ENCODED_SIZE]) -> Self {
        let low = CanonicalUint::from_le_bytes(bytes[..ENCODED_SIZE].try_into().unwrap());
        let high = CanonicalUint::from_le_bytes(bytes[ENCODED_SIZE..].try_into().unwrap());
        Self::from_montgomery(raw_product_sum::<M>(
            &low.limbs(),
            &M::R2,
            &high.limbs(),
            &M::R3,
        ))
    }

    /// Returns the canonical fixed-width integer representation.
    pub fn to_canonical_uint(self) -> CanonicalUint {
        CanonicalUint::from_limbs(self.canonical_limbs())
    }

    /// Encodes the ordinary field integer as 32 canonical little-endian bytes.
    pub fn to_bytes(self) -> [u8; ENCODED_SIZE] {
        self.to_canonical_uint().to_le_bytes()
    }

    /// Returns the reduced little-endian limbs of `self * 2^256 mod p`.
    ///
    /// These are storage words; use [`Self::to_bytes`] for protocol encoding.
    #[inline]
    pub const fn montgomery_limbs(&self) -> [u64; 4] {
        self.limbs
    }

    /// Constructs a field element from reduced Montgomery limbs.
    ///
    /// This reverses [`Self::montgomery_limbs`] without changing the limbs.
    ///
    /// # Panics
    ///
    /// Panics if the integer in `limbs` is at least the modulus. In a const
    /// expression this produces a compile error.
    pub const fn from_montgomery_limbs(limbs: [u64; 4]) -> Self {
        assert!(
            !u256::ge(&limbs, &M::MODULUS),
            "Montgomery limbs must be a canonical residue"
        );
        Self {
            limbs,
            marker: PhantomData,
        }
    }

    /// Converts a canonical `0x`-prefixed, 64-digit hexadecimal integer.
    ///
    /// The most significant digit comes first. Both letter cases are
    /// accepted, matching [`u256::from_hex`]. Use [`fp_hex!`](crate::fp_hex)
    /// or [`fq_hex!`](crate::fq_hex) to require compile-time evaluation.
    ///
    /// # Panics
    ///
    /// Panics if the prefix, digit count, or digits are invalid, or the
    /// integer is at least the modulus. In a const expression this produces
    /// a compile error.
    pub const fn from_hex(value: &str) -> Self {
        let canonical = u256::from_hex(value);
        assert!(
            !u256::ge(&canonical, &M::MODULUS),
            "field constants must be canonical residues"
        );
        Self::from_montgomery_limbs(m255::mul(&M::MODULUS, &canonical, &M::R2))
    }

    /// Returns the parity of the canonical integer representative.
    pub fn is_odd(&self) -> bool {
        self.to_canonical_uint().bit(0) == Some(true)
    }
}

// Raw operands may exceed p. The parameter bundle checks the constant bounds
// for both callers, establishing a*b+c*d < pR without field constructors.
#[inline]
fn raw_product_sum<M: PrimeModulus>(
    a: &[u64; 4],
    b: &[u64; 4],
    c: &[u64; 4],
    d: &[u64; 4],
) -> [u64; 4] {
    let mut sum = multiply_wide(a, b);
    let product = multiply_wide(c, d);
    let mut carry = 0;
    for (limb, term) in sum.iter_mut().zip(product) {
        (*limb, carry) = adc(*limb, term, carry);
    }
    debug_assert_eq!(carry, 0);
    montgomery_reduce::<M>(sum)
}
