//! Montgomery-form arithmetic for the two Pasta prime fields.
//!
//! One generic [`PastaField`] serves both fields through the [`PrimeModulus`]
//! marker types, keeping the base and scalar fields distinct at compile time
//! even though the curves form a cycle.
//!
//! ```compile_fail
//! use zakura_udon::field::{Fp, Fq};
//! let _ = Fp::ONE.add(&Fq::ONE);
//! ```
//!
//! A stored field value is the reduced residue `x * R mod p`, where
//! `R = 2^256`. Protocol bytes and [`CanonicalUint`] instead represent `x`
//! itself. Arithmetic is variable-time; this module provides no constant-time
//! guarantee for secret inputs.

use core::{fmt, marker::PhantomData};

mod algorithms;
mod encoding;
mod inversion;
mod montgomery;
mod parameters;
mod products;
mod safegcd;
mod uint;
mod word;

pub use parameters::{PallasBase, PallasScalar, PrimeModulus};
pub use products::ProductSum;
pub use uint::CanonicalUint;

use montgomery::{montgomery_multiply, montgomery_square, reduce_once};
use parameters::TWO_ADICITY;
use word::{adc, subtract_limbs};

#[cfg(test)]
mod tests;

const ENCODED_SIZE: usize = 32;

/// An element of a Pasta prime field stored as four Montgomery limbs.
///
/// Values hold `x * 2^256 mod p` in `[0, p)`. This unique reduced
/// representation makes structural equality valid; ordering and debug output
/// convert back to the ordinary integer `x`.
///
/// For [`bento::Pod`] storage, store the [`montgomery_limbs`](Self::montgomery_limbs)
/// as `[u64; 4]` and validate them with
/// [`from_montgomery_limbs`](Self::from_montgomery_limbs). Field elements do not
/// implement `Pod`, because arbitrary bytes can violate the reduced-residue
/// invariant. Use [`to_bytes`](Self::to_bytes) for canonical protocol encoding.
///
/// ```compile_fail
/// let _: &zakura_udon::field::Fp = bento::AlignedBytes([0; 32]).as_value();
/// ```
#[derive(Clone, Copy, Eq, PartialEq)]
#[repr(transparent)]
pub struct PastaField<M: PrimeModulus> {
    limbs: [u64; 4],
    marker: PhantomData<M>,
}

/// The Pallas coordinate field and Vesta scalar field.
pub type Fp = PastaField<PallasBase>;
/// The Vesta coordinate field and Pallas scalar field.
pub type Fq = PastaField<PallasScalar>;

impl<M: PrimeModulus> Ord for PastaField<M> {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        // Multiplication by R preserves equality, but not integer ordering.
        self.to_canonical_uint().cmp(&other.to_canonical_uint())
    }
}

impl<M: PrimeModulus> PartialOrd for PastaField<M> {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl<M: PrimeModulus> fmt::Debug for PastaField<M> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let limbs = self.canonical_limbs();
        write!(
            formatter,
            "0x{:016x}{:016x}{:016x}{:016x}",
            limbs[3], limbs[2], limbs[1], limbs[0]
        )
    }
}

impl<M: PrimeModulus> Default for PastaField<M> {
    fn default() -> Self {
        Self::zero()
    }
}

impl<M: PrimeModulus> PastaField<M> {
    /// The additive identity in stored Montgomery form.
    pub const ZERO: Self = Self {
        limbs: [0; 4],
        marker: PhantomData,
    };
    /// The multiplicative identity, whose stored limbs are `R mod p`.
    pub const ONE: Self = Self {
        limbs: M::R,
        marker: PhantomData,
    };

    #[inline]
    const fn from_montgomery(limbs: [u64; 4]) -> Self {
        debug_assert!(!bento::const_arithmetic::u256::ge(&limbs, &M::MODULUS));
        Self {
            limbs,
            marker: PhantomData,
        }
    }

    /// Returns the additive identity.
    pub const fn zero() -> Self {
        Self::ZERO
    }

    /// Returns the multiplicative identity.
    pub const fn one() -> Self {
        Self::ONE
    }

    /// Embeds an unsigned 64-bit integer; every such integer is below both moduli.
    pub fn from_u64(value: u64) -> Self {
        Self::from_canonical_limbs([value, 0, 0, 0])
    }

    /// Returns whether this value is zero.
    #[inline]
    pub const fn is_zero(&self) -> bool {
        self.limbs[0] == 0 && self.limbs[1] == 0 && self.limbs[2] == 0 && self.limbs[3] == 0
    }

    /// Returns `self + rhs`.
    #[inline]
    pub fn add(&self, rhs: &Self) -> Self {
        let mut limbs = [0; 4];
        let mut carry = 0;
        for (index, limb) in limbs.iter_mut().enumerate() {
            (*limb, carry) = adc(self.limbs[index], rhs.limbs[index], carry);
        }
        debug_assert_eq!(carry, 0);
        Self::from_montgomery(reduce_once::<M>(limbs))
    }

    /// Returns `self - rhs`.
    #[inline]
    pub fn sub(&self, rhs: &Self) -> Self {
        let (mut limbs, borrow) = subtract_limbs(&self.limbs, &rhs.limbs);
        // Restore the modulus exactly when subtraction borrowed.
        let mask = borrow.wrapping_neg();
        let mut carry = 0;
        for (limb, modulus) in limbs.iter_mut().zip(M::MODULUS) {
            (*limb, carry) = adc(*limb, modulus & mask, carry);
        }
        Self::from_montgomery(limbs)
    }

    /// Returns the additive inverse.
    #[inline]
    pub fn neg(&self) -> Self {
        // p - a masked to zero for a = 0: the zero test becomes mask
        // arithmetic instead of a branch on the operand value.
        let (limbs, borrow) = subtract_limbs(&M::MODULUS, &self.limbs);
        debug_assert_eq!(borrow, 0);
        let nonzero = self.limbs[0] | self.limbs[1] | self.limbs[2] | self.limbs[3];
        let mask = u64::from(nonzero != 0).wrapping_neg();
        Self::from_montgomery([
            limbs[0] & mask,
            limbs[1] & mask,
            limbs[2] & mask,
            limbs[3] & mask,
        ])
    }

    /// Returns `self * rhs`.
    #[inline]
    pub fn mul(&self, rhs: &Self) -> Self {
        Self::from_montgomery(montgomery_multiply::<M>(&self.limbs, &rhs.limbs))
    }

    /// Returns `self * self`.
    #[inline(always)]
    pub fn square(&self) -> Self {
        Self::from_montgomery(montgomery_square::<M>(&self.limbs))
    }

    /// Returns `2 * self`.
    #[inline]
    pub fn double(&self) -> Self {
        // Both Pasta moduli are below 2^255, so doubling a canonical
        // Montgomery representative cannot overflow these four limbs.
        let limbs = [
            self.limbs[0] << 1,
            (self.limbs[1] << 1) | (self.limbs[0] >> 63),
            (self.limbs[2] << 1) | (self.limbs[1] >> 63),
            (self.limbs[3] << 1) | (self.limbs[2] >> 63),
        ];
        debug_assert_eq!(self.limbs[3] >> 63, 0);
        Self::from_montgomery(reduce_once::<M>(limbs))
    }

    /// Returns `3 * self`.
    #[inline]
    pub fn triple(&self) -> Self {
        self.double().add(self)
    }

    /// Returns `4 * self`.
    #[inline]
    pub fn mul_by_4(&self) -> Self {
        self.double().double()
    }

    /// Returns `8 * self`.
    #[inline]
    pub fn mul_by_8(&self) -> Self {
        self.double().double().double()
    }

    /// Returns `self * multiplier + addend`.
    #[inline]
    pub fn mul_add(&self, multiplier: &Self, addend: &Self) -> Self {
        self.mul(multiplier).add(addend)
    }

    /// Returns `self * multiplier - subtrahend`.
    #[inline]
    pub fn mul_sub(&self, multiplier: &Self, subtrahend: &Self) -> Self {
        self.mul(multiplier).sub(subtrahend)
    }

    /// Raises this value to an unsigned exponent; exponent zero returns one.
    ///
    /// The multiplication schedule depends on the exponent.
    pub fn pow_u64(&self, exponent: u64) -> Self {
        crate::field::algorithms::pow_u64(self, exponent)
    }

    /// Embeds a signed 64-bit integer, including `i64::MIN`.
    #[inline]
    pub fn from_i64(coefficient: i64) -> Self {
        let value = Self::from_u64(coefficient.unsigned_abs());
        if coefficient < 0 { value.neg() } else { value }
    }

    /// Computes a square root with Tonelli-Shanks, or `None` for a nonsquare.
    ///
    /// Either root may be returned; branches depend on the input.
    pub fn sqrt(&self) -> Option<Self> {
        if self.is_zero() {
            return Some(Self::zero());
        }

        // With p - 1 = t * 2^32, the fixed exponent is (t - 1) / 2.
        // This initializes x = self^((t + 1) / 2) and t = self^t with one
        // exponentiation. The exponent is fixed per field, so the
        // multiplication schedule is planned at compile time.
        let w = M::pow_sqrt_exponent(self);
        crate::field::algorithms::tonelli_shanks_with_roots(
            self,
            w,
            |k| Self::from_montgomery(M::ROOTS[k as usize]),
            TWO_ADICITY,
        )
    }
}

// Inline the wrappers so generic exponentiation uses the specialized kernels.
impl<M: PrimeModulus> crate::field::algorithms::Field for PastaField<M> {
    #[inline(always)]
    fn zero() -> Self {
        Self::zero()
    }
    #[inline(always)]
    fn one() -> Self {
        Self::one()
    }
    #[inline(always)]
    fn is_zero(&self) -> bool {
        self.is_zero()
    }
    #[inline(always)]
    fn mul(&self, rhs: &Self) -> Self {
        self.mul(rhs)
    }
    #[inline(always)]
    fn square(&self) -> Self {
        self.square()
    }
}
