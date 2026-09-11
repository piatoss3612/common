//! Root and power tables for field and inversion parameters.

use super::super::U256;
use super::{add, assert_modulus, one, pow2_mod};

/// Builds consecutive powers of a reduced Montgomery base.
///
/// If `base` represents the integer `b`, entry `k` is the reduced Montgomery
/// representation of `b^k`: `b^k * 2^256 mod modulus`, for `0 <= k < N`.
/// Entry zero represents one, including for a zero base. With `N = 0`, the
/// result is empty; the modulus and base are still checked. Primality is not
/// required.
///
/// # Panics
///
/// Panics if [`assert_modulus`] rejects `modulus`, or if `base >= modulus`.
pub const fn powers<const N: usize>(modulus: &U256, base: &U256) -> [U256; N] {
    let context = super::MontgomeryContext::new(*modulus);
    assert!(
        !super::super::u256::ge(base, modulus),
        "base must be reduced"
    );
    let mut table = [[0; 4]; N];
    if N > 0 {
        table[0] = context.one();
        let mut k = 1;
        while k < N {
            table[k] = context.mul(&table[k - 1], base);
            k += 1;
        }
    }
    table
}

/// Builds forward and inverse root tables indexed by logarithmic order.
///
/// Returns `(roots, inverse_roots)`, each of length `N = two_adicity + 1`.
/// Entry `k` in `roots` is the reduced Montgomery representation of
/// `generator^((modulus - 1) / 2^k)`. The corresponding inverse is in
/// `inverse_roots`. Entry zero represents one and entry `two_adicity` in
/// `roots` matches [`super::two_adic_root_of_unity`]. `generator` is an
/// ordinary integer, interpreted modulo `modulus`.
///
/// The caller must supply a prime modulus and a generator of its nonzero
/// residues. Primality and generator order are not checked. The exact
/// two-adicity of `modulus - 1` is checked by [`super::super::u256::odd_cofactor`].
///
/// # Panics
///
/// Panics if `modulus` is outside the [supported domain](super::assert_modulus),
/// if [`super::super::u256::odd_cofactor`] rejects `two_adicity`, or if
/// `N != two_adicity + 1`.
pub const fn two_adic_root_tables<const N: usize>(
    modulus: &U256,
    generator: u64,
    two_adicity: u32,
) -> ([U256; N], [U256; N]) {
    use super::super::u256::{odd_cofactor, sub_u64};
    // Share Montgomery setup across both tables, then square down from the
    // highest order instead of exponentiating each entry independently.
    let context = super::MontgomeryContext::new(*modulus);
    let exponent = odd_cofactor(modulus, two_adicity);
    assert!(
        N == two_adicity as usize + 1,
        "root table length must equal two_adicity + 1"
    );
    let mut roots = [[0; 4]; N];
    let mut inverses = roots;
    let mut index = N - 1;
    roots[index] = context.pow(&context.from_u64(generator), &exponent);
    inverses[index] = context.pow(&roots[index], &sub_u64(modulus, 2));
    while index > 0 {
        roots[index - 1] = context.mul(&roots[index], &roots[index]);
        inverses[index - 1] = context.mul(&inverses[index], &inverses[index]);
        index -= 1;
    }
    (roots, inverses)
}

/// Builds a table of inverse powers of two in Montgomery form.
///
/// Entry `k`, for `0 <= k < N`, is the reduced Montgomery representation of
/// `2^-k`: the integer `2^(256 - k) mod modulus`. Entry zero is therefore
/// the Montgomery representation of one. Primality is not required.
///
/// At most 257 entries can be generated, covering inverse powers through
/// `2^-256`. With `N = 0`, the result is empty; the modulus is still checked.
///
/// # Panics
///
/// Panics if `N > 257`, or if `modulus` is outside the
/// [supported domain](super::assert_modulus).
///
/// # Examples
///
/// ```
/// # use zakura_bento_core as bento;
/// use bento::const_arithmetic::{U256, m255};
///
/// const MODULUS: U256 = [97, 0, 0, 0];
/// const INVERSES: [U256; 3] = m255::inverse_powers_of_two(&MODULUS);
/// assert_eq!(INVERSES[0], m255::one(&MODULUS));
/// assert_eq!(INVERSES[1], m255::two_inverse(&MODULUS));
/// assert_eq!(INVERSES[2], m255::from_u64(&MODULUS, 73)); // 4 * 73 mod 97 = 1
/// ```
pub const fn inverse_powers_of_two<const N: usize>(modulus: &U256) -> [U256; N] {
    assert_modulus(modulus);
    assert!(N <= 257, "inverse powers must fit the Montgomery radix");
    let mut table = [[0; 4]; N];
    if N > 0 {
        // Work backward from the smallest exponent, sharing the doublings
        // across entries. Every nonempty table needs only 256 doublings total.
        let mut k = N - 1;
        table[k] = pow2_mod(modulus, 256 - k as u32);
        while k > 0 {
            table[k - 1] = add(modulus, &table[k], &table[k]);
            k -= 1;
        }
    }
    table
}

/// Builds Montgomery correction factors for batched safegcd inversion.
///
/// Use with a kernel whose batch transition matrix is scaled by `2^62` and
/// whose Bézout coefficient update divides by `2^64` modulo `modulus`.
/// This introduces an extra factor of `2^-2` per batch. For `1 <= batches <= N`,
/// entry `batches - 1` is the reduced Montgomery representation of
/// `2^(2 * batches)`: the integer `2^(256 + 2 * batches) mod modulus`.
/// Multiplying with [`super::mul`] by that entry corrects the accumulated
/// factor while preserving the coefficient's ordinary or Montgomery form.
/// The consumer must account for its kernel's initial scaling and final sign.
///
/// The consumer chooses `N` to cover its inversion kernel's batch limit.
/// Primality is not required. With `N = 0`, the result is empty; the modulus
/// is still checked.
///
/// # Panics
///
/// Panics if `modulus` is outside the
/// [supported domain](super::assert_modulus).
///
/// # Examples
///
/// ```
/// # use zakura_bento_core as bento;
/// use bento::const_arithmetic::{U256, m255};
///
/// const MODULUS: U256 = [97, 0, 0, 0];
/// const CORRECTIONS: [U256; 2] = m255::safegcd_corrections_62_64(&MODULUS);
/// assert_eq!(CORRECTIONS[0], m255::from_u64(&MODULUS, 4)); // One batch.
/// assert_eq!(CORRECTIONS[1], m255::from_u64(&MODULUS, 16)); // Two batches.
/// ```
pub const fn safegcd_corrections_62_64<const N: usize>(modulus: &U256) -> [U256; N] {
    assert_modulus(modulus);
    let mut table = [[0; 4]; N];
    if N > 0 {
        let mut value = one(modulus);
        let mut batch = 0;
        while batch < N {
            // Each batch needs two more doublings, with no exponent counter
            // or repeated derivation of the Montgomery radix.
            value = add(modulus, &value, &value);
            value = add(modulus, &value, &value);
            table[batch] = value;
            batch += 1;
        }
    }
    table
}
