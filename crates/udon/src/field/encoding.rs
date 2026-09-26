//! Sampling and canonical integer access through the field traits.

use super::Field;

/// Samples a field element by reducing 64 bytes from the caller's source.
///
/// Calls `fill` exactly once with the entire buffer. The callback must fill it
/// with uniformly random bytes; cryptographic use requires a cryptographically
/// secure source. Reduction follows [`Field::from_uniform_bytes`], without
/// rejection sampling or additional draws. The modulus must have at most 384
/// bits for the documented `2^-128` bound on statistical distance from uniform.
pub fn random<F: Field>(fill: impl FnOnce(&mut [u8; 64])) -> F {
    const {
        assert!(
            F::NUM_BITS <= 384,
            "sampling requires a modulus of at most 384 bits"
        )
    };
    let mut bytes = [0u8; 64];
    fill(&mut bytes);
    F::from_uniform_bytes(&bytes)
}

/// Returns the low 64 bits of the canonical integer representative.
pub fn low_u64<F: Field>(value: &F) -> u64 {
    let bytes = value.to_bytes();
    let bytes = bytes.as_ref();
    let mut low = [0u8; 8];
    let count = bytes.len().min(low.len());
    low[..count].copy_from_slice(&bytes[..count]);
    u64::from_le_bytes(low)
}
