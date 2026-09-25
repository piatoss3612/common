//! Descriptor for stored field representations and artifact filenames.

/// Expands to the stored representation's descriptor as a string literal.
///
/// Use this in [`concat!`] paths passed to [`bento::embed_array!`] or
/// [`bento::embed_struct!`]. Generators name artifacts with [`STORED_FORM`];
/// consumers select them with this macro. See [`STORED_FORM`] for the storage
/// contract and the descriptor's scope.
///
/// ```
/// use zakura_udon::{STORED_FORM, stored_form};
///
/// const FILENAME: &str = concat!("roots-", stored_form!(), ".bin");
/// assert_eq!(FILENAME, "roots-mont-u64x4.bin");
/// assert_eq!(FILENAME, format!("roots-{STORED_FORM}.bin"));
/// ```
#[macro_export]
macro_rules! stored_form {
    () => {
        "mont-u64x4"
    };
}

/// The permanent filename token for the stored field representation.
///
/// Field elements store a Montgomery representative congruent to `x * 2^256`
/// modulo `p` as four little-endian `u64` limbs, least significant limb first.
/// Here `x` is the canonical integer representing the field element and `p` is
/// its field's prime modulus. `Loose` representatives are below `2p`; `Reduced`
/// representatives are below `p`. The marker occupies no bytes.
///
/// This descriptor identifies storage, not canonical protocol bytes, a modulus,
/// or the surrounding record's schema. In particular, [`Fp`](crate::field::Fp)
/// and [`Fq`](crate::field::Fq), in either reduction state, share a descriptor;
/// the artifact format identifies the modulus and state. Storage preserves the
/// exact limbs of the constructed value. See
/// [`PastaField`](crate::field::PastaField) for the mathematical
/// storage invariants and [`bento::Pod`] for layout and endianness requirements.
///
/// Use [`stored_form!`] where a string literal is required, such as [`concat!`].
pub const STORED_FORM: &str = stored_form!();
