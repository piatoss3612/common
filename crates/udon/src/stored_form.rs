//! Descriptors for stored field representations and artifact filenames.

/// Expands to the active stored representation's descriptor as a string literal.
///
/// Use this in [`concat!`] paths passed to [`bento::embed_array!`] or
/// [`bento::embed_struct!`]. Generators name artifacts with
/// [`StoredForm::descriptor`]; consumers select them with this macro. See
/// [`StoredForm`] for the meaning and scope of these descriptors.
///
/// ```
/// use zakura_udon::{StoredForm, stored_form};
///
/// const FILENAME: &str = concat!("roots-", stored_form!(), ".bin");
/// assert_eq!(FILENAME, "roots-mont-u64x4.bin");
/// assert_eq!(stored_form!(), StoredForm::ACTIVE.descriptor());
/// ```
#[macro_export]
macro_rules! stored_form {
    () => {
        "mont-u64x4"
    };
}

/// The active stored representation's descriptor; see [`stored_form!`].
pub const STORED_FORM: &str = stored_form!();

/// Registered representations of stored field elements.
///
/// Artifact generators construct field values normally, then use this registry
/// to name their output. Each descriptor permanently identifies a field storage
/// representation. It does not identify canonical protocol bytes, a modulus,
/// or the surrounding record's schema. In particular, [`Fp`](crate::field::Fp)
/// and [`Fq`](crate::field::Fq) share a descriptor; the artifact format must
/// distinguish them.
///
/// Selecting a descriptor does not convert values or validate artifact contents.
/// See [`PastaField`](crate::field::PastaField) for the field storage contract.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StoredForm {
    /// Reduced Montgomery storage, identified by `mont-u64x4`.
    ///
    /// Stores `x * 2^256 mod p` as four little-endian `u64` limbs, least
    /// significant limb first. Here `x` is the canonical integer representing
    /// the field element and `p` is its field's prime modulus.
    MontU64x4,
}

impl StoredForm {
    /// Every registered stored representation.
    pub const ALL: &'static [Self] = &[Self::MontU64x4];

    /// The representation used by this compilation's field elements.
    pub const ACTIVE: Self = Self::MontU64x4;

    /// The permanent filename token for this representation.
    pub const fn descriptor(self) -> &'static str {
        match self {
            Self::MontU64x4 => "mont-u64x4",
        }
    }

    /// Selects the representation for a Cargo target's pointer width.
    ///
    /// Build scripts pass `CARGO_CFG_TARGET_POINTER_WIDTH`, which describes
    /// the target rather than the build host. The argument is currently ignored:
    /// every string selects [`Self::MontU64x4`]. This does not check target
    /// support; storage operations validate endianness and layout through
    /// [`bento::Pod`].
    pub fn for_target(pointer_width: &str) -> Self {
        let _ = pointer_width;
        Self::MontU64x4
    }
}

// Keep the literal needed by concat! aligned with the value-level registry.
const _: () = {
    let literal = stored_form!().as_bytes();
    let active = StoredForm::ACTIVE.descriptor().as_bytes();
    assert!(literal.len() == active.len());
    let mut index = 0;
    while index < literal.len() {
        assert!(literal[index] == active[index]);
        index += 1;
    }
};
