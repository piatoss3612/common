//! An artifact schema chosen by the downstream owner, shared with its generator.
//!
//! These records describe a 16-point subgroup and expansion to the 64-point
//! coset shifted by `zeta`. The generator and consumer must agree on those
//! parameters. This owner serializes Udon's semantic descriptors into a concrete
//! POD header; POD embedding itself checks only the target layout.

use udon::{
    fft::{
        CosetDomain, ExpansionScaleArtifact, ExpansionScaleNormalization, FftError,
        TableRequirements, Tables, TablesMut, TwiddleArtifact, TwiddleDescription, TwiddleStorage,
    },
    field::{Fp, Fq, PallasBase, PallasScalar, PastaField, PrimeModulus},
};

pub const SIZE: usize = 16;
pub const EXTENDED_SIZE: usize = 64;
pub const TWIDDLES: TwiddleDescription = TwiddleDescription {
    size: SIZE,
    inverse: false,
    storage: TwiddleStorage::StagePacked,
};
const PACKED: usize = match TWIDDLES.requirements() {
    Ok(fields) => fields,
    Err(_) => panic!("unsupported packed table"),
};
const REQUIREMENTS: TableRequirements = match TableRequirements::for_size(SIZE) {
    Ok(required) => required,
    Err(_) => panic!("unsupported table size"),
};

/// Metadata for this owner's FFT table records.
///
/// [`Self::schema_version`] must be 2 for this record layout; [`Self::version`]
/// identifies Udon's semantic descriptors. The owner assigns `twiddle_kind = 2`
/// to stage-packed tables, `inverse = 0` to forward roots, and `normalization = 1`
/// to scales for unscaled inverse output. Other flag values are unsupported.
/// `modulus` and the Montgomery-encoded `shift` store limbs least significant first.
#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
pub struct Header {
    pub modulus: [u64; 4],
    pub shift: [u64; 4],
    pub version: u64,
    pub montgomery_bits: u64,
    pub base_size: u64,
    pub extended_size: u64,
    pub twiddle_size: u64,
    pub twiddle_kind: u64,
    pub schema_version: u64,
    pub inverse: u64,
    pub normalization: u64,
}

impl Header {
    fn new<M: PrimeModulus>() -> Self {
        let artifact = TwiddleArtifact::for_field::<M>(TWIDDLES);
        Self {
            modulus: artifact.modulus,
            shift: PastaField::<M>::zeta().montgomery_limbs(),
            version: u64::from(artifact.version),
            montgomery_bits: u64::from(artifact.montgomery_bits),
            base_size: SIZE as u64,
            extended_size: EXTENDED_SIZE as u64,
            twiddle_size: TWIDDLES.size as u64,
            twiddle_kind: 2,
            schema_version: 2,
            inverse: 0,
            normalization: 1,
        }
    }

    /// Checks the owner's schema and Udon metadata against the intended domains.
    ///
    /// Returns [`FftError::InvalidTables`] for unsupported encodings or metadata
    /// inconsistent with field `M`, [`TWIDDLES`], or the expansion from [`SIZE`]
    /// to `extended`. Table entries require separate mathematical validation.
    pub fn validate<M: PrimeModulus>(self, extended: CosetDomain<M>) -> Result<(), FftError> {
        // Reject unknown encodings before converting the owner's integer flags
        // into Udon's semantic types. Integrity checks would be another layer.
        if self.schema_version != 2
            || self.twiddle_kind != 2
            || self.inverse != 0
            || self.normalization != 1
            || self.twiddle_size != SIZE as u64
            || self.base_size != SIZE as u64
            || self.extended_size != EXTENDED_SIZE as u64
        {
            return Err(FftError::InvalidTables);
        }
        let version = u32::try_from(self.version).map_err(|_| FftError::InvalidTables)?;
        let montgomery_bits =
            u32::try_from(self.montgomery_bits).map_err(|_| FftError::InvalidTables)?;
        TwiddleArtifact {
            version,
            modulus: self.modulus,
            montgomery_bits,
            twiddles: TWIDDLES,
        }
        .validate::<M>()?;
        ExpansionScaleArtifact {
            version,
            modulus: self.modulus,
            montgomery_bits,
            base_size: SIZE,
            extended_size: EXTENDED_SIZE,
            shift: self.shift,
            normalization: ExpansionScaleNormalization::UnscaledInverse,
        }
        .validate(SIZE, extended, ExpansionScaleNormalization::UnscaledInverse)
    }
}

macro_rules! record {
    ($name:ident, $field:ty, $modulus:ty) => {
        #[repr(C)]
        #[derive(Clone, Copy, bento::Pod)]
        pub struct $name {
            pub header: Header,
            pub forward: [$field; REQUIREMENTS.twiddles],
            pub inverse: [$field; REQUIREMENTS.twiddles],
            pub finish: [$field; REQUIREMENTS.twiddles],
            pub scales: [$field; REQUIREMENTS.inverse_scales],
            pub residues: [$field; EXTENDED_SIZE],
            pub packed: [$field; PACKED],
        }

        impl $name {
            pub fn empty() -> Self {
                Self {
                    header: Header::new::<$modulus>(),
                    forward: [<$field>::ZERO; REQUIREMENTS.twiddles],
                    inverse: [<$field>::ZERO; REQUIREMENTS.twiddles],
                    finish: [<$field>::ZERO; REQUIREMENTS.twiddles],
                    scales: [<$field>::ZERO; REQUIREMENTS.inverse_scales],
                    residues: [<$field>::ZERO; EXTENDED_SIZE],
                    packed: [<$field>::ZERO; PACKED],
                }
            }

            pub fn destinations(&mut self) -> TablesMut<'_, $modulus> {
                TablesMut {
                    forward: Some(&mut self.forward),
                    inverse: Some(&mut self.inverse),
                    inverse_finish: Some(&mut self.finish),
                    inverse_scales: Some(&mut self.scales),
                }
            }

            pub fn tables(&self) -> Tables<'_, $modulus> {
                Tables {
                    forward: Some(&self.forward),
                    inverse: Some(&self.inverse),
                    inverse_finish: Some(&self.finish),
                    inverse_scales: Some(&self.scales),
                }
            }
        }
    };
}

record!(FpTables, Fp, PallasBase);
record!(FqTables, Fq, PallasScalar);
