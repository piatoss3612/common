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
    storage: TwiddleStorage::Factored { low_len: 4 },
};
const FACTORED: usize = match TWIDDLES.requirements() {
    Ok(fields) => fields,
    Err(_) => panic!("unsupported factored table"),
};
const REQUIREMENTS: TableRequirements = match TableRequirements::for_size(SIZE) {
    Ok(required) => required,
    Err(_) => panic!("unsupported table size"),
};

/// Version one of this owner's schema; all flags and dimensions are explicit.
///
/// The owner assigns `twiddle_kind = 1` to factored tables, `inverse = 0` to
/// forward roots, and `normalization = 1` to scales for unscaled inverse output.
/// Other flag values are unsupported. `modulus` and the Montgomery-encoded
/// `shift` store limbs least significant first.
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
    pub low_len: u64,
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
            twiddle_kind: 1,
            low_len: 4,
            inverse: 0,
            normalization: 1,
        }
    }

    pub fn validate<M: PrimeModulus>(self, extended: CosetDomain<M>) -> Result<(), FftError> {
        // Reject unknown encodings before converting the owner's integer flags
        // into Udon's semantic types. Integrity checks would be another layer.
        if self.twiddle_kind != 1
            || self.low_len != 4
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
            pub permutation: [u32; REQUIREMENTS.permutation],
            pub forward: [$field; REQUIREMENTS.twiddles],
            pub inverse: [$field; REQUIREMENTS.twiddles],
            pub finish: [$field; REQUIREMENTS.twiddles],
            pub scales: [$field; REQUIREMENTS.inverse_scales],
            pub residues: [$field; EXTENDED_SIZE],
            pub factored: [$field; FACTORED],
        }

        impl $name {
            pub fn empty() -> Self {
                Self {
                    header: Header::new::<$modulus>(),
                    permutation: [0; REQUIREMENTS.permutation],
                    forward: [<$field>::ZERO; REQUIREMENTS.twiddles],
                    inverse: [<$field>::ZERO; REQUIREMENTS.twiddles],
                    finish: [<$field>::ZERO; REQUIREMENTS.twiddles],
                    scales: [<$field>::ZERO; REQUIREMENTS.inverse_scales],
                    residues: [<$field>::ZERO; EXTENDED_SIZE],
                    factored: [<$field>::ZERO; FACTORED],
                }
            }

            pub fn destinations(&mut self) -> TablesMut<'_, $modulus> {
                TablesMut {
                    bit_reversed: Some(&mut self.permutation),
                    forward: Some(&mut self.forward),
                    inverse: Some(&mut self.inverse),
                    inverse_finish: Some(&mut self.finish),
                    inverse_scales: Some(&mut self.scales),
                }
            }

            pub fn tables(&self) -> Tables<'_, $modulus> {
                Tables {
                    bit_reversed: Some(&self.permutation),
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
