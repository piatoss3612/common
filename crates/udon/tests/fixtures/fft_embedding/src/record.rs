//! An artifact schema chosen by the downstream owner, shared with its generator.
//!
//! These records describe a 16-point subgroup and expansion to the 64-point
//! coset shifted by `zeta`. The generator and consumer must agree on those
//! parameters through this shared schema and its artifact filenames.

use udon::{
    fft::{TableRequirements, Tables, TablesMut, TwiddleDescription, TwiddleStorage},
    field::{Fp, Fq, PallasBase, PallasScalar},
};

pub const SIZE: usize = 16;
pub const EXTENDED_SIZE: usize = 64;
pub const TWIDDLES: TwiddleDescription = TwiddleDescription {
    size: SIZE,
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

macro_rules! record {
    ($name:ident, $field:ty, $modulus:ty) => {
        #[repr(C)]
        #[derive(Clone, Copy, bento::Pod)]
        pub struct $name {
            pub forward: [$field; REQUIREMENTS.twiddles],
            pub inverse: [$field; REQUIREMENTS.twiddles],
            pub finish: [$field; REQUIREMENTS.twiddles],
            pub residues: [$field; EXTENDED_SIZE],
            pub packed: [$field; PACKED],
        }

        impl $name {
            pub fn empty() -> Self {
                Self {
                    forward: [<$field>::ZERO; REQUIREMENTS.twiddles],
                    inverse: [<$field>::ZERO; REQUIREMENTS.twiddles],
                    finish: [<$field>::ZERO; REQUIREMENTS.twiddles],
                    residues: [<$field>::ZERO; EXTENDED_SIZE],
                    packed: [<$field>::ZERO; PACKED],
                }
            }

            pub fn destinations(&mut self) -> TablesMut<'_, $modulus> {
                TablesMut {
                    forward: Some(&mut self.forward),
                    inverse: Some(&mut self.inverse),
                    inverse_finish: Some(&mut self.finish),
                }
            }

            pub const fn tables(&self) -> Tables<'_, $modulus> {
                Tables {
                    forward: Some(&self.forward),
                    inverse: Some(&self.inverse),
                    inverse_finish: Some(&self.finish),
                }
            }
        }
    };
}

record!(FpTables, Fp, PallasBase);
record!(FqTables, Fq, PallasScalar);
