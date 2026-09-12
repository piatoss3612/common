//! An artifact schema chosen by the downstream owner, shared with its generator.
//!
//! These records describe a 16-point subgroup and expansion to the 64-point
//! coset shifted by `zeta`. The generator and consumer must agree on those
//! parameters: the record stores only table entries, and POD embedding checks
//! layout without validating their mathematical contents.

use udon::{
    fft::{TableRequirements, Tables, TablesMut},
    field::{Fp, Fq, PallasBase, PallasScalar},
};

pub const SIZE: usize = 16;
pub const EXTENDED_SIZE: usize = 64;
const REQUIREMENTS: TableRequirements = match TableRequirements::for_size(SIZE) {
    Ok(required) => required,
    Err(_) => panic!("unsupported table size"),
};

macro_rules! record {
    ($name:ident, $field:ty, $modulus:ty) => {
        #[repr(C)]
        #[derive(Clone, Copy, bento::Pod)]
        pub struct $name {
            pub permutation: [u32; REQUIREMENTS.permutation],
            pub forward: [$field; REQUIREMENTS.twiddles],
            pub inverse: [$field; REQUIREMENTS.twiddles],
            pub finish: [$field; REQUIREMENTS.twiddles],
            pub scales: [$field; REQUIREMENTS.inverse_scales],
            pub residues: [$field; EXTENDED_SIZE],
        }

        impl $name {
            pub fn empty() -> Self {
                Self {
                    permutation: [0; REQUIREMENTS.permutation],
                    forward: [<$field>::ZERO; REQUIREMENTS.twiddles],
                    inverse: [<$field>::ZERO; REQUIREMENTS.twiddles],
                    finish: [<$field>::ZERO; REQUIREMENTS.twiddles],
                    scales: [<$field>::ZERO; REQUIREMENTS.inverse_scales],
                    residues: [<$field>::ZERO; EXTENDED_SIZE],
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
