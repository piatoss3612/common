//! Canonical integer reference fields for testing representation-independent APIs.
//!
//! Arithmetic uses `BigUint`; these types are only test consumers of Udon's
//! traits. The moduli are from the zkcrypto BLS12-381 and Jubjub descriptions:
//! <https://github.com/zkcrypto/bls12_381#curve-description> and
//! <https://github.com/zkcrypto/jubjub#curve-description>.

// Field and curve consumers exercise different subsets of these reference types.
#![allow(dead_code)]

use core::{iter, ops};
use num_bigint::BigUint;
use zakura_udon::{
    fft::{Domain, FftError},
    field::Field,
};

macro_rules! binary {
    ($name:ident, $trait:ident, $method:ident, $assign:ident, $assign_method:ident, $body:expr) => {
        impl ops::$trait<&Self> for $name {
            type Output = Self;

            fn $method(self, rhs: &Self) -> Self {
                Self::from_integer($body(self.integer(), rhs.integer(), Self::modulus()))
            }
        }

        impl ops::$trait for $name {
            type Output = Self;

            fn $method(self, rhs: Self) -> Self {
                ops::$trait::$method(self, &rhs)
            }
        }

        impl ops::$assign<&Self> for $name {
            fn $assign_method(&mut self, rhs: &Self) {
                *self = ops::$trait::$method(*self, rhs);
            }
        }

        impl ops::$assign for $name {
            fn $assign_method(&mut self, rhs: Self) {
                ops::$assign::$assign_method(self, &rhs);
            }
        }
    };
}

macro_rules! aggregate {
    ($name:ident, $trait:ident, $method:ident, $identity:ident, $op:tt) => {
        impl iter::$trait for $name {
            fn $method<I: Iterator<Item = Self>>(values: I) -> Self {
                values.fold(Self::$identity, |acc, value| acc $op value)
            }
        }

        impl<'a> iter::$trait<&'a Self> for $name {
            fn $method<I: Iterator<Item = &'a Self>>(values: I) -> Self {
                values.copied().$method()
            }
        }
    };
}

macro_rules! reference_field {
    (
        $name:ident, $limbs:literal, $bytes:literal, $bits:literal, $modulus:expr,
        generator = $generator:literal, two_adicity = $two_adicity:literal,
        root = $root:expr, inverse_root = $inverse_root:expr,
        two_inverse = $two_inverse:expr, delta = $delta:expr, zeta = $zeta:expr
    ) => {
        #[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
        pub struct $name([u64; $limbs]);

        impl $name {
            fn integer(self) -> BigUint {
                BigUint::from_bytes_le(&self.0.map(u64::to_le_bytes).concat())
            }

            fn modulus() -> BigUint {
                BigUint::from_bytes_le(&Self::MODULUS.map(u64::to_le_bytes).concat())
            }

            fn from_integer(integer: BigUint) -> Self {
                let digits = (integer % Self::modulus()).to_u64_digits();
                let mut result = Self::ZERO;
                result.0[..digits.len()].copy_from_slice(&digits);
                result
            }
        }

        binary!($name, Add, add, AddAssign, add_assign, |a, b, _p| a + b);
        binary!($name, Sub, sub, SubAssign, sub_assign, |a, b, p| a + p - b);
        binary!($name, Mul, mul, MulAssign, mul_assign, |a, b, _p| a * b);
        aggregate!($name, Sum, sum, ZERO, +);
        aggregate!($name, Product, product, ONE, *);

        impl ops::Neg for $name {
            type Output = Self;

            fn neg(self) -> Self {
                Self::from_integer(Self::modulus() - self.integer())
            }
        }

        impl From<u64> for $name {
            fn from(integer: u64) -> Self {
                Self::from_integer(integer.into())
            }
        }

        impl Field for $name {
            const ZERO: Self = Self([0; $limbs]);
            const ONE: Self = {
                let mut limbs = [0; $limbs];
                limbs[0] = 1;
                Self(limbs)
            };
            fn is_zero(&self) -> bool {
                *self == Self::ZERO
            }

            fn square(&self) -> Self {
                *self * self
            }

            fn mul_add(&self, multiplier: &Self, addend: &Self) -> Self {
                Self::from_integer(self.integer() * multiplier.integer() + addend.integer())
            }

            fn double(&self) -> Self {
                *self + self
            }

            fn invert(&self) -> Option<Self> {
                self.integer().modinv(&Self::modulus()).map(Self::from_integer)
            }

            fn batch_invert(values: &mut [Self], scratch: &mut [Self]) {
                Self::batch_invert_groups(&mut [values], scratch)
            }

            fn batch_invert_groups(groups: &mut [impl AsMut<[Self]>], scratch: &mut [Self]) {
                let length: usize = groups.iter_mut().map(|group| group.as_mut().len()).sum();
                let batch_size = scratch.len().max(1);
                // Exact integer division gives each cofactor independently of
                // Udon's prefix-product algorithm. The integer model allocates;
                // scratch only selects the number of shared inversions here.
                for start in (0..length).step_by(batch_size) {
                    let mut product = BigUint::from(1u8);
                    let mut nonzero = false;
                    for value in groups
                        .iter_mut()
                        .flat_map(|group| group.as_mut())
                        .skip(start)
                        .take(batch_size)
                    {
                        if !value.is_zero() {
                            product *= value.integer();
                            nonzero = true;
                        }
                    }
                    if !nonzero {
                        continue;
                    }
                    let inverse = product.modinv(&Self::modulus()).unwrap();
                    for value in groups
                        .iter_mut()
                        .flat_map(|group| group.as_mut())
                        .skip(start)
                        .take(batch_size)
                    {
                        if !value.is_zero() {
                            *value = Self::from_integer((&product / value.integer()) * &inverse);
                        }
                    }
                }
            }

            fn sqrt(&self) -> Option<Self> {
                sqrt(self.integer(), Self::modulus()).map(Self::from_integer)
            }

            fn pow_u64(&self, exponent: u64) -> Self {
                Self::from_integer(self.integer().modpow(&exponent.into(), &Self::modulus()))
            }

            fn sum_of_products_slice(lhs: &[Self], rhs: &[Self]) -> Self {
                assert_eq!(lhs.len(), rhs.len(), "inner product lengths must agree");
                Self::sum_of_product_pairs(lhs.iter().zip(rhs))
            }

            fn sum_of_product_pairs<'a>(
                pairs: impl IntoIterator<Item = (&'a Self, &'a Self)>,
            ) -> Self {
                Self::from_integer(
                    pairs.into_iter().map(|(lhs, rhs)| lhs.integer() * rhs.integer()).sum(),
                )
            }

            fn from_u128(value: u128) -> Self {
                Self::from_integer(value.into())
            }

            type Repr = [u8; $bytes];
            type Limbs = [u64; $limbs];
            type Bits = [bool; $bytes * 8];

            const MODULUS: Self::Limbs = $modulus;
            const NUM_BITS: u32 = $bits;
            const CAPACITY: u32 = $bits - 1;

            fn is_odd(&self) -> bool {
                self.0[0] & 1 == 1
            }

            fn to_bytes(&self) -> Self::Repr {
                let digits = self.integer().to_bytes_le();
                let mut bytes = [0; $bytes];
                bytes[..digits.len()].copy_from_slice(&digits);
                bytes
            }

            fn from_bytes(bytes: Self::Repr) -> Option<Self> {
                let integer = BigUint::from_bytes_le(&bytes);
                (integer < Self::modulus()).then(|| Self::from_integer(integer))
            }

            fn from_uniform_bytes(bytes: &[u8; 64]) -> Self {
                Self::from_integer(BigUint::from_bytes_le(bytes))
            }

            fn random(fill: impl FnOnce(&mut [u8; 64])) -> Self {
                const {
                    assert!(
                        Self::NUM_BITS <= 384,
                        "sampling requires a modulus of at most 384 bits"
                    );
                }
                let mut bytes = [0u8; 64];
                fill(&mut bytes);
                Self::from_uniform_bytes(&bytes)
            }

            fn from_limbs(limbs: Self::Limbs) -> Option<Self> {
                let integer = BigUint::from_bytes_le(&limbs.map(u64::to_le_bytes).concat());
                (integer < Self::modulus()).then(|| Self::from_integer(integer))
            }

            fn to_le_bits(&self) -> Self::Bits {
                let integer = self.integer();
                core::array::from_fn(|index| integer.bit(index as u64))
            }

            const TWO_ADICITY: u32 = $two_adicity;
            const MULTIPLICATIVE_GENERATOR: Self = {
                let mut limbs = [0; $limbs];
                limbs[0] = $generator;
                Self(limbs)
            };
            const ROOT_OF_UNITY: Self = Self($root);
            const ROOT_OF_UNITY_INVERSE: Self = Self($inverse_root);
            const TWO_INVERSE: Self = Self($two_inverse);
            const DELTA: Self = Self($delta);
            const ZETA: Self = Self($zeta);

            fn domain(log_size: u32) -> Result<Domain<Self>, FftError> {
                Domain::from_field(log_size)
            }

            fn fft(domain: Domain<Self>, values: &mut [Self]) {
                assert_eq!(values.len(), domain.size());
                dft(values, domain.root(), Self::ONE);
            }

            fn ifft(domain: Domain<Self>, values: &mut [Self]) {
                assert_eq!(values.len(), domain.size());
                dft(values, domain.inverse_root(), domain.size_inverse());
            }

            fn lagrange_evaluations(
                domain: Domain<Self>,
                point: Self,
                evaluations: &mut [Self],
                scratch: &mut [Self],
            ) -> Option<usize> {
                assert!(
                    evaluations.len() <= domain.size(),
                    "Lagrange evaluations exceed the domain size"
                );
                if let Some(index) = domain.elements().position(|node| node == point) {
                    evaluations.fill(Self::ZERO);
                    if let Some(value) = evaluations.get_mut(index) {
                        *value = Self::ONE;
                    }
                    return Some(index);
                }
                for (value, node) in evaluations.iter_mut().zip(domain.elements()) {
                    *value = point - node;
                }
                Self::batch_invert(evaluations, scratch);
                let scale = domain.vanishing(point) * domain.size_inverse();
                for (value, node) in evaluations.iter_mut().zip(domain.elements()) {
                    *value *= scale * node;
                }
                None
            }

            fn root_of_unity(log_size: u32) -> Option<Self> {
                (log_size <= Self::TWO_ADICITY).then(|| {
                    let exponent = BigUint::from(1u8) << (Self::TWO_ADICITY - log_size) as usize;
                    Self::from_integer(
                        Self::ROOT_OF_UNITY.integer().modpow(&exponent, &Self::modulus()),
                    )
                })
            }

            fn root_of_unity_inverse(log_size: u32) -> Option<Self> {
                (log_size <= Self::TWO_ADICITY).then(|| {
                    let exponent = BigUint::from(1u8) << (Self::TWO_ADICITY - log_size) as usize;
                    Self::from_integer(
                        Self::ROOT_OF_UNITY_INVERSE.integer().modpow(&exponent, &Self::modulus()),
                    )
                })
            }

            fn power_of_two_inverse(log_size: u32) -> Self {
                Self::TWO_INVERSE.pow_u64(u64::from(log_size))
            }

            type Accumulator = BigUint;

            fn mul_accumulate(accumulator: &mut BigUint, lhs: &Self, rhs: &Self) {
                *accumulator += lhs.integer() * rhs.integer();
            }

            fn reduce(accumulator: BigUint) -> Self {
                Self::from_integer(accumulator)
            }
        }
    };
}

reference_field!(
    BlsBase,
    6,
    48,
    381,
    [
        0xb9fe_ffff_ffff_aaab,
        0x1eab_fffe_b153_ffff,
        0x6730_d2a0_f6b0_f624,
        0x6477_4b84_f385_12bf,
        0x4b1b_a7b6_434b_acd7,
        0x1a01_11ea_397f_e69a
    ],
    generator = 2,
    two_adicity = 1,
    root = [
        0xb9fe_ffff_ffff_aaaa,
        0x1eab_fffe_b153_ffff,
        0x6730_d2a0_f6b0_f624,
        0x6477_4b84_f385_12bf,
        0x4b1b_a7b6_434b_acd7,
        0x1a01_11ea_397f_e69a
    ],
    inverse_root = [
        0xb9fe_ffff_ffff_aaaa,
        0x1eab_fffe_b153_ffff,
        0x6730_d2a0_f6b0_f624,
        0x6477_4b84_f385_12bf,
        0x4b1b_a7b6_434b_acd7,
        0x1a01_11ea_397f_e69a
    ],
    two_inverse = [
        0xdcff_7fff_ffff_d556,
        0x0f55_ffff_58a9_ffff,
        0xb398_6950_7b58_7b12,
        0xb23b_a5c2_79c2_895f,
        0x258d_d3db_21a5_d66b,
        0x0d00_88f5_1cbf_f34d
    ],
    delta = [4, 0, 0, 0, 0, 0],
    zeta = [
        0x2e01_ffff_fffe_fffe,
        0xde17_d813_620a_0002,
        0xddb3_a93b_e6f8_9688,
        0xba69_c607_6a0f_77ea,
        0x5f19_672f_df76_ce51,
        0
    ]
);
// This is also Jubjub's base field.
reference_field!(
    BlsScalar,
    4,
    32,
    255,
    [
        0xffff_ffff_0000_0001,
        0x53bd_a402_fffe_5bfe,
        0x3339_d808_09a1_d805,
        0x73ed_a753_299d_7d48
    ],
    generator = 7,
    two_adicity = 32,
    root = [
        0x3829_971f_439f_0d2b,
        0xb636_8350_8c22_80b9,
        0xd09b_6819_22c8_13b4,
        0x16a2_a19e_dfe8_1f20
    ],
    inverse_root = [
        0x0fb4_d6e1_3cf1_9a78,
        0x6f67_d4a2_b566_f833,
        0xed4f_2f74_a35d_0168,
        0x0538_a6f6_6e19_c653
    ],
    two_inverse = [
        0x7fff_ffff_8000_0001,
        0xa9de_d201_7fff_2dff,
        0x199c_ec04_04d0_ec02,
        0x39f6_d3a9_94ce_bea4
    ],
    delta = [
        0x6c08_3479_5901_89d7,
        0xf650_2437_c6a0_9c00,
        0x43ca_b354_fabb_0062,
        0x0863_4d0a_a021_aaf8
    ],
    zeta = [0x0000_0000_ffff_ffff, 0xac45_a401_0001_a402, 0, 0]
);
reference_field!(
    JubjubScalar,
    4,
    32,
    252,
    [
        0xd097_0e5e_d6f7_2cb7,
        0xa668_2093_ccc8_1082,
        0x0667_3b01_0134_3b00,
        0x0e7d_b4ea_6533_afa9
    ],
    generator = 6,
    two_adicity = 1,
    root = [
        0xd097_0e5e_d6f7_2cb6,
        0xa668_2093_ccc8_1082,
        0x0667_3b01_0134_3b00,
        0x0e7d_b4ea_6533_afa9
    ],
    inverse_root = [
        0xd097_0e5e_d6f7_2cb6,
        0xa668_2093_ccc8_1082,
        0x0667_3b01_0134_3b00,
        0x0e7d_b4ea_6533_afa9
    ],
    two_inverse = [
        0x684b_872f_6b7b_965c,
        0x5334_1049_e664_0841,
        0x8333_9d80_809a_1d80,
        0x073e_da75_3299_d7d4
    ],
    delta = [36, 0, 0, 0],
    zeta = [
        0x59bf_ba86_63cd_b2c4,
        0x7242_494d_4a86_d10b,
        0xb133_fd7d_bbf3_e5fa,
        0x07e1_d690_6f41_bf2f
    ]
);
// A byte encoding shorter than a limb catches accidental truncation.
reference_field!(
    Small,
    1,
    1,
    5,
    [17],
    generator = 3,
    two_adicity = 4,
    root = [3],
    inverse_root = [6],
    two_inverse = [9],
    delta = [1],
    zeta = [1]
);
reference_field!(
    SmallScalar,
    1,
    1,
    3,
    [5],
    generator = 2,
    two_adicity = 2,
    root = [2],
    inverse_root = [3],
    two_inverse = [3],
    delta = [1],
    zeta = [1]
);

// An independent quadratic transform using the integer reference arithmetic.
fn dft<F: Field>(values: &mut [F], root: F, scale: F) {
    let original = values.to_vec();
    for (i, value) in values.iter_mut().enumerate() {
        *value = original
            .iter()
            .enumerate()
            .map(|(j, coefficient)| *coefficient * root.pow_u64((i * j) as u64))
            .sum::<F>()
            * scale;
    }
}

fn sqrt(value: BigUint, p: BigUint) -> Option<BigUint> {
    let zero = BigUint::from(0u8);
    let one = BigUint::from(1u8);
    if value == zero {
        return Some(zero);
    }
    if value.modpow(&((&p - &one) >> 1usize), &p) != one {
        return None;
    }
    let mut odd = &p - &one;
    let mut order = 0;
    while !odd.bit(0) {
        odd >>= 1usize;
        order += 1;
    }
    let mut nonresidue = BigUint::from(2u8);
    while nonresidue.modpow(&((&p - &one) >> 1usize), &p) == one {
        nonresidue += &one;
    }
    let mut c = nonresidue.modpow(&odd, &p);
    let mut root = value.modpow(&((&odd + &one) >> 1usize), &p);
    let mut remainder = value.modpow(&odd, &p);
    while remainder != one {
        let mut squared = remainder.clone();
        let mut index = 0;
        while squared != one {
            squared = &squared * &squared % &p;
            index += 1;
        }
        let factor = c.modpow(&(&one << (order - index - 1)), &p);
        root = root * &factor % &p;
        c = &factor * &factor % &p;
        remainder = remainder * &c % &p;
        order = index;
    }
    Some(root)
}
