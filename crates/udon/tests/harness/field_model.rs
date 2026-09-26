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
    fft::Domain,
    field::{FftField, Field, PrimeField},
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
    ($name:ident, $limbs:literal, $bytes:literal, $bits:literal, $modulus:expr) => {
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

            fn double(&self) -> Self {
                *self + self
            }

            fn invert(&self) -> Option<Self> {
                self.integer().modinv(&Self::modulus()).map(Self::from_integer)
            }

            fn batch_invert(values: &mut [Self], scratch: &mut [Self]) {
                Self::batch_invert_groups(&mut [values], scratch)
            }

            fn sqrt(&self) -> Option<Self> {
                sqrt(self.integer(), Self::modulus()).map(Self::from_integer)
            }

            fn pow_u64(&self, exponent: u64) -> Self {
                Self::from_integer(self.integer().modpow(&exponent.into(), &Self::modulus()))
            }


        }

        impl PrimeField for $name {
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

            fn to_le_bits(&self) -> Self::Bits {
                let integer = self.integer();
                core::array::from_fn(|index| integer.bit(index as u64))
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
    ]
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
    ]
);
// A byte encoding shorter than a limb catches accidental truncation.
reference_field!(Small, 1, 1, 5, [17]);
reference_field!(SmallScalar, 1, 1, 3, [5]);

impl FftField for Small {
    const TWO_ADICITY: u32 = 4;
    const MULTIPLICATIVE_GENERATOR: Self = Self([3]);
    const ROOT_OF_UNITY: Self = Self([3]);
    const ROOT_OF_UNITY_INVERSE: Self = Self([6]);
    const TWO_INVERSE: Self = Self([9]);
    const DELTA: Self = Self::ONE;

    fn fft(domain: Domain<Self>, values: &mut [Self]) {
        assert_eq!(values.len(), domain.size());
        Self::dft(values, domain.root(), Self::ONE);
    }

    fn ifft(domain: Domain<Self>, values: &mut [Self]) {
        assert_eq!(values.len(), domain.size());
        Self::dft(values, domain.inverse_root(), domain.size_inverse());
    }
}

impl Small {
    // An independent quadratic transform over integers modulo 17.
    fn dft(values: &mut [Self], root: Self, scale: Self) {
        let mut original = [Self::ZERO; 16];
        let original = &mut original[..values.len()];
        original.copy_from_slice(values);
        for (i, value) in values.iter_mut().enumerate() {
            *value = original
                .iter()
                .enumerate()
                .map(|(j, coefficient)| *coefficient * root.pow_u64((i * j) as u64))
                .sum::<Self>()
                * scale;
        }
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
