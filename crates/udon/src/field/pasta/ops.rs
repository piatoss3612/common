//! Operator forms of the field arithmetic and the field trait instances.
//!
//! Every operator forwards to the inherent method of the same meaning, so
//! nothing here adds an arithmetic path. Binary operators take operands in the
//! same reduction state and return loose values, like the inherent methods;
//! assignment operators therefore apply to loose values only. Keeping both
//! operands in one state lets the right operand's state follow the left one,
//! so `x * PastaField::from_u64(3)` needs no annotation; mixed states use the
//! inherent methods. The [`Field`], [`FftField`], and [`DeferredField`]
//! instances delegate the same way, so generic code reaches the same kernels
//! as direct callers.

use core::{
    iter::{Product, Sum},
    ops,
};

use crate::{
    exec::{ExecutionOptions, SerialExecutor},
    fft::{Domain, Transform},
};

use super::parameters::TWO_ADICITY;
use super::{
    CanonicalUint, CubeRootField, DeferredField, FftField, Field, PastaField, PrimeField,
    PrimeModulus, ProductSum, ReductionState,
};

macro_rules! forward_binary_operator {
    ($trait:ident, $method:ident) => {
        impl<M: PrimeModulus, S: ReductionState> ops::$trait for PastaField<M, S> {
            type Output = PastaField<M>;

            #[inline]
            fn $method(self, rhs: Self) -> PastaField<M> {
                PastaField::$method(&self, &rhs)
            }
        }

        impl<M: PrimeModulus, S: ReductionState> ops::$trait<&PastaField<M, S>>
            for PastaField<M, S>
        {
            type Output = PastaField<M>;

            #[inline]
            fn $method(self, rhs: &Self) -> PastaField<M> {
                PastaField::$method(&self, rhs)
            }
        }

        impl<M: PrimeModulus, S: ReductionState> ops::$trait<PastaField<M, S>>
            for &PastaField<M, S>
        {
            type Output = PastaField<M>;

            #[inline]
            fn $method(self, rhs: PastaField<M, S>) -> PastaField<M> {
                PastaField::$method(self, &rhs)
            }
        }

        impl<M: PrimeModulus, S: ReductionState> ops::$trait<&PastaField<M, S>>
            for &PastaField<M, S>
        {
            type Output = PastaField<M>;

            #[inline]
            fn $method(self, rhs: &PastaField<M, S>) -> PastaField<M> {
                PastaField::$method(self, rhs)
            }
        }
    };
}

macro_rules! forward_assign_operator {
    ($trait:ident, $method:ident, $inherent:ident) => {
        impl<M: PrimeModulus> ops::$trait for PastaField<M> {
            #[inline]
            fn $method(&mut self, rhs: Self) {
                *self = PastaField::$inherent(self, &rhs);
            }
        }

        impl<M: PrimeModulus> ops::$trait<&PastaField<M>> for PastaField<M> {
            #[inline]
            fn $method(&mut self, rhs: &Self) {
                *self = PastaField::$inherent(self, rhs);
            }
        }
    };
}

forward_binary_operator!(Add, add);
forward_binary_operator!(Sub, sub);
forward_binary_operator!(Mul, mul);
forward_assign_operator!(AddAssign, add_assign, add);
forward_assign_operator!(SubAssign, sub_assign, sub);
forward_assign_operator!(MulAssign, mul_assign, mul);

impl<M: PrimeModulus, S: ReductionState> ops::Neg for PastaField<M, S> {
    type Output = PastaField<M>;

    #[inline]
    fn neg(self) -> PastaField<M> {
        PastaField::neg(&self)
    }
}

impl<M: PrimeModulus, S: ReductionState> ops::Neg for &PastaField<M, S> {
    type Output = PastaField<M>;

    #[inline]
    fn neg(self) -> PastaField<M> {
        PastaField::neg(self)
    }
}

impl<M: PrimeModulus, S: ReductionState> From<u64> for PastaField<M, S> {
    #[inline]
    fn from(value: u64) -> Self {
        Self::from_u64(value)
    }
}

impl<M: PrimeModulus, T: ReductionState> Sum<PastaField<M, T>> for PastaField<M> {
    fn sum<I: Iterator<Item = PastaField<M, T>>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |sum, term| PastaField::add(&sum, &term))
    }
}

impl<'a, M: PrimeModulus, T: ReductionState> Sum<&'a PastaField<M, T>> for PastaField<M> {
    fn sum<I: Iterator<Item = &'a PastaField<M, T>>>(iter: I) -> Self {
        iter.fold(Self::ZERO, |sum, term| PastaField::add(&sum, term))
    }
}

impl<M: PrimeModulus, T: ReductionState> Product<PastaField<M, T>> for PastaField<M> {
    fn product<I: Iterator<Item = PastaField<M, T>>>(iter: I) -> Self {
        iter.fold(Self::ONE, |product, factor| {
            PastaField::mul(&product, &factor)
        })
    }
}

impl<'a, M: PrimeModulus, T: ReductionState> Product<&'a PastaField<M, T>> for PastaField<M> {
    fn product<I: Iterator<Item = &'a PastaField<M, T>>>(iter: I) -> Self {
        iter.fold(Self::ONE, |product, factor| {
            PastaField::mul(&product, factor)
        })
    }
}

impl<M: PrimeModulus> Field for PastaField<M> {
    const ZERO: Self = Self::ZERO;
    const ONE: Self = Self::ONE;
    fn is_zero(&self) -> bool {
        PastaField::is_zero(self)
    }

    fn square(&self) -> Self {
        PastaField::square(self)
    }

    fn double(&self) -> Self {
        PastaField::double(self)
    }

    fn invert(&self) -> Option<Self> {
        PastaField::invert(self)
    }

    fn batch_invert(values: &mut [Self], scratch: &mut [Self]) {
        super::batch_invert_groups(&mut [values], scratch)
    }

    fn sqrt(&self) -> Option<Self> {
        self.reduce().sqrt().map(PastaField::into_loose)
    }

    fn pow_u64(&self, exponent: u64) -> Self {
        PastaField::pow_u64(self, exponent)
    }

    fn from_u128(value: u128) -> Self {
        PastaField::from_u128(value)
    }

    fn sum_of_product_pairs<'a>(pairs: impl IntoIterator<Item = (&'a Self, &'a Self)>) -> Self {
        PastaField::sum_of_product_pairs(pairs)
    }
}

impl<M: PrimeModulus> PrimeField for PastaField<M> {
    type Repr = [u8; 32];
    type Limbs = [u64; 4];
    type Bits = [bool; 256];

    const MODULUS: [u64; 4] = M::MODULUS;
    const NUM_BITS: u32 = 256 - M::MODULUS[3].leading_zeros();
    const CAPACITY: u32 = 255 - M::MODULUS[3].leading_zeros();

    fn is_odd(&self) -> bool {
        PastaField::is_odd(self)
    }

    fn to_bytes(&self) -> [u8; 32] {
        PastaField::to_bytes(*self)
    }

    fn from_bytes(bytes: [u8; 32]) -> Option<Self> {
        PastaField::from_bytes(bytes)
    }

    fn from_uniform_bytes(bytes: &[u8; 64]) -> Self {
        PastaField::from_wide_bytes_reduced(bytes)
    }

    fn from_limbs(limbs: [u64; 4]) -> Option<Self> {
        PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs))
    }

    fn to_le_bits(&self) -> Self::Bits {
        let bytes = PastaField::to_bytes(*self);
        core::array::from_fn(|index| (bytes[index / 8] >> (index % 8)) & 1 == 1)
    }
}

impl<M: PrimeModulus> FftField for PastaField<M> {
    const TWO_ADICITY: u32 = TWO_ADICITY;
    const MULTIPLICATIVE_GENERATOR: Self = Self::MULTIPLICATIVE_GENERATOR;
    const ROOT_OF_UNITY: Self = match Self::root_of_unity(TWO_ADICITY) {
        Some(root) => root,
        None => panic!("the two-adicity is a supported root order"),
    };
    const ROOT_OF_UNITY_INVERSE: Self = match Self::root_of_unity_inverse(TWO_ADICITY) {
        Some(root) => root,
        None => panic!("the two-adicity is a supported root order"),
    };
    const TWO_INVERSE: Self = Self::TWO_INVERSE;
    const DELTA: Self = Self::DELTA;

    fn fft(domain: Domain<Self>, values: &mut [Self]) {
        Transform::new(domain.subgroup())
            .forward(
                values,
                ExecutionOptions::default(),
                &SerialExecutor,
                &mut [],
            )
            .expect("a serial subgroup transform supports empty scratch");
    }

    fn ifft(domain: Domain<Self>, values: &mut [Self]) {
        Transform::new(domain.subgroup())
            .inverse(
                values,
                ExecutionOptions::default(),
                &SerialExecutor,
                &mut [],
            )
            .expect("a serial subgroup transform supports empty scratch");
    }

    fn root_of_unity(log_size: u32) -> Option<Self> {
        PastaField::root_of_unity(log_size)
    }

    fn root_of_unity_inverse(log_size: u32) -> Option<Self> {
        PastaField::root_of_unity_inverse(log_size)
    }

    fn power_of_two_inverse(log_size: u32) -> Self {
        PastaField::power_of_two_inverse(log_size)
    }
}

impl<M: PrimeModulus> DeferredField for PastaField<M> {
    type Accumulator = ProductSum<M>;

    fn mul_accumulate(accumulator: &mut ProductSum<M>, lhs: &Self, rhs: &Self) {
        accumulator.add_product(lhs, rhs);
    }

    fn reduce(accumulator: ProductSum<M>) -> Self {
        accumulator.finish()
    }
}

impl<M: PrimeModulus> CubeRootField for PastaField<M> {
    const ZETA: Self = Self::ZETA;
}
