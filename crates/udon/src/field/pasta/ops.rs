//! Operator forms of the native field arithmetic.
//!
//! Every operator forwards to the inherent method of the same meaning, so
//! nothing here adds an arithmetic path. Binary operators take operands in the
//! same reduction state and return loose values, like the inherent methods;
//! assignment operators therefore apply to loose values only. Keeping both
//! operands in one state lets the right operand's state follow the left one,
//! so `x * PastaField::from_u64(3)` needs no annotation; mixed states use the
//! inherent methods.

use core::{
    iter::{Product, Sum},
    ops,
};

use super::{PastaField, PrimeModulus, ReductionState};

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
