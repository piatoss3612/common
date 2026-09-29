//! Field elements for consumers of the optional arithmetic traits.

use super::Field;
use crate::{
    exec::{ExecutionOptions, SerialExecutor},
    fft::{Domain, FftError, Transform},
    field::{CanonicalUint, PastaField, PrimeModulus, ProductSum, pasta::TWO_ADICITY},
};
use core::{
    iter::{Product, Sum},
    ops,
};

/// A Pasta field element implementing the optional [`super::Field`] interface.
///
/// Native [`PastaField`] arithmetic is explicit. This wrapper supplies operators
/// for generic consumers, forwarding each operation to the native methods.
/// Both representations have identical layout and invariants. Slice views
/// preserve the original allocation, so bulk operations use native kernels
/// without staging or copying elements. No implicit dereferencing is provided;
/// use [`Self::as_inner`] when calling native APIs.
/// Equality compares field values, reducing both native representatives.
#[derive(Clone, Copy, bento::Pod)]
#[repr(transparent)]
pub struct FieldAdapter<M: PrimeModulus>(pub(crate) PastaField<M>);

impl<M: PrimeModulus> PartialEq for FieldAdapter<M> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        self.0.reduce() == other.0.reduce()
    }
}

impl<M: PrimeModulus> Eq for FieldAdapter<M> {}

impl<M: PrimeModulus> Default for FieldAdapter<M> {
    fn default() -> Self {
        Self(PastaField::ZERO)
    }
}

impl<M: PrimeModulus> FieldAdapter<M> {
    /// Borrows a native value as a consumer element without copying.
    pub fn from_ref(value: &PastaField<M>) -> &Self {
        // SAFETY: Self has the same layout and validity as its sole native
        // field. The returned shared reference preserves the input lifetime.
        unsafe { &*(core::ptr::from_ref(value).cast::<Self>()) }
    }

    /// Wraps a native value without changing its representation.
    pub const fn new(value: PastaField<M>) -> Self {
        Self(value)
    }

    /// Returns the native value.
    pub const fn into_inner(self) -> PastaField<M> {
        self.0
    }

    /// Borrows the native value.
    pub const fn as_inner(&self) -> &PastaField<M> {
        &self.0
    }

    /// Mutably borrows the native value.
    pub fn as_inner_mut(&mut self) -> &mut PastaField<M> {
        &mut self.0
    }

    /// Borrows native values as consumer elements without copying.
    pub fn from_slice(values: &[PastaField<M>]) -> &[Self] {
        // SAFETY: Self is transparent over PastaField<M> with identical
        // validity and alignment. The slice keeps its length and borrow; the
        // original reference exclusively controls mutation when mutable.
        unsafe { core::slice::from_raw_parts(values.as_ptr().cast::<Self>(), values.len()) }
    }

    /// Mutably borrows native values as consumer elements without copying.
    pub fn from_slice_mut(values: &mut [PastaField<M>]) -> &mut [Self] {
        // SAFETY: Self is transparent over PastaField<M> with identical
        // validity and alignment. The slice keeps its length and borrow; the
        // original reference exclusively controls mutation when mutable.
        unsafe { core::slice::from_raw_parts_mut(values.as_mut_ptr().cast::<Self>(), values.len()) }
    }

    /// Borrows the native buffer beneath consumer elements without copying.
    pub fn as_slice(values: &[Self]) -> &[PastaField<M>] {
        // SAFETY: Self is transparent over PastaField<M> with identical
        // validity and alignment. The slice keeps its length and borrow; the
        // original reference exclusively controls mutation when mutable.
        unsafe {
            core::slice::from_raw_parts(values.as_ptr().cast::<PastaField<M>>(), values.len())
        }
    }

    /// Mutably borrows the native buffer beneath consumer elements without copying.
    pub fn as_slice_mut(values: &mut [Self]) -> &mut [PastaField<M>] {
        // SAFETY: Self is transparent over PastaField<M> with identical
        // validity and alignment. The slice keeps its length and borrow; the
        // original reference exclusively controls mutation when mutable.
        unsafe {
            core::slice::from_raw_parts_mut(
                values.as_mut_ptr().cast::<PastaField<M>>(),
                values.len(),
            )
        }
    }
}

impl<M: PrimeModulus> From<PastaField<M>> for FieldAdapter<M> {
    fn from(value: PastaField<M>) -> Self {
        Self(value)
    }
}

impl<M: PrimeModulus> From<u64> for FieldAdapter<M> {
    fn from(value: u64) -> Self {
        Self(PastaField::from_u64(value))
    }
}

macro_rules! binary {
    ($trait:ident, $method:ident) => {
        impl<M: PrimeModulus> ops::$trait for FieldAdapter<M> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: Self) -> Self {
                Self(self.0.$method(&rhs.0))
            }
        }
        impl<M: PrimeModulus> ops::$trait<&Self> for FieldAdapter<M> {
            type Output = Self;
            #[inline]
            fn $method(self, rhs: &Self) -> Self {
                Self(self.0.$method(&rhs.0))
            }
        }
        impl<M: PrimeModulus> ops::$trait<FieldAdapter<M>> for &FieldAdapter<M> {
            type Output = FieldAdapter<M>;
            #[inline]
            fn $method(self, rhs: FieldAdapter<M>) -> Self::Output {
                FieldAdapter(self.0.$method(&rhs.0))
            }
        }
        impl<M: PrimeModulus> ops::$trait for &FieldAdapter<M> {
            type Output = FieldAdapter<M>;
            #[inline]
            fn $method(self, rhs: Self) -> Self::Output {
                FieldAdapter(self.0.$method(&rhs.0))
            }
        }
    };
}
macro_rules! assign {
    ($trait:ident, $method:ident, $native:ident) => {
        impl<M: PrimeModulus> ops::$trait for FieldAdapter<M> {
            #[inline]
            fn $method(&mut self, rhs: Self) {
                self.0 = self.0.$native(&rhs.0);
            }
        }
        impl<M: PrimeModulus> ops::$trait<&Self> for FieldAdapter<M> {
            #[inline]
            fn $method(&mut self, rhs: &Self) {
                self.0 = self.0.$native(&rhs.0);
            }
        }
    };
}
binary!(Add, add);
binary!(Sub, sub);
binary!(Mul, mul);
assign!(AddAssign, add_assign, add);
assign!(SubAssign, sub_assign, sub);
assign!(MulAssign, mul_assign, mul);

impl<M: PrimeModulus> ops::Neg for FieldAdapter<M> {
    type Output = Self;
    #[inline]
    fn neg(self) -> Self {
        Self(self.0.neg())
    }
}
impl<M: PrimeModulus> ops::Neg for &FieldAdapter<M> {
    type Output = FieldAdapter<M>;
    #[inline]
    fn neg(self) -> Self::Output {
        FieldAdapter(self.0.neg())
    }
}

macro_rules! fold {
    ($trait:ident, $method:ident, $identity:ident, $native:ident) => {
        impl<M: PrimeModulus> $trait for FieldAdapter<M> {
            fn $method<I: Iterator<Item = Self>>(iter: I) -> Self {
                Self(iter.fold(PastaField::$identity, |acc, value| acc.$native(&value.0)))
            }
        }
        impl<'a, M: PrimeModulus> $trait<&'a Self> for FieldAdapter<M> {
            fn $method<I: Iterator<Item = &'a Self>>(iter: I) -> Self {
                Self(iter.fold(PastaField::$identity, |acc, value| acc.$native(&value.0)))
            }
        }
    };
}
fold!(Sum, sum, ZERO, add);
fold!(Product, product, ONE, mul);

impl<M: PrimeModulus> core::fmt::Debug for FieldAdapter<M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

impl<M: PrimeModulus> Field for FieldAdapter<M> {
    type Repr = [u8; 32];

    type Limbs = [u64; 4];

    type Bits = [bool; 256];

    type Accumulator = ProductSum<M>;

    const ZERO: Self = Self(PastaField::ZERO);

    const ONE: Self = Self(PastaField::ONE);

    const MODULUS: [u64; 4] = M::MODULUS;

    const NUM_BITS: u32 = 256 - M::MODULUS[3].leading_zeros();

    const CAPACITY: u32 = 255 - M::MODULUS[3].leading_zeros();

    const TWO_ADICITY: u32 = TWO_ADICITY;

    const MULTIPLICATIVE_GENERATOR: Self = Self(PastaField::MULTIPLICATIVE_GENERATOR);

    const ROOT_OF_UNITY: Self = match PastaField::root_of_unity(TWO_ADICITY) {
        Some(root) => Self(root),
        None => panic!("the two-adicity is a supported root order"),
    };

    const ROOT_OF_UNITY_INVERSE: Self = match PastaField::root_of_unity_inverse(TWO_ADICITY) {
        Some(root) => Self(root),
        None => panic!("the two-adicity is a supported root order"),
    };

    const TWO_INVERSE: Self = Self(PastaField::TWO_INVERSE);

    const DELTA: Self = Self(PastaField::DELTA);

    const ZETA: Self = Self(PastaField::ZETA);

    fn is_zero(&self) -> bool {
        self.0.is_zero()
    }

    #[inline(always)]
    fn square(&self) -> Self {
        Self(self.0.square())
    }

    #[inline]
    fn mul_add(&self, multiplier: &Self, addend: &Self) -> Self {
        Self(self.0.mul_add(&multiplier.0, &addend.0))
    }

    #[inline(always)]
    fn double(&self) -> Self {
        Self(self.0.double())
    }

    fn invert(&self) -> Option<Self> {
        self.0.invert().map(Self)
    }

    fn batch_invert(values: &mut [Self], scratch: &mut [Self]) {
        crate::field::pasta::batch_invert(Self::as_slice_mut(values), Self::as_slice_mut(scratch))
    }

    fn batch_invert_groups(groups: &mut [impl AsMut<[Self]>], scratch: &mut [Self]) {
        crate::field::pasta::invert_groups(
            groups,
            scratch,
            Self::ONE,
            |value| value.0.is_zero(),
            |value| value.0.invert().map(Self),
            |a, b| Self(a.0.mul(&b.0)),
        )
    }

    fn sqrt(&self) -> Option<Self> {
        self.0.reduce().sqrt().map(|value| Self(value.into_loose()))
    }

    fn pow_u64(&self, exponent: u64) -> Self {
        Self(self.0.pow_u64(exponent))
    }

    fn from_u128(value: u128) -> Self {
        Self(PastaField::from_u128(value))
    }

    fn sum_of_products_slice(lhs: &[Self], rhs: &[Self]) -> Self {
        Self(PastaField::sum_of_products_slice(
            Self::as_slice(lhs),
            Self::as_slice(rhs),
        ))
    }

    fn sum_of_product_pairs<'a>(pairs: impl IntoIterator<Item = (&'a Self, &'a Self)>) -> Self {
        Self(PastaField::sum_of_product_pairs(
            pairs.into_iter().map(|(a, b)| (&a.0, &b.0)),
        ))
    }

    fn is_odd(&self) -> bool {
        self.0.is_odd()
    }

    fn to_bytes(&self) -> [u8; 32] {
        self.0.to_bytes()
    }

    fn from_bytes(bytes: [u8; 32]) -> Option<Self> {
        PastaField::from_bytes(bytes).map(Self)
    }

    fn from_uniform_bytes(bytes: &[u8; 64]) -> Self {
        Self(PastaField::from_wide_bytes_reduced(bytes))
    }

    fn random(fill: impl FnOnce(&mut [u8; 64])) -> Self {
        Self(crate::field::random::<M>(fill))
    }

    fn from_limbs(limbs: [u64; 4]) -> Option<Self> {
        PastaField::from_canonical_uint(CanonicalUint::from_limbs(limbs)).map(Self)
    }

    fn to_le_bits(&self) -> Self::Bits {
        let bytes = self.0.to_bytes();
        core::array::from_fn(|index| (bytes[index / 8] >> (index % 8)) & 1 == 1)
    }

    fn domain(log_size: u32) -> Result<Domain<Self>, FftError> {
        Domain::<PastaField<M>>::new(log_size).map(|domain| domain.map(Self))
    }

    fn fft(domain: Domain<Self>, values: &mut [Self]) {
        Transform::new(domain.map(Self::into_inner).subgroup())
            .forward(
                Self::as_slice_mut(values),
                ExecutionOptions::default(),
                &SerialExecutor,
                &mut [],
            )
            .expect("a serial subgroup transform supports empty scratch");
    }

    fn ifft(domain: Domain<Self>, values: &mut [Self]) {
        Transform::new(domain.map(Self::into_inner).subgroup())
            .inverse(
                Self::as_slice_mut(values),
                ExecutionOptions::default(),
                &SerialExecutor,
                &mut [],
            )
            .expect("a serial subgroup transform supports empty scratch");
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
        crate::fft::generic::evaluate_lagrange_with_index(
            domain.map(Self::into_inner).subgroup(),
            &point.0,
            Self::as_slice_mut(evaluations),
            Self::as_slice_mut(scratch),
        )
        .expect("a validated prefix fits the domain and output")
    }

    fn root_of_unity(log_size: u32) -> Option<Self> {
        PastaField::root_of_unity(log_size).map(Self)
    }

    fn root_of_unity_inverse(log_size: u32) -> Option<Self> {
        PastaField::root_of_unity_inverse(log_size).map(Self)
    }

    fn power_of_two_inverse(log_size: u32) -> Self {
        Self(PastaField::power_of_two_inverse(log_size))
    }

    #[inline]
    fn mul_accumulate(accumulator: &mut ProductSum<M>, lhs: &Self, rhs: &Self) {
        accumulator.add_product(&lhs.0, &rhs.0);
    }

    #[inline]
    fn reduce(accumulator: ProductSum<M>) -> Self {
        Self(accumulator.finish())
    }
}
