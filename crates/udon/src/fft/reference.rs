//! Textbook, in-place radix-2 transforms over twiddle-scalable values.
//!
//! These algorithms are useful in downstream generators and as correctness
//! oracles. They use no auxiliary buffers; allocations and timing behavior of
//! caller-provided arithmetic and cloning depend on those implementations.
//! Callers supply roots and inverse lengths. The optimized Pasta API is
//! [`super::Transform`].
//!
//! Every [`Field`] is its own twiddle domain and butterfly value, so field
//! elements transform over their own field. [`ProjectivePoint`] values
//! transform over their curve's scalar field, supporting coefficient and
//! Lagrange basis conversion in artifact generators. Group arithmetic is
//! variable-time and needs no allocation; callers can batch-normalize outputs
//! with [`crate::curve::batch_normalize`]. [`super::Domain`] binds these
//! transforms to a domain's root and normalization.

use crate::{
    curve::{PastaCurve, ProjectivePoint},
    field::{Field, PastaField},
};

#[cfg(test)]
mod tests;

/// The scalar domain containing a transform's roots of unity.
///
/// For correct transforms, these operations must agree with multiplication in
/// a commutative ring: [`Self::ONE`] is its multiplicative identity, and
/// [`Self::multiply`] is associative and commutative. [`Self::square`] must
/// agree with multiplying a value by itself. A field is sufficient, and every
/// [`Field`] implements this trait through its operators.
/// The transform's root requirements are documented on [`transform`]. These
/// algebraic laws are not checked and are not memory-safety requirements.
pub trait Twiddle: Copy {
    /// The multiplicative identity.
    const ONE: Self;
    /// Multiplies two scalars.
    fn multiply(&self, rhs: &Self) -> Self;
    /// Squares a scalar.
    fn square(&self) -> Self;
}

/// Values supporting addition, negation, and multiplication by a twiddle.
///
/// For correct transforms, values must form an additive commutative group with
/// the scalar action of [`Twiddle`]'s ring. Scaling must distribute over value
/// and scalar addition, compose according to scalar multiplication, and leave
/// the value unchanged for [`Twiddle::ONE`]. Cloning must preserve the value.
/// Violating these unchecked laws can produce incorrect transform results.
/// Every [`Field`] is a butterfly value over itself.
pub trait Butterfly<T: Twiddle>: Clone {
    /// Scales this value.
    fn scaled(&self, twiddle: &T) -> Self;
    /// Adds two values.
    fn add(&self, rhs: &Self) -> Self;
    /// Negates this value.
    fn negated(&self) -> Self;
}

// The field's operators forward to its kernels, so these instances add no
// arithmetic path.
impl<F: Field> Twiddle for F {
    const ONE: Self = <F as Field>::ONE;
    fn multiply(&self, rhs: &Self) -> Self {
        *self * rhs
    }
    fn square(&self) -> Self {
        Field::square(self)
    }
}

impl<F: Field> Butterfly<F> for F {
    fn scaled(&self, twiddle: &F) -> Self {
        *self * twiddle
    }
    fn add(&self, rhs: &Self) -> Self {
        *self + rhs
    }
    fn negated(&self) -> Self {
        -*self
    }
}

impl<C: PastaCurve> Butterfly<PastaField<C::Scalar>> for ProjectivePoint<C> {
    fn scaled(&self, twiddle: &PastaField<C::Scalar>) -> Self {
        self.mul(twiddle)
    }
    fn add(&self, rhs: &Self) -> Self {
        self.add(rhs)
    }
    fn negated(&self) -> Self {
        self.neg()
    }
}

/// Computes `output[j] = sum(input[i] * root^(i*j), i = 0..n)` in place.
///
/// Here `n = values.len()`, and `*` means [`Butterfly::scaled`]. Both input and
/// output are in natural order. Supply a principal root of order `n`: its
/// multiplicative order is exactly `n`, and `sum(root^(i*k), i = 0..n) = 0`
/// for every `0 < k < n`. For `n > 1`, also require `root^(n/2) = -1` so that
/// each radix-two butterfly can use subtraction for its second output. The
/// character sums alone do not imply this when `n` is not invertible in the
/// scalar ring. A field root of exact order `n` satisfies all these conditions.
/// A forward transform does not otherwise require an invertible length.
/// Root validity is not checked; a wrong root can give an incorrect transform.
/// A singleton is the identity.
///
/// # Panics
///
/// Panics before mutation if the length is zero or not a power of two. Panics
/// in caller-provided arithmetic or cloning may leave partial results.
pub fn transform<T: Twiddle, V: Butterfly<T>>(values: &mut [V], root: &T) {
    let size = values.len();
    assert!(
        size.is_power_of_two(),
        "transform length must be a power of two"
    );
    let log_size = size.ilog2();
    for index in 0..size {
        let reversed = super::bit_reverse(index, log_size);
        if index < reversed {
            values.swap(index, reversed);
        }
    }
    for stage in 1..=log_size {
        let block = 1usize << stage;
        let mut step = *root;
        for _ in stage..log_size {
            step = step.square();
        }
        for chunk in values.chunks_exact_mut(block) {
            let (left, right) = chunk.split_at_mut(block / 2);
            // The first twiddle is one. The scalar action's identity law lets
            // even an expensive value type bypass its scaling machinery here.
            let (first_left, left) = left.split_first_mut().unwrap();
            let (first_right, right) = right.split_first_mut().unwrap();
            let original = first_left.clone();
            *first_left = original.add(first_right);
            *first_right = original.add(&first_right.negated());
            let mut twiddle = step;
            for (left, right) in left.iter_mut().zip(right) {
                let product = right.scaled(&twiddle);
                let original = left.clone();
                *left = original.add(&product);
                *right = original.add(&product.negated());
                twiddle = twiddle.multiply(&step);
            }
        }
    }
}

/// Transforms at `inverse_root`, then scales every output by `size_inverse`.
///
/// To undo [`transform`], supply the multiplicative inverse of a root meeting
/// its full contract and of the scalar `values.len()`. The length must be
/// invertible in the scalar ring. Neither scalar is validated. Both sides use
/// natural order.
///
/// # Panics
///
/// Panics on the same invalid lengths as [`transform`], before mutation.
/// Panics in caller-provided arithmetic or cloning may leave partial results.
pub fn inverse_transform<T: Twiddle, V: Butterfly<T>>(
    values: &mut [V],
    inverse_root: &T,
    size_inverse: &T,
) {
    transform(values, inverse_root);
    for value in values {
        *value = value.scaled(size_inverse);
    }
}
