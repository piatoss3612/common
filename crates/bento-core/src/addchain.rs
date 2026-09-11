//! Support for scaling values with addition chains.
//!
//! [`AdditionChain`] supplies the operations needed to combine multiples of a
//! value. Implement it to support scaling for a type or an adapter.

/// A support interface for scaling a value with an addition chain.
///
/// An addition chain builds a positive multiple of one input by combining
/// previously computed multiples. This interface supplies the operations needed
/// to follow such a chain without requiring an identity or inverse operation.
///
/// [`add`](Self::add) must be associative, [`double`](Self::double) must equal
/// adding a value to itself, and cloning must preserve the value. Addition need
/// not be commutative: all values in a chain are multiples of one input, so they
/// commute given associativity. For exponentiation, implement
/// [`double`](Self::double) as squaring and [`add`](Self::add) as multiplication.
///
/// Use the fully qualified trait path in implementations to avoid introducing
/// its method names into method lookup elsewhere in the module.
pub trait AdditionChain: Clone {
    /// Returns the sum of this value with itself.
    fn double(&self) -> Self;

    /// Returns the sum of this value and `rhs`, preserving both inputs.
    fn add(&self, rhs: &Self) -> Self;

    /// Doubles this value `count` times; zero returns a clone.
    ///
    /// Overrides must have the same value as repeated calls to `double`.
    fn double_n(&self, count: usize) -> Self {
        if count == 0 {
            return self.clone();
        }
        let mut value = self.double();
        for _ in 1..count {
            value = value.double();
        }
        value
    }

    /// Doubles this value `count` times, then adds `rhs`.
    ///
    /// Overrides may fuse the operations but must preserve their value.
    fn double_n_add(&self, count: usize, rhs: &Self) -> Self {
        if count == 0 {
            self.add(rhs)
        } else {
            self.double_n(count).add(rhs)
        }
    }
}
