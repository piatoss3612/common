//! Support for scaling values with addition chains.

/// Operations used by `bento::addition_chain!` to scale a value.
///
/// Implement this on an internal type or wrapper in the consuming crate using
/// the fully qualified path, such as `impl bento::addchain::AdditionChain for Value`.
/// Avoid importing the trait so its method names do not affect method lookup
/// elsewhere in the module. This trait provides support for the macro.
///
/// `add` must be associative, `self.double()` must be equivalent to
/// `self.add(self)`, and cloning must preserve the value. No identity or inverse
/// operation is required. Addition need not be commutative: all values in a
/// chain are multiples of one input, so they commute given associativity.
/// For exponentiation, implement `double` as squaring and `add` as multiplication.
pub trait AdditionChain: Clone {
    /// Return the sum of this value with itself.
    fn double(&self) -> Self;

    /// Return the sum of this value and `rhs`, preserving both inputs.
    fn add(&self, rhs: &Self) -> Self;
}
