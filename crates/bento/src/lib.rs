//! Shared POD storage utilities and compile-time support for field implementations.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub use bento_core::*;

/// Scale a value using a compile-time addition chain.
///
/// `addition_chain!(value_expression, scalar)` evaluates the expression once,
/// taking ownership of its result, and returns the same type scaled by `scalar`.
/// The scalar must be a nonzero, unsuffixed integer literal: decimal, hexadecimal,
/// octal, and binary are accepted, with underscores and an optional trailing
/// comma. Literals wider than 128 bits are supported; the scalar never becomes
/// a target integer. Constants, expressions, negative values, and zero are not
/// accepted. Zero would require an identity operation.
///
/// The value's type must implement [`addchain::AdditionChain`], whose supertrait is
/// `Clone`, with `double(&self) -> Self` and
/// `add(&self, rhs: &Self) -> Self`. Addition must be associative, doubling must
/// equal adding a value to itself, and cloning must preserve the value.
/// Implement this support trait on an internal type or adapter using the fully
/// qualified path, `impl bento::addchain::AdditionChain for Value`. Avoid importing the
/// trait so its method names do not affect method lookup elsewhere in the
/// module. `Copy` is not required. The generated code also uses qualified trait
/// calls, so inherent methods named `double` or `add` do not affect it.
///
/// The host planner compares left-to-right sliding windows of widths one through
/// six, counting additions and doublings equally, including preparation of odd
/// multiples. Ties prefer wider windows. This heuristic does not guarantee a
/// shortest chain and does not account for cloning or storage costs. The output
/// prepares the needed odd multiples, clones the initial accumulator once, and
/// performs a fixed sequence of trait calls. Scalar one clones the input without
/// adding or doubling. Each replacement drops the previous accumulator before
/// the next operation. The input and precomputed values drop when the generated
/// block finishes. There are no generated runtime loops or branches; timing and
/// allocation behavior still depend on the type's operations and destructors.
///
/// ```
/// # use zakura_bento as bento;
/// #[derive(Clone, Debug, PartialEq)] // Deliberately not Copy.
/// struct Value(u64);
///
/// impl bento::addchain::AdditionChain for Value {
///     fn double(&self) -> Self { Self(self.0 * 2) }
///     fn add(&self, rhs: &Self) -> Self { Self(self.0 + rhs.0) }
/// }
///
/// let value = Value(7);
/// assert_eq!(bento::addition_chain!(value.clone(), 0xb5), Value(1267));
/// assert_eq!(value, Value(7));
/// ```
///
/// Exponentiation uses the same chain by interpreting doubling as squaring
/// and addition as multiplication:
///
/// ```
/// # use zakura_bento as bento;
/// #[derive(Clone, Debug, PartialEq)]
/// struct Power(u64);
///
/// impl bento::addchain::AdditionChain for Power {
///     fn double(&self) -> Self { Self(self.0 * self.0 % 65_521) }
///     fn add(&self, rhs: &Self) -> Self { Self(self.0 * rhs.0 % 65_521) }
/// }
///
/// assert_eq!(bento::addition_chain!(Power(3), 9), Power(19_683));
/// ```
pub use bento_macros::addition_chain;
