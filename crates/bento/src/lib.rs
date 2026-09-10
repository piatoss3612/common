//! Compile-time support for cryptographic arithmetic.
//!
//! Use [`addition_chain!`] to scale a value by a fixed positive integer. The
//! [`addchain`] module defines the operations that a value supplies for scaling.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub use bento_core::*;

/// Scales a value using a compile-time addition chain.
///
/// A fixed scalar allows the sequence of operations to be chosen at compile time.
/// This macro scales values through the [`addchain::AdditionChain`] support
/// interface, which defines the required operations and their laws.
///
/// `addition_chain!(value_expression, scalar)` evaluates the expression once,
/// taking ownership of its result, and returns the same type scaled by `scalar`.
/// The scalar must be a nonzero, unsuffixed integer literal: decimal, hexadecimal,
/// octal, and binary are accepted, with underscores and an optional trailing
/// comma. Literals wider than 128 bits are supported. Constants, expressions,
/// negative values, and zero are not accepted.
///
/// Implement [`addchain::AdditionChain`] on the value's type or on an adapter, as
/// shown below. The trait need not be imported to invoke the macro.
///
/// # Examples
///
/// ```
/// # use zakura_bento as bento;
/// #[derive(Clone, Debug, PartialEq)]
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
/// The exponentiation interpretation described by [`addchain::AdditionChain`]
/// can be used to raise a value to a fixed power:
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
