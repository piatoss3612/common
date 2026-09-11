//! Compile-time support for cryptographic arithmetic.
//!
//! Use [`addition_chain!`] to scale a value by a fixed positive integer. The
//! [`addchain`] module defines the operations that a value supplies for scaling.
//! [`const_arithmetic::u256`] provides full-width integer arithmetic, and
//! [`const_arithmetic::m255`] provides Montgomery arithmetic for odd moduli
//! below `2^255`. Use them to derive field and curve constants at compile time.
//!
//! See [`const_arithmetic::m255::two_adic_root_of_unity`] for an example deriving
//! a field constant from a modulus and a generator.
//!
//! Store records with the [`Pod`](trait@Pod) trait and its
//! [derive macro](macro@Pod). [`bytes_of`] and [`bytes_of_slice`] expose their
//! stored representation; [`embed_struct!`] and [`embed_array!`] include files as
//! typed static data. [`AlignedBytes`] provides aligned storage when the bytes
//! are already available.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

extern crate self as zakura_bento;

pub use bento_core::{AlignedBytes, MAX_ALIGN, Pod, addchain, bytes_of, bytes_of_slice};

#[doc(inline)]
pub use bento_core::const_arithmetic;

#[doc(hidden)]
pub use bento_macros::addition_chain as __addition_chain;

#[doc(inline)]
pub use bento_core::{embed_array, embed_struct};

/// Derives the [`Pod`](trait@Pod) storage contract for a struct.
///
/// Use this derive for records that generators write as bytes and consumers
/// embed as static data. It implements the [`Pod`](trait@Pod) contract for
/// `repr(C)` and `repr(transparent)` structs whose fields implement
/// [`Pod`](trait@Pod). Packed structs, enums, and unions are not supported.
/// Explicit `repr(align(...))` is allowed within [`MAX_ALIGN`].
///
/// The struct must also implement [`Copy`] and [`Sync`] and satisfy `'static`.
/// Generic structs are supported. A [`core::marker::PhantomData<T>`] field needs
/// `T: Sync + 'static`, but does not require `T` to implement [`Pod`](trait@Pod)
/// or [`Copy`]. Deriving [`Copy`] separately can impose additional bounds.
///
/// Deriving [`Pod`](trait@Pod) leaves layout validation to [`Pod::ASSERT_LAYOUT`].
/// A padded or over-aligned type can still be constructed and used normally;
/// converting it to bytes fails during compilation.
///
/// The derive checks representation and layout. It cannot check invariants
/// assumed by methods on the record: every bit pattern must be memory-safe for
/// its safe operations, as required by the [trait's safety contract](trait@Pod).
///
/// # Support paths
///
/// Ordinary facade dependency aliases are discovered automatically. Use
/// `#[pod(crate = path)]` on the struct for build-only dependencies, direct core
/// dependencies, support reached through another crate's re-exports, or
/// ambiguous dependency arrangements. The path must expose the
/// [`Pod`](trait@Pod) trait. Discovery reads manifest names; it does not determine
/// which dependencies Cargo enabled.
///
/// # Examples
///
/// ```
/// # use zakura_bento as bento;
/// #[repr(C)]
/// #[derive(Clone, Copy, bento::Pod)]
/// struct Record {
///     low: u16,
///     high: u16,
///     value: u32,
/// }
///
/// let record = Record { low: 0x0201, high: 0x0403, value: 0x0807_0605 };
/// assert_eq!(bento::bytes_of(&record), &[1, 2, 3, 4, 5, 6, 7, 8]);
///
/// static BYTES: bento::AlignedBytes<8> = bento::AlignedBytes([1, 2, 3, 4, 5, 6, 7, 8]);
/// static STORED: &Record = BYTES.as_value();
/// assert_eq!(STORED.value, record.value);
/// ```
///
/// Concrete layouts can also be checked explicitly, without converting bytes:
///
/// ```compile_fail
/// # use zakura_bento as bento;
/// #[repr(C)]
/// #[derive(Clone, Copy, bento::Pod)]
/// struct Padded(u8, u32);
///
/// const _: () = <Padded as bento::Pod>::ASSERT_LAYOUT;
/// ```
pub use bento_macros::Pod;

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
/// Alternatively, `tonelli_shanks("0x…", s)` derives `(t - 1) / 2` from
/// `p - 1 = t * 2^s`. Supply an odd modulus above two as `0x` followed by
/// exactly 64 hexadecimal digits, and its exact two-adicity as an unsuffixed
/// literal in `1..=255`. Primality is not checked; a zero derived exponent is
/// rejected. This syntax evaluates no arbitrary Rust constants or expressions.
///
/// Two optional named arguments follow the scalar:
///
/// - `chain = |input| { ...; result }` supplies a schedule. Each statement must
///   bind a unique plain name to `double(previous, positive_literal_count)` or
///   `add(previous, previous)`. References must name the input or earlier
///   bindings. The final expression names the result. Integer replay checks
///   every coefficient and rejects schedules that overshoot or do not compute
///   the requested scalar. The closure is syntax for the generator; it does not
///   execute Rust code.
/// - `emission = compact | unrolled | batched` selects code generation.
///   `compact` is the default and emits loops for long runs of doublings;
///   `unrolled` emits individual operations. `batched` uses
///   [`double_n`](addchain::AdditionChain::double_n) and
///   [`double_n_add`](addchain::AdditionChain::double_n_add), whose default
///   implementations preserve the same arithmetic. Types may override these
///   hooks to reduce the cost of intermediate representations.
///
/// Implement [`addchain::AdditionChain`] on the value's type or on an adapter, as
/// shown below. The trait need not be imported to invoke the macro.
///
/// Dependency aliases and re-exports of this macro retain the facade's support
/// path, including in build scripts.
///
/// The operation schedule depends on the scalar literal, not the input value.
/// Its exact sequence and operation count are implementation details; a shortest
/// chain is not guaranteed. A fixed schedule does not establish constant-time
/// behavior for input values: cloning, arithmetic, and dropping values also
/// depend on the supplied type.
///
/// # Panics
///
/// Panics from the input expression or the supplied type's operations propagate
/// to the caller.
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
/// assert_eq!(bento::addition_chain!(Power(3), 9, chain = |x| {
///     let eight = double(x, 3);
///     let nine = add(eight, x);
///     nine
/// }, emission = batched), Power(19_683));
/// ```
#[macro_export]
macro_rules! addition_chain {
    ($($input:tt)*) => {
        $crate::__addition_chain!(crate = $crate; $($input)*)
    };
}
