//! Direct byte storage and embedding of trusted values.
//!
//! # Background
//!
//! Artifact generators write already-constructed values that consumers access
//! directly from static bytes. Embedding preserves those values, including any
//! representation state encoded in their types. It performs no runtime
//! validation, normalization, or initialization. [`Pod`] expresses the layout
//! and memory-safety requirements for sharing values this way.
//!
//! # Design
//!
//! Layout checks depend on concrete generic arguments and the compilation target.
//! [`Pod::ASSERT_LAYOUT`] allows generic layouts to be checked when concrete
//! types are used. Every conversion forces const evaluation, including empty
//! views; the compiler can also evaluate concrete assertions earlier.

use core::marker::PhantomData;

mod layout;
mod macros;
mod storage;

pub use layout::PodLayout;
use layout::assert_little_endian;
pub use storage::{AlignedBytes, MAX_ALIGN, bytes_of, bytes_of_slice};

/// A type whose values can be stored as bytes and used directly after embedding.
///
/// Stored records use a fixed little-endian layout so generators can write bytes
/// that consumers read without decoding or allocation. This contract supports
/// shared byte views through [`bytes_of`] and [`bytes_of_slice`], and typed
/// static views through [`AlignedBytes`].
/// The bytes come from trusted values of the same type. Construction establishes
/// the values' invariants; storage preserves their exact representation without
/// rechecking or changing it.
///
/// Layout validation is conditional: a type can implement this trait even when
/// [`ASSERT_LAYOUT`] fails. Every byte conversion evaluates that assertion at
/// compile time before using the layout. Use `cargo build` or `cargo test` to
/// exercise these checks: `cargo check` can miss failures that require code
/// generation.
///
/// Implementations are provided for `u8`, `u16`, `u32`, `u64`, arrays, and
/// [`PhantomData<T>`] with `T: ?Sized + Sync + 'static`. Integer size and alignment
/// must both equal their byte width; unsupported target layouts fail validation.
/// Arrays validate their element type even when empty. [`PhantomData`] does not
/// store or validate its marker type's layout.
///
/// Generator and consumer must agree on type definitions, representation
/// attributes, and type parameters. A marker describing a representation bound,
/// for example, is part of the stored type even when it occupies no bytes.
///
/// # Safety
///
/// After successful evaluation of [`ASSERT_LAYOUT`], an implementation guarantees
/// all of the following:
///
/// - The layout is a supported fixed-width primitive, array, [`PhantomData`],
///   or a record with `repr(C)` or `repr(transparent)`.
/// - There are no padding or uninitialized bytes, and every bit pattern is valid
///   to read and share: no references, pointers, `bool`s, restricted
///   discriminants, or interior mutability.
/// - Alignment is at most [`MAX_ALIGN`].
/// - Size, alignment, and field offsets are stable across supported targets for
///   the same type definition and representation attributes.
/// - The stored representation is the little-endian in-memory layout.
///
/// Every admitted bit pattern must be memory-safe for all safe operations on the
/// resulting type, including operations implemented internally with unsafe code.
/// This memory-safety obligation also applies to arbitrary bytes supplied to
/// [`AlignedBytes`]; mathematical invariants are not a substitute for it.
///
/// Unsafe consumers must force compile-time evaluation before relying on these
/// guarantees. Use `const { T::ASSERT_LAYOUT; size_of::<T>() }` as the conversion's
/// element size, so using the size requires evaluating the assertion. A `T: Pod`
/// bound or an unused unit-valued assertion alone is insufficient. Handwritten
/// implementations must validate nested types too; an empty assertion is
/// appropriate only when every obligation has already been established
/// independently.
/// Implementations must retain the default `__LAYOUT` metadata, which measures
/// `Self` for generated record checks. Violating these requirements can cause
/// undefined behavior.
///
/// [`ASSERT_LAYOUT`]: Self::ASSERT_LAYOUT
pub unsafe trait Pod: Copy + Sync + Sized + 'static {
    /// Layout metadata used by generated implementations.
    ///
    /// Implementations must retain the default, which measures `Self` in core.
    /// Substituting another type's metadata violates the unsafe contract.
    const __LAYOUT: PodLayout = PodLayout::of::<Self>();

    /// Establishes the layout contract for this concrete type.
    ///
    /// Must fail during const evaluation if any requirement of [`Pod`] is not
    /// met. Unsafe consumers must evaluate this assertion even for empty arrays
    /// and slices.
    const ASSERT_LAYOUT: ();
}

macro_rules! primitive_pod {
    ($($ty:ty => $bytes:literal),+ $(,)?) => {$(
        // SAFETY: Fixed-width unsigned integers have no padding and accept all
        // bit patterns. The assertions check their size, alignment, and byte order.
        unsafe impl Pod for $ty {
            const ASSERT_LAYOUT: () = {
                assert_little_endian();
                assert!(size_of::<Self>() == $bytes, "unsupported Pod primitive size");
                assert!(align_of::<Self>() == $bytes, "unsupported Pod primitive alignment");
                assert!(align_of::<Self>() <= MAX_ALIGN, "over-aligned Pod type");
            };
        }
    )+};
}

primitive_pod!(u8 => 1, u16 => 2, u32 => 4, u64 => 8);

// SAFETY: Arrays inherit element validity and have no inter-element padding.
// Validate `T` even when `LEN` is zero so empty arrays enforce its contract too.
unsafe impl<T: Pod, const LEN: usize> Pod for [T; LEN] {
    const ASSERT_LAYOUT: () = T::ASSERT_LAYOUT;
}

// SAFETY: `PhantomData` has size zero and alignment one and stores no `T`.
// Its layout is independent of `T`, so `T` need not implement `Pod`.
unsafe impl<T: ?Sized + Sync + 'static> Pod for PhantomData<T> {
    const ASSERT_LAYOUT: () = {
        assert_little_endian();
        assert!(size_of::<Self>() == 0 && align_of::<Self>() == 1);
    };
}
