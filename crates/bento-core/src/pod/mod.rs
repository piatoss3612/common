//! Layout validation and byte storage for embedded records.
//!
//! # Background
//!
//! Artifact generators write records that consumers access directly from static
//! bytes. This requires agreement on byte order and layout, and excludes padding
//! and types whose bit patterns can be invalid. [`Pod`] expresses that contract.
//!
//! # Design
//!
//! Layout checks depend on concrete generic arguments and the compilation target.
//! [`Pod::ASSERT_LAYOUT`] defers them until a byte conversion instantiates the
//! type. Every conversion forces const evaluation, including empty views.
//!
//! - [`storage`] supplies aligned storage and shared byte views.
//! - [`macros`] declares typed statics backed by included files.

use core::marker::PhantomData;

mod macros;
mod storage;
pub use storage::{AlignedBytes, MAX_ALIGN, bytes_of, bytes_of_slice};

/// A type whose validated layout can be read directly from embedded bytes.
///
/// Stored records use a fixed little-endian layout so generators can write bytes
/// that consumers read without decoding or allocation. This contract supports
/// shared byte views through [`bytes_of`] and [`bytes_of_slice`], and typed
/// static views through [`AlignedBytes`].
///
/// Layout validation is conditional: a type can implement this trait even when
/// [`ASSERT_LAYOUT`] fails. Every byte conversion evaluates that assertion at
/// compile time before using the layout. Use `cargo build` or `cargo test` to
/// exercise these checks: `cargo check` can miss failures that require code
/// generation.
///
/// Implementations are provided for `u8`, `u16`, `u32`, `u64`, arrays, and
/// [`PhantomData`]. Integer size and alignment must both equal their byte width;
/// unsupported target layouts fail validation. Arrays validate their element
/// type even when empty. [`PhantomData`] does not store or validate its marker
/// type.
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
/// Generator and consumer must agree on type definitions and representation
/// attributes. This trait does not check library invariants such as canonical
/// residues or curve membership; those remain the responsibility of the types'
/// libraries and artifact generators.
///
/// Unsafe consumers must force compile-time evaluation with
/// `const { T::ASSERT_LAYOUT };` before relying on these guarantees. A
/// `T: Pod` bound alone is insufficient. Handwritten implementations must
/// validate nested types too; an empty assertion is appropriate only when
/// every obligation has already been established independently.
/// Violating these requirements can cause undefined behavior.
///
/// [`ASSERT_LAYOUT`]: Self::ASSERT_LAYOUT
pub unsafe trait Pod: Copy + Sync + Sized + 'static {
    /// Establishes the layout contract for this concrete type.
    ///
    /// Must fail during const evaluation if any requirement of [`Pod`] is not
    /// met. Unsafe consumers must evaluate this assertion even for empty arrays
    /// and slices.
    const ASSERT_LAYOUT: ();
}

/// Checks whether the target uses the required little-endian storage convention.
const fn assert_little_endian() {
    assert!(
        cfg!(target_endian = "little"),
        "embedded static data requires little-endian in-memory layout"
    );
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
