//! Aligned storage and the casts between records and their byte representation.
//!
//! Typed views borrow static storage; byte views borrow the original value or
//! slice. [`AlignedBytes::as_value`] and [`bytes_of_slice`] check layout during
//! compilation; the array and single-value helpers delegate to them. Stored
//! values are trusted and are never inspected or transformed.

use super::Pod;

/// A byte buffer aligned for every validated [`Pod`] type.
///
/// Use this buffer when bytes need to be viewed as typed static data. Its first
/// byte is aligned to [`MAX_ALIGN`]; the length counts payload bytes and excludes
/// any padding added to the buffer itself. File embedding is available through
/// [`embed_array!`](crate::embed_array) and [`embed_struct!`](crate::embed_struct).
#[repr(C, align(64))]
pub struct AlignedBytes<const N: usize>(pub [u8; N]);

/// The maximum supported [`Pod`] alignment, in bytes.
///
/// [`AlignedBytes`] guarantees this alignment for every buffer length.
pub const MAX_ALIGN: usize = align_of::<AlignedBytes<0>>();

impl<const N: usize> AlignedBytes<N> {
    /// Views the stored bytes as an array without copying them.
    ///
    /// The element type must pass [`Pod::ASSERT_LAYOUT`] during compilation,
    /// including when `LEN` is zero. A byte length other than
    /// `size_of::<[T; LEN]>()` is a compilation error, including in runtime calls.
    #[track_caller]
    pub const fn as_array<T: Pod, const LEN: usize>(&'static self) -> &'static [T; LEN] {
        self.as_value()
    }

    /// Views the stored bytes as a value without copying them.
    ///
    /// The type must pass [`Pod::ASSERT_LAYOUT`] during compilation. A byte
    /// length other than `size_of::<T>()` is a compilation error, including in
    /// runtime calls. The returned reference points directly into this buffer.
    #[track_caller]
    pub const fn as_value<T: Pod>(&'static self) -> &'static T {
        // Consume the const result in the pointer expression: unused unit-valued
        // assertions can be discarded during optimization, even for empty types.
        let offset = const {
            super::assert_little_endian();
            let () = T::ASSERT_LAYOUT;
            assert!(align_of::<T>() <= MAX_ALIGN, "over-aligned Pod type");
            assert!(
                size_of::<T>() == N,
                "embedded byte length must equal the requested type's size"
            );
            0
        };

        // SAFETY: `repr(C)` puts the byte array at offset zero, so its pointer is
        // aligned to `MAX_ALIGN`. The assertions establish sufficient alignment
        // and exactly `size_of::<T>()` initialized bytes. The validated `Pod`
        // contract permits every bit pattern and shared access. The storage is
        // borrowed for `'static` and cannot be mutated while the view exists.
        unsafe { &*self.0.as_ptr().add(offset).cast::<T>() }
    }
}

/// Returns the stored representation of a value without copying it.
///
/// The view borrows the value and contains its little-endian bytes with no
/// header or length prefix. Generators can write these bytes to a file for
/// [`embed_struct!`](crate::embed_struct). The type must pass
/// [`Pod::ASSERT_LAYOUT`] during compilation.
pub fn bytes_of<T: Pod>(value: &T) -> &[u8] {
    bytes_of_slice(core::slice::from_ref(value))
}

/// Returns the concatenated stored representations of a slice's elements.
///
/// The view borrows the slice and contains no header or length prefix.
/// Generators can write it to a file for [`embed_array!`](crate::embed_array).
/// Empty slices and slices of zero-sized types produce empty byte views.
/// The element type must pass [`Pod::ASSERT_LAYOUT`] during compilation,
/// including for empty slices.
pub fn bytes_of_slice<T: Pod>(values: &[T]) -> &[u8] {
    // As in as_value, consuming the const result keeps validation tied to the
    // conversion even when a byte-view wrapper is inlined.
    let element_size = const {
        super::assert_little_endian();
        let () = T::ASSERT_LAYOUT;
        size_of::<T>()
    };

    // SAFETY: The validated `Pod` contract excludes padding and uninitialized
    // bytes. Slice elements are contiguous, and their count times the validated
    // element size is their total byte length, which cannot overflow for a valid
    // slice. The pointer remains non-null and aligned for `u8` even for an
    // empty slice or zero-sized `T`. The view borrows the original allocation,
    // whose bytes cannot be mutated through a shared reference.
    unsafe {
        core::slice::from_raw_parts(values.as_ptr().cast::<u8>(), values.len() * element_size)
    }
}

#[cfg(test)]
mod tests;
