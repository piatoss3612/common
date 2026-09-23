//! Target layout metadata and checks used by POD implementations.

use super::{MAX_ALIGN, Pod};

/// Target layout metadata for generated record checks.
///
/// This describes layout; it does not certify that a type implements `Pod`.
/// Measurements and assertions live here so a consumer cannot replace them by
/// shadowing `core` or re-exporting `Pod` alongside counterfeit helpers.
#[doc(hidden)]
pub struct Layout {
    size: usize,
    align: usize,
}

impl Layout {
    pub(super) const fn of<T>() -> Self {
        Self {
            size: size_of::<T>(),
            align: align_of::<T>(),
        }
    }

    /// Evaluates a nested field's [`Pod::ASSERT_LAYOUT`].
    ///
    /// Generated implementations call this on the containing record's `__LAYOUT`
    /// before [`Self::assert_record`]. The receiver keeps concrete field checks
    /// dependent on that record's generic arguments.
    ///
    /// # Panics
    ///
    /// Panics when `T`'s layout assertion fails.
    pub const fn assert_field<T: Pod>(&self) {
        T::ASSERT_LAYOUT
    }

    /// Checks a record's byte order, total field size, and alignment.
    ///
    /// The derive must first validate the representation and recursively
    /// evaluate every field's `ASSERT_LAYOUT`; metadata alone does not establish
    /// field validity.
    ///
    /// # Panics
    ///
    /// Panics on a big-endian target, field-size overflow, padding, or alignment
    /// above [`MAX_ALIGN`].
    pub const fn assert_record(&self, fields: &[Self]) {
        assert_little_endian();
        let mut size = 0usize;
        let mut index = 0;
        while index < fields.len() {
            // Even release const evaluation must reject overflow.
            size = match size.checked_add(fields[index].size) {
                Some(size) => size,
                None => panic!("Pod field sizes overflow"),
            };
            index += 1;
        }
        assert!(self.size == size, "Pod struct must have no padding");
        assert!(self.align <= MAX_ALIGN, "over-aligned Pod type");
    }
}

/// Checks whether the target uses the required little-endian storage convention.
pub(super) const fn assert_little_endian() {
    assert!(
        cfg!(target_endian = "little"),
        "embedded static data requires little-endian in-memory layout"
    );
}
