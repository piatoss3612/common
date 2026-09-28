//! Shared scalar and digit access without requiring contiguous retained storage.

use core::ops::Range;

pub(super) trait Storage<T>: Copy {
    fn len(self) -> usize;
    fn get(self, index: usize) -> T
    where
        T: Copy;
    fn slice(self, range: Range<usize>) -> Self;
    fn contiguous(&self) -> Option<&[T]>;
    fn is_empty(self) -> bool {
        self.len() == 0
    }
    fn iter(self) -> impl Iterator<Item = T>
    where
        T: Copy,
    {
        (0..self.len()).map(move |i| self.get(i))
    }
    fn chunks_exact(self, size: usize) -> impl Iterator<Item = Self> {
        (0..self.len() / size).map(move |i| self.slice(i * size..(i + 1) * size))
    }
}
impl<T> Storage<T> for &[T] {
    fn len(self) -> usize {
        <[T]>::len(self)
    }
    fn get(self, index: usize) -> T
    where
        T: Copy,
    {
        self[index]
    }
    fn slice(self, range: Range<usize>) -> Self {
        &self[range]
    }
    fn contiguous(&self) -> Option<&[T]> {
        Some(self)
    }
}
