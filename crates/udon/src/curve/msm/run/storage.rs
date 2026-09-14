//! Shared scalar and digit access without requiring contiguous retained storage.

use crate::exec::run::ReadView;
use core::ops::Range;

pub(crate) trait Storage<T>: Copy {
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

pub(super) struct Fragmented<'a, T> {
    view: &'a dyn ReadView<T>,
    start: usize,
    len: usize,
}
impl<T> Copy for Fragmented<'_, T> {}
impl<T> Clone for Fragmented<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<'a, T> Fragmented<'a, T> {
    pub fn new(view: &'a dyn ReadView<T>, len: usize) -> Self {
        assert!(len <= view.len());
        Self {
            view,
            start: 0,
            len,
        }
    }
}
impl<T> Storage<T> for Fragmented<'_, T> {
    fn len(self) -> usize {
        self.len
    }
    fn get(self, index: usize) -> T
    where
        T: Copy,
    {
        assert!(index < self.len);
        *self
            .view
            .get(self.start + index)
            .expect("initialized retained view")
    }
    fn slice(self, range: Range<usize>) -> Self {
        assert!(range.start <= range.end && range.end <= self.len);
        Self {
            view: self.view,
            start: self.start + range.start,
            len: range.len(),
        }
    }
    fn contiguous(&self) -> Option<&[T]> {
        self.view.contiguous(self.start..self.start + self.len)
    }
}
