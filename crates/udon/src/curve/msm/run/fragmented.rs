//! Retained task-fragment views exposed through the shared [`Storage`] access trait.

use super::super::storage::Storage;
use crate::exec::run::ReadView;
use core::ops::Range;

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
    pub(super) fn new(view: &'a dyn ReadView<T>, len: usize) -> Self {
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
