//! Owned scratch buffers and join-width measurement shared by MSM tests.

use super::{
    AffinePoint, PastaCurve, PastaField, ProjectivePoint, Requirements, ScalarStorage, Scratch,
};
use crate::exec::{Executor, SerialExecutor};
use std::{vec, vec::Vec};

pub(super) struct Buffers<C: PastaCurve> {
    pub(super) scalars: Vec<ScalarStorage<C>>,
    pub(super) digits: Vec<u8>,
    pub(super) affine: Vec<AffinePoint<C>>,
    pub(super) projective: Vec<ProjectivePoint<C>>,
    pub(super) field: Vec<PastaField<C::Base>>,
    pub(super) indices: Vec<usize>,
}

impl<C: PastaCurve> Buffers<C> {
    pub(super) fn new(r: Requirements) -> Self {
        Self {
            scalars: vec![ScalarStorage::ZERO; r.scalars + 1],
            digits: vec![73; r.digits + 1],
            affine: vec![AffinePoint::GENERATOR; r.affine + 1],
            projective: vec![ProjectivePoint::GENERATOR; r.projective + 1],
            field: vec![PastaField::ONE; r.field + 1],
            indices: vec![73; r.indices + 1],
        }
    }
    pub(super) fn borrow(&mut self) -> Scratch<'_, C> {
        Scratch {
            scalars: &mut self.scalars,
            digits: &mut self.digits,
            affine: &mut self.affine,
            projective: &mut self.projective,
            field: &mut self.field,
            indices: &mut self.indices,
        }
    }
    pub(super) fn tails(&self, r: Requirements) {
        assert!(self.scalars[r.scalars] == ScalarStorage::ZERO);
        assert_eq!(self.digits[r.digits], 73);
        assert_eq!(self.affine[r.affine], AffinePoint::GENERATOR);
        assert_eq!(self.projective[r.projective], ProjectivePoint::GENERATOR);
        assert_eq!(
            (self.field[r.field]).reduce(),
            (PastaField::<_>::ONE).reduce()
        );
        assert_eq!(self.indices[r.indices], 73);
    }
}

pub(super) struct Pool;
impl Executor for Pool {
    fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
    where
        L: FnOnce() -> A + Send,
        R: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        rayon::join(left, right)
    }
}

/// Measures the widest set of independent leaves exposed by a join tree.
///
/// Jobs run sequentially: consecutive joins take a maximum; joined branches add.
/// This checks the allowance without relying on OS scheduling or worker counts.
pub(super) struct JoinWidth(pub(super) core::sync::atomic::AtomicUsize);
impl JoinWidth {
    pub(super) fn measure<R>(&self, work: impl FnOnce() -> R) -> (R, usize) {
        use core::sync::atomic::{AtomicUsize, Ordering};
        struct Restore<'a>(&'a AtomicUsize, usize);
        impl Drop for Restore<'_> {
            fn drop(&mut self) {
                self.0.store(self.1, Ordering::Relaxed);
            }
        }
        let _restore = Restore(&self.0, self.0.swap(1, Ordering::Relaxed));
        let result = work();
        (result, self.0.load(Ordering::Relaxed))
    }
}
impl Executor for JoinWidth {
    fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
    where
        L: FnOnce() -> A + Send,
        R: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        let ((a, left), (b, right)) =
            SerialExecutor.join(|| self.measure(left), || self.measure(right));
        self.0
            .fetch_max(left + right, core::sync::atomic::Ordering::Relaxed);
        (a, b)
    }
}
