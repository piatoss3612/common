//! Typed storage and nonblocking bundle acquisition for incremental MSMs.

use spin::{RwLock, RwLockReadGuard as Read, RwLockWriteGuard as Write};
use std::{vec, vec::Vec};
use zakura_udon::{
    curve::{AffinePoint, PastaCurve, ProjectivePoint},
    exec::execution::ReadView,
    field::PastaField,
    msm::{
        Requirements, ScalarStorage, Scratch,
        execution::{Buffers, MsmPlan, Request, Resources},
    },
};

pub(crate) struct Work<C: PastaCurve> {
    affine: Vec<AffinePoint<C>>,
    projective: Vec<ProjectivePoint<C>>,
    field: Vec<PastaField<C::Base>>,
    indices: Vec<usize>,
}

impl<C: PastaCurve> Work<C> {
    pub(crate) fn new(requirements: impl Iterator<Item = Requirements>) -> Self {
        let mut counts = [0; 4];
        for r in requirements {
            for (max, n) in
                counts
                    .iter_mut()
                    .zip([r.affine(), r.projective(), r.field(), r.indices()])
            {
                *max = (*max).max(n);
            }
        }
        Self {
            affine: vec![AffinePoint::GENERATOR; counts[0]],
            projective: vec![ProjectivePoint::IDENTITY; counts[1]],
            field: vec![PastaField::ZERO; counts[2]],
            indices: vec![0; counts[3]],
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        size_of::<Self>()
            + self.affine.capacity() * size_of::<AffinePoint<C>>()
            + self.projective.capacity() * size_of::<ProjectivePoint<C>>()
            + self.field.capacity() * size_of::<PastaField<C::Base>>()
            + self.indices.capacity() * size_of::<usize>()
    }
}

pub(crate) struct Arena<C: PastaCurve> {
    records: Vec<RwLock<Vec<ScalarStorage<C>>>>,
    digits: Vec<RwLock<Vec<u8>>>,
    partials: Vec<RwLock<[ProjectivePoint<C>; 1]>>,
    buckets: Vec<RwLock<Vec<ProjectivePoint<C>>>>,
}

impl<C: PastaCurve> Arena<C> {
    pub(crate) fn new(plan: MsmPlan<C>) -> Self {
        Self::for_plans(core::iter::once(plan))
    }

    pub(crate) fn for_plans(plans: impl Iterator<Item = MsmPlan<C>>) -> Self {
        let mut scalars = 0;
        let mut digits = 0;
        let mut windows = 0;
        let mut buckets = 0;
        for plan in plans {
            let r = plan.retained();
            scalars = scalars.max(r.scalars());
            if r.scalars() != 0 {
                digits = digits.max(r.digits() / r.scalars() * 256);
            }
            windows = windows.max(plan.output_slots());
            if plan.output_slots() != 0 {
                buckets = buckets.max((r.projective() - plan.output_slots()) / plan.output_slots());
            }
        }
        assert!(scalars.div_ceil(256) <= 32);
        Self {
            records: (0..scalars.div_ceil(256))
                .map(|_| RwLock::new(vec![ScalarStorage::ZERO; 256]))
                .collect(),
            digits: (0..scalars.div_ceil(256))
                .map(|_| RwLock::new(vec![0; digits]))
                .collect(),
            partials: (0..windows)
                .map(|_| RwLock::new([ProjectivePoint::IDENTITY]))
                .collect(),
            buckets: (0..windows)
                .map(|_| RwLock::new(vec![ProjectivePoint::IDENTITY; buckets]))
                .collect(),
        }
    }

    pub(crate) fn bytes(&self) -> usize {
        size_of::<Self>()
            + self.records.capacity() * size_of::<RwLock<Vec<ScalarStorage<C>>>>()
            + self
                .records
                .iter()
                .map(|s| s.read().capacity() * size_of::<ScalarStorage<C>>())
                .sum::<usize>()
            + self.digits.capacity() * size_of::<RwLock<Vec<u8>>>()
            + self
                .digits
                .iter()
                .map(|s| s.read().capacity())
                .sum::<usize>()
            + self.partials.capacity() * size_of::<RwLock<[ProjectivePoint<C>; 1]>>()
            + self.buckets.capacity() * size_of::<RwLock<Vec<ProjectivePoint<C>>>>()
            + self
                .buckets
                .iter()
                .map(|b| b.read().capacity() * size_of::<ProjectivePoint<C>>())
                .sum::<usize>()
    }

    pub(crate) fn acquire<'a>(
        &'a self,
        request: Request<'_>,
        work: &'a [RwLock<Work<C>>],
    ) -> Option<Lease<'a, C>> {
        let mut lease = Lease {
            scalars: request.scratch.scalars(),
            digit_len: request.scratch.digits(),
            records: Reads::new(request.read_scalars, 256),
            digits: Reads::new(
                request.read_digits,
                request
                    .read_digits
                    .checked_div(request.terms)
                    .map_or(1, |digits| (digits * 256).max(1)),
            ),
            write_records: None,
            write_digits: None,
            work: None,
            buckets: None,
            output: None,
            partials: Partials {
                guards: core::array::from_fn(|_| None),
                len: request.read_partials,
            },
        };
        if request.scratch.scalars() > 0 {
            lease.write_records = Some(self.records[request.scalar_start / 256].try_write()?);
        }
        if request.scratch.digits() > 0 {
            lease.write_digits = Some(self.digits[request.scalar_start / 256].try_write()?);
        }
        lease.records.acquire(&self.records)?;
        lease.digits.acquire(&self.digits)?;
        if request.scratch.affine()
            | request.scratch.projective()
            | request.scratch.field()
            | request.scratch.indices()
            != 0
        {
            lease.work = Some(work.iter().filter_map(RwLock::try_write).find(|slot| {
                slot.affine.len() >= request.scratch.affine()
                    && slot.projective.len() >= request.scratch.projective()
                    && slot.field.len() >= request.scratch.field()
                    && slot.indices.len() >= request.scratch.indices()
            })?);
        }
        if let Some(slot) = request.bucket_start.checked_div(request.buckets) {
            lease.buckets = Some(self.buckets[slot].try_write()?);
        }
        if let Some(slot) = request.output_slot {
            lease.output = Some(self.partials[slot].try_write()?);
        }
        for (target, slot) in lease
            .partials
            .guards
            .iter_mut()
            .zip(&self.partials)
            .take(request.read_partials)
        {
            *target = Some(slot.try_read()?);
        }
        Some(lease)
    }
}

struct Reads<'a, T> {
    guards: [Option<Read<'a, Vec<T>>>; 32],
    len: usize,
    fragment: usize,
}
impl<'a, T> Reads<'a, T> {
    fn new(len: usize, fragment: usize) -> Self {
        Self {
            guards: core::array::from_fn(|_| None),
            len,
            fragment,
        }
    }
    fn acquire(&mut self, slots: &'a [RwLock<Vec<T>>]) -> Option<()> {
        for (target, slot) in self
            .guards
            .iter_mut()
            .zip(slots)
            .take(self.len.div_ceil(self.fragment))
        {
            *target = Some(slot.try_read()?);
        }
        Some(())
    }
}
impl<T> ReadView<T> for Reads<'_, T> {
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, i: usize) -> Option<&T> {
        if i >= self.len {
            return None;
        }
        self.guards
            .get(i / self.fragment)?
            .as_ref()?
            .get(i % self.fragment)
    }
    fn contiguous(&self, range: core::ops::Range<usize>) -> Option<&[T]> {
        if range.is_empty() {
            return Some(&[]);
        }
        if range.end > self.len || range.start / self.fragment != (range.end - 1) / self.fragment {
            return None;
        }
        self.guards
            .get(range.start / self.fragment)?
            .as_ref()?
            .get(range.start % self.fragment..(range.end - 1) % self.fragment + 1)
    }
}
struct Partials<'a, C: PastaCurve> {
    guards: [Option<Read<'a, [ProjectivePoint<C>; 1]>>; 32],
    len: usize,
}

impl<C: PastaCurve> ReadView<ProjectivePoint<C>> for Partials<'_, C> {
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, index: usize) -> Option<&ProjectivePoint<C>> {
        if index >= self.len {
            return None;
        }
        self.guards.get(index)?.as_ref().map(|guard| &guard[0])
    }
}

pub(crate) struct Lease<'a, C: PastaCurve> {
    scalars: usize,
    digit_len: usize,
    records: Reads<'a, ScalarStorage<C>>,
    digits: Reads<'a, u8>,
    write_records: Option<Write<'a, Vec<ScalarStorage<C>>>>,
    write_digits: Option<Write<'a, Vec<u8>>>,
    work: Option<Write<'a, Work<C>>>,
    buckets: Option<Write<'a, Vec<ProjectivePoint<C>>>>,
    output: Option<Write<'a, [ProjectivePoint<C>; 1]>>,
    partials: Partials<'a, C>,
}

impl<C: PastaCurve> Lease<'_, C> {
    pub(crate) fn preparation(&self) -> (&[ScalarStorage<C>], &[u8]) {
        (
            self.write_records
                .as_ref()
                .map_or(&[], |guard| &guard[..self.scalars]),
            self.write_digits
                .as_ref()
                .map_or(&[], |guard| &guard[..self.digit_len]),
        )
    }
}

impl<C: PastaCurve> Resources<C> for Lease<'_, C> {
    fn buffers(&mut self) -> Buffers<'_, C> {
        let (affine, projective, field, indices) = match self.work.as_mut() {
            Some(guard) => {
                let Work {
                    affine,
                    projective,
                    field,
                    indices,
                } = &mut **guard;
                (
                    affine.as_mut_slice(),
                    projective.as_mut_slice(),
                    field.as_mut_slice(),
                    indices.as_mut_slice(),
                )
            }
            None => (&mut [][..], &mut [][..], &mut [][..], &mut [][..]),
        };
        Buffers {
            records: &self.records,
            digits: &self.digits,
            scratch: Scratch::new(
                self.write_records
                    .as_mut()
                    .map_or(&mut [], |guard| &mut guard[..self.scalars]),
                self.write_digits
                    .as_mut()
                    .map_or(&mut [], |guard| &mut guard[..self.digit_len]),
                affine,
                projective,
                field,
                indices,
            ),
            buckets: self
                .buckets
                .as_mut()
                .map_or(&mut [], |guard| guard.as_mut_slice()),
            output: self
                .output
                .as_mut()
                .map_or(&mut [], |guard| guard.as_mut_slice()),
            partials: &self.partials,
        }
    }
}
