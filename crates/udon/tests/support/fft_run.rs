//! Movable guards for fragmented FFT storage. Acquisition never waits.

use core::ops::Range;
use spin::{RwLock, RwLockReadGuard as Read, RwLockWriteGuard as Write};
use std::{vec, vec::Vec};
use zakura_udon::{
    exec::run::ReadView,
    fft::run::{Bank, Buffers, FftPlan, Request, Resources},
    field::{PastaField, PrimeModulus},
};

pub struct Arena<M: PrimeModulus> {
    pub values: Vec<RwLock<Vec<PastaField<M>>>>,
    snapshot: RwLock<Vec<PastaField<M>>>,
    tile: usize,
}

impl<M: PrimeModulus> Arena<M> {
    pub fn new(plan: FftPlan<'_, M>) -> Self {
        Self {
            values: (0..plan.fragments())
                .map(|_| RwLock::new(vec![PastaField::ZERO; plan.tile()]))
                .collect(),
            snapshot: RwLock::new(vec![PastaField::ZERO; plan.retained_fields()]),
            tile: plan.tile(),
        }
    }

    pub fn write(&self, input: &[PastaField<M>]) {
        for (slot, chunk) in self.values.iter().zip(input.chunks(self.tile)) {
            slot.write()[..chunk.len()].copy_from_slice(chunk);
        }
    }

    pub fn bytes(&self) -> usize {
        size_of::<Self>()
            + self.values.capacity() * size_of::<RwLock<Vec<PastaField<M>>>>()
            + (self
                .values
                .iter()
                .map(|s| s.read().capacity())
                .sum::<usize>()
                + self.snapshot.read().capacity())
                * size_of::<PastaField<M>>()
    }

    pub fn acquire<'a>(
        &'a self,
        request: &Request<'_>,
        input: &'a [PastaField<M>],
        factor: &'a [PastaField<M>],
    ) -> Option<Lease<'a, M>> {
        let (values, range) = match request.write.0 {
            Bank::Values => (
                self.values[request.write.1.start / self.tile].try_write()?,
                request.write.1.start % self.tile
                    ..request.write.1.start % self.tile + request.write.1.len(),
            ),
            Bank::Snapshot => (self.snapshot.try_write()?, request.write.1.clone()),
            Bank::Input => return None,
        };
        let pair = request
            .pair
            .as_ref()
            .map(|range| self.values[range.start / self.tile].try_write())
            .transpose_option()?;
        let source = match &request.read {
            None => Source::Slice(&[]),
            Some((Bank::Input, range)) => Source::Slice(&input[range.clone()]),
            Some((Bank::Values, range)) => {
                let start = range.start / self.tile;
                let end = range.end.div_ceil(self.tile);
                let mut guards = core::array::from_fn(|_| None);
                for (guard, values) in guards
                    .get_mut(..end - start)?
                    .iter_mut()
                    .zip(&self.values[start..end])
                {
                    *guard = Some(values.try_read()?);
                }
                Source::Fragments {
                    guards,
                    range: range.start % self.tile..range.start % self.tile + range.len(),
                    tile: self.tile,
                }
            }
            Some((Bank::Snapshot, range)) => {
                Source::Guard(self.snapshot.try_read()?, range.clone())
            }
        };
        let factor = request
            .factor
            .as_ref()
            .map_or(&[][..], |range| &factor[range.clone()]);
        Some(Lease {
            values,
            range,
            pair,
            source,
            factor,
        })
    }
}

trait TransposeOption<T> {
    fn transpose_option(self) -> Option<Option<T>>;
}
impl<T> TransposeOption<T> for Option<Option<T>> {
    fn transpose_option(self) -> Option<Option<T>> {
        match self {
            None => Some(None),
            Some(value) => value.map(Some),
        }
    }
}

#[expect(
    clippy::large_enum_variant,
    reason = "borrowed guards use fixed task envelopes"
)]
enum Source<'a, M: PrimeModulus> {
    Slice(&'a [PastaField<M>]),
    Guard(Read<'a, Vec<PastaField<M>>>, Range<usize>),
    Fragments {
        guards: [Option<Read<'a, Vec<PastaField<M>>>>; 128],
        range: Range<usize>,
        tile: usize,
    },
}
impl<M: PrimeModulus> ReadView<PastaField<M>> for Source<'_, M> {
    fn len(&self) -> usize {
        match self {
            Self::Slice(s) => s.len(),
            Self::Guard(_, range) | Self::Fragments { range, .. } => range.len(),
        }
    }
    fn get(&self, index: usize) -> Option<&PastaField<M>> {
        match self {
            Self::Slice(s) => s.get(index),
            Self::Guard(g, range) => g[range.clone()].get(index),
            Self::Fragments {
                guards,
                range,
                tile,
            } => {
                if index >= range.len() {
                    return None;
                }
                let index = range.start + index;
                guards[index / tile].as_ref()?.get(index % tile)
            }
        }
    }
}

pub struct Lease<'a, M: PrimeModulus> {
    values: Write<'a, Vec<PastaField<M>>>,
    range: Range<usize>,
    pair: Option<Write<'a, Vec<PastaField<M>>>>,
    source: Source<'a, M>,
    factor: &'a [PastaField<M>],
}

impl<M: PrimeModulus> Resources<M> for Lease<'_, M> {
    fn buffers(&mut self) -> Buffers<'_, M> {
        Buffers {
            values: &mut self.values[self.range.clone()],
            pair: self.pair.as_mut().map_or(&mut [], |p| p.as_mut_slice()),
            source: &self.source,
            factor: &self.factor,
        }
    }
}
