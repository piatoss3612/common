//! Fixed fragment banks shared by dependent FFT runs, with no claim allocation.
use core::ops::Range;
use spin::{RwLock, RwLockReadGuard as Read, RwLockWriteGuard as Write};
use std::{vec, vec::Vec};
use zakura_udon::{
    exec::run::ReadView,
    fft::run::{Bank, Buffers, Request, Resources},
    field::{PastaField, PrimeModulus},
};

pub(crate) struct Banks<M: PrimeModulus> {
    banks: Vec<Vec<RwLock<Vec<PastaField<M>>>>>,
    tiles: Vec<usize>,
}

impl<M: PrimeModulus> Banks<M> {
    pub(crate) fn new(sizes: &[usize], tile: usize, contiguous: Range<usize>) -> Self {
        assert!(sizes.iter().all(|size| size.div_ceil(tile) <= 32));
        let tiles: Vec<_> = sizes
            .iter()
            .enumerate()
            .map(|(i, &size)| {
                if contiguous.contains(&i) {
                    size.max(1)
                } else {
                    tile
                }
            })
            .collect();
        Self {
            banks: sizes
                .iter()
                .zip(&tiles)
                .map(|(size, &tile)| {
                    (0..size.div_ceil(tile))
                        .map(|_| RwLock::new(vec![PastaField::ZERO; tile]))
                        .collect()
                })
                .collect(),
            tiles,
        }
    }
    pub(crate) fn write(&self, bank: usize, values: &[PastaField<M>]) {
        for (slot, values) in self.banks[bank].iter().zip(values.chunks(self.tiles[bank])) {
            slot.write()[..values.len()].copy_from_slice(values);
        }
    }
    pub(crate) fn read(&self, bank: usize) -> Vec<PastaField<M>> {
        self.banks[bank]
            .iter()
            .flat_map(|s| s.read().clone())
            .collect()
    }
    fn source(&self, bank: usize, range: Range<usize>) -> Option<Source<'_, M>> {
        let mut result = Source {
            slots: core::array::from_fn(|_| None),
            len: range.len(),
            tile: self.tiles[bank],
            offset: range.start % self.tiles[bank],
        };
        if range.is_empty() {
            return Some(result);
        }
        for (entry, slot) in result.slots.iter_mut().zip(
            &self.banks[bank][range.start / self.tiles[bank]..range.end.div_ceil(self.tiles[bank])],
        ) {
            *entry = Some(slot.try_read()?);
        }
        Some(result)
    }
    pub(crate) fn acquire(
        &self,
        task: &Request<'_>,
        input: usize,
        values: usize,
        snapshot: usize,
        factor: usize,
    ) -> Option<Lease<'_, M>> {
        let bank = |bank| match bank {
            Bank::Input => input,
            Bank::Values => values,
            Bank::Snapshot => snapshot,
        };
        let write_bank = bank(task.write.0);
        let write_tile = self.tiles[write_bank];
        let write = self.banks[write_bank][task.write.1.start / write_tile].try_write()?;
        let pair = match &task.pair {
            Some(range) => Some(self.banks[values][range.start / self.tiles[values]].try_write()?),
            None => None,
        };
        let source = match &task.read {
            Some((b, range)) => self.source(bank(*b), range.clone())?,
            None => self.source(0, 0..0)?,
        };
        let factor = self.source(factor, task.factor.clone().unwrap_or(0..0))?;
        Some(Lease {
            write,
            pair,
            source,
            factor,
            range: task.write.1.start % write_tile
                ..task.write.1.start % write_tile + task.write.1.len(),
        })
    }
}

struct Source<'a, M: PrimeModulus> {
    slots: [Option<Read<'a, Vec<PastaField<M>>>>; 32],
    len: usize,
    tile: usize,
    offset: usize,
}
impl<M: PrimeModulus> ReadView<PastaField<M>> for Source<'_, M> {
    fn len(&self) -> usize {
        self.len
    }
    fn get(&self, index: usize) -> Option<&PastaField<M>> {
        if index >= self.len {
            return None;
        }
        self.slots[(index + self.offset) / self.tile]
            .as_ref()?
            .get((index + self.offset) % self.tile)
    }
}

pub(crate) struct Lease<'a, M: PrimeModulus> {
    write: Write<'a, Vec<PastaField<M>>>,
    pair: Option<Write<'a, Vec<PastaField<M>>>>,
    source: Source<'a, M>,
    factor: Source<'a, M>,
    range: Range<usize>,
}
impl<M: PrimeModulus> Resources<M> for Lease<'_, M> {
    fn buffers(&mut self) -> Buffers<'_, M> {
        Buffers {
            values: &mut self.write[self.range.clone()],
            pair: self
                .pair
                .as_mut()
                .map_or(&mut [], |s| &mut s[self.range.clone()]),
            source: &self.source,
            factor: &self.factor,
        }
    }
}
