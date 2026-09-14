//! Fixed fragment banks shared by dependent FFT runs, with no claim allocation.
use core::ops::Range;
use spin::{RwLock, RwLockReadGuard as Read, RwLockWriteGuard as Write};
use zakura_udon::{
    exec::run::ReadView,
    fft::run::{Bank, Buffers, Request, Resources},
    field::{PastaField, PrimeModulus},
};

pub struct Banks<M: PrimeModulus> {
    banks: Vec<Vec<RwLock<Vec<PastaField<M>>>>>,
    tile: usize,
}

impl<M: PrimeModulus> Banks<M> {
    pub fn new(sizes: &[usize], tile: usize) -> Self {
        assert!(sizes.iter().all(|size| size.div_ceil(tile) <= 32));
        Self {
            banks: sizes
                .iter()
                .map(|size| {
                    (0..size.div_ceil(tile))
                        .map(|_| RwLock::new(vec![PastaField::ZERO; tile]))
                        .collect()
                })
                .collect(),
            tile,
        }
    }
    pub fn write(&self, bank: usize, values: &[PastaField<M>]) {
        for (slot, values) in self.banks[bank].iter().zip(values.chunks(self.tile)) {
            slot.write()[..values.len()].copy_from_slice(values);
        }
    }
    pub fn read(&self, bank: usize) -> Vec<PastaField<M>> {
        self.banks[bank]
            .iter()
            .flat_map(|s| s.read().clone())
            .collect()
    }
    fn source(&self, bank: usize, range: Range<usize>) -> Option<Source<'_, M>> {
        let mut result = Source {
            slots: core::array::from_fn(|_| None),
            len: range.len(),
            tile: self.tile,
            offset: range.start % self.tile,
        };
        if range.is_empty() {
            return Some(result);
        }
        for (entry, slot) in result
            .slots
            .iter_mut()
            .zip(&self.banks[bank][range.start / self.tile..range.end.div_ceil(self.tile)])
        {
            *entry = Some(slot.try_read()?);
        }
        Some(result)
    }
    pub fn acquire(
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
        let write = self.banks[bank(task.write.0)][task.write.1.start / self.tile].try_write()?;
        let pair = match &task.pair {
            Some(range) => Some(self.banks[values][range.start / self.tile].try_write()?),
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
            range: task.write.1.start % self.tile
                ..task.write.1.start % self.tile + task.write.1.len(),
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

pub struct Lease<'a, M: PrimeModulus> {
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
