//! Stage schedules partition paired slices without aliasing.
//!
//! A guard owns all loose storage until terminal stores complete; executor joins
//! never receive a field slice or callback that can observe a partially reduced
//! region.

use super::executor::for_chunks;
use super::{
    Codelet, Executor, InputOrder, InverseScale, PastaField, Plan, PrimeModulus,
    TwiddleDescription, TwiddleStorage, TwiddleTable, reverse,
};
use crate::field::fft::{
    Guard, butterfly, butterfly_dif, divide_by_power_of_two, normalize, scale,
};

#[derive(Clone, Copy)]
pub(super) struct StageKernel<'a, 'b, M: PrimeModulus> {
    pub plan: Plan<'a, M>,
    pub inverse: bool,
    pub dif: bool,
    pub scale: InverseScale,
    pub codelet: Codelet,
    pub twiddles: Option<TwiddleTable<'a, M>>,
    pub output_order: InputOrder,
    pub factor: Option<&'b [PastaField<M>]>,
}

// Four useful mathematical schedules, rather than a Cartesian product of all
// request options: forward decimation in time (DIT), forward decimation in
// frequency (DIF), normalized inverse DIT, and raw inverse DIT. Initialization,
// tables, and terminal products retain separate policies.
struct Schedule<'k, 'a, 'b, M: PrimeModulus, const MODE: u8>(&'k StageKernel<'a, 'b, M>);

impl<'a, 'b, M: PrimeModulus, const MODE: u8> core::ops::Deref for Schedule<'_, 'a, 'b, M, MODE> {
    type Target = StageKernel<'a, 'b, M>;
    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<M: PrimeModulus> StageKernel<'_, '_, M> {
    pub fn run<E: Executor>(
        &self,
        values: &mut [PastaField<M>],
        first: usize,
        tasks: usize,
        executor: &E,
    ) {
        debug_assert!(!self.inverse || !self.dif);
        if self.inverse {
            match self.scale {
                InverseScale::Normalized => {
                    Schedule::<_, 2>(self).run(values, first, tasks, executor)
                }
                InverseScale::Unscaled => {
                    Schedule::<_, 3>(self).run(values, first, tasks, executor)
                }
            }
        } else if self.dif {
            Schedule::<_, 1>(self).run(values, first, tasks, executor);
        } else {
            Schedule::<_, 0>(self).run(values, first, tasks, executor);
        }
    }

    pub fn stockham<E: Executor>(
        &self,
        values: &mut [PastaField<M>],
        scratch: &mut [PastaField<M>],
        tasks: usize,
        executor: &E,
    ) {
        if !self.inverse {
            Schedule::<_, 0>(self).stockham(values, scratch, tasks, executor);
        } else if self.scale == InverseScale::Normalized {
            Schedule::<_, 2>(self).stockham(values, scratch, tasks, executor);
        } else {
            Schedule::<_, 3>(self).stockham(values, scratch, tasks, executor);
        }
    }

    pub fn untwist<E: Executor>(&self, values: &mut [PastaField<M>], tasks: usize, executor: &E) {
        debug_assert!(self.inverse);
        if self.scale == InverseScale::Normalized {
            Schedule::<_, 2>(self).untwist(values, tasks, executor);
        } else {
            Schedule::<_, 3>(self).untwist(values, tasks, executor);
        }
    }
}

// Specialize the provider family at each stage, rather than dispatching among
// dense/factored/recurrence strategies inside every butterfly.
trait Powers<M: PrimeModulus>: Copy + Sync {
    fn at(self, index: usize) -> PastaField<M>;
    fn next(self, index: usize, previous: PastaField<M>) -> PastaField<M>;
}

#[derive(Clone, Copy)]
struct Recurrence<'a, M: PrimeModulus> {
    step: PastaField<M>,
    seed: Option<Lookup<'a, M>>,
}
impl<M: PrimeModulus> Powers<M> for Recurrence<'_, M> {
    fn at(self, index: usize) -> PastaField<M> {
        self.seed
            .map_or_else(|| self.step.pow_u64(index as u64), |seed| seed.at(index))
    }
    fn next(self, _: usize, previous: PastaField<M>) -> PastaField<M> {
        previous.mul(&self.step)
    }
}

#[derive(Clone, Copy)]
struct Lookup<'a, M: PrimeModulus> {
    table: TwiddleTable<'a, M>,
    block: usize,
    inverse: bool,
}
impl<M: PrimeModulus> Powers<M> for Lookup<'_, M> {
    fn at(self, index: usize) -> PastaField<M> {
        self.table.power(self.block, index, self.inverse)
    }
    fn next(self, index: usize, _: PastaField<M>) -> PastaField<M> {
        self.at(index)
    }
}

#[derive(Clone, Copy)]
struct DensePowers<'a, M: PrimeModulus> {
    values: &'a [PastaField<M>],
    stride: usize,
    offset: usize,
    half: usize,
    conjugate: bool,
}
impl<M: PrimeModulus> Powers<M> for DensePowers<'_, M> {
    fn at(self, index: usize) -> PastaField<M> {
        if index == 0 {
            return PastaField::ONE;
        }
        let index = if self.conjugate {
            self.half - index
        } else {
            index
        };
        let power = self.values[self.offset + index * self.stride];
        if self.conjugate { power.neg() } else { power }
    }
    fn next(self, index: usize, _: PastaField<M>) -> PastaField<M> {
        self.at(index)
    }
}

#[derive(Clone, Copy)]
struct FactoredPowers<'a, M: PrimeModulus> {
    dense: DensePowers<'a, M>,
    low_len: usize,
    low_count: usize,
}
impl<M: PrimeModulus> Powers<M> for FactoredPowers<'_, M> {
    fn at(self, index: usize) -> PastaField<M> {
        if index == 0 {
            return PastaField::ONE;
        }
        let index = if self.dense.conjugate {
            self.dense.half - index
        } else {
            index
        };
        let exponent = index * self.dense.stride;
        let power = self.dense.values[exponent % self.low_len]
            .mul(&self.dense.values[self.low_count + exponent / self.low_len]);
        if self.dense.conjugate {
            power.neg()
        } else {
            power
        }
    }
    fn next(self, index: usize, _: PastaField<M>) -> PastaField<M> {
        self.at(index)
    }
}

// Dispatch table representation once per stage. Seed reconstruction happens
// only at each task's first power; the inner recurrence has no table switch.
macro_rules! dispatch_powers {
    ($kernel:ident, $block:ident, $method:ident, $($argument:expr),+ $(,)?) => {{
        if let Some(table) = $kernel.table($block) {
            let description = table.description();
            let dense = DensePowers { values: table.as_slice(), stride: description.size / $block,
                offset: 0, half: $block / 2, conjugate: description.inverse != $kernel.inverse() };
            match description.storage {
                TwiddleStorage::Dense => $kernel.$method($($argument,)+ dense),
                TwiddleStorage::StagePacked => $kernel.$method($($argument,)+ DensePowers { stride: 1, offset: $block / 2 - 1, ..dense }),
                TwiddleStorage::Factored { low_len } => $kernel.$method($($argument,)+ FactoredPowers { dense, low_len, low_count: low_len.min(description.size / 2) }),
                TwiddleStorage::ChunkSeeds { .. } => $kernel.$method($($argument,)+ Recurrence { step: $kernel.step($block), seed: Some(Lookup { table, block: $block, inverse: $kernel.inverse() }) }),
            }
        } else {
            $kernel.$method($($argument,)+ Recurrence { step: $kernel.step($block), seed: None });
        }
    }};
}

impl<'a, M: PrimeModulus, const MODE: u8> Schedule<'_, 'a, '_, M, MODE> {
    const DIF: bool = MODE == 1;

    fn inverse(&self) -> bool {
        MODE >= 2
    }
    fn table(&self, block: usize) -> Option<TwiddleTable<'a, M>> {
        if let Some(table) = self.twiddles {
            return (table.description().size >= block).then_some(table);
        }
        let direct = if self.inverse() {
            self.plan.tables.inverse
        } else {
            self.plan.tables.forward
        };
        let opposite = if self.inverse() {
            self.plan.tables.forward
        } else {
            self.plan.tables.inverse
        };
        let (values, inverse) = direct
            .map(|values| (values, self.inverse()))
            .or_else(|| opposite.map(|values| (values, !self.inverse())))?;
        Some(
            TwiddleTable::bind(
                TwiddleDescription {
                    size: self.plan.domain.size(),
                    inverse,
                    storage: TwiddleStorage::Dense,
                },
                values,
            )
            .unwrap(),
        )
    }
    fn step(&self, block: usize) -> PastaField<M> {
        if self.inverse() {
            PastaField::root_of_unity_inverse(block.ilog2()).unwrap()
        } else {
            PastaField::root_of_unity(block.ilog2()).unwrap()
        }
    }

    pub fn run<E: Executor>(
        &self,
        values: &mut [PastaField<M>],
        first: usize,
        tasks: usize,
        executor: &E,
    ) {
        let guard = Guard::new(values);
        let size = guard.values.len();
        let radix = match self.codelet {
            Codelet::Radix2 => 2,
            Codelet::Radix4 => 4,
            Codelet::Radix8 => 8,
        }
        .min(size);
        if size == 1 || first > size {
            self.finish_region(guard.values, 0, Self::DIF);
        } else if Self::DIF {
            let mut block = size;
            while block > radix || radix <= 2 && block >= 2 {
                self.stage(guard.values, block, tasks, executor);
                block /= 2;
            }
            if radix > 2 {
                self.codelets(guard.values, radix, tasks, executor);
            }
        } else {
            let mut block = first;
            if first == 2 && radix > 2 {
                self.codelets(guard.values, radix, tasks, executor);
                block = radix * 2;
            }
            while block <= size {
                self.stage(guard.values, block, tasks, executor);
                block *= 2;
            }
        }
        guard.disarm();
        let native = if Self::DIF {
            InputOrder::BitReversed
        } else {
            InputOrder::Natural
        };
        if self.output_order != native {
            self.plan.permute(values);
        }
    }

    fn stage<E: Executor>(
        &self,
        values: &mut [PastaField<M>],
        block: usize,
        tasks: usize,
        executor: &E,
    ) {
        dispatch_powers!(self, block, stage_with, values, block, tasks, executor);
    }

    fn stage_with<E: Executor, P: Powers<M>>(
        &self,
        values: &mut [PastaField<M>],
        block: usize,
        tasks: usize,
        executor: &E,
        powers: P,
    ) {
        let chunks = values.len() / block;
        let inner_tasks = (tasks / chunks.min(tasks)).max(1);
        let terminal = if Self::DIF {
            block == 2
        } else {
            block == values.len()
        };
        for_chunks(values, block, tasks, executor, &|chunk, values| {
            let (left, right) = values.split_at_mut(block / 2);
            paired(
                left,
                right,
                0,
                inner_tasks,
                executor,
                &|start, left, right| {
                    let mut power = powers.at(start);
                    let mut low_scale = if terminal {
                        self.inverse_factor(chunk * block + start)
                    } else {
                        PastaField::ONE
                    };
                    let mut high_scale = if terminal {
                        self.inverse_factor(chunk * block + start + block / 2)
                    } else {
                        PastaField::ONE
                    };
                    for (offset, (left, right)) in left.iter_mut().zip(right).enumerate() {
                        let index = start + offset;
                        let twiddle = (index != 0).then_some(&power);
                        if Self::DIF {
                            butterfly_dif(left, right, twiddle);
                        } else {
                            butterfly(left, right, twiddle);
                        }
                        if terminal {
                            *left = self.finish(*left, chunk * block + index, low_scale, Self::DIF);
                            *right = self.finish(
                                *right,
                                chunk * block + index + block / 2,
                                high_scale,
                                Self::DIF,
                            );
                            if self.inverse() {
                                low_scale = low_scale.mul(&self.plan.domain.inverse_shift());
                                high_scale = high_scale.mul(&self.plan.domain.inverse_shift());
                            }
                        }
                        if index + 1 < block / 2 {
                            power = powers.next(index + 1, power);
                        }
                    }
                },
            );
        });
    }

    fn inverse_factor(&self, index: usize) -> PastaField<M> {
        if self.inverse() && self.plan.domain.inverse_shift() != PastaField::ONE {
            self.plan.domain.inverse_shift().pow_u64(index as u64)
        } else {
            PastaField::ONE
        }
    }

    #[inline]
    fn finish(
        &self,
        value: PastaField<M>,
        index: usize,
        untwist: PastaField<M>,
        bit_reversed: bool,
    ) -> PastaField<M> {
        if self.inverse() {
            let value = if MODE == 2 {
                divide_by_power_of_two(value, self.plan.domain.domain().log_size())
            } else {
                normalize(value)
            };
            if untwist == PastaField::ONE {
                value
            } else {
                value.mul(&untwist)
            }
        } else if let Some(factor) = self.factor {
            let desired_reversed = self.output_order == InputOrder::BitReversed;
            let index = if desired_reversed == bit_reversed {
                index
            } else {
                reverse(index, self.plan.domain.domain().log_size())
            };
            scale(value, &factor[index])
        } else {
            normalize(value)
        }
    }

    fn finish_region(&self, values: &mut [PastaField<M>], start: usize, bit_reversed: bool) {
        let mut untwist = self.inverse_factor(start);
        for (offset, value) in values.iter_mut().enumerate() {
            *value = self.finish(*value, start + offset, untwist, bit_reversed);
            if self.inverse() {
                untwist = untwist.mul(&self.plan.domain.inverse_shift());
            }
        }
    }

    pub fn untwist<E: Executor>(&self, values: &mut [PastaField<M>], tasks: usize, executor: &E) {
        let chunk = values.len().div_ceil(tasks);
        for_chunks(values, chunk, tasks, executor, &|job, values| {
            self.finish_region(values, job * chunk, false)
        });
    }

    fn codelets<E: Executor>(
        &self,
        values: &mut [PastaField<M>],
        radix: usize,
        tasks: usize,
        executor: &E,
    ) {
        if radix == 4 {
            self.codelets_with::<E, 4>(values, tasks, executor);
        } else {
            self.codelets_with::<E, 8>(values, tasks, executor);
        }
    }

    fn codelets_with<E: Executor, const RADIX: usize>(
        &self,
        values: &mut [PastaField<M>],
        tasks: usize,
        executor: &E,
    ) {
        let terminal = Self::DIF || values.len() == RADIX;
        let fourth = self.step(4);
        let eighth = if RADIX == 8 {
            self.step(8)
        } else {
            PastaField::ONE
        };
        let powers = [fourth, eighth, eighth.mul(&fourth)];
        for_chunks(values, RADIX, tasks, executor, &|chunk, values| {
            let mut local = [PastaField::ZERO; 8];
            local[..RADIX].copy_from_slice(values);
            let schedule = if RADIX == 4 { &RADIX4[..] } else { &RADIX8[..] };
            // Emit constant-index operations from the schedule the tests
            // interpret. Constant indices allow specialization; register
            // allocation depends on the target compiler.
            macro_rules! emit {
                ($($index:literal),+ $(,)?) => { $( {
                    let index = if Self::DIF { schedule.len() - 1 - $index } else { $index };
                    self.codelet_step(&mut local, schedule[index], powers);
                } )+ };
            }
            if RADIX == 4 {
                emit!(0, 1, 2, 3);
            } else {
                emit!(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11);
            }
            values.copy_from_slice(&local[..RADIX]);
            if terminal {
                self.finish_region(values, chunk * RADIX, Self::DIF);
            }
        });
    }
    #[inline(always)]
    fn codelet_step(
        &self,
        values: &mut [PastaField<M>; 8],
        operation: Step,
        powers: [PastaField<M>; 3],
    ) {
        let mut left = values[operation.left];
        let mut right = values[operation.right];
        let twiddle = match (operation.block, operation.exponent) {
            (_, 0) => None,
            (4, 1) | (8, 2) => Some(&powers[0]),
            (8, 1) => Some(&powers[1]),
            (8, 3) => Some(&powers[2]),
            _ => unreachable!(),
        };
        if Self::DIF {
            butterfly_dif(&mut left, &mut right, twiddle);
        } else {
            butterfly(&mut left, &mut right, twiddle);
        }
        values[operation.left] = left;
        values[operation.right] = right;
    }

    pub fn stockham<E: Executor>(
        &self,
        values: &mut [PastaField<M>],
        scratch: &mut [PastaField<M>],
        tasks: usize,
        executor: &E,
    ) {
        if values.len() == 1 {
            self.finish_region(values, 0, false);
            return;
        }
        let mut in_values = true;
        let mut block = 2;
        while block <= values.len() {
            if in_values {
                self.stockham_stage(values, scratch, block, tasks, executor);
            } else {
                self.stockham_stage(scratch, values, block, tasks, executor);
            }
            in_values = !in_values;
            block *= 2;
        }
        if !in_values {
            values.copy_from_slice(scratch);
        }
        if self.output_order == InputOrder::BitReversed {
            self.plan.permute(values);
        }
    }
    fn stockham_stage<E: Executor>(
        &self,
        input: &[PastaField<M>],
        output: &mut [PastaField<M>],
        block: usize,
        tasks: usize,
        executor: &E,
    ) {
        dispatch_powers!(
            self,
            block,
            stockham_with,
            input,
            output,
            block,
            tasks,
            executor
        );
    }
    #[allow(clippy::too_many_arguments)]
    fn stockham_with<E: Executor, P: Powers<M>>(
        &self,
        input: &[PastaField<M>],
        output: &mut [PastaField<M>],
        block: usize,
        tasks: usize,
        executor: &E,
        powers: P,
    ) {
        let half = block / 2;
        let inner_tasks = (tasks / (input.len() / block).min(tasks)).max(1);
        for_chunks(output, block, tasks, executor, &|chunk, output| {
            let (left, right) = output.split_at_mut(half);
            paired(
                left,
                right,
                0,
                inner_tasks,
                executor,
                &|start, left, right| {
                    let mut power = powers.at(start);
                    let terminal = block == input.len();
                    let mut low_scale = if terminal {
                        self.inverse_factor(chunk * block + start)
                    } else {
                        PastaField::ONE
                    };
                    let mut high_scale = if terminal {
                        self.inverse_factor(chunk * block + start + half)
                    } else {
                        PastaField::ONE
                    };
                    for (offset, (left, right)) in left.iter_mut().zip(right).enumerate() {
                        let index = start + offset;
                        let source = chunk * half + index;
                        let mut low = input[source];
                        let mut high = input[source + input.len() / 2];
                        butterfly(&mut low, &mut high, (index != 0).then_some(&power));
                        if block == input.len() {
                            *left = self.finish(low, chunk * block + index, low_scale, false);
                            *right =
                                self.finish(high, chunk * block + index + half, high_scale, false);
                            if self.inverse() {
                                low_scale = low_scale.mul(&self.plan.domain.inverse_shift());
                                high_scale = high_scale.mul(&self.plan.domain.inverse_shift());
                            }
                        } else {
                            // Every ping-pong store is canonical, so a panic leaves
                            // both caller-visible buffers reduced without a guard.
                            *left = normalize(low);
                            *right = normalize(high);
                        }
                        if index + 1 < half {
                            power = powers.next(index + 1, power);
                        }
                    }
                },
            );
        });
    }
}

fn paired<
    M: PrimeModulus,
    E: Executor,
    F: Fn(usize, &mut [PastaField<M>], &mut [PastaField<M>]) + Sync,
>(
    left: &mut [PastaField<M>],
    right: &mut [PastaField<M>],
    offset: usize,
    tasks: usize,
    executor: &E,
    work: &F,
) {
    if tasks <= 1 || left.len() <= 32 {
        work(offset, left, right);
    } else {
        let mid = left.len() / 2;
        let (ll, lr) = left.split_at_mut(mid);
        let (rl, rr) = right.split_at_mut(mid);
        executor.join(
            || paired(ll, rl, offset, tasks / 2, executor, work),
            || paired(lr, rr, offset + mid, tasks - tasks / 2, executor, work),
        );
    }
}

#[derive(Clone, Copy)]
pub(super) struct Step {
    pub left: usize,
    pub right: usize,
    pub block: usize,
    pub exponent: usize,
}

const fn schedule<const N: usize>(radix: usize) -> [Step; N] {
    assert!(radix.is_power_of_two() && radix <= 8);
    assert!(N == radix / 2 * radix.ilog2() as usize);
    let mut result = [Step {
        left: 0,
        right: 0,
        block: 0,
        exponent: 0,
    }; N];
    let mut count = 0;
    let mut block = 2;
    while block <= radix {
        let mut start = 0;
        while start < radix {
            let mut index = 0;
            while index < block / 2 {
                result[count] = Step {
                    left: start + index,
                    right: start + index + block / 2,
                    block,
                    exponent: index,
                };
                assert!(result[count].right < radix && result[count].exponent < block / 2);
                count += 1;
                index += 1;
            }
            start += block;
        }
        block *= 2;
    }
    result
}
pub(super) const RADIX4: [Step; 4] = schedule(4);
pub(super) const RADIX8: [Step; 12] = schedule(8);
