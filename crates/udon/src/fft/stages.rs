//! Stage schedules partition paired slices without aliasing.
//!
//! A guard owns all loose storage until terminal stores complete; executor joins
//! never receive a field slice or callback that can observe a partially reduced
//! region.

use super::finish::{Factors, InverseFinish};
use super::{
    Codelet, ElementOrder, Executor, InverseScale, PastaField, Plan, PrimeModulus,
    TwiddleDescription, TwiddleStorage, TwiddleTable, reverse,
};
use crate::exec::{TaskBudget, for_each_chunk_mut};
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
    pub output_order: ElementOrder,
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
    // Explicit tables also cover bounded, column-major panels. Dispatch once
    // per stage; the default recurrence path shares transform::Kernel::cross.
    pub fn columns(
        &self,
        values: &mut [PastaField<M>],
        first: usize,
        tile: usize,
        first_block: usize,
    ) {
        let rows = self.plan.domain.size() / tile;
        let guard = Guard::new(values);
        let mut block = (tile * 2).max(first_block);
        while block <= self.plan.domain.size() {
            let table = twiddle_table(self.plan, self.twiddles, self.inverse, block).map(|table| {
                let description = table.description();
                let packed = description.storage == TwiddleStorage::StagePacked;
                DensePowers {
                    values: table.as_slice(),
                    stride: if packed { 1 } else { description.size / block },
                    offset: if packed { block / 2 - 1 } else { 0 },
                    half: block / 2,
                    conjugate: description.inverse != self.inverse,
                }
            });
            let root = if self.inverse {
                PastaField::root_of_unity_inverse(block.ilog2())
            } else {
                PastaField::root_of_unity(block.ilog2())
            }
            .unwrap();
            let row_step = root.pow_u64(tile as u64);
            let distance = block / tile / 2;
            for (column, lane) in guard.values.chunks_exact_mut(rows).enumerate() {
                let seed = table.map_or_else(
                    || root.pow_u64((first + column) as u64),
                    |powers| powers.at(first + column),
                );
                for group in lane.chunks_exact_mut(distance * 2) {
                    let (left, right) = group.split_at_mut(distance);
                    let mut power = seed;
                    for (row, (left, right)) in left.iter_mut().zip(right).enumerate() {
                        let exponent = row * tile + first + column;
                        butterfly(left, right, (exponent != 0).then_some(&power));
                        if row + 1 < distance {
                            power = table.map_or_else(
                                || power.mul(&row_step),
                                |powers| powers.at(exponent + tile),
                            );
                        }
                    }
                }
            }
            if block == self.plan.domain.size() {
                break;
            }
            block *= 2;
        }
        if self.inverse && self.scale == InverseScale::Normalized {
            let factors = Factors::untwist(self.plan.domain);
            let row_step = self.plan.domain.inverse_shift().pow_u64(tile as u64);
            for (column, lane) in guard.values.chunks_exact_mut(rows).enumerate() {
                let mut power = factors.at(first + column);
                for value in lane {
                    *value = divide_by_power_of_two(*value, self.plan.domain.domain().log_size());
                    if power != PastaField::ONE {
                        *value = value.mul(&power);
                    }
                    power = power.mul(&row_step);
                }
            }
        }
    }

    // A detached tile pair has no executor or child resource requests. The
    // same provider selection serves complete stages and incremental runs.
    pub fn pair(
        &self,
        left: &mut [PastaField<M>],
        right: &mut [PastaField<M>],
        start: usize,
        block: usize,
    ) {
        if self.inverse {
            Schedule::<_, 3>(self).pair(left, right, start, block);
        } else if self.dif {
            Schedule::<_, 1>(self).pair(left, right, start, block);
        } else {
            Schedule::<_, 0>(self).pair(left, right, start, block);
        }
    }

    // Continues an initialized transform, including the fused interpolation
    // paths. Kernel-local calls use `run` with a serial executor.
    pub fn drive<E: Executor>(
        &self,
        values: &mut [PastaField<M>],
        first: usize,
        options: super::ExecutionOptions,
        executor: &E,
    ) {
        let nz = |n| core::num::NonZeroUsize::new(n).unwrap();
        let mut request = super::TransformRequest::new(if self.inverse {
            super::Direction::Inverse
        } else {
            super::Direction::Forward
        });
        request.inverse_scale = self.scale;
        request.output_order = self.output_order;
        let mut plan =
            super::run::FftPlan::new(self.plan, request, nz(options.tile_len), self.codelet)
                .expect("validated stage request")
                .with_contiguous_permutation()
                .resume(
                    first,
                    if self.dif {
                        ElementOrder::Natural
                    } else {
                        ElementOrder::BitReversed
                    },
                );
        if let Some(table) = self.twiddles {
            plan = plan.with_twiddles(table);
        }
        plan.execute(
            None,
            values,
            self.factor,
            &mut [],
            nz(options.max_tasks),
            executor,
        )
        .expect("validated stage storage");
    }

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
}

pub(super) fn twiddle_table<'a, M: PrimeModulus>(
    plan: Plan<'a, M>,
    explicit: Option<TwiddleTable<'a, M>>,
    inverse: bool,
    block: usize,
) -> Option<TwiddleTable<'a, M>> {
    if let Some(table) = explicit {
        return (table.description().size >= block).then_some(table);
    }
    let (direct, opposite) = if inverse {
        (plan.tables.inverse, plan.tables.forward)
    } else {
        (plan.tables.forward, plan.tables.inverse)
    };
    let (values, direction) = direct
        .map(|values| (values, inverse))
        .or_else(|| opposite.map(|values| (values, !inverse)))?;
    Some(
        TwiddleTable::bind_trusted(
            TwiddleDescription {
                size: plan.domain.size(),
                inverse: direction,
                storage: TwiddleStorage::Dense,
            },
            values,
        )
        .unwrap(),
    )
}

// Specialize the provider family at each stage, rather than dispatching among
// table lookups and recurrence inside every butterfly.
trait Powers<M: PrimeModulus>: Copy + Sync {
    fn at(self, index: usize) -> PastaField<M>;
    fn next(self, index: usize, previous: PastaField<M>) -> PastaField<M>;
}

#[derive(Clone, Copy)]
struct Recurrence<M: PrimeModulus> {
    step: PastaField<M>,
}
impl<M: PrimeModulus> Powers<M> for Recurrence<M> {
    fn at(self, index: usize) -> PastaField<M> {
        self.step.pow_u64(index as u64)
    }
    fn next(self, _: usize, previous: PastaField<M>) -> PastaField<M> {
        previous.mul(&self.step)
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
        // For a stage root w of order 2 * half, w^half = -1. Thus
        // w^(-i) = -w^(half-i) for 0 < i < half, so either table direction
        // supplies the other. The identity at index zero was handled above.
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

// Dispatch table representation once per stage. Computed powers start each
// task independently and advance by recurrence within its region.
macro_rules! dispatch_powers {
    ($kernel:ident, $block:ident, $method:ident, $($argument:expr),+ $(,)?) => {{
        if let Some(table) = $kernel.table($block) {
            let description = table.description();
            let dense = DensePowers { values: table.as_slice(), stride: description.size / $block,
                offset: 0, half: $block / 2, conjugate: description.inverse != $kernel.inverse() };
            match description.storage {
                TwiddleStorage::Dense => $kernel.$method($($argument,)+ dense),
                TwiddleStorage::StagePacked => $kernel.$method($($argument,)+ DensePowers { stride: 1, offset: $block / 2 - 1, ..dense }),
            }
        } else {
            $kernel.$method($($argument,)+ Recurrence { step: $kernel.step($block) });
        }
    }};
}

impl<'a, M: PrimeModulus, const MODE: u8> Schedule<'_, 'a, '_, M, MODE> {
    const DIF: bool = MODE == 1;

    fn inverse(&self) -> bool {
        MODE >= 2
    }
    fn table(&self, block: usize) -> Option<TwiddleTable<'a, M>> {
        // Immutable table handles were validated at binding, not per task.
        twiddle_table(self.plan, self.twiddles, self.inverse(), block)
    }
    fn step(&self, block: usize) -> PastaField<M> {
        if self.inverse() {
            PastaField::root_of_unity_inverse(block.ilog2()).unwrap()
        } else {
            PastaField::root_of_unity(block.ilog2()).unwrap()
        }
    }

    fn pair(
        &self,
        left: &mut [PastaField<M>],
        right: &mut [PastaField<M>],
        start: usize,
        block: usize,
    ) {
        let left = Guard::new(left);
        let right = Guard::new(right);
        dispatch_powers!(self, block, pair_with, left.values, right.values, start);
        // Both guards canonicalize even when an internal assertion unwinds.
    }

    fn pair_with<P: Powers<M>>(
        &self,
        left: &mut [PastaField<M>],
        right: &mut [PastaField<M>],
        start: usize,
        powers: P,
    ) {
        let mut power = powers.at(start);
        let len = left.len();
        for (offset, (left, right)) in left.iter_mut().zip(right).enumerate() {
            let index = start + offset;
            let twiddle = (index != 0).then_some(&power);
            if Self::DIF {
                butterfly_dif(left, right, twiddle);
            } else {
                butterfly(left, right, twiddle);
            }
            if offset + 1 < len {
                power = powers.next(index + 1, power);
            }
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
            ElementOrder::BitReversed
        } else {
            ElementOrder::Natural
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
        if MODE == 2
            && block == values.len()
            && matches!(
                InverseFinish::select(self.plan.domain, self.plan.tables.inverse_finish.is_some()),
                InverseFinish::ScaledInputs
            )
        {
            if self.plan.tables.inverse_finish.is_some() {
                self.finish_stage(
                    values,
                    tasks,
                    executor,
                    Recurrence {
                        step: PastaField::ONE,
                    },
                );
            } else {
                dispatch_powers!(self, block, finish_stage, values, tasks, executor);
            }
        } else {
            dispatch_powers!(self, block, stage_with, values, block, tasks, executor);
        }
    }

    fn stage_with<E: Executor, P: Powers<M>>(
        &self,
        values: &mut [PastaField<M>],
        block: usize,
        tasks: usize,
        executor: &E,
        powers: P,
    ) {
        let terminal = if Self::DIF {
            block == 2
        } else {
            block == values.len()
        };
        for_each_chunk_mut(
            values,
            block,
            TaskBudget::new(tasks).unwrap(),
            executor,
            |chunk, values, inner| {
                let (left, right) = values.split_at_mut(block / 2);
                paired(left, right, 0, inner, executor, &|start, left, right| {
                    let mut power = powers.at(start);
                    let factors = if terminal && self.inverse() {
                        Factors::untwist(self.plan.domain)
                    } else {
                        Factors::Identity
                    };
                    let mut low_scale = factors.at(chunk * block + start);
                    let mut high_scale = factors.at(chunk * block + start + block / 2);
                    let len = left.len();
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
                            if self.inverse() && offset + 1 < len {
                                low_scale = factors.next(chunk * block + index + 1, low_scale);
                                high_scale =
                                    factors.next(chunk * block + index + 1 + block / 2, high_scale);
                            }
                        }
                        if offset + 1 < len {
                            power = powers.next(index + 1, power);
                        }
                    }
                });
            },
        );
    }

    fn finish_stage<E: Executor, P: Powers<M>>(
        &self,
        values: &mut [PastaField<M>],
        tasks: usize,
        executor: &E,
        powers: P,
    ) {
        let half = values.len() / 2;
        let factors = Factors::normalized(self.plan);
        let upper = match factors {
            Factors::Table { upper, .. } => upper,
            _ => Factors::untwist(self.plan.domain).at(half),
        };
        let combined = self.plan.tables.inverse_finish;
        let (left, right) = values.split_at_mut(half);
        paired(
            left,
            right,
            0,
            TaskBudget::new(tasks).unwrap(),
            executor,
            &|start, left, right| {
                let mut low_scale = factors.at(start);
                let mut twiddle = if combined.is_none() {
                    powers.at(start)
                } else {
                    PastaField::ONE
                };
                let len = left.len();
                for (offset, (left, right)) in left.iter_mut().zip(right).enumerate() {
                    let index = start + offset;
                    let high_scale = combined.map_or_else(
                        || {
                            if index == 0 {
                                low_scale
                            } else {
                                twiddle.mul(&low_scale)
                            }
                        },
                        |table| table[index],
                    );
                    // The low factor includes the inverse size and shift; the
                    // high factor also includes the inverse twiddle. Scaling
                    // inputs here replaces normalization of the outputs.
                    let low = scale(*left, &low_scale);
                    let high = scale(*right, &high_scale);
                    *left = low.add(&high);
                    *right = low.sub(&high);
                    if upper != PastaField::ONE {
                        *right = right.mul(&upper);
                    }
                    if offset + 1 < len {
                        low_scale = factors.next(index + 1, low_scale);
                        if combined.is_none() {
                            twiddle = powers.next(index + 1, twiddle);
                        }
                    }
                }
            },
        );
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
            let desired_reversed = self.output_order == ElementOrder::BitReversed;
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
        let scaled = MODE == 2
            && self.plan.tables.inverse_scales.is_some()
            && matches!(
                InverseFinish::select(self.plan.domain, false),
                InverseFinish::ScaledInputs
            );
        let factors = if scaled {
            Factors::normalized(self.plan)
        } else if self.inverse() {
            Factors::untwist(self.plan.domain)
        } else {
            Factors::Identity
        };
        let mut factor = factors.at(start);
        let len = values.len();
        for (offset, value) in values.iter_mut().enumerate() {
            *value = if scaled {
                scale(*value, &factor)
            } else {
                self.finish(*value, start + offset, factor, bit_reversed)
            };
            if self.inverse() && offset + 1 < len {
                factor = factors.next(start + offset + 1, factor);
            }
        }
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
        for_each_chunk_mut(
            values,
            RADIX,
            TaskBudget::new(tasks).unwrap(),
            executor,
            |chunk, values, _| {
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
            },
        );
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
}

fn paired<
    M: PrimeModulus,
    E: Executor,
    F: Fn(usize, &mut [PastaField<M>], &mut [PastaField<M>]) + Sync,
>(
    left: &mut [PastaField<M>],
    right: &mut [PastaField<M>],
    offset: usize,
    budget: TaskBudget,
    executor: &E,
    work: &F,
) {
    if budget == TaskBudget::SERIAL || left.len() <= 32 {
        work(offset, left, right);
    } else {
        let mid = left.len() / 2;
        let (ll, lr) = left.split_at_mut(mid);
        let (rl, rr) = right.split_at_mut(mid);
        let (left_budget, right_budget) = budget.split_at(budget.get() / 2).unwrap();
        executor.join(
            || paired(ll, rl, offset, left_budget, executor, work),
            || paired(lr, rr, offset + mid, right_budget, executor, work),
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
