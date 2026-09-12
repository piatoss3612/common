use super::executor::{Geometry, for_chunks};
use super::{
    BoundTables, CoefficientView, CosetDomain, ExecutionOptions, Executor, FftError, PastaField,
    PrimeModulus, ScratchRequirements, Tables, check_len, check_prefix, reverse,
};
use crate::field::fft::{
    Guard, butterfly, divide_by_power_of_two, normalize, scale as scale_loose,
};

/// Reusable transform metadata borrowing caller-prepared tables.
///
/// For size `n`, root `w`, and shift `s` from [`Self::domain`], the forward
/// transform maps coefficients `c[i]` to `sum(c[i] * (s * w^j)^i, i = 0..n)`
/// at evaluation index `j`. The inverse recovers the coefficients, including
/// division by `n` and removal of the shift, unless an explicit
/// [`super::InverseScale`] selects an unscaled inverse. Natural order means
/// increasing degree for coefficients and increasing `j` for evaluations;
/// methods accepting other orders document them. A singleton transform preserves
/// its sole value.
///
/// Plans can be shared across executions with independent mutable buffers.
/// Every full input and output slice must contain exactly `n` fields.
/// [`Self::forward_prefix`] and [`Self::inverse_prefix`] accept shorter inputs.
/// Scratch for direct transforms must meet [`Self::scratch_requirements`];
/// [`Self::configure`] provides separate requirements for prepared operations.
/// Incorrect buffer lengths return
/// [`FftError::LengthMismatch`]; insufficient scratch returns
/// [`FftError::ScratchTooSmall`]. Invalid execution options or storage overflow
/// return the errors described by [`Self::scratch_requirements`].
///
/// Table contents follow [`Tables`]' validity contract. The module's
/// [validation and working-storage rules](super) apply to all executions,
/// including errors and panics.
#[derive(Clone, Copy)]
pub struct Plan<'a, M: PrimeModulus> {
    pub(super) domain: CosetDomain<M>,
    pub(super) tables: Tables<'a, M>,
}

impl<M: PrimeModulus> core::fmt::Debug for Plan<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Plan")
            .field("domain", &self.domain)
            .field("tables", &self.tables)
            .finish()
    }
}

impl<'a, M: PrimeModulus> Plan<'a, M> {
    /// Constructs a plan using the domain retained by its table handle.
    ///
    /// Prepare tables with [`super::TablesMut::prepare`] or check imported
    /// entries with [`Tables::bind`]. Construction does not rescan contents.
    /// Raw table descriptors cannot be used without binding their domain:
    ///
    /// ```compile_fail
    /// use zakura_udon::{field::PallasBase, fft::{Plan, Tables}};
    /// let raw = Tables::<PallasBase>::default();
    /// let plan = Plan::new(raw);
    /// ```
    pub const fn new(tables: BoundTables<'a, M>) -> Self {
        Self {
            domain: tables.domain(),
            tables: tables.tables(),
        }
    }

    /// Constructs a plan that computes powers and permutations as needed.
    pub fn without_tables(domain: CosetDomain<M>) -> Self {
        Self {
            domain,
            tables: Tables::default(),
        }
    }

    /// The evaluation domain, including its coset shift.
    pub const fn domain(self) -> CosetDomain<M> {
        self.domain
    }

    /// Required temporary field storage for either transform direction.
    ///
    /// Delegates to [`ExecutionOptions::requirements`] with this domain's size.
    /// That query also sizes arrays in const contexts without constructing a plan.
    ///
    /// Returns [`FftError::InvalidExecution`] for invalid [`ExecutionOptions`],
    /// or [`FftError::SizeOverflow`] if the scratch field slice would exceed
    /// `isize::MAX` bytes or its element count overflows `usize`.
    pub const fn scratch_requirements(
        self,
        options: ExecutionOptions,
    ) -> Result<ScratchRequirements, FftError> {
        options.requirements(self.domain.size())
    }

    pub(super) fn check(
        self,
        buffer: &'static str,
        len: usize,
        options: ExecutionOptions,
        scratch_len: usize,
    ) -> Result<usize, FftError> {
        check_len(buffer, len, self.domain.size())?;
        let required = self.scratch_requirements(options)?;
        required.check(scratch_len)?;
        Ok(required.field_elements)
    }

    /// Replaces natural-order coefficients with natural-order coset evaluations.
    ///
    /// Buffer lengths, errors, and the evaluation formula are defined by [`Plan`].
    pub fn forward<E: Executor>(
        self,
        values: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let required = self.check("values", values.len(), options, scratch.len())?;
        self.scale_coefficients(values, PastaField::ONE);
        self.permute(values);
        self.run(
            values,
            options,
            executor,
            &mut scratch[..required],
            Run::forward(2),
        );
        Ok(())
    }

    /// Replaces natural-order evaluations with normalized polynomial coefficients.
    ///
    /// Both sides use the ordering, lengths, and error contract of [`Plan`].
    pub fn inverse<E: Executor>(
        self,
        values: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let required = self.check("values", values.len(), options, scratch.len())?;
        self.permute(values);
        self.run(
            values,
            options,
            executor,
            &mut scratch[..required],
            Run::inverse(&[]),
        );
        Ok(())
    }

    /// Interpolates evaluations already stored in bit-reversed order.
    ///
    /// This avoids a permutation when a caller scatters evaluations directly
    /// into their working positions with [`super::Class::scatter`]. Evaluation
    /// row `j` must be stored at the reversal of its low `log2(n)` bits, where
    /// `n` is the domain size. Output coefficients are in increasing degree
    /// order, with the same normalization, lengths, and errors as [`Self::inverse`].
    pub fn inverse_bit_reversed<E: Executor>(
        self,
        values: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let required = self.check("values", values.len(), options, scratch.len())?;
        self.run(
            values,
            options,
            executor,
            &mut scratch[..required],
            Run::inverse(&[]),
        );
        Ok(())
    }

    /// Preserves the coefficients and writes their transform into `output`.
    ///
    /// Accepts ordinary coefficients or a [`CoefficientView`], whose scale is
    /// folded into output initialization.
    /// Both input and output must have the domain size. Ordering, scratch
    /// requirements, and errors are the same as for [`Self::forward`].
    pub fn forward_into<'input, E: Executor>(
        self,
        input: impl Into<CoefficientView<'input, M>>,
        output: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let input = input.into();
        let extra = input.normalization_factor();
        let input = input.as_slice();
        check_len("input", input.len(), self.domain.size())?;
        let required = self.check("output", output.len(), options, scratch.len())?;
        let first = self.fill_prefix(input, output, self.domain.shift(), None, extra);
        self.run(
            output,
            options,
            executor,
            &mut scratch[..required],
            Run::forward(first),
        );
        Ok(())
    }

    /// Preserves the evaluations and writes their interpolation into `output`.
    ///
    /// Both slices must have the domain size. Ordering, scratch requirements,
    /// and errors are the same as for [`Self::inverse`].
    pub fn inverse_into<E: Executor>(
        self,
        input: &[PastaField<M>],
        output: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        check_len("input", input.len(), self.domain.size())?;
        let required = self.check("output", output.len(), options, scratch.len())?;
        self.scatter(input, output);
        self.run(
            output,
            options,
            executor,
            &mut scratch[..required],
            Run::inverse(&[]),
        );
        Ok(())
    }

    /// Evaluates a coefficient prefix, treating the remaining coefficients as zero.
    ///
    /// Input uses increasing degree order. Accepts ordinary coefficients or a
    /// [`CoefficientView`], including one retained by a smaller expansion; its
    /// scale is folded into initialization.
    ///
    /// An empty prefix produces the zero polynomial; a prefix longer than the
    /// domain returns [`FftError::InvalidPrefix`]. Output uses natural evaluation
    /// order and must have the full domain size. Scratch requirements and other
    /// errors are those of [`Self::forward`], even for an empty prefix.
    pub fn forward_prefix<'input, E: Executor>(
        self,
        coefficients: impl Into<CoefficientView<'input, M>>,
        output: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let coefficients = coefficients.into();
        let extra = coefficients.normalization_factor();
        let coefficients = coefficients.as_slice();
        let required = self.check("output", output.len(), options, scratch.len())?;
        check_prefix(coefficients.len(), 0, self.domain.size())?;
        if coefficients.is_empty() {
            output.fill(PastaField::ZERO);
            return Ok(());
        }
        let first = self.fill_prefix(coefficients, output, self.domain.shift(), None, extra);
        self.run(
            output,
            options,
            executor,
            &mut scratch[..required],
            Run::forward(first),
        );
        Ok(())
    }

    pub(super) fn scatter(self, input: &[PastaField<M>], output: &mut [PastaField<M>]) {
        for (index, &value) in input.iter().enumerate() {
            output[self.reversed(index)] = value;
        }
    }

    pub(super) fn permute(self, values: &mut [PastaField<M>]) {
        for index in 0..values.len() {
            let destination = self.reversed(index);
            if index < destination {
                values.swap(index, destination);
            }
        }
    }

    pub(super) fn reversed(self, index: usize) -> usize {
        reverse(index, self.domain.domain().log_size())
    }

    pub(super) fn scale_coefficients(self, values: &mut [PastaField<M>], extra: PastaField<M>) {
        let shift = self.domain.shift();
        if shift == PastaField::ONE && extra == PastaField::ONE {
            return;
        }
        if self.domain.inverse_scale_cycle.is_some() {
            let cycle = [extra, extra.mul(&shift), extra.mul(&shift.square())];
            for (index, value) in values.iter_mut().enumerate() {
                if index % 3 != 0 || extra != PastaField::ONE {
                    *value = value.mul(&cycle[index % 3]);
                }
            }
        } else {
            let mut scale = extra;
            let len = values.len();
            for (index, value) in values.iter_mut().enumerate() {
                *value = value.mul(&scale);
                if index + 1 < len {
                    scale = scale.mul(&shift);
                }
            }
        }
    }

    // Also used by residue expansion: each residue has a different scaling base
    // but shares this plan's subgroup twiddles and bit-reversal permutation.
    pub(super) fn fill_prefix(
        self,
        coefficients: &[PastaField<M>],
        output: &mut [PastaField<M>],
        shift: PastaField<M>,
        scales: Option<&[PastaField<M>]>,
        extra: PastaField<M>,
    ) -> usize {
        if coefficients.is_empty() {
            output.fill(PastaField::ZERO);
            return output.len() * 2;
        }
        let width = coefficients.len().next_power_of_two();
        let chunk_len = output.len() / width;
        // After bit reversal, each chunk starts with one coefficient and a
        // zero suffix. Its first log2(chunk_len) butterfly rounds only copy
        // that coefficient, so broadcasting skips those rounds. Visiting the
        // coefficients in degree order also permits a scaling progression.
        let mut scale = extra;
        let normalized = extra != PastaField::ONE;
        let identity = shift == PastaField::ONE;
        let cycle = if scales.is_none() && !identity && coefficients.len() > 1 {
            let squared = shift.square();
            (squared.mul(&shift) == PastaField::ONE)
                .then(|| [extra, extra.mul(&shift), extra.mul(&squared)])
        } else {
            None
        };
        for (index, value) in coefficients.iter().enumerate() {
            let value = if let Some(table) = scales {
                let scale = if normalized {
                    table[index].mul(&extra)
                } else {
                    table[index]
                };
                value.mul(&scale)
            } else if identity {
                if normalized {
                    value.mul(&extra)
                } else {
                    *value
                }
            } else if let Some(cycle) = cycle {
                if index % 3 == 0 && !normalized {
                    *value
                } else {
                    value.mul(&cycle[index % 3])
                }
            } else {
                let value = value.mul(&scale);
                if index + 1 < coefficients.len() {
                    scale = scale.mul(&shift);
                }
                value
            };
            let destination = self.reversed(index);
            output[destination..destination + chunk_len].fill(value);
        }
        for index in coefficients.len()..width {
            let destination = self.reversed(index);
            output[destination..destination + chunk_len].fill(PastaField::ZERO);
        }
        chunk_len * 2
    }

    pub(super) fn run<E: Executor>(
        self,
        values: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
        run: Run<'_, '_, M>,
    ) {
        if values.len() == 1 || run.first > values.len() {
            for (index, value) in values.iter_mut().enumerate() {
                if run.normalized {
                    *value = value.mul(&self.domain.inverse_scale(index));
                    for lift in run.lifts {
                        if let Some(coefficient) = lift.values.get(index) {
                            *value = value.add(coefficient);
                        }
                    }
                }
                if let Some(factor) = run.factor {
                    *value = value.mul(&factor[index]);
                }
            }
            return;
        }
        let geometry = options.geometry(values.len());
        // Division removes the subgroup's general scaling products. For a
        // cubic shift, pair it with periodic untwisting unless a retained
        // combined finish table supplies the input factors directly.
        let finish = if run.normalized && self.domain.shift() == PastaField::ONE {
            InverseFinish::Subgroup
        } else if run.normalized
            && self.domain.inverse_scale_cycle.is_some()
            && self.tables.inverse_finish.is_none()
        {
            InverseFinish::Periodic([
                PastaField::ONE,
                self.domain.inverse_shift(),
                self.domain.inverse_shift().square(),
            ])
        } else {
            InverseFinish::ScaledInputs
        };
        let kernel = Kernel {
            plan: self,
            run,
            finish,
        };
        if kernel.run.first <= geometry.tile_len {
            for_chunks(
                values,
                geometry.tile_len,
                options.max_tasks,
                executor,
                &|_, tile| kernel.local(tile),
            );
        }
        if geometry.tiles == 1 {
            return;
        }
        let job_len = geometry.tiles * geometry.columns;
        let mut first_column = 0;
        while first_column < geometry.tile_len {
            let count = (geometry.tile_len - first_column).min(geometry.columns * geometry.jobs);
            let jobs = count.div_ceil(geometry.columns);
            let work = &mut scratch[..jobs * job_len];
            // All readers finish before any scatter writer is scheduled. Each
            // scratch job stores contiguous lanes in column-major order.
            for_chunks(work, job_len, options.max_tasks, executor, &|job, work| {
                let column = first_column + job * geometry.columns;
                let columns = geometry.columns.min(geometry.tile_len - column);
                let work = &mut work[..columns * geometry.tiles];
                // Bounded rectangles reuse adjacent source columns and short
                // destination spans instead of streaming a full strided lane.
                for tile_start in (0..geometry.tiles).step_by(8) {
                    for column_start in (0..columns).step_by(8) {
                        for tile in tile_start..(tile_start + 8).min(geometry.tiles) {
                            for offset in column_start..(column_start + 8).min(columns) {
                                work[offset * geometry.tiles + tile] =
                                    values[tile * geometry.tile_len + column + offset];
                            }
                        }
                    }
                }
                kernel.cross(work, column, &geometry);
            });
            for_chunks(
                values,
                geometry.tile_len,
                options.max_tasks,
                executor,
                &|tile, output| {
                    for offset in 0..count {
                        let job = offset / geometry.columns;
                        let column = offset % geometry.columns;
                        output[first_column + offset] =
                            work[job * job_len + column * geometry.tiles + tile];
                    }
                },
            );
            first_column += count;
        }
    }
}

pub(super) struct Run<'a, 'b, M: PrimeModulus> {
    inverse: bool,
    normalized: bool,
    first: usize,
    lifts: &'a [super::Class<'b, M>],
    factor: Option<&'a [PastaField<M>]>,
}

impl<'a, 'b, M: PrimeModulus> Run<'a, 'b, M> {
    pub(super) fn set_first(&mut self, first: usize) {
        self.first = first;
    }
    pub(super) fn forward(first: usize) -> Self {
        Self {
            inverse: false,
            normalized: false,
            first,
            lifts: &[],
            factor: None,
        }
    }
    pub(super) fn inverse(lifts: &'a [super::Class<'b, M>]) -> Self {
        Self {
            inverse: true,
            normalized: true,
            first: 2,
            lifts,
            factor: None,
        }
    }
    pub(super) fn inverse_unscaled() -> Self {
        Self {
            inverse: true,
            normalized: false,
            first: 2,
            lifts: &[],
            factor: None,
        }
    }
    pub(super) fn forward_product(first: usize, factor: &'a [PastaField<M>]) -> Self {
        Self {
            factor: Some(factor),
            ..Self::forward(first)
        }
    }
}

struct Kernel<'a, 'b, 'c, M: PrimeModulus> {
    plan: Plan<'a, M>,
    run: Run<'b, 'c, M>,
    finish: InverseFinish<M>,
}

#[derive(Clone, Copy)]
enum InverseFinish<M: PrimeModulus> {
    Subgroup,
    Periodic([PastaField<M>; 3]),
    ScaledInputs,
}

impl<M: PrimeModulus> Kernel<'_, '_, '_, M> {
    fn root(&self) -> PastaField<M> {
        if self.run.inverse {
            self.plan.domain.domain().inverse_root()
        } else {
            self.plan.domain.domain().root()
        }
    }

    // Canonical roots are nested: the root of order 2^k is the full
    // domain root raised to size / 2^k. The field already stores this ladder.
    fn stage_root(&self, log_size: u32) -> PastaField<M> {
        if self.run.inverse {
            PastaField::root_of_unity_inverse(log_size).unwrap()
        } else {
            PastaField::root_of_unity(log_size).unwrap()
        }
    }

    fn twiddles(&self) -> Option<&[PastaField<M>]> {
        if self.run.inverse {
            self.plan.tables.inverse
        } else {
            self.plan.tables.forward
        }
    }

    fn needs_scale_progression(&self) -> bool {
        self.run.normalized
            && matches!(self.finish, InverseFinish::ScaledInputs)
            && self.plan.tables.inverse_scales.is_none()
            && self.plan.domain.inverse_scale_cycle.is_none()
    }

    fn needs_finish_twiddles(&self) -> bool {
        self.run.normalized
            && self.plan.tables.inverse.is_none()
            && (!matches!(self.finish, InverseFinish::ScaledInputs)
                || self.plan.tables.inverse_finish.is_none())
    }

    // The guard spans all local rounds. Normalizing only at the boundary keeps
    // the loose arithmetic optimization while preventing a public slice escape.
    #[inline(never)]
    fn local(&self, values: &mut [PastaField<M>]) {
        let whole_inverse = self.run.normalized && values.len() == self.plan.domain.size();
        let whole_forward = !self.run.inverse && values.len() == self.plan.domain.size();
        let last = if whole_inverse {
            values.len() / 2
        } else {
            values.len()
        };
        let guard = Guard::new(values);
        let mut block = self.run.first;
        while block <= last {
            let stride = self.plan.domain.size() / block;
            let step = if self.twiddles().is_some() {
                PastaField::ONE
            } else {
                self.stage_root(block.trailing_zeros())
            };
            for chunk in guard.values.chunks_exact_mut(block) {
                let (left, right) = chunk.split_at_mut(block / 2);
                let mut power = PastaField::ONE;
                for (index, (left, right)) in left.iter_mut().zip(right).enumerate() {
                    if index == 0 {
                        butterfly(left, right, None);
                    } else {
                        if self.twiddles().is_none() {
                            power = power.mul(&step);
                        }
                        let twiddle = self.twiddles().map_or(power, |table| table[index * stride]);
                        butterfly(left, right, Some(&twiddle));
                    }
                    if whole_forward && block == self.plan.domain.size() {
                        *left = self.forward_store(*left, index);
                        *right = self.forward_store(*right, index + block / 2);
                    }
                }
            }
            block *= 2;
        }
        if whole_inverse {
            let (left, right) = guard.values.split_at_mut(self.plan.domain.size() / 2);
            let mut scale = self.plan.domain.domain().size_inverse();
            let mut twiddle = PastaField::ONE;
            let right_scale = self.right_scale();
            for (index, (left, right)) in left.iter_mut().zip(right).enumerate() {
                self.finish(left, right, index, scale, twiddle, right_scale);
                if index + 1 < self.plan.domain.size() / 2 && self.needs_scale_progression() {
                    scale = scale.mul(&self.plan.domain.inverse_shift());
                }
                if index + 1 < self.plan.domain.size() / 2 && self.needs_finish_twiddles() {
                    twiddle = twiddle.mul(&self.root());
                }
            }
            guard.disarm();
        } else if whole_forward {
            guard.disarm();
        }
    }

    fn forward_store(&self, value: PastaField<M>, index: usize) -> PastaField<M> {
        self.run.factor.map_or_else(
            || normalize(value),
            |factor| scale_loose(value, &factor[index]),
        )
    }

    fn finish(
        &self,
        left: &mut PastaField<M>,
        right: &mut PastaField<M>,
        index: usize,
        scale: PastaField<M>,
        twiddle: PastaField<M>,
        right_scale: PastaField<M>,
    ) {
        if matches!(self.finish, InverseFinish::ScaledInputs) {
            self.finish_scaled(left, right, index, scale, twiddle, right_scale);
        } else {
            let twiddle = self
                .plan
                .tables
                .inverse
                .map_or(twiddle, |table| table[index]);
            butterfly(left, right, (index != 0).then_some(&twiddle));
            let log_size = self.plan.domain.domain().log_size();
            *left = divide_by_power_of_two(*left, log_size);
            *right = divide_by_power_of_two(*right, log_size);
            if let InverseFinish::Periodic(cycle) = self.finish {
                for (value, degree) in [
                    (&mut *left, index),
                    (&mut *right, index + self.plan.domain.size() / 2),
                ] {
                    if degree % 3 != 0 {
                        *value = value.mul(&cycle[degree % 3]);
                    }
                }
            }
        }
        // Add canonical coefficients only after normalization and untwisting.
        // Equal-size lifts also contribute to the upper terminal store.
        for lift in self.run.lifts {
            if let Some(coefficient) = lift.values.get(index) {
                *left = left.add(coefficient);
            }
            if let Some(coefficient) = lift.values.get(index + self.plan.domain.size() / 2) {
                *right = right.add(coefficient);
            }
        }
    }

    fn finish_scaled(
        &self,
        left: &mut PastaField<M>,
        right: &mut PastaField<M>,
        index: usize,
        scale: PastaField<M>,
        twiddle: PastaField<M>,
        right_scale: PastaField<M>,
    ) {
        // At coefficient i, scale the low input by n^-1 * shift^-i and
        // the high input by that times root^-i. The upper output also needs
        // shift^(-n/2). Loose-input multiplication avoids reducing each input
        // solely to meet the ordinary field multiplication contract.
        let scale = self.plan.tables.inverse_scales.map_or_else(
            || {
                if self.plan.domain.inverse_scale_cycle.is_some() {
                    self.plan.domain.inverse_scale(index)
                } else {
                    scale
                }
            },
            |table| table[index],
        );
        let high_scale = self.plan.tables.inverse_finish.map_or_else(
            || {
                self.plan
                    .tables
                    .inverse
                    .map_or(twiddle, |table| table[index])
                    .mul(&scale)
            },
            |table| table[index],
        );
        let low = scale_loose(*left, &scale);
        let high = scale_loose(*right, &high_scale);
        *left = low.add(&high);
        let difference = low.sub(&high);
        *right = if right_scale == PastaField::ONE {
            difference
        } else {
            difference.mul(&right_scale)
        };
    }

    fn right_scale(&self) -> PastaField<M> {
        if self.run.normalized && matches!(self.finish, InverseFinish::ScaledInputs) {
            self.plan
                .domain
                .inverse_shift()
                .pow_u64((self.plan.domain.size() / 2) as u64)
        } else {
            PastaField::ONE
        }
    }

    fn cross(&self, values: &mut [PastaField<M>], first_column: usize, geometry: &Geometry) {
        let guard = Guard::new(values);
        let scale_progression = self.needs_scale_progression();
        let finish_twiddles = self.needs_finish_twiddles();
        let mut column_scale = if scale_progression {
            self.plan.domain.inverse_scale(first_column)
        } else {
            PastaField::ONE
        };
        let mut column_twiddle = if finish_twiddles {
            self.root().pow_u64(first_column as u64)
        } else {
            PastaField::ONE
        };
        let tile_scale = if scale_progression {
            self.plan
                .domain
                .inverse_shift()
                .pow_u64(geometry.tile_len as u64)
        } else {
            PastaField::ONE
        };
        let tile_twiddle = if finish_twiddles {
            self.stage_root(geometry.tiles.trailing_zeros())
        } else {
            PastaField::ONE
        };
        let right_scale = self.right_scale();
        // One seed per active stage and column block, then a recurrence
        // across columns. Pasta domains have at most 32 radix-2 stages.
        // Each tuple holds the column seed, column step, and tile step.
        let mut stages = [(PastaField::ONE, PastaField::ONE, PastaField::ONE); 32];
        if self.twiddles().is_none() {
            for (stage, state) in stages[..geometry.tiles.trailing_zeros() as usize]
                .iter_mut()
                .enumerate()
            {
                let block = (2 << stage) * geometry.tile_len;
                if block >= self.run.first
                    && !(self.run.normalized && block == self.plan.domain.size())
                {
                    let column_step = self.stage_root(block.trailing_zeros());
                    *state = (
                        column_step.pow_u64(first_column as u64),
                        column_step,
                        self.stage_root(stage as u32 + 1),
                    );
                }
            }
        }
        let columns = guard.values.len() / geometry.tiles;
        for (offset, lane) in guard.values.chunks_exact_mut(geometry.tiles).enumerate() {
            let column = first_column + offset;
            let mut distance = 1;
            while distance < geometry.tiles {
                let stride = geometry.tiles / (2 * distance);
                if 2 * distance * geometry.tile_len >= self.run.first {
                    if self.run.normalized && distance == geometry.tiles / 2 {
                        let (left, right) = lane.split_at_mut(distance);
                        let mut scale = column_scale;
                        let mut twiddle = column_twiddle;
                        for (tile, (left, right)) in left.iter_mut().zip(right).enumerate() {
                            self.finish(
                                left,
                                right,
                                tile * geometry.tile_len + column,
                                scale,
                                twiddle,
                                right_scale,
                            );
                            if tile + 1 < distance && scale_progression {
                                scale = scale.mul(&tile_scale);
                            }
                            if tile + 1 < distance && finish_twiddles {
                                twiddle = twiddle.mul(&tile_twiddle);
                            }
                        }
                    } else {
                        let (first, column_step, step) = stages[distance.trailing_zeros() as usize];
                        for group in lane.chunks_exact_mut(distance * 2) {
                            let (left, right) = group.split_at_mut(distance);
                            let mut power = first;
                            for (tile, (left, right)) in left.iter_mut().zip(right).enumerate() {
                                let index = (tile * geometry.tile_len + column) * stride;
                                if index == 0 {
                                    butterfly(left, right, None);
                                } else {
                                    let twiddle =
                                        self.twiddles().map_or(power, |table| table[index]);
                                    butterfly(left, right, Some(&twiddle));
                                }
                                if !self.run.inverse && distance == geometry.tiles / 2 {
                                    let index = tile * geometry.tile_len + column;
                                    *left = self.forward_store(*left, index);
                                    *right = self
                                        .forward_store(*right, index + self.plan.domain.size() / 2);
                                }
                                if self.twiddles().is_none() && tile + 1 < distance {
                                    power = power.mul(&step);
                                }
                            }
                        }
                        if self.twiddles().is_none() && offset + 1 < columns {
                            stages[distance.trailing_zeros() as usize].0 = first.mul(&column_step);
                        }
                    }
                }
                distance *= 2;
            }
            if offset + 1 < columns && scale_progression {
                column_scale = column_scale.mul(&self.plan.domain.inverse_shift());
            }
            if offset + 1 < columns && finish_twiddles {
                column_twiddle = column_twiddle.mul(&self.root());
            }
        }
        if self.run.normalized || !self.run.inverse {
            guard.disarm();
        }
    }
}
