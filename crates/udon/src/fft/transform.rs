use super::executor::{Geometry, for_chunks};
use super::{
    CosetDomain, ExecutionOptions, Executor, FftError, PastaField, PrimeModulus,
    ScratchRequirements, Tables, check_len, check_prefix, reverse,
};
use crate::field::fft::{Guard, butterfly, normalize};

/// Reusable transform metadata borrowing caller-prepared tables.
///
/// For size `n`, root `w`, and shift `s` from [`Self::domain`], the forward
/// transform maps coefficients `c[i]` to `sum(c[i] * (s * w^j)^i, i = 0..n)`
/// at evaluation index `j`. The inverse recovers the coefficients, including
/// division by `n` and removal of the shift. Coefficients are ordered by
/// increasing degree; evaluations use increasing `j` unless a method specifies
/// bit-reversed input. A singleton transform preserves its sole value.
///
/// Plans can be shared across executions with independent mutable buffers.
/// Every full input and output slice must contain exactly `n` fields; only
/// [`Self::forward_prefix`] accepts fewer coefficients. Scratch must meet
/// [`Self::scratch_requirements`]. Incorrect buffer lengths return
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
    /// Binds tables after checking their lengths, without regenerating them.
    ///
    /// Returns [`FftError::LengthMismatch`] if any supplied table has the wrong
    /// length. Use [`Tables::validate`] to check contents against `domain`.
    pub fn new(domain: CosetDomain<M>, tables: Tables<'a, M>) -> Result<Self, FftError> {
        tables.check_shape(domain)?;
        Ok(Self { domain, tables })
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
    /// Both slices must have the domain size. Ordering, scratch requirements,
    /// and errors are the same as for [`Self::forward`].
    pub fn forward_into<E: Executor>(
        self,
        input: &[PastaField<M>],
        output: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        check_len("input", input.len(), self.domain.size())?;
        self.check("output", output.len(), options, scratch.len())?;
        output.copy_from_slice(input);
        self.forward(output, options, executor, scratch)
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
        self.check("output", output.len(), options, scratch.len())?;
        output.copy_from_slice(input);
        self.inverse(output, options, executor, scratch)
    }

    /// Evaluates a coefficient prefix, treating the remaining coefficients as zero.
    ///
    /// `coefficients[i]` is the coefficient of degree `i`. An empty prefix
    /// produces the zero polynomial; a prefix longer than the domain returns
    /// [`FftError::InvalidPrefix`]. Output uses natural evaluation order and must
    /// have the full domain size. Scratch requirements and other errors are
    /// those of [`Self::forward`], even for an empty prefix.
    pub fn forward_prefix<E: Executor>(
        self,
        coefficients: &[PastaField<M>],
        output: &mut [PastaField<M>],
        options: ExecutionOptions,
        executor: &E,
        scratch: &mut [PastaField<M>],
    ) -> Result<(), FftError> {
        let required = self.check("output", output.len(), options, scratch.len())?;
        check_prefix(coefficients.len(), 0, self.domain.size())?;
        if coefficients.is_empty() {
            output.fill(PastaField::ZERO);
            return Ok(());
        }
        let first = self.fill_prefix(
            coefficients,
            output,
            self.domain.shift(),
            None,
            PastaField::ONE,
        );
        self.run(
            output,
            options,
            executor,
            &mut scratch[..required],
            Run::forward(first),
        );
        Ok(())
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
        self.tables.bit_reversed.map_or_else(
            || reverse(index, self.domain.domain().log_size()),
            |table| table[index] as usize,
        )
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
            for value in values {
                *value = value.mul(&scale);
                scale = scale.mul(&shift);
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
        for index in 0..width {
            let value = coefficients.get(index).map_or(PastaField::ZERO, |value| {
                let scale = scales.map_or(scale, |table| {
                    if normalized {
                        table[index].mul(&extra)
                    } else {
                        table[index]
                    }
                });
                value.mul(&scale)
            });
            let destination = self.reversed(index);
            output[destination..destination + chunk_len].fill(value);
            if scales.is_none() {
                scale = scale.mul(&shift);
            }
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
        if values.len() == 1 {
            return;
        }
        let geometry = options.geometry(values.len());
        let kernel = Kernel { plan: self, run };
        for_chunks(
            values,
            geometry.tile_len,
            options.max_tasks,
            executor,
            &|_, tile| kernel.local(tile),
        );
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
                for (offset, lane) in work.chunks_exact_mut(geometry.tiles).enumerate() {
                    for (tile, value) in lane.iter_mut().enumerate() {
                        *value = values[tile * geometry.tile_len + column + offset];
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
}

impl<'a, 'b, M: PrimeModulus> Run<'a, 'b, M> {
    pub(super) fn forward(first: usize) -> Self {
        Self {
            inverse: false,
            normalized: false,
            first,
            lifts: &[],
        }
    }
    pub(super) fn inverse(lifts: &'a [super::Class<'b, M>]) -> Self {
        Self {
            inverse: true,
            normalized: true,
            first: 2,
            lifts,
        }
    }
    pub(super) fn inverse_unscaled() -> Self {
        Self {
            inverse: true,
            normalized: false,
            first: 2,
            lifts: &[],
        }
    }
}

struct Kernel<'a, 'b, 'c, M: PrimeModulus> {
    plan: Plan<'a, M>,
    run: Run<'b, 'c, M>,
}

impl<M: PrimeModulus> Kernel<'_, '_, '_, M> {
    fn root(&self) -> PastaField<M> {
        if self.run.inverse {
            self.plan.domain.domain().inverse_root()
        } else {
            self.plan.domain.domain().root()
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
            && self.plan.tables.inverse_scales.is_none()
            && self.plan.domain.inverse_scale_cycle.is_none()
    }

    fn needs_finish_twiddles(&self) -> bool {
        self.run.normalized
            && self.plan.tables.inverse_finish.is_none()
            && self.plan.tables.inverse.is_none()
    }

    // The guard spans all local rounds. Normalizing only at the boundary keeps
    // the loose arithmetic optimization while preventing a public slice escape.
    #[inline(never)]
    fn local(&self, values: &mut [PastaField<M>]) {
        let whole_inverse = self.run.normalized && values.len() == self.plan.domain.size();
        let last = if whole_inverse {
            values.len() / 2
        } else {
            values.len()
        };
        let guard = Guard { values };
        let mut block = self.run.first;
        while block <= last {
            let stride = self.plan.domain.size() / block;
            let step = if self.twiddles().is_some() {
                PastaField::ONE
            } else {
                self.root().pow_u64(stride as u64)
            };
            for chunk in guard.values.chunks_exact_mut(block) {
                let (left, right) = chunk.split_at_mut(block / 2);
                let mut power = PastaField::ONE;
                for (index, (left, right)) in left.iter_mut().zip(right).enumerate() {
                    if index == 0 {
                        butterfly(left, right, None);
                    } else {
                        let twiddle = self.twiddles().map_or(power, |table| table[index * stride]);
                        butterfly(left, right, Some(&twiddle));
                    }
                    if self.twiddles().is_none() {
                        power = power.mul(&step);
                    }
                }
            }
            block *= 2;
        }
        if whole_inverse {
            let (left, right) = guard.values.split_at_mut(self.plan.domain.size() / 2);
            let mut scale = self.plan.domain.domain().size_inverse();
            let mut twiddle = PastaField::ONE;
            let right_scale = self.plan.domain.inverse_shift().pow_u64(left.len() as u64);
            for (index, (left, right)) in left.iter_mut().zip(right).enumerate() {
                self.finish(left, right, index, scale, twiddle, right_scale);
                if self.needs_scale_progression() {
                    scale = scale.mul(&self.plan.domain.inverse_shift());
                }
                if self.needs_finish_twiddles() {
                    twiddle = twiddle.mul(&self.root());
                }
            }
        }
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
        // For the final inverse butterfly at coefficient i, scale the low
        // input by n^-1 * shift^-i and the high input by that times root^-i.
        // The upper output also needs shift^(-n/2). Every lift fits in the
        // lower half, so its normalized coefficients can be added only there.
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
        let low = normalize(*left).mul(&scale);
        let high = normalize(*right).mul(&high_scale);
        *left = low.add(&high);
        let difference = low.sub(&high);
        *right = if right_scale == PastaField::ONE {
            difference
        } else {
            difference.mul(&right_scale)
        };
        for lift in self.run.lifts {
            if let Some(coefficient) = lift.values.get(index) {
                *left = left.add(coefficient);
            }
        }
    }

    fn cross(&self, values: &mut [PastaField<M>], first_column: usize, geometry: &Geometry) {
        let guard = Guard { values };
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
            self.root().pow_u64(geometry.tile_len as u64)
        } else {
            PastaField::ONE
        };
        let right_scale = if self.run.normalized {
            self.plan
                .domain
                .inverse_shift()
                .pow_u64((self.plan.domain.size() / 2) as u64)
        } else {
            PastaField::ONE
        };
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
                            if scale_progression {
                                scale = scale.mul(&tile_scale);
                            }
                            if finish_twiddles {
                                twiddle = twiddle.mul(&tile_twiddle);
                            }
                        }
                    } else {
                        let (first, step) = if self.twiddles().is_none() {
                            (
                                self.root().pow_u64((column * stride) as u64),
                                self.root().pow_u64((geometry.tile_len * stride) as u64),
                            )
                        } else {
                            (PastaField::ONE, PastaField::ONE)
                        };
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
                                if self.twiddles().is_none() {
                                    power = power.mul(&step);
                                }
                            }
                        }
                    }
                }
                distance *= 2;
            }
            if scale_progression {
                column_scale = column_scale.mul(&self.plan.domain.inverse_shift());
            }
            if finish_twiddles {
                column_twiddle = column_twiddle.mul(&self.root());
            }
        }
    }
}
