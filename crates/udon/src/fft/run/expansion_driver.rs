use super::super::{ExpansionOrder, ExpansionStorage, ScratchRequirements, assert_length};
use super::expansion::ExpansionPlan;
use super::*;
use crate::exec::Executor;

impl<M: PrimeModulus> ExpansionPlan<'_, M> {
    /// Scratch field count for contiguous execution.
    ///
    /// The inverse and residues reuse scratch; coefficient storage from
    /// [`Self::coefficient_fields`] is separate. The task budget is divided across
    /// residues and within each transform.
    pub const fn scratch_fields(&self) -> usize {
        self.scratch_fields
    }

    pub(crate) fn scratch_fields_with(&self, max_tasks: NonZeroUsize) -> usize {
        // Retained fields per residue are bounded by the base domain, so the
        // combined scratch fits within the validated extended domain.
        self.snapshot_fields() * self.residues().min(max_tasks.get())
    }

    fn check(
        &self,
        input: usize,
        output: usize,
        factor: Option<&[PastaField<M>]>,
        scratch: usize,
        max_tasks: NonZeroUsize,
    ) {
        let input_len = if self.storage == ExpansionStorage::Coefficients {
            match self.support {
                InputSupport::Full => self.base_size(),
                InputSupport::Prefix(n) => n,
            }
        } else {
            self.base_size()
        };
        assert_length("input", input_len, input);
        assert_length("output", self.base_size() * self.residues(), output);
        if let Some(factor) = factor {
            assert_length("factor", output, factor.len());
        }
        ScratchRequirements {
            field_elements: self.scratch_fields_with(max_tasks),
        }
        .check(scratch)
    }

    /// Expands preserved input into contiguous residue blocks.
    ///
    /// `input` follows the order and support selected by [`Self::new`]: exactly
    /// the declared coefficient prefix length, or [`Self::base_size`] entries
    /// for full coefficients or evaluations. A prefix's missing suffix is zero.
    /// `output` must contain `base_size() * residues()` fields and receives
    /// evaluations in the selected [`ExpansionOrder`].
    ///
    /// `coefficients` must have exactly [`Self::coefficient_fields`] entries,
    /// including an empty slice for modes without a separate workspace. A
    /// workspace retains natural coefficients with the scale selected by
    /// [`ExpansionStorage`]. Returns a view of those coefficients, or `None`
    /// for other modes. `factor`, when present, must have the output length and
    /// physical order; it is multiplied pointwise into the output evaluations.
    /// Scratch must meet [`Self::scratch_fields`].
    ///
    /// Disposable input requires [`Self::execute_disposable`]. Panics if the storage
    /// mode or buffers do not meet these requirements, before mutation. No allocation
    /// occurs. Scoped tasks finish unwinding before a panic propagates; fields remain
    /// canonical, but results may be incomplete.
    pub fn execute<'c, E: Executor>(
        self,
        input: &[PastaField<M>],
        output: &mut [PastaField<M>],
        coefficients: &'c mut [PastaField<M>],
        factor: Option<&[PastaField<M>]>,
        scratch: &mut [PastaField<M>],
        executor: &E,
    ) -> Option<super::super::CoefficientView<'c, M>> {
        self.execute_with(
            input,
            output,
            coefficients,
            factor,
            scratch,
            NonZeroUsize::new(self.budget.get()).unwrap(),
            executor,
        )
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Internal driver receives disjoint buffers and a task allowance."
    )]
    pub(crate) fn execute_with<'c, E: Executor>(
        self,
        input: &[PastaField<M>],
        output: &mut [PastaField<M>],
        coefficients: &'c mut [PastaField<M>],
        factor: Option<&[PastaField<M>]>,
        scratch: &mut [PastaField<M>],
        max_tasks: NonZeroUsize,
        executor: &E,
    ) -> Option<super::super::CoefficientView<'c, M>> {
        self.check(input.len(), output.len(), factor, scratch.len(), max_tasks);
        assert_length(
            "coefficients",
            self.coefficient_fields(),
            coefficients.len(),
        );
        match self.storage {
            ExpansionStorage::DisposableInput { .. } => {
                panic!("disposable input requires execute_disposable")
            }
            ExpansionStorage::Coefficients => {
                self.residue_transforms(input, output, factor, 0, scratch, max_tasks, executor)
            }
            ExpansionStorage::CoefficientWorkspace { .. } => {
                self.inverse().execute_with(
                    Some(input),
                    coefficients,
                    None,
                    scratch,
                    max_tasks,
                    executor,
                );
                self.residue_transforms(
                    coefficients,
                    output,
                    factor,
                    0,
                    scratch,
                    max_tasks,
                    executor,
                );
            }
            ExpansionStorage::ReuseOutput => {
                let (first, rest) = output.split_at_mut(self.base_size());
                let (first_factor, rest_factor) = factor.map_or((None, None), |f| {
                    let (a, b) = f.split_at(self.base_size());
                    (Some(a), Some(b))
                });
                // Nested roots make residue zero identical to the original base
                // evaluations whenever the two coset shifts agree.
                let copy_first = self.expansion.extended.shift().reduce()
                    == self.expansion.base.domain().shift().reduce();
                if !rest.is_empty() || !copy_first {
                    self.inverse().execute_with(
                        Some(input),
                        first,
                        None,
                        scratch,
                        max_tasks,
                        executor,
                    );
                    self.residue_transforms(
                        first,
                        rest,
                        rest_factor,
                        1,
                        scratch,
                        max_tasks,
                        executor,
                    );
                }
                if copy_first {
                    let output_order = if self.order == ExpansionOrder::Residues {
                        ElementOrder::Natural
                    } else {
                        ElementOrder::BitReversed
                    };
                    for (i, value) in first.iter_mut().enumerate() {
                        let source = if self.input_order == output_order {
                            i
                        } else {
                            reverse(i, self.base_size().ilog2())
                        };
                        *value = input[source];
                        if let Some(factor) = first_factor {
                            *value = value.mul(&factor[i]);
                        }
                    }
                } else {
                    self.transform(0, true).execute_with(
                        None,
                        first,
                        first_factor,
                        scratch,
                        max_tasks,
                        executor,
                    );
                }
            }
        }
        match self.storage {
            ExpansionStorage::CoefficientWorkspace { scale } => {
                Some(super::super::CoefficientView::new(coefficients, scale))
            }
            _ => None,
        }
    }

    /// Expands evaluations while reusing input for natural coefficients.
    ///
    /// Requires [`ExpansionStorage::DisposableInput`]. Input must contain exactly
    /// [`Self::base_size`] evaluations in the order selected by [`Self::new`]. It
    /// retains coefficients with that mode's inverse scale. Output, factor, and scratch
    /// follow [`Self::execute`]. Panics if the storage mode or buffers do not meet
    /// these requirements, before mutation. Executor panic behavior follows
    /// [`Self::execute`]; refill input evaluations before retrying.
    pub fn execute_disposable<'c, E: Executor>(
        self,
        input: &'c mut [PastaField<M>],
        output: &mut [PastaField<M>],
        factor: Option<&[PastaField<M>]>,
        scratch: &mut [PastaField<M>],
        executor: &E,
    ) -> super::super::CoefficientView<'c, M> {
        self.execute_disposable_with(
            input,
            output,
            factor,
            scratch,
            NonZeroUsize::new(self.budget.get()).unwrap(),
            executor,
        )
    }

    pub(crate) fn execute_disposable_with<'c, E: Executor>(
        self,
        input: &'c mut [PastaField<M>],
        output: &mut [PastaField<M>],
        factor: Option<&[PastaField<M>]>,
        scratch: &mut [PastaField<M>],
        max_tasks: NonZeroUsize,
        executor: &E,
    ) -> super::super::CoefficientView<'c, M> {
        self.check(input.len(), output.len(), factor, scratch.len(), max_tasks);
        let ExpansionStorage::DisposableInput { scale } = self.storage else {
            panic!("execute_disposable requires disposable input");
        };
        self.inverse()
            .execute_with(None, input, None, scratch, max_tasks, executor);
        self.residue_transforms(input, output, factor, 0, scratch, max_tasks, executor);
        super::super::CoefficientView::new(input, scale)
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Scoped residue jobs carry disjoint output and scratch."
    )]
    fn residue_transforms<E: Executor>(
        self,
        coefficients: &[PastaField<M>],
        output: &mut [PastaField<M>],
        factor: Option<&[PastaField<M>]>,
        first: usize,
        scratch: &mut [PastaField<M>],
        max_tasks: NonZeroUsize,
        executor: &E,
    ) {
        let count = output.len() / self.base_size();
        let jobs = count.min(max_tasks.get());
        if jobs == 0 {
            return;
        }
        let inner = NonZeroUsize::new(max_tasks.get() / jobs).unwrap();
        self.visit(
            coefficients,
            output,
            factor,
            first,
            scratch,
            jobs,
            inner,
            executor,
        );
    }

    #[expect(
        clippy::too_many_arguments,
        reason = "Scoped residue jobs carry disjoint output and scratch."
    )]
    fn visit<E: Executor>(
        self,
        coefficients: &[PastaField<M>],
        output: &mut [PastaField<M>],
        factor: Option<&[PastaField<M>]>,
        first: usize,
        scratch: &mut [PastaField<M>],
        jobs: usize,
        inner: NonZeroUsize,
        executor: &E,
    ) {
        let size = self.base_size();
        if jobs == 1 {
            for (i, block) in output.chunks_exact_mut(size).enumerate() {
                let factor = factor.map(|f| &f[i * size..(i + 1) * size]);
                self.transform(first + i, false).execute_with(
                    Some(coefficients),
                    block,
                    factor,
                    scratch,
                    inner,
                    executor,
                );
            }
        } else {
            let middle = output.len() / size / 2;
            let (left, right) = output.split_at_mut(middle * size);
            let (a, b) = scratch.split_at_mut(jobs / 2 * self.snapshot_fields());
            let (fa, fb) = factor.map_or((None, None), |f| {
                let (a, b) = f.split_at(middle * size);
                (Some(a), Some(b))
            });
            executor.join(
                || self.visit(coefficients, left, fa, first, a, jobs / 2, inner, executor),
                || {
                    self.visit(
                        coefficients,
                        right,
                        fb,
                        first + middle,
                        b,
                        jobs - jobs / 2,
                        inner,
                        executor,
                    )
                },
            );
        }
    }
}
