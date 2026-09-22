use super::*;
use crate::exec::Executor;
use crate::exec::run::Reserved;

struct Lease<'a, C: PastaCurve> {
    records: &'a [ScalarStorage<C>],
    digits: &'a [u8],
    scratch: Scratch<'a, C>,
    buckets: &'a mut [ProjectivePoint<C>],
    output: &'a mut [ProjectivePoint<C>],
    partials: &'a [ProjectivePoint<C>],
}

impl<C: PastaCurve> Resources<C> for Lease<'_, C> {
    fn buffers(&mut self) -> Buffers<'_, C> {
        Buffers {
            records: &self.records,
            digits: &self.digits,
            scratch: self.scratch.reborrow(),
            buckets: self.buckets,
            output: self.output,
            partials: &self.partials,
        }
    }
}

fn empty<C: PastaCurve>() -> Scratch<'static, C> {
    Scratch::new(&mut [], &mut [], &mut [], &mut [], &mut [], &mut [])
}

fn dispatch<'a, 'i, 'b, C: PastaCurve, E: Executor>(
    run: &mut MsmRun<'a, 'i, C>,
    requests: &mut [Option<Request<'a>>],
    leases: impl Iterator<Item = Lease<'b, C>>,
    executor: &E,
) {
    crate::exec::run::dispatch!(
        run,
        requests,
        leases,
        executor,
        "validated input and complete resources"
    );
}

// A structured join owns every window output and the requested scratch bank
// before reserving work. Each branch lends its bundle to successive bounded
// kernels, reclaiming it after each return. This avoids a join barrier after
// every scratch-sized wave. These are driver loops, not arithmetic kernels;
// application schedulers keep using independently published run tasks.
struct Windows<'a, 'i, 'b, C: PastaCurve> {
    claims: &'b mut [Option<Reserved<'a, MsmKernel<'i, C>>>],
    outputs: &'b mut [ProjectivePoint<C>],
    receipts: &'b mut [Option<Completion<'a, (), MsmOutput<C>>>],
    records: &'b [ScalarStorage<C>],
    digits: &'b [u8],
    scratch: Scratch<'b, C>,
    work: Requirements,
    leases: usize,
}

impl<C: PastaCurve> Windows<'_, '_, '_, C> {
    fn execute<E: Executor>(mut self, executor: &E) {
        if self.leases == 1 {
            for ((claim, output), receipt) in
                self.claims.iter_mut().zip(self.outputs).zip(self.receipts)
            {
                let mut task = claim.take().unwrap().bind(Lease {
                    records: self.records,
                    digits: self.digits,
                    scratch: self.scratch.reborrow(),
                    buckets: &mut [],
                    output: core::slice::from_mut(output),
                    partials: &[],
                });
                task.execute().expect("fresh window");
                *receipt = Some(task.finish().release_resources());
            }
            return;
        }
        let left_leases = self.leases / 2;
        let middle = self.claims.len() / 2;
        let (left_claims, right_claims) = self.claims.split_at_mut(middle);
        let (left_outputs, right_outputs) = self.outputs.split_at_mut(middle);
        let (left_receipts, right_receipts) = self.receipts.split_at_mut(middle);
        let (left_scratch, right_scratch) =
            schedule::split_scratch(self.scratch, self.work.times::<C>(left_leases).unwrap());
        executor.join(
            || {
                Windows {
                    claims: left_claims,
                    outputs: left_outputs,
                    receipts: left_receipts,
                    records: self.records,
                    digits: self.digits,
                    scratch: left_scratch,
                    work: self.work,
                    leases: left_leases,
                }
                .execute(executor)
            },
            || {
                Windows {
                    claims: right_claims,
                    outputs: right_outputs,
                    receipts: right_receipts,
                    records: self.records,
                    digits: self.digits,
                    scratch: right_scratch,
                    work: self.work,
                    leases: self.leases - left_leases,
                }
                .execute(executor)
            },
        );
    }
}

impl<C: PastaCurve> MsmPlan<C> {
    /// Typed workspace for the contiguous driver under this plan's task budget.
    ///
    /// Includes retained intermediates and simultaneous temporary bundles.
    /// Metadata, buffer tails, alignment, and stack/executor costs are separate.
    pub fn requirements(&self) -> Requirements {
        self.requirements_with(NonZeroUsize::new(self.job.budget.get()).unwrap())
    }

    pub(crate) fn requirements_with(&self, leases: NonZeroUsize) -> Requirements {
        let count = self.output_slots().min(leases.get()).min(32);
        let r = self
            .retained
            .plus(
                self.temporary()
                    .times::<C>(count)
                    .expect("bounded task count"),
            )
            .expect("bounded plan workspace")
            .times::<C>(1)
            .expect("bounded plan workspace");
        r.bytes::<C>().expect("bounded plan workspace");
        r
    }

    /// Executes bounded tasks with contiguous storage and a scoped executor.
    ///
    /// Scratch must meet [`Self::requirements`]. Every executing task owns a distinct
    /// bundle; nested work never obtains storage by worker identity. The input must
    /// satisfy the plan's requirements (see [`Self::for_input`]). Incompatible
    /// input or insufficient scratch panics before
    /// mutation; unused scratch tails are untouched. An empty input returns identity.
    /// The driver allocates nothing. A panic waits for joined work before propagating;
    /// scratch can be reused after unwinding.
    pub fn execute<E: Executor>(
        self,
        input: Input<'_, C>,
        executor: &E,
        scratch: Scratch<'_, C>,
    ) -> ProjectivePoint<C> {
        self.execute_with(
            input,
            NonZeroUsize::new(self.job.budget.get()).unwrap(),
            executor,
            scratch,
        )
    }

    pub(crate) fn execute_with<E: Executor>(
        self,
        input: Input<'_, C>,
        leases: NonZeroUsize,
        executor: &E,
        scratch: Scratch<'_, C>,
    ) -> ProjectivePoint<C> {
        super::super::assert_length("input", self.terms, input.len());
        assert!(
            self.accepts(input),
            "input must match the plan's preparation"
        );
        let scratch = scratch.checked(self.requirements_with(leases));
        if input.is_empty() {
            return ProjectivePoint::IDENTITY;
        }
        let (retained, mut temporary) = schedule::split_scratch(scratch, self.retained);
        let Scratch {
            scalars,
            digits,
            projective,
            ..
        } = retained;
        let (partials, buckets) = projective.split_at_mut(self.output_slots());
        let mut identity = Identity::new();
        let mut slots = [const { TaskStorage::EMPTY }; 32];
        let mut run = MsmRun::new(self, input, &mut identity, &mut slots);
        let mut requests = [None; 32];
        loop {
            if let Some(result) = run.result() {
                return result;
            }
            let limit = if run.kind == WorkKind::Window && !self.options.streaming() {
                self.output_slots().min(32)
            } else {
                leases.get().min(32)
            };
            let count = run.ready(&mut requests[..limit]);
            let first = requests[0]
                .as_ref()
                .expect("structured run progress")
                .window;
            match run.kind {
                WorkKind::Prepare => {
                    let scalar_size = if matches!(input.scalars, Scalars::Prepared(_)) {
                        0
                    } else {
                        recode::CHUNK
                    };
                    let digit_size = if self.cached(input, run.geometry).is_some() {
                        0
                    } else {
                        recode::CHUNK * run.geometry.stride()
                    };
                    fn pieces<T>(slice: &mut [T], size: usize) -> impl Iterator<Item = &mut [T]> {
                        let mut rest = Some(slice);
                        core::iter::from_fn(move || {
                            let slice = rest.take().unwrap();
                            let n = size.min(slice.len());
                            let (head, tail) = slice.split_at_mut(n);
                            rest = Some(tail);
                            Some(head)
                        })
                    }
                    let leases = pieces(scalars, scalar_size)
                        .zip(pieces(digits, digit_size))
                        .skip(first)
                        .take(count)
                        .map(|(scalars, digits)| Lease {
                            records: &[],
                            digits: &[],
                            scratch: Scratch::new(
                                scalars,
                                digits,
                                &mut [],
                                &mut [],
                                &mut [],
                                &mut [],
                            ),
                            buckets: &mut [],
                            output: &mut [],
                            partials: &[],
                        });
                    dispatch(&mut run, &mut requests[..count], leases, executor);
                }
                WorkKind::Window if !self.options.streaming() => {
                    let mut claims = core::array::from_fn::<_, 32, _>(|_| None);
                    let mut receipts = core::array::from_fn::<_, 32, _>(|_| None);
                    for (claim, request) in claims.iter_mut().zip(&mut requests[..count]) {
                        let request = request.take().unwrap();
                        let kernel = run.kernel(request).expect("ready window");
                        *claim = Some(
                            run.frontier
                                .reserve(request.key, kernel)
                                .expect("owned window"),
                        );
                    }
                    Windows {
                        claims: &mut claims[..count],
                        outputs: &mut partials[first..first + count],
                        receipts: &mut receipts[..count],
                        records: scalars,
                        digits,
                        scratch: temporary.reborrow(),
                        work: self.temporary(),
                        leases: leases.get().min(count),
                    }
                    .execute(executor);
                    let mut error = None;
                    for receipt in &mut receipts[..count] {
                        error = error.or(run
                            .complete(receipt.take().unwrap())
                            .expect("window receipt")
                            .error);
                    }
                    assert!(error.is_none(), "validated input and complete resources");
                }
                WorkKind::Window | WorkKind::Collapse => {
                    let leases = partials
                        .iter_mut()
                        .zip(buckets.chunks_exact_mut(run.geometry.buckets()))
                        .skip(first)
                        .take(count)
                        .map(|(output, buckets)| Lease {
                            records: scalars,
                            digits,
                            scratch: empty(),
                            buckets,
                            output: core::slice::from_mut(output),
                            partials: &[],
                        });
                    dispatch(&mut run, &mut requests[..count], leases, executor);
                }
                WorkKind::Reduce => {
                    let lease = Lease {
                        records: &[],
                        digits: &[],
                        scratch: empty(),
                        buckets: &mut [],
                        output: &mut [],
                        partials,
                    };
                    dispatch(
                        &mut run,
                        &mut requests[..count],
                        core::iter::once(lease),
                        executor,
                    );
                }
            }
        }
    }
}
