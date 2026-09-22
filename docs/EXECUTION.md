# Scheduling bounded arithmetic work

Udon's run APIs let an application schedule MSMs, FFTs, their consumers, and
application kernels on one pool with a fixed storage provision. Completing a
task exposes its local successors. A worker and its returned scratch can serve
any compatible ready task, without changing arithmetic plans or redistributing
per-operation budgets. The target library remains `no_std`, allocation-free,
and safe Rust. The application supplies synchronization and storage.

The internal [mixed-driver fixture](../crates/udon/tests/support/mixed_run.rs)
combines two unequal MSMs, two unequal FFTs, application work, immediate consumers,
and round-challenge fences. It repeats eleven shrinking rounds without growing
its provision. It uses a private frontier for application work; downstream
schedulers supply their own application-task state. The
[execution tests](../crates/udon/tests/execution)
check independent arithmetic results, one-scratch progress, small dispatch
queues, scoped borrows, cancellation, and failure draining. The
[performance report](EXECUTION_PERFORMANCE.md) records measurements and limits.

## Four responsibilities

| Owner | Responsibility | Lifetime |
| --- | --- | --- |
| Arithmetic plan | Geometry, legal subdivisions, dependency rules, resource requirements | Reusable across invocations; immutable tables may outlive rounds |
| Operation run | Bound input, compact frontier, retained-state dependencies, completion | One invocation, through its last consumers |
| Resource provider | Initialized typed blocks and exclusive or shared leases | Provision may span rounds; scratch lease lasts one task |
| Application scheduler | Admission, ready-work selection, priorities, fairness, dispatch, wakeups | Shared by all work kinds |

[`exec::run`](../crates/udon/src/exec/run/mod.rs) contains the common task and
completion protocol. [`curve::msm::run`](../crates/udon/src/curve/msm/run.rs) and
[`fft::run`](../crates/udon/src/fft/run.rs) supply arithmetic plans and runs.
Plans resolve implementation choices from mathematical inputs, physical storage
layout, and `exec::ExecutionOptions`. Arithmetic runs use private frontiers over
caller-owned `TaskStorage`, with
public `Task` and `Completion` envelopes. Applications own the dependency and
admission policy for their kernels. No arithmetic task receives an executor or
waits for a child task.

## Dispatch and ownership

The coordinator performs this sequence:

1. Enumerate a bounded page of ready requests across admitted runs. A request
   describes logical ranges and a complete resource bundle.
2. Acquire the bundle without blocking, including an output slot and dispatch
   and completion capacity. Roll back partial acquisitions if any part fails.
   Leave that request ready and consider other compatible work.
3. Claim the request. Its identity and epoch are checked before acquisition.
   A successful claim produces a detached task owning the complete bundle.
4. Execute one bounded kernel on a scoped worker. Return its owned completion
   to the coordinator.
5. Publish completion, return actual leases, update accounting, and dispatch
   newly ready work. Release retained blocks only after their final consumers.

The application must treat resource acquisition and admission accounting as one
coordinator transaction. Counting compatible blocks cannot create references or
establish that a particular block contains a particular result.
Resource providers must supply the data and storage described by each request.
Kernels assert buffer requirements before writing. Matching logical slots to the
requested data is the provider's arithmetic-correctness contract.
Incorrect data does not permit a Rust memory-safety violation.

A task owns safe mutable slices or application lease guards. Its kernel may
borrow those guards temporarily, but the detached task does not borrow the run's
mutable coordinator state. This permits several tasks from one run to reside
in a scoped queue. Sendability follows the kernel and resource bundle, and
inputs need no `'static` lifetime. Scratch is never selected by thread identity;
nested execution cannot borrow a still-live scratch lease a second time.

FFT and MSM `Buffers` are transient kernel views. Their direct `Resources`
implementations allow local execution, but erased `ReadView` references do not
promise `Sync`. A movable owner can retain typed borrowed slices and create the
views on the worker in `Resources::buffers`. The small
[borrowed-owner checks](../crates/udon/tests/execution/borrowed.rs) demonstrate
both protocols with non-static storage and scoped threads.

The test [fragment provider](../crates/udon/tests/support/fft_run.rs) uses
preallocated `spin::RwLock` fragments and nonblocking acquisition. Shared read
views retain their guards; disjoint writes take exclusive guards. `spin` is a
development dependency only. The production protocol also accepts ordinary
borrowed slices and does not require locks, atomics, allocation, or a runtime.

### Failure and cancellation

Workers call `Task::execute` through a mutable reference. Before entering the
kernel, the task marks itself failed; only normal return changes that status to
success. A worker using unwinding catches it while retaining the task envelope,
then calls `finish` and returns the failed receipt. Calling `finish` before
execution cancels that task. Executing twice is rejected.

`Outcome::Success` means the kernel returned normally; its output can still be
an arithmetic `Err`. Publication inspects both the outcome and the output.
FFT kernels return unit; MSM kernels can reject invalid indices in produced
sources. Setup validation precedes driver writes; each incremental task asserts
its resource requirements before writing. A later task failure leaves earlier
writes in place.
An arithmetic error, failed execution, or cancellation poisons the run and
stops new claims. Other outstanding receipts remain drainable. The application
must join workers and drop actual guards before releasing their accounting or
reusing storage. Failed in-place data must be refilled. FFT arithmetic preserves
the loose field bound at every step, including on unwind, without a cleanup
pass. A failed task does not promise an unchanged or valid polynomial result.

Foreign or stale publication returns the intact receipt to its caller. Tickets
are not forgeable or cloneable. Dropping or forgetting a task does not publish
it: its frontier and accounting stay occupied. This may stall an abandoned run,
but memory safety does not depend on a destructor running. Abandon the run only
after all accessible detached tasks have been drained or ended. Fresh identity
storage cannot be rebound while accessible tickets still borrow it.

[`TaskError`](../crates/udon/src/exec/run/task.rs) reports a retained provision
exceeding the plan's workspace ceiling (`Storage`), unrepresentable storage sizes
or exhausted epoch identity arithmetic (`Overflow`), and invalid task or run
transitions. Plan compatibility and metadata capacity are caller contracts.
Transition errors do not report arithmetic kernel results.

## Bounded readiness and retained results

Each arithmetic run represents its task range with caller-owned ring metadata.
Its private frontier stores one state per live index even when a phase contains
millions of tasks. Ready request enumeration scans at most the fixed frontier
capacity. Successive dependency phases reuse that storage with checked epochs,
rejecting old keys.

Completion can arrive out of order. Consecutive completed indices retire in
order before their slots are reused. The window includes space for the earliest
missing result, so later partials cannot occupy all retained slots and prevent
its production. This introduces bounded head-of-line blocking inside a run;
other runs remain selectable. A scheduler must page through every relevant slot
and ready index before deciding that all work is resource-blocked. Reset cursors
after publication, since it may change epochs and expose successors.

`MsmRun` retains one term chunk. `ParallelMsmRun<SLOTS>` adds a fixed number of
independent chunk slots, each with its own preparation and window dependencies.
The slot count bounds retained storage; it does not assign workers. There is
one logical output per chunk/window partial, followed by a defined reduction.
Results are not replicated per worker. Streaming MSM instead retains each
window's buckets across bounded deposits and collapses them at the end.

`FftRun` retains its own stage and buffer barriers. Local transforms, paired
tiles, permutation copies, gathers, and final scaling are bounded tasks. Udon
selects column panel dimensions and their retained count from the task budget
and workspace ceiling. Requests report the required ranges. Natural-order permutation
either uses a retained snapshot or exclusively leases a contiguous bank for
bounded index swaps. The latter saves snapshot storage but serializes that
run's permutation tasks.

`StorageLayout` declares the values bank's physical fragmentation and whether
whole-bank leases are possible. It constrains which initialization and
permutation schedules Udon can select; applications need not choose those
schedules themselves.

`ExpansionRun<SLOTS>` publishes each completed residue independently and reuses
its transform slot. When coefficients occupy output block zero, that block is
transformed last so other residue readers cannot lose their input.
`InterpolationRun<CLASSES>` exposes independent inverse transforms and bounded
coefficient additions. A completed lift can reduce once the output inverse is
ready, and its last read releases that lift independently of unrelated inverses.
Same-domain consuming interpolation merges evaluations before their
representative inverse, including groups smaller than the output. These are
data dependencies, as are round-challenge fences.

### Inputs published by application tasks

`ProducedInput` binds base storage and a logical term count without borrowing
unfinished scalar or index rows. `MsmPlan::for_produced` resolves arithmetic
from that descriptor, resource constraints, and the maximum consecutive source
fragment the provider can lease. `MsmRun::new_produced` and
`new_produced_partition` expose arithmetic requests normally. The provider
declines a claim until the request's whole source range is available, then
leases immutable fragments through `Resources::with_source` and `SourceBuffers`.
This lets a field kernel publish one fragment and release its recoding tasks
while other field kernels still own disjoint writable fragments.

Source views use request-local indices: element zero corresponds to
`Request::offset`, and preparation requires `Request::terms` raw field scalars.
Indexed preparation and windows also require that many base indices. Kernels
assert resource requirements and validate indices before writing; matching each
fragment to its logical range remains the provider's responsibility. Keep
published values unchanged through their last consumer. A source may cross
several fragments; `ReadView::contiguous_prefix` exposes the first contiguous
part of a range so the kernel can recode each part directly into retained scalar
and digit banks.
Preparation writes those retained banks in place. Publication only transfers
their ownership; it need not copy the prepared data under a scheduler lock.

`new_partition` binds a range of an existing borrowed input, with
global scalar and base-index offsets. Nonempty partial ranges discard a
whole-input digit cache while preserving reusable scalar records; a plan that
requires that cache rejects those ranges. Empty ranges complete with the
identity. Completed produced runs can rebind to a new input and range; stale
requests still fail their epoch check.

## Admission with a progress reservation

Count three storage lifetimes separately: persistent plans/tables,
operation-retained intermediates, and task-leased scratch. A task's result can
move from an exclusive write lease to retained shared input; completion alone
does not free it while consumers remain.

For MSM, `MsmPlan::retained_for_slots` bounds the retained buffers for the chosen
number of active chunks, while `temporary` bounds one executing task's scratch.
Count that temporary bundle once per simultaneous lease. The contiguous driver
combines one retained chunk with temporary bundles selected from its task
budget in `requirements`. Extra retained slots must fit the plan's workspace
ceiling together with those temporary bundles. FFT plans likewise count
retained snapshots and expansion coefficient workspace. Add
metadata, queues, buffer alignment, and unused provider capacity when deciding
what the application can admit. See
[`MsmPlan`](../crates/udon/src/curve/msm/run.rs) for the individual query contracts.

The [test fixture's admission policy](../crates/udon/tests/support/admission.rs)
charges every provisioned block, including idle capacity,
padding, lease metadata, retained handoff banks, run/frontier state, envelopes,
and bounded queues. Classes distinguish exact types and capacities: bytes of
`Fp` storage cannot satisfy an `Fq` lease. Inputs, external final outputs, and
persistent immutable plans/tables may be accounted separately. Runtime stacks,
thread-pool internals, allocator overhead, and process RSS are outside this
working-storage ceiling. The reference fixture charges its intermediate output
banks because they feed later work.

For every admitted pipeline segment `i`, declare:

- `R_i`: the maximum simultaneously retained blocks through a release frontier,
  including all intermediate consumers in that segment.
- `T_i`: the componentwise maximum complete temporary bundle of any one task
  needed to reach that frontier.

That fixture admits another segment only when, componentwise,

```text
sum(R_i) + max(T_i) <= arena capacity
```

This is a conservative sufficient condition. Existing tasks may temporarily
occupy the spare capacity. After their bounded kernels complete and their
leases return, every admitted segment's next declared bundle fits even if all
segments hold their full retained maxima. New retained outputs are bounded by
their segment's reservation. Admission therefore cannot consume the space
needed to execute the next task through that release frontier.

The progress argument additionally requires truthful profiles, finite kernels,
acyclic internal dependencies, eventual external fences, and fair selection.
Reserve a consumer in the producer's segment when that producer retains data
needed to admit the consumer. Admitting producers alone can deadlock: all memory
fills with intermediates while their consumers cannot enter. Reserving a byte
total without compatible block types has the same problem. Workers holding one
lease while waiting for another can deadlock; workers repeatedly retrying one
blocked request can livelock or starve compatible work. The complete-bundle
transaction and fair enumeration avoid those scheduler-induced failures.

Applications may account for additional capacity classes or prioritize work by
measured arithmetic and traffic costs. These scheduling policies belong to the
application; Udon requests describe logical ranges and required resources.

## Arithmetic subdivisions

Udon selects task size as part of the arithmetic plan. Increasing workers or scratch leases
does not alter an existing plan's decomposition. Smaller grains expose more
parallelism and faster interleaving, but they add real work:

| Decomposition | Arithmetic cost | Retained or temporary cost |
| --- | --- | --- |
| Independent MSM term chunks | Each chunk repeats `W` bucket collapses and window recombination | One prepared chunk and `W` partials per live chunk slot, plus shared task scratch |
| Streaming MSM deposits | Collapses each window once; deposits remain bounded by the term grain | All `W` bucket sets retained through the final collapse |
| FFT tile stages | Same butterflies; extra dispatch and stage publication as tiles shrink | Two exclusive tiles for a paired task |
| FFT column panels | Same transform work plus gather/scatter traffic | `columns * fragments()` fields per retained panel |
| Fragmented FFT permutation | Bounded copies and gathers | A full `N`-field snapshot when order conversion requires it |

For `n` terms, term grain `g`, `W` windows and `B` buckets per window, independent
chunks perform about `ceil(n/g) * W` collapses instead of `W`. Their extra
collapse work scales as `(ceil(n/g) - 1) * W * B` group operations, plus partial
recombination; the exact constant depends on the bucket backend. Streaming
avoids this repetition at a retained floor of roughly `W * B` projective points.
For a radix-2 FFT with `N` fields and tile `t`, the local stage has `ceil(N/t)`
tasks and each remaining cross-tile stage has about `N/(2t)` paired tasks.
Traffic and publication costs can dominate the arithmetic saved by concurrency.

The implementation weighs these costs against scalar shape, source availability,
physical fragmentation, scratch capacity, and task allowance. A source fragment
or buffer limit can force a smaller subdivision; the caller states that limit
and Udon resolves the arithmetic. The performance report records the evidence
behind current choices. Private differential tests retain forced variants to
check and measure alternative implementations.

## Synchronous execution and integration

`MsmPlan`, `BatchPlan`, `FftPlan`, `ExpansionPlan`, and `InterpolationPlan`
provide synchronous execution over caller-owned storage and `Executor::join`.
These drivers divide the plan's total task allowance and reuse the bounded
arithmetic kernels. The MSM driver owns disjoint scratch bundles and can execute
successive ready windows with one bundle inside a joined branch. Preparation
and streaming deposits can use the full structured task allowance without a
temporary window scratch bundle.

The synchronous return contract waits for the complete operation. Use the run
interfaces for application-wide priorities, immediate consumer dispatch, or one
shared scratch bank across heterogeneous operations. `TaskBudget` helpers remain
available for structured partitioning. Admission and queue policy belong to the
application in either case.

An application can migrate one operation at a time: keep its existing immutable
tables, preplan the round shapes, provision retained banks and typed task
scratch, and bind runs to caller-owned identities and task metadata. Then route
their requests and application kernels through one coordinator. Keep each scratch
provider and admission segment alive until its last consumer returns. Rebind
completed MSM and FFT run metadata for the next preplanned shape; no worker
change requires plan reconstruction. The reference pool and providers are test
adapters illustrating this integration, not an exported application runtime.
