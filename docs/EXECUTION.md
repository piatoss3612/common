# Scheduling bounded arithmetic work

Udon's run APIs let an application schedule MSMs, FFTs, their consumers, and
application kernels on one pool with a fixed storage provision. Completing a
task exposes its local successors. A worker and its returned scratch can serve
any compatible ready task, without changing arithmetic plans or redistributing
per-operation budgets. The target library remains `no_std`, allocation-free,
and safe Rust. The application supplies synchronization and storage.

The executable [mixed driver](../crates/udon/tests/support/mixed_run.rs) combines
two unequal MSMs, two unequal FFTs, application work, immediate result consumers,
and round-challenge fences. It repeats eleven shrinking rounds without growing
its provision. The [execution tests](../crates/udon/tests/execution/main.rs)
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
admission protocol. [`curve::msm::run`](../crates/udon/src/curve/msm/run.rs) and
[`fft::run`](../crates/udon/src/fft/run.rs) supply arithmetic plans and runs.
Application work implements `Kernel<R>` for an owned resource bundle `R` and
uses the same `Frontier`, `Task`, and `Completion` types. No task receives an
executor or waits for a child task.

## Dispatch and ownership

The coordinator performs this sequence:

1. Enumerate a bounded page of ready requests across admitted runs. A request
   describes its work kind, logical ranges, and complete resource bundle.
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
coordinator transaction. `Admission` counts compatible blocks; it cannot create
references or establish that a particular block contains a particular result.
Arithmetic resource traits validate lengths before writing. Matching logical
slots to the requested data is the provider's arithmetic-correctness contract.
Incorrect data does not permit a Rust memory-safety violation.

A task owns safe mutable slices or application lease guards. Its kernel may
borrow those guards temporarily, but the detached task does not borrow the run's
mutable coordinator state. This permits several tasks from one run to reside
in a scoped queue. Sendability follows the kernel and resource bundle, and
inputs need no `'static` lifetime. Scratch is never selected by thread identity;
nested execution cannot borrow a still-live scratch lease a second time.

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

Failed publication poisons the run and stops new claims. Other outstanding
receipts remain drainable. The application must join workers and drop actual
guards before releasing their accounting or reusing storage. Failed in-place
data must be refilled. FFT tasks restore canonical field representation on
unwind; they do not promise an unchanged or valid polynomial result.

Foreign or stale publication returns the intact receipt to its caller. Tickets
are not forgeable or cloneable. Dropping or forgetting a task does not publish
it: its frontier and accounting stay occupied. This may stall an abandoned run,
but memory safety does not depend on a destructor running. Abandon the run only
after all accessible detached tasks have been drained or ended. Fresh identity
storage cannot be rebound while accessible tickets still borrow it.

## Bounded readiness and retained results

`Frontier` represents a task range with caller-owned ring metadata. It exposes
only the next `capacity()` indices and stores one state per live index, even
when the phase has millions of tasks. Ready ranges encode consecutive indices;
claimed or completed indices split them. Enumeration scans at most the fixed
frontier capacity. Successive dependency phases reuse that storage with checked
epochs, rejecting old keys.

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
tiles, permutation copies, gathers, and final scaling are bounded tasks. Optional
column panels have a declared size of `columns * fragments()` fields per task
and a separately configured retained panel count. Natural-order permutation
either uses a retained snapshot or exclusively leases a contiguous bank for
bounded index swaps. The latter saves snapshot storage but serializes that
run's permutation tasks.

Separate-input initialization normally gathers into disjoint destination tiles.
Explicit scatter initialization instead reads consecutive input tiles while
exclusively leasing the contiguous destination bank. Its work remains bounded
by the tile size, but those initialization tasks serialize within that run.

`ExpansionRun<SLOTS>` publishes each completed residue independently and reuses
its transform slot. When coefficients occupy output block zero, that block is
transformed last so other residue readers cannot lose their input.
`InterpolationRun<CLASSES>` exposes independent inverse transforms and bounded
coefficient additions. A completed lift can reduce once the output inverse is
ready, and its last read releases that lift independently of unrelated inverses.
Same-domain consuming interpolation merges evaluations before the output
inverse. These are real data dependencies, as are round-challenge fences.

## Admission with a progress reservation

Count three storage lifetimes separately: persistent plans/tables,
operation-retained intermediates, and task-leased scratch. A task's result can
move from an exclusive write lease to retained shared input; completion alone
does not free it while consumers remain.

`ArenaLayout<N>` charges every provisioned block, including idle capacity,
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

`Admission` admits another segment only when, componentwise,

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

The implementation accepts additional application-defined capacity classes.
For example, bandwidth-heavy kernels can require a bandwidth token, and a
cache-sensitive batch can reserve a limited number of cache tokens. Request
`WorkEstimate` values supply arithmetic, traffic, and active-set hints. These
are estimates, not measured hardware reservations. The scheduler chooses and
calibrates such policies; the default driver does not enforce bandwidth limits.

## Choosing arithmetic grain

Task size is part of the arithmetic plan. Increasing workers or scratch leases
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

The default MSM grain is capped at 8,192 terms. Automatic full-width geometry
uses width 11 at grains of at least 4,096. Small geometries stay fused. These
choices reflect the measured cases in the performance report, not a universal
minimum grain. Explicit widths, grains, accumulation modes, and FFT columns
remain available for application measurements. Algorithm selection should be
made with the retained and temporary provision together; oversubscribing a
smaller, more expensive kernel need not improve throughput.

## Synchronous compatibility and migration

Existing synchronous signatures and caller-owned storage remain available.
MSM execution and FFT transforms use the same bounded kernels and run state
through structured drivers over `Executor::join`. The MSM driver owns disjoint
scratch bundles explicitly and can execute successive ready windows with a
bundle inside a joined branch. It publishes that group's receipts after join.
Preparation and streaming deposits can use the full structured task allowance;
their retained buffers do not require a temporary window scratch bundle.
Setting `with_memory_limit(usize::MAX)` no longer selects a separate MSM queue
policy or duplicates logical window results per worker.

The existing synchronous batch and interpolation helpers still partition their
borrowed scratch and nested task allowances; their structured return contract
cannot expose a partially completed batch to its caller. Use the run interfaces
for application-wide priorities, immediate consumer dispatch, or one shared
scratch bank across heterogeneous operations. Their frontiers and provision
replace static inner budget shares. The old `TaskBudget` helpers remain useful
for callers deliberately choosing structured partitioning.

An application can migrate one operation at a time: keep its existing immutable
tables, preplan the round shapes, provision retained banks and typed task
scratch, and bind runs to caller-owned identities/frontiers. Then route their
requests and application kernels through one coordinator. Keep each scratch
provider and admission segment alive until its last consumer returns. Rebind
completed MSM and FFT run metadata for the next preplanned shape; no worker
change requires plan reconstruction. The reference pool and providers are test
adapters illustrating this integration, not an exported application runtime.
