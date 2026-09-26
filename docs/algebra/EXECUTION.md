# Composing and scheduling arithmetic

[Algebra reference](../ALGEBRA.md). The
[`exec`](../../crates/udon/src/exec/mod.rs),
[`msm::execution`](../../crates/udon/src/msm/execution/mod.rs), and
[`fft::execution`](../../crates/udon/src/fft/execution/mod.rs) APIs preserve the arithmetic
described in the other chapters while changing how its pieces become
available. Choose a synchronous call when its result is the next useful
boundary; choose an incremental run when independent work or downstream
consumers can proceed between those boundaries. Physical fragments are
storage boundaries, not automatically independent mathematical results.

## Scoped operations

### `ExecutionOptions`

`ExecutionOptions` constrains one operation without changing its sum or
polynomial. Start with `DEFAULT`, select a concurrency allowance with
`with_task_budget`, and optionally bound arithmetic workspace with
`with_memory_limit`; `task_budget` and `memory_limit` expose those choices.
Use a common limit when several algebraically equivalent plans must fit
the same workspace. The ceiling covers used arithmetic scratch and
retained intermediates, while inputs, outputs, persistent preparation,
and scheduling storage remain separately owned. Giving concurrent
operations the same options gives each its own allowance; it does not
bound their combined storage.

### `TaskBudget`

`TaskBudget::SERIAL` means one work partition; `new` constructs another
nonzero allowance and `get` reads it. Use `split_at` to assign unequal
shares to two independent calculations, or `partition` to choose a
number of outer jobs and an equal nested allowance for each. For
example, several independent polynomial commitments can split the
outer allowance while each MSM uses its own share internally. Pass
that nested share onward instead of restarting the full allowance at
every level. A budget describes permitted decomposition, not reserved
threads or an application-wide semaphore.

### `Executor` and `SerialExecutor`

`Executor::join` runs two scoped jobs and returns their results in
left/right order. It expresses independence: disjoint partial MSMs
can run before their points are added, and independent transforms can
run before their evaluation products are formed. Both jobs finish or
unwind before the call ends, so they can borrow caller-owned buffers.
`SerialExecutor` provides the same composition synchronously. A pool
adapter must support nested joins even with one worker; queuing children
and blocking every parent worker does not satisfy that contract.

### `for_each_mut` and `for_each_chunk_mut`

These apply an independent operation to every item or consecutive
nonempty-sized chunk, passing its logical index and nested task budget.
Use items that own separate polynomial buffers, point batches, or
partial `ProductSum` accumulators; after the call, combine results
according to the required algebra. `for_each_chunk_mut`'s index names
the chunk starting at `index*chunk_len`, including a possibly shorter
last chunk. Callback order is unspecified, so a recurrence such as
running fraction prefixes cannot be parallelized merely by putting
its original sequential loop inside these callbacks.

### `MsmPlan`

`MsmPlan` resolves how to evaluate one sum `sum_i [a_i]G_i` under
`ExecutionOptions`. Use `new(terms, options)` for changing scalar rows
and ordinary bases of the same length, `for_input` when retained scalar
preparation or compact bases are part of the invariant, and
`for_produced` when scalar or index fragments will arrive later.
`execute` performs the complete sum with an `Executor`; its input must
still satisfy the facts used in planning. A plan does not retain the
input borrow, but a plan relying on short prepared scalars, a digit
cache, or compact bases cannot silently discard that requirement.

`requirements` sizes the synchronous workspace. For incremental
execution, `retained` sizes one active chunk, `retained_for_slots`
sizes independent retained chunks, and `temporary` sizes one executing
task's bundle. `grain`, `preparation_terms`, and `output_slots` describe
the resulting subdivision. These counts support different ways to
accumulate the same group sum; they do not add algebraic terms or
assign a chunk to a particular worker.

### `BatchPlan`, `JobStorage`, and `WorkerStorage`

`BatchPlan` evaluates several independent `Input`s and writes one
projective result per input, in input order. Choose it when the
application needs a vector of MSM results whose lengths, bases, or
scalars may differ. This is distinct from combining all terms into
one MSM, and from `SharedScalarInput`, which states that every output
uses the same scalar vector. If only the sum of the results is needed,
linearity may instead permit combining their inputs or folding rows
that share a basis.

`storage_len` sizes metadata initialized with `JobStorage::EMPTY` and
`WorkerStorage::EMPTY`; `new` borrows the input list and that metadata.
`requirements` and `temporary_bytes` describe arithmetic workspace,
and `execute` can reuse the retained schedule with fresh output and
scratch buffers. The plan retains these particular scalar rows and
bases. Changing their values requires constructing new inputs and a
new plan; retaining a `Selection` is the appropriate way to keep a
base mapping across changing rows.

## Detached task protocol

### `Identity`, `TaskStorage`, and `TaskKey`

The shared [`exec::execution`](../../crates/udon/src/exec/execution/mod.rs)
protocol distinguishes a task's invocation and dependency epoch from
its arithmetic index. Supply a separate `Identity::new()` for each
simultaneously bound frontier and initialize its bounded metadata with
`TaskStorage::EMPTY`. A `TaskKey` belongs to that frontier and epoch;
`index()` supports paging ready work and routing completions. Metadata
can be reused after completion without making old keys valid again.
Use these identities to keep partial sums or transform stages attached
to the invocation that produced them, even when a scheduler interleaves
several operations over reusable storage.

### `Kernel`, `Task`, and `Completion`

A `Kernel::execute` performs one bounded work unit using an already
acquired resource bundle and returns its associated `Output`.
Claiming a ready Udon request produces a `Task` that owns the kernel
and that bundle independently of the coordinator. Call
`Task::execute` once, then consume it with `finish` and publish the
resulting `Completion` to the originating run. `Completion::key` routes
the receipt and `outcome` reports how execution ended. The run, rather
than a worker's return alone, decides which successors or mathematical
results are now ready.

Keep the task outside an unwind-catching closure so its resources can
be returned after failure. A kernel must finish without waiting for
additional scarce resources; dependencies belong in coordinator state.
This permits arithmetic and application work to share a scheduler:
acquire complete bundles before dispatch, then publish results that
enable subsequent work. Dropping a task does not publish its receipt
or advance the operation.

### `Outcome`, `TaskError`, and `PublishError`

`Outcome::Success` means the kernel returned normally, `Failed` means
execution began but unwound, and `Cancelled` means the task was returned
without execution. A normal return may itself contain an arithmetic
error, so examine the operation's publication too. `TaskError` reports
storage or lifecycle problems such as stale keys, repeated claims,
incomplete dependencies, and exhausted identity arithmetic; it does
not assert that a polynomial identity or a divisibility claim holds.
`PublishError` returns a rejected receipt intact for correct routing
or draining. After failure, drain outstanding tasks and refill any
partly overwritten in-place operands before treating them as new
mathematical inputs.

### `ReadView`

`ReadView<T>` presents one logical vector across shared fragments.
`len`, `is_empty`, and `get` describe that vector; `contiguous` may
expose an exact requested range, and `contiguous_prefix` may expose
only its initial consecutive portion. Use it when completed producer
fragments jointly form one scalar row, coefficient bank, or array of
partial sums. Its indexing must preserve the original mathematical
order across fragment boundaries. Fragmentation neither permutes
coefficients nor changes an FFT layout; slices and arrays already
provide the ordinary contiguous implementations.

## Incremental group sums

### `MsmRun`

`MsmRun::new` binds one planned input. `ready` and `ready_from` expose
bounded requests; `try_claim` detaches one only if its complete bundle
is available; `complete` publishes a receipt and exposes successors.
`result` becomes the final projective sum, with identity for an empty
input. Use this protocol when the application's scheduler should
interleave MSM work with transforms, other sums, or consumers of earlier
results. A pending acquisition can be declined without occupying a
worker. `is_failed` and `inflight` distinguish failure from outstanding
receipts that still need returning.

`new_partition` selects a range of terms from an existing input, using
the full input's plan and index mapping. The identity
`sum_all [a_i]G_i = sum_partitions (sum_in_partition [a_i]G_i)` lets
the application reduce disjoint partition results itself. Cover each
intended term exactly once. Partial ranges can reuse scalar records
but discard a whole-input digit cache, so a plan requiring that cache
cannot bind them. `rebind` reuses successfully completed metadata for
another compatible invocation; storage reuse must also wait for the
old result's consumers.

### `ParallelMsmRun` and `ChunkRequest`

`ParallelMsmRun<SLOTS>` retains several independent term chunks of one
MSM, each with its own preparation and window dependencies. It still
returns one group sum. Use it when several chunks should make progress
independently while bounding the number of retained partial results;
`SLOTS` counts retained chunks, not workers. It requires a plan that
supports independent chunks. A streaming plan that accumulates into
retained window buckets uses `MsmRun` instead.

`new`, `ready`, `ready_slot_from`, `try_claim`, and `complete` follow
the common protocol. `ChunkRequest` supplies both the slot and its
underlying request; publish to that slot and release returned resources
before leasing its next chunk. `result` is available after every chunk
retires, even though individual chunks can finish out of order.
`is_failed`, `inflight`, and `rebind` provide lifecycle inspection and
completed-metadata reuse. Visit all slots when searching for ready
work; a blocked chunk need not block another chunk's arithmetic.

### `ProducedInput` and `SourceBuffers`

`ProducedInput::dense` or `indexed` fixes the base bank and term count
without borrowing unfinished scalar or index arrays. `len` and
`is_empty` describe the eventual sum. Use it when a field computation
will produce coefficients for an MSM in separately publishable
fragments. Plan with `MsmPlan::for_produced`, then bind with
`MsmRun::new_produced` or `new_produced_partition`. The provider declines
acquisition until every source element needed by a request is available;
`SourceBuffers` supplies scalar and optional index `ReadView`s local
to that request's term range, starting at logical index zero.

The algebra is the same dense or indexed MSM; availability is the
additional fact. Published sources must remain unchanged through
their consumers' last reads, and the plan's declared source fragment
size must match what the provider can lease. An intermediate FFT stage
is not a finished coefficient fragment merely because a worker has
stopped writing it. `rebind_produced` and `rebind_produced_partition`
reuse completed metadata for subsequent produced rows while retaining
stale-key detection.

### MSM `Request`, `Resources`, and `Buffers`

An MSM `Request` identifies a term range, retained scalar and digit
ranges, optional buckets and partial-result slots, source reads, and
temporary `Requirements`. Implement `Resources::buffers` to borrow
those exact views from an owned bundle; `with_source` additionally
supplies produced scalars and indices. `Buffers` holds prepared
records, recoding digits, scratch, retained buckets, an output slot,
and partial sums. This separation follows the computation: preparation
describes coefficients, window tasks form group contributions, and
reduction adds those contributions. Correct capacities alone do not
prove that a buffer contains the coefficient row or partial sum named
by the request; preserve that association in the provider.

### `MsmKernel`, `MsmOutput`, and MSM `Published`

`MsmKernel` is the bounded arithmetic returned by a successful MSM
claim; its opaque `MsmOutput` is interpreted when the receipt is
published. `Published` returns the owned `resources`, an optional
arithmetic `error`, the execution `outcome`, and a `result` only when
this publication finishes the whole MSM. Use that final-result event
to enable a consumer of the group sum. A completed window or chunk
is generally only a contribution, not the final point. Returned
resources can be released or handed off according to their remaining
consumers rather than being tied to the worker that executed them.

## Incremental polynomial operations

### `FftRun`

`FftRun::new` binds an [`FftPlan`](FFT.md#executionfftplan) to frontier
metadata; its `product` flag requests the plan's terminal elementwise
factor in physical output order. `ready`, `ready_from`, `try_claim`,
and `complete` expose and publish bounded stages until `is_complete`
establishes that the requested transform is available. Use it to
interleave independent transforms or other arithmetic with the stages
of a larger transform. Individual butterfly completions do not imply
that their values are finished evaluations or coefficients.

`is_failed` and `inflight` support failure and draining; `rebind`
reuses completed metadata for another transform. As with synchronous
execution, direction, coset, order, and normalization come from the
plan. Scheduling stages differently changes none of those meanings.
Reuse the field banks only after all consumers of the previous
polynomial representation have finished.

### FFT `Bank`, `Request`, `Resources`, and `Buffers`

FFT requests name `Bank::Input`, `Values`, or `Snapshot`, together
with an exclusive write range, an optional paired write range, a
shared read range, and a terminal factor range. A provider implements
`Resources::buffers` to supply `Buffers` with the corresponding
`values`, `pair`, `source`, and `factor` views. Use the named ranges,
not worker identities, to associate storage with the transform.
Paired writes express a butterfly's joint update; snapshots preserve
values during order conversion; factor reads express the requested
post-transform multiplication. The same logical operation can use
contiguous storage or safe fragments without changing node order.

### `FftKernel` and FFT `Published`

`FftKernel` executes the bounded task selected by a claim. Publishing
its receipt returns `Published` with the owned `resources`, optional
transition `error`, execution `outcome`, and a `complete` flag. Only
successful completion of the operation makes the full planned
transform ready for consumers. Use this distinction when a subsequent
MSM or pointwise product expects finished coefficients or evaluations;
`Outcome::Success` for one task establishes only that its own kernel
returned normally. Release returned leases before acquiring successor
bundles from the same storage.

### `ExpansionRun`, `ExpansionRequest`, and `ExpansionBank`

`ExpansionRun<SLOTS>` executes an
[`ExpansionPlan`](FFT.md#expansionstorage-and-executionexpansionplan) through
a bounded set of inverse and residue-transform slots. Each completed
residue represents the original bounded-degree polynomial on one specified
target coset.
Use it when consumers can process residues separately rather than
waiting for the entire larger domain. With output reuse, the block
holding coefficients remains shared until the other residues have
read it, then becomes the last residue to be transformed.

`new`, `ready_slot_from`, `try_claim`, and `complete` expose that
progress; `is_complete`, `is_failed`, and `inflight` inspect it.
`ExpansionRequest` maps a slot's ordinary FFT accesses to
`ExpansionBank::Input`, `Coefficients`, or `Output(block)`.
`ExpansionPublished` contains the base task publication, the newly
ready physical `residue` when present, and whole-operation `complete`.
Respect the plan's residue and inner element order when identifying
the published nodes. Slots bound retained state; each completed
residue can enable its own consumer independently of later residues.

### `InterpolationRun`

`InterpolationRun<CLASSES>` incrementally computes the polynomial sum
described by
[`execution::InterpolationPlan`](FFT.md#executioninterpolationplan-for-a-sum-of-classes).
`new` binds transform and addition frontiers per class.
`ready_transform_from` and `try_claim_transform` expose inverses;
`ready_addition_from` and `try_claim_addition` expose merges and
coefficient additions. With consuming interpolation, compatible
evaluation vectors can be added before their shared inverse;
different-domain classes are interpolated before their coefficients
join the sum. This is linearity with explicit domain matching.

`complete_transform` and `complete_addition` update dependencies
without imposing a barrier across unrelated inverses. `state` reports
each `ClassState`; consumed classes no longer promise a retained
polynomial, and output class zero becomes coefficients only when its
whole sum is complete. `is_complete`, `is_failed`, and `inflight`
distinguish that final polynomial from partial progress. Use this run
when separately arriving classes can be merged or released while
other classes are still being transformed.

### `AdditionRequest`, `AdditionKernel`, and `InterpolationPublished`

`AdditionRequest` names a target class, shared source range, exclusive
destination range, task key, and whether the addition acts on
evaluations or coefficients. `AdditionKernel` uses the same FFT
`Resources` interface to perform that bounded addition. Evaluation
merges require the plan's matching domains and orders; coefficient
additions align powers of `X`, with shorter polynomials zero-extended.
The two kinds of addition express the same linearity at different
points in the interpolation pipeline.

`InterpolationPublished` returns resources and status, plus a
`coefficients` event when a class inverse finishes, a `released`
event after a lift's last read by this interpolation, and `complete`
when output class zero holds the final sum. A representative's
coefficients can already include consumed same-domain classes, and
class zero can still be awaiting additions. Recycle a released bank
only after its other application consumers also finish; an arithmetic
readiness event and the end of every consumer's lifetime are different
conditions. The [execution guide](../EXECUTION.md) develops the storage
and scheduling contracts in more detail.
