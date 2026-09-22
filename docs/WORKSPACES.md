# Owning arithmetic workspaces

Udon borrows storage and execution resources from its caller. An owning
workspace can retain allocations across operations while lending the slices
each operation needs. Keep immutable tables separate from mutable buffers so
independent workspaces can share them.

The [workspace tests](../crates/udon/tests/workspaces/main.rs) demonstrate these
relationships with ordinary `Vec` storage and a caller-selected Rayon pool.
The examples use deterministic inputs to check arithmetic and capacity reuse;
allocation policy and application-wide resource limits belong to the caller.

For scheduling across operations with a shared scratch provision, use the
[incremental run protocol](EXECUTION.md). Its plans resolve arithmetic from
resource constraints, and its task leases let returned scratch serve any
compatible ready operation. The structured workspace examples below retain
their explicit nested budget and disjoint-buffer policies.

## Scoped execution

The [Rayon adapter](../crates/udon/tests/support/workspaces/executor.rs) borrows
a pool and enters it with `ThreadPool::install` before each `rayon::join`.
This selects the intended pool even when a call originates on another pool's
worker. The adapter supports borrowed results and nested joins with one worker.
The [`Executor` contract](../crates/udon/src/exec.rs) defines completion and
panic handling; its tests check that successful results are dropped when another
branch panics.

Independent operations with separate scratch can each use the full task budget
on a bounded, cooperative pool. The example `with_side_work` passes that budget
to both an MSM batch and fixed-base side work; with one task, it runs both
branches through `SerialExecutor`. The pool bounds executing workers, while
each operation plans its own scratch. Account for the combined storage of all
simultaneous operations.

Use [`for_each_task_mut`](../crates/udon/src/exec.rs) to expose independent buffer
owners or borrowed tiles as separate jobs without dividing their nested budgets.
When simultaneous operations must fit one combined task allowance, split it
with `TaskBudget::split_at` or use `for_each_mut`. The polynomial example passes
the inner allowance supplied by `for_each_mut` into each FFT operation. See
[scratch and execution](FFT.md#scratch-and-execution) for that nested budget
contract.

Applications decide how independent requests share a pool and which context,
such as allocation attribution, must be installed around each job. A task
budget divides work within an operation; it does not cap all concurrent requests.

## Owning an MSM workspace

The [MSM workspace](../crates/udon/tests/support/workspaces/msm.rs) holds
initialized scratch and plan metadata. Its `prepare` method accepts
`exec::ExecutionOptions`, sizes metadata, and constructs a `msm::run::BatchPlan`,
then grows scratch to that plan's requirements. The ceiling covers the used
arithmetic workspace; metadata and excess workspace capacity are separate. The
[grouped-job guide](CURVES.md#grouped-jobs-and-other-work) explains planning and
how long plans borrow scalar rows.

The returned run borrows inputs, metadata and scratch. The owner stores no
references into itself. Dropping the run releases its borrows so the owner can
prepare another batch. The run exposes scratch as slices, so workspace growth
stays in preparation. The example uses `Vec::resize`; callers can choose fallible
growth or another allocation policy at this boundary.

Retained capacity can exceed the current plan's requirements. Warm the complete
sequence of shapes and options before expecting capacities to stabilize: a
smaller batch can select scratch geometry that a larger batch did not need.
The tests repeat eleven rounds of halving input lengths and check that the
second sequence needs no capacity growth. They report required temporary bytes
separately from retained vector capacity. The
[memory limit documentation](CURVES.md#sizing-and-reusing-scratch) defines which
storage the planner accounts for.

## Polynomial storage and liveness

The [FFT workspace](../crates/udon/tests/support/workspaces/fft.rs) retains
coefficients between expansion and product operations. Table owners lend plans
after preparation, and sequential operations reuse one scratch buffer sized by
the maximum of their requirement queries. Simultaneous workspaces need disjoint
mutable storage.

The example expands a base domain of `N = 2048` rows to `8N` rows with the
order-three coset shift
[`PastaField::ZETA`](../crates/udon/src/field/parameters.rs). Its storage choices
are:

| Buffer | Fields | Lifetime and copies |
| --- | --- | --- |
| Base evaluations | `N` | Borrowed input is copied into the coefficient workspace before the inverse |
| Retained coefficients | `N` | Reused for another expansion and a prefix transform |
| Expanded evaluations | `8N` | Borrowed by the product operation |
| Product output | `8N` | Separate from the borrowed expanded factor |
| Four class buffers | `N + 2N + 4N + 8N` | One allocation split into disjoint classes and overwritten by interpolation |
| Scratch | Maximum queried requirement | Reused after each scoped operation finishes |
| Tables | Queried per domain | Shared immutably between independent workspaces |

At 32 bytes per field, the retained coefficients, expansion, product and class
buffers total 2 MiB per workspace, excluding scratch, tables, input and excess
capacity. These sizes describe the example's chosen lifetimes. Applications
can release or reuse buffers once their consumers finish; a borrowed residue
chunk keeps its entire contiguous allocation alive.

The retained inverse uses `InverseScale::Unscaled`. Carry its `CoefficientView`'s
normalization factor into later plans, as described under
[residue expansion](FFT.md#residue-expansion-and-layouts). Borrowing the view
keeps the coefficient buffer live while output and scratch remain reusable.

The example's `ClassBuilder` tracks which disjoint residues have been submitted
before exposing a class to interpolation. Initialized memory does not establish
producer completion: unwritten entries may contain earlier results. Tracking
residue identities also rejects duplicate submissions. After a panic, refill
affected buffers before starting interpolation again; see the
[interpolation contract](FFT.md#fused-class-interpolation).

## Generating commitment bases

An inverse group FFT converts coefficient commitment bases into natural-order
Lagrange bases for the same domain. Use the curve's scalar field for roots and
inverse lengths, then batch-normalize the projective outputs into caller-owned
storage. The [reference transform](../crates/udon/src/fft/reference.rs) defines
root and ordering requirements.

The [curve artifact consumer](../crates/udon/tests/fixtures/curve_embedding/)
generates a small structured reference string (SRS) for both curves and embeds
coefficient and Lagrange bases as `PreparedAffinePoint` arrays. The fixture uses
known generator multiples solely as deterministic test data. Its consumer
tests basis order against a direct DFT and checks commitment agreement across
both bases. Bento POD establishes storage layout at compile time and borrows
the constructed values directly; the
[artifact owner](POD.md#format-ownership) defines their schema.

The workspace tests run in the normal suite. The artifact consumer uses the
separate [slow consumer command](TESTING.md#slow-consumer-tests).
