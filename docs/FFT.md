# Field FFTs

Udon's [`fft` module](../crates/udon/src/fft/mod.rs) provides power-of-two field FFTs
for both Pasta fields. Transforms are always available, including in `no_std`
builds without an allocator. No feature flag is required.

The caller owns all mutable buffers, optional tables, and parallel execution.
A plan borrows tables and keeps the domain constants needed for execution.
Table preparation writes into caller storage; Udon's FFT setup and execution
do not allocate. An executor's resource use belongs to the caller. Arithmetic
is variable-time, with no constant-time guarantee for secret inputs; see the
[field contract](../crates/udon/src/field/mod.rs).

## Domains and transform order

[`Domain`](../crates/udon/src/fft/domain.rs) constructs a subgroup with the
canonical Pasta root for its size. Use `Domain::new(k)` for `2^k` elements or
`Domain::for_size(n)` for an element count. The constructor documents supported
orders and target address-space limits. Size one is supported.

`domain.subgroup()` selects shift one. `domain.coset(shift)` accepts any nonzero
loose field element, including shifts inside the subgroup. Both representations
of zero are rejected. For size `n`, canonical
`root`, and coefficients `c[i]`, natural evaluation row `j` is

```text
sum(c[i] * (shift * root^j)^i, i = 0..n), for 0 <= j < n.
```

[`Transform::forward`](../crates/udon/src/fft/transform.rs) replaces coefficients
with these evaluations. `inverse` removes both the shift and the domain-size factor,
returning the original coefficients. Both input and output use natural order.
For separate inputs, prefixes, other orders, or inverse normalization, pass a
`TransformRequest` to `Transform::execute`. `InputStorage::Preserve` reads a
separate immutable input and writes the output; `InPlace` overwrites its input.
`ElementOrder` specifies natural or bit-reversed positions independently for
input and output. These are mathematical layout facts, independent of the
arithmetic schedule.

`InputSupport::Prefix(k)` declares that natural input positions from `k` onward
are zero. A forward prefix describes low-degree coefficients; an inverse prefix
describes evaluation positions, independently of the polynomial's degree. Empty
prefixes represent zero. Output always has the full domain size.
`InverseScale::Unscaled` returns `n * c[i]` while still removing `shift^i`.
The [transform contract](../crates/udon/src/fft/transform.rs) defines validation
and mutation guarantees.

[`EvaluationView`](../crates/udon/src/fft/layout.rs) binds a slice to a coset and
an `EvaluationLayout`: natural rows, bit-reversed rows, or residue-major rows.
`get` maps a logical natural row; `get_extended_row` checks the same-shift
subdomain relationship before looking up a larger domain's row. Pointwise
operations require matching domains and layouts, including the field type,
canonical root, size, and shift. Binding checks dimensions and declared
semantics; it cannot establish which polynomial produced the evaluations.

The [`reference` module](../crates/udon/src/fft/reference.rs) retains a simple
generic transform through `Twiddle` and `Butterfly`. It takes explicit roots;
its inverse also takes the inverse size. This is useful as an independent
schedule or for a downstream value type. Its trait contracts define the
algebraic laws that a custom implementation must satisfy.

## Transform plans and task budgets

[`run::FftPlan`](../crates/udon/src/fft/run.rs) resolves a `TransformRequest`
from a domain/table `Transform`, `StorageLayout`, and shared
[`ExecutionOptions`](../crates/udon/src/exec.rs). The options supply a total task
budget and optional workspace byte ceiling. The default is serial execution
with no extra ceiling. Udon selects local arithmetic, permutations, and column
panels; the caller does not configure those implementation choices.

`StorageLayout::Contiguous` describes a bank that can be leased as a whole.
`Fragments` declares the provider's actual power-of-two fragment length and
whether it can also lease the whole bank. The resolved plan's resource requests
must be satisfiable by that provider. Scratch banks can require different
fragmentation from the values bank; their resource requests specify the needed
ranges.

The plan borrows tables and is reusable across compatible executions. Working
buffers are borrowed by a synchronous call or individual task resources, while
an incremental run borrows its own frontier storage. `retained_fields()` sizes
the plan's fixed arithmetic scratch. Construction and execution allocate
nothing. See the [execution guide](EXECUTION.md) for transferable resource
owners, admission, and failure boundaries.

A forward transform can write bit-reversed evaluations for an inverse that
accepts that order directly. Equally ordered pointwise products preserve this
composition.
`execute` accepts an optional factor slice in the output's physical order;
the caller establishes its domain and layout. `EvaluationView` provides checked
domain and layout bindings when constructing these pipelines. The
[module example](../crates/udon/src/fft/mod.rs) demonstrates a product followed
by interpolation without an intervening scatter.

`execute_batch` transforms consecutive full-domain polynomials in place. It
requires disposable, full-support input. `batch_fields(count)` sizes independent
scratch partitions under the plan's total task budget and workspace ceiling.
Empty batches need no scratch. Small polynomials can be scheduled independently
while sharing one borrowed set of tables.

## Optional tables and downstream storage

[`Tables`](../crates/udon/src/fft/tables.rs) borrows independently optional
tables; its field documentation defines their entry formulas. Begin with
`Transform::new(domain)`, then prepare tables if repeated execution justifies
their storage and setup cost. Obtain destination lengths from
the const query `TableRequirements::for_size(n)` and fill the chosen subset with
`TablesMut::prepare`. The [module examples](../crates/udon/src/fft/mod.rs) show
table arrays sized by Udon at compile time. The query accepts the
same sizes as `Domain::for_size`, applies to both fields and all coset shifts,
and needs no domain construction. `TableRequirements::for_domain` is a
convenience wrapper for an existing domain.

`TablesMut::prepare` returns a `Transform` handle tied to the generating coset.
For stored slices, `Tables::bind` asserts their lengths and returns the same
handle. It is a const function and trusts the entries produced by preparation.
Use `bound.for_coset(other_coset)` to reuse ordinary forward and inverse
twiddles on another coset of the same subgroup. This takes constant work and
retains the original borrows.
Changing the shift drops inverse-finish and inverse-scaling tables, whose entries
depend on that shift; the same domain retains every table. A different subgroup
size is rejected.

Every table family has one `bind` operation. Binding attaches trusted entries
to their domain or sequence description and checks only dimensions and
configuration. It performs no field reduction or mathematical content scan.
The artifact's schema identifies the field and table conventions; Bento preserves
the exact representations generated with those types.

Transform plans additionally accept
[`TwiddleTable`](../crates/udon/src/fft/powers.rs) through `with_twiddles`.
`TwiddleDescription::requirements` sizes each representation; `prepare` and
`bind` produce handles from preparation or trusted stored entries. For table
domain size `N > 1`, retained field counts are listed below; size one needs no
entries in either representation.

| Storage | Fields | Access cost |
| --- | --- | --- |
| `Dense` | `N/2` | One lookup |
| `StagePacked` | `N-1` | Stage-contiguous lookup; choose a smaller table size to retain only local stages |

All table domains use the nested canonical Pasta roots. A larger table serves
a smaller transform through the canonical root stride. A smaller table serves
local stages while larger stages regenerate their powers. Forward and inverse
transforms can share either root orientation. A table describes subgroup powers
independently of the transform's coset shift. Column panels can also use the
explicit twiddle provider.

`PowerTable` describes entry `i` as `first * step^i`, including deliberate scaling.
Forward coefficient tables use `first = 1`, `step = shift`, and exactly the
transform size. `PowerTable::prepare` and `bind` return handles directly;
`with_forward_scales` checks their compatibility without rescanning contents.
Seeds and entries use loose field representations.

Callers can prepare table arrays at runtime and lend their slices, or prepare
them in a downstream build script and embed them through [Bento POD](POD.md).
The executable [FFT embedding consumer](../crates/udon/tests/fixtures/fft_embedding)
shows the complete build-to-runtime path for both fields:

1. Its [record schema](../crates/udon/tests/fixtures/fft_embedding/src/record.rs)
   owns concrete POD arrays for a chosen domain and expansion ratio.
2. Its [build script](../crates/udon/tests/fixtures/fft_embedding/build.rs) prepares
   the arrays with Udon, writes `bento::bytes_of(&record)`, and names the file
   using `STORED_FORM`.
3. Its [consumer library](../crates/udon/tests/fixtures/fft_embedding/src/lib.rs)
   uses `bento::embed_struct!` with `udon::stored_form!()`, borrows tables directly
   from the embedded record, and executes with stack-owned buffers.

The owner chooses filenames, dimensions, and format versions. The fixture's
shared schema fixes the field types, domain sizes, root orientation, scale
normalization, shift, and layout. Its consumer borrows the generated entries
directly. These conventions belong to the artifact owner's schema; Udon supplies
mathematical table handles without rechecking the stored values.
Native build-time and runtime preparation produce identical borrowed handles;
use build scripts for large artifacts and const evaluation for small schedules
or storage requirements.
Field POD storage and embedding follow the representation and target layout
contracts in the [field storage guide](POD.md#storing-field-elements). Ordinary
FFT arithmetic also builds on big-endian targets; the
[portability checks](TESTING.md#test-roles) compile those calls
but do not execute them.

## Scratch and execution

[`ExecutionOptions`](../crates/udon/src/exec.rs) is shared by FFTs and MSMs.
`with_task_budget` provides the allowance for an entire operation, including
nested transforms. `with_memory_limit` bounds used arithmetic scratch and
retained intermediates. Inputs, outputs, persistent tables, metadata, unused
buffer tails, and executor resources are outside that ceiling. Supplied buffer
capacities are always hard limits.

`transform.scratch_requirements(options)` reports preferred initialized field
storage for direct full transforms. Direct execution can select a smaller
workspace when the supplied scratch is shorter, including a scratch-free path
for contiguous transforms. A resolved `FftPlan` instead requires its reported
`retained_fields()` so its resource requests remain stable. Fragmented storage
can require a snapshot even with a serial budget.

Scratch arrays filled with `Fp::ZERO` or `Fq::ZERO` suffice. Unused tails remain
untouched. The [module contract](../crates/udon/src/fft/mod.rs) defines scratch
reuse, validation boundaries, and recovery on unwind.

Arbitrary nonzero coset shifts are supported, but coefficient scaling costs
depend on the shift and strategy. The [benchmarks](TESTING.md#fft-benchmarks)
include scaling when comparing subgroup transforms, the order-three shift
`zeta`, and the generic shift 7.

FFTs use [`exec::Executor`](../crates/udon/src/exec.rs) for scoped joins. Its trait
documentation defines completion, panic handling, and progress during nested
calls. `SerialExecutor` can also exercise tiled transforms with the queried
scratch requirement.

For concurrency across polynomials or separately owned tiles, use
`exec::for_each_mut`; use `exec::for_each_chunk_mut` for contiguous chunks. Each
callback receives a `TaskBudget` for its nested work. Pass it through
`ExecutionOptions::with_task_budget`, and give concurrent transforms disjoint
scratch. The
[execution module's example](../crates/udon/src/exec.rs) demonstrates this with
separate tiles. Divide budgets between simultaneous application operations;
copying a budget does not reserve or limit threads.

[`fft::run`](../crates/udon/src/fft/run.rs) exposes bounded transform, expansion,
and interpolation work to an application scheduler. Each run owns its buffer
barriers; completed transforms and residue blocks can ready their consumers
while other operations continue. The [execution guide](EXECUTION.md) explains
fragment leases, retained snapshots, admission, and integration with MSM and
application work.

## Residue expansion and layouts

[`Expansion`](../crates/udon/src/fft/expansion.rs) accepts a base subgroup plan
and a coset of equal or larger size. Write the base size as `n`, the extended
size as `N`, and their ratio as `r = N/n`, a supported power of two including
one. Let `g` be the extended coset shift and `w_N` its canonical root. Residue
`s` consists of natural rows `s + r*k`, where `0 <= k < n` and `0 <= s < r`.
Coefficient indices `i` range over `0 <= i < n`.

Output is residue-major: all rows for residue zero, then residue one, and so on.
For natural extended row `j`, its storage index is

```text
(j % r) * n + j / r.
```

[`ResidueLayout` and `EvaluationView`](../crates/udon/src/fft/layout.rs) provide
checked index conversion, residue slices, and lookup at rows of a still larger
domain with the same shift. Such lookup rejects rows outside the smaller
domain. Copy helpers convert between natural and residue order into distinct
caller buffers. Ordinary slice ranges provide coefficient tiles.

`Expansion::coefficients` accepts any prefix fitting the base domain. An optional
table holds exactly `extended_size` residue scales; `prepare_scales` fills caller
storage and returns an `ExpansionScales` handle with `Coefficients`
normalization. Pass the handle to `Expansion::new` or `with_scales`.

Direct expansion accepts the same `ExecutionOptions` as transforms. Udon divides
one task allowance across residues and their inner transforms and fits their
combined scratch within the byte ceiling. `expansion.coefficient_scratch(options)`
reports preferred scratch for `coefficients` and `short_product`; execution
adapts to smaller supplied capacity. Output is the residue workspace, so no
full zero-padded working buffer is needed.

Use `Expansion::evaluations` when the input is already evaluated on the base
subgroup. It preserves that input and needs no separate coefficient buffer;
`expansion.evaluation_scratch(options)` reports its preferred temporary storage.
The first output residue holds coefficients while the remaining residues read
them. [`ExpansionScales`](../crates/udon/src/fft/expansion_scales.rs) records the
scale convention and domain. `Expansion::new` accepts an optional handle, and
`Expansion::with_scales` attaches one to an existing expansion:

| Normalization | Entry `(s,i)` |
| --- | --- |
| `Coefficients` | `(g*w_N^s)^i` |
| `UnscaledInverse` | `n^-1 * (g*w_N^s)^i` |

Both need `N` fields. `ExpansionScales::prepare` writes either convention;
`bind` checks the configuration and asserts the stored length without inspecting
entries. Both coefficient and
evaluation inputs can use either convention. Residue initialization accounts for
the table's factor and the input's normalization; the table does not select the
inverse's scale. See the
[`Expansion::with_scales` contract](../crates/udon/src/fft/expansion.rs).

Scratch is reused between the inverse and residue phases; the query accounts
for concurrent residues. Buffer lengths and resource constraints are checked
before writing, including when the input and output domains are equal.

`short_product` expands a nonempty short prefix and multiplies a supplied
residue-major factor into each completed residue. It accepts an `EvaluationView`
and checks both the ordered coset domain and residue layout before any writes
or executor joins. `Expansion::view` binds a factor slice to those conventions;
the caller establishes that its values evaluate the intended polynomial.
Scheduling and scratch requirements are the same as `coefficients`.
If the product will be interpolated, choose an extended domain larger than its
degree to recover all coefficients.

One use is a polynomial `p` of degree less than `n` whose base-subgroup
evaluations vanish outside `t` selected rows, where `0 < t <= n`. Let `H` be the
monic degree-`t` polynomial vanishing on those selected points. Then
`S(X) = (X^n - 1) / H(X)` vanishes on every other base point, and `p = q * S`
for a polynomial `q` of degree less than `t`. Precompute the extended-domain
evaluations of `S` and pass the short coefficients of `q` to `short_product`.
This uses coefficient support; an inverse `TransformRequest` prefix instead
describes evaluation positions and does not recover `q`. Consumers of `p`'s
coefficients still need the full product.

[`run::ExpansionPlan`](../crates/udon/src/fft/run/expansion.rs) fixes liveness,
input support, ordering, and resource constraints. `coefficient_fields()` reports
separate coefficient workspace; `scratch_fields()` reports synchronous transform
scratch. The workspace ceiling covers both. The plan divides its total task
allowance across concurrent residues and their transforms.

| Storage policy | Input | Additional coefficient fields | Scheduling dependency |
| --- | --- | --- | --- |
| `Coefficients` | Immutable coefficient prefix | Zero | All residues independent |
| `ReuseOutput` | Immutable base evaluations | Zero | Residue zero waits until the others finish reading it |
| `CoefficientWorkspace` | Immutable base evaluations | `base_size` | All residues independent after one inverse |
| `DisposableInput` | Mutable base evaluations | Zero | Input becomes coefficient storage; all residues independent after one inverse |

Use `execute` for immutable input, supplying a coefficient buffer for
`CoefficientWorkspace`, or `execute_disposable` to consume mutable input.
The two policies retaining coefficients select an explicit `InverseScale` and
return a [`CoefficientView`](../crates/udon/src/fft/layout.rs) borrowing only that
buffer. `Normalized` retains `c[i]`; `Unscaled` retains `n * c[i]`, where `n` is
the source base size. Both use increasing degree order and loose Montgomery
representations. The view's `normalization_factor()` recovers `c[i]`; output and
scratch can be reused while the view is live.

Pass the view directly to `Transform::execute` or `Expansion::coefficients`.
For run plans consuming coefficient slices, attach its normalization factor
with `FftPlan::with_input_scale` or `ExpansionPlan::with_coefficient_scale`.
Passing only `view.as_slice()` loses the scale information. Scale tables use the
conventions above; initialization accounts for their normalization and the
retained coefficient scale. The extra coefficient buffer removes a scheduling
dependency; its latency benefit depends on the workload and target.

`ExpansionOrder::Residues` uses `ResidueLayout`. `ExpansionOrder::BitReversed`
reverses both the residue blocks and their inner row order. For `r=2^a` residues
and `n=2^b` rows, let `bit_reverse_d(x)` reverse the low `d` bits of `x`:

```text
bit_reverse_(a+b)(s + 2^a*k) = 2^b * bit_reverse_a(s) + bit_reverse_b(k).
```

The resulting vector can feed a full inverse accepting bit-reversed input.
Bind an `EvaluationView` with the selected layout for checked row lookup.
The optional factor slice passed to execution must use that same physical
layout and coset; the caller establishes those semantics.

For bounded-memory consumers, `Expansion::residue(s, order)` borrows a descriptor
that evaluates a coefficient prefix into one reusable base-sized output. Its
`domain` identifies the selected coset; `order` applies only inside that residue.
The caller chooses which residues to retain for rotations or other dependencies.
It can also fill separately owned residue buffers, scheduled through
`exec::for_each_mut` with shared coefficients. Whole-expansion operations use
contiguous output, so choose storage according to the consumer's access pattern.

## Fused class interpolation

[`run::InterpolationPlan`](../crates/udon/src/fft/run/interpolation.rs) combines
classes supplied as `(Transform, ElementOrder)` pairs. Entry zero is the output;
other classes may be smaller and have different nonzero coset shifts. The caller
supplies storage layout, whether lift buffers may be consumed, and one set of
resource constraints. Udon resolves the inverse transforms and their scheduling.

`execute` accepts arrays of mutable class buffers and scratch slices.
`snapshot_fields(class)` sizes each scratch slice.
The output becomes the coefficient sum, with smaller vectors implicitly
zero-padded. Without `consume`, lifts retain their individual coefficients.
With `consume`, equal-domain evaluation vectors merge before one inverse per
distinct domain, including groups smaller than the output. Lift contents are
then unspecified. The implementation may fuse inverse work with coefficient
additions or schedule them separately within the resource constraints.

Applications own producer completeness and scatter. For nested domains with the
same shift, derive scatter positions in natural row order. If an extended domain
has `r` residues of `n` rows, a window in residue `s` starting at inner offset `k`
starts at natural row `s + r*k` and advances by `r`. A subdomain smaller by `d`
contains extended rows divisible by `d`. When `d` divides `r` and `s`, its window
starts at `s/d + (r/d)*k` with stride `r/d`. Map each resulting index into the
selected element order before writing. The
[workspace fixture](../crates/udon/tests/support/workspaces/fft.rs) checks producer
completion before exposing its buffers to interpolation.

Validation precedes mutation. A panic can leave partial results; refill affected
buffers with evaluations before retrying. `InterpolationRun` additionally tracks
class phases through `ClassState` while tasks and consumers complete.

For correctness coverage and performance commands, see the
[testing guide](TESTING.md) and [strategy measurements](FFT_PERFORMANCE.md).
