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

`domain.subgroup()` selects shift one. `domain.coset(shift)` accepts any nonzero,
reduced field element, including shifts inside the subgroup; it rejects
unreduced Montgomery representations before arithmetic. For size `n`, canonical
`root`, and coefficients `c[i]`, natural evaluation row `j` is

```text
sum(c[i] * (shift * root^j)^i, i = 0..n), for 0 <= j < n.
```

[`Plan::forward`](../crates/udon/src/fft/transform.rs) replaces coefficients with
these evaluations. `inverse` removes both the shift and the domain-size factor,
returning the original coefficients. Both input and output use natural order.
`forward_into` and `inverse_into` preserve a separate input slice and overwrite
the caller's output.
`inverse_bit_reversed` accepts evaluations already placed
at bit-reversed positions and omits the input permutation.

[`ElementOrder`](../crates/udon/src/fft/layout.rs) describes natural or
bit-reversed storage for either coefficients or evaluations. Configured
transforms select this order independently for input and output.

Use `forward_prefix` when only the low-degree coefficients are present. It
treats omitted coefficients as zero, and an empty prefix as the zero polynomial.
Its output needs the full domain size and the queried scratch, including for
empty input. The `Plan` documentation defines
buffer lengths, scratch requirements, and errors for each transform method.

For explicit input support, order, and normalization, use `run::FftPlan` with
a `TransformRequest`. `InverseScale::Unscaled` returns `n * c[i]` and still
removes `shift^i`. An inverse prefix declares natural evaluation positions
with an omitted zero suffix, independently of the polynomial's degree.

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

[`run::FftPlan`](../crates/udon/src/fft/run.rs) fixes a `TransformRequest`, tile
size, and codelet over a domain/table `Plan`. The request selects direction,
full or prefix support, input and output order, inverse scaling, and one
`InputStorage` choice. `InPlace` reads and overwrites the values bank, including
a prefix's unused tail. `Preserve` reads an immutable separate input and writes
the values bank. A prefix requires natural input order. The default request
uses `InPlace`; `execute` requires `Some(input)` exactly for `Preserve`.

Both plans borrow tables and are reusable across executions. Working buffers
are borrowed by a synchronous call or individual task resources, while an
incremental run borrows its own frontier storage. Plan configuration and buffer
validation have separate lifetimes; see the [execution guide](EXECUTION.md)
for transferable resource owners and failure boundaries.

`retained_fields()` reports initialized scratch fields for synchronous execution.
The application accounts separately for inputs, outputs, borrowed tables, and
executor resources. Tile geometry is independent of worker count. Pass a nonzero
total task allowance to `execute`; the driver divides it across scoped work.
No setup or execution allocation occurs.

The default stage path supports radix-2/4/8 codelets. `with_columns` selects
column panels and a retained panel count. Natural-order in-place permutation
normally retains a snapshot; `with_contiguous_permutation` uses bounded index
swaps on the contiguous bank instead. Separate-output initialization gathers
destination tiles by default; `with_scatter_initialization` selects consecutive
input tiles. Forward coset scaling generates powers by recurrence or borrows a
`PowerTable` through `with_forward_scales`.

Natural coefficients to bit-reversed evaluations use decimation in frequency
(DIF). A matching inverse using decimation in time (DIT) accepts those evaluations
directly. Equally ordered pointwise products preserve this composition.
`execute` accepts an optional factor slice in the output's physical order;
the caller establishes its domain and layout. `EvaluationView` provides checked
domain and layout bindings when constructing these pipelines. The
[module example](../crates/udon/src/fft/mod.rs) demonstrates a product followed
by interpolation without an intervening scatter.

`execute_batch` transforms consecutive full-domain polynomials in place. It
requires disposable, full-support input. `batch_fields(count, tasks)` sizes
independent scratch partitions under the same total task allowance passed to
execution. Empty batches need no scratch. Small polynomials can be scheduled
independently while sharing one borrowed set of tables.

## Optional tables and downstream storage

[`Tables`](../crates/udon/src/fft/tables.rs) borrows independently optional
tables; its field documentation defines their entry formulas. Begin with
`Plan::without_tables`, then prepare tables if repeated execution justifies
their storage and setup cost. Obtain destination lengths from
the const query `TableRequirements::for_size(n)` and fill the chosen subset with
`TablesMut::prepare`. The [module examples](../crates/udon/src/fft/mod.rs) show
table and scratch arrays sized by Udon at compile time. The query accepts the
same sizes as `Domain::for_size`, applies to both fields and all coset shifts,
and needs no domain construction. `TableRequirements::for_domain` is a
convenience wrapper for an existing domain.

`TablesMut::prepare` returns a `BoundTables` handle tied to the generating coset.
For imported slices, `Tables::bind` checks lengths and every mathematical entry,
including reduced Montgomery limbs, then returns the same handle. `Plan::new`
uses the handle's domain. Native generation requires no content scan; checked
imports require linear work, with no entry validation during execution.
Use `bound.for_coset(other_coset)?.plan()` to reuse validated ordinary forward
and inverse twiddles on another coset of the same subgroup. This takes constant
work and retains the original borrows, without inspecting entries again.
Changing the shift drops inverse-finish and inverse-scaling tables, whose entries
depend on that shift; the same domain retains every table. A different subgroup
size is rejected.

Every table family also offers `bind_trusted`, which relies on the caller to
establish correct entries elsewhere. Each constructor documents the dimensions
or seeds it still checks and the caller's obligations. Incorrect contents can
cause wrong results or panics; these APIs remain memory safe. An explicit
`validate` method remains available to check an existing handle. Bento checks the
storage layout; artifact metadata identifies the field and conventions; checked
binding establishes the mathematical contents. These are separate checks.

Transform plans additionally accept
[`TwiddleTable`](../crates/udon/src/fft/powers.rs) through `with_twiddles`.
`TwiddleDescription::requirements` sizes each representation; `prepare` and
`bind` produce handles from native generation or checked imports. For table
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
transform size. `PowerTable::prepare` and `bind` return checked handles;
`with_forward_scales` checks their compatibility without rescanning contents.
Both seeds must be reduced, even for empty sequences.

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

The owner chooses filenames, dimensions, format versions, and integrity checks.
The fixture's header records the field modulus, Montgomery radix, domain sizes,
root orientation, scale normalization, shift, and layout. Its consumer validates
that metadata before binding the tables. These conventions belong to the
artifact owner's schema; Udon supplies mathematical table handles.
Consumers check semantic compatibility and mathematical entries separately from
Bento's target-layout checks and the owner's transport-integrity policy.
Native build-time and runtime preparation produce identical borrowed handles;
use build scripts for large artifacts and const evaluation for small schedules
or storage requirements.
Field POD storage and embedding follow the representation and target layout
contracts in the [field storage guide](POD.md#storing-field-elements). Ordinary
FFT arithmetic also builds on big-endian targets; the
[portability checks](TESTING.md#test-roles) compile those calls
but do not execute them.

## Scratch and execution

[`ExecutionOptions`](../crates/udon/src/fft/execution.rs) selects a power-of-two
local tile length, columns per cross-tile task, and a task budget. Its const
`requirements(size)` query sizes arrays before a domain or plan exists;
`plan.scratch_requirements(options)` returns the same requirement. Scratch is
initialized field storage, so arrays filled with `Fp::ZERO` or `Fq::ZERO`
suffice. The [module contract](../crates/udon/src/fft/mod.rs) defines scratch
reuse, the scope of validation and mutation guarantees, and recovery on unwind.

`ExecutionOptions::serial()` uses one whole-transform tile, makes no executor
calls, and needs zero scratch. `ExecutionOptions::default()` uses 1,024-element
tiles, 64 columns per task, and one task. Increasing its `max_tasks` enables
concurrency for transforms larger than one tile. Increasing only `max_tasks` on
`serial()` retains its whole-transform tile; set a smaller `tile_len` as well.
The current tiled implementation reserves one
rectangular scratch partition per concurrent cross-tile job: tiles × columns
× jobs. Obtain counts from the query so buffers follow changes to that geometry.
For example, the current requirement for 16,384 elements with 2,048-element
tiles, 128 columns per task, and four tasks is 4,096 fields, or 128 KiB.
This is in addition to
input, output, tables, class descriptors, and any executor resources.

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
callback receives a `TaskBudget` for its nested work. Pass that budget's `.get()`
to the FFT task limit, and give concurrent transforms disjoint scratch. The
[execution module's example](../crates/udon/src/exec.rs) demonstrates this with
separate tiles. Divide budgets between simultaneous application operations;
copying a budget does not reserve or limit threads.

[`fft::run`](../crates/udon/src/fft/run.rs) exposes bounded transform, expansion,
and interpolation work to an application scheduler. Each run owns its buffer
barriers; completed transforms and residue blocks can ready their consumers
while other operations continue. Tile and column geometry are independent of
worker count. The [execution guide](EXECUTION.md) explains fragment leases,
retained snapshots, admission, and integration with MSM and application work.

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
storage and returns a checked `ExpansionScales` handle with `Coefficients`
normalization. Pass the handle to `Expansion::new` or `with_scales`;
`validate_scales` is an optional explicit audit.

The direct expansion methods accept
[`ExpansionOptions`](../crates/udon/src/fft/expansion.rs).
`max_residue_tasks` limits concurrent residues, while `transform` supplies
`ExecutionOptions` for every base-size transform. Both levels use the caller's
executor, including nested joins. Their task limits multiply: two concurrent
residues with eight transform tasks each permit up to sixteen concurrent
partitions. Set `max_residue_tasks` to one to reuse one transform's scratch
across residues while retaining parallel work within each transform. Set
`transform` to `ExecutionOptions::serial()` to parallelize only across residues,
using no scratch. `ExpansionOptions::serial()` needs no scratch or joins.

For example, a base size of 2²⁰ expanded 2× can use the following settings to
allow sixteen tasks within each residue. The current scratch requirement is
32,768 fields (1 MiB), obtained at compile time:

```rust
use zakura_udon::fft::{ExecutionOptions, ExpansionOptions};

const OPTIONS: ExpansionOptions = ExpansionOptions {
    max_residue_tasks: 1,
    transform: ExecutionOptions {
        tile_len: 4096,
        columns_per_task: 8,
        max_tasks: 16,
    },
};
const SCRATCH: usize = match OPTIONS.evaluation_requirements(1 << 20, 1 << 21) {
    Ok(required) => required.field_elements,
    Err(_) => panic!("unsupported expansion configuration"),
};
```

Query `ExpansionOptions::coefficient_requirements(base_size, extended_size)`
before calling `coefficients` or `short_product`, or use
`expansion.coefficient_scratch(options)` once the expansion exists. The current
implementation reserves one base transform's scratch partition per concurrent
residue. Output holds the coefficients and evaluations; there is no separate
zero-padded working buffer.

Use `Expansion::evaluations` when the input is already evaluated on the base
subgroup. It preserves that input and needs no separate coefficient buffer;
query `ExpansionOptions::evaluation_requirements(base_size, extended_size)` or
`expansion.evaluation_scratch(options)` for its temporary storage requirement.
The first output residue holds coefficients while the remaining residues read
them. [`ExpansionScales`](../crates/udon/src/fft/expansion_scales.rs) records the
scale convention and domain. `Expansion::new` accepts an optional handle, and
`Expansion::with_scales` attaches one to an existing expansion:

| Normalization | Entry `(s,i)` | Base inverse for evaluation input |
| --- | --- | --- |
| `Coefficients` | `(g*w_N^s)^i` | Normalized once |
| `UnscaledInverse` | `n^-1 * (g*w_N^s)^i` | Omits the base-size factor |

Both need `N` fields. `ExpansionScales::prepare` writes either convention;
`bind` checks imported entries before returning a handle. Normalized coefficient
input ignores `UnscaledInverse` tables and generates ordinary powers. Unscaled
coefficient views can use either convention, with normalization adjusted during
initialization; see [`Expansion::with_scales`](../crates/udon/src/fft/expansion.rs).

Scratch is reused between the inverse and residue phases; the query accounts
for concurrent residues. The required scratch and execution options are checked
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
input support, ordering, tile size, and codelet. `coefficient_fields()` reports
separate coefficient workspace; `scratch_fields(tasks)` reports synchronous
transform scratch. The caller accounts for both. Execution divides one total
task allowance across concurrent residues and their transforms.

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
the source base size. Both use increasing degree order and reduced Montgomery
representations. The view's `normalization_factor()` recovers `c[i]`; output and
scratch can be reused while the view is live.

Pass the view directly to `Plan::forward_prefix` or `Expansion::coefficients`.
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

The resulting vector is already in the input order of a full inverse DIT.
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
full-support, normalized inverse `FftPlan`s with natural coefficient output.
Entry zero is the output; other classes may be smaller and have different
nonzero coset shifts. Every class tile must match the output tile or its own
smaller size. Each input may use natural or bit-reversed order.

`execute` accepts arrays of mutable class buffers and scratch slices plus one
total task allowance. `snapshot_fields(class)` sizes each scratch slice.
The output becomes the coefficient sum, with smaller vectors implicitly
zero-padded. Without `consume`, lifts retain their individual coefficients.
With `consume`, equal-domain evaluation vectors merge before one inverse per
distinct domain, including groups smaller than the output. Lift contents are
then unspecified. The serial radix-2 path retains fused inverse/addition kernels;
parallel execution schedules independent inverses and coefficient additions.

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
