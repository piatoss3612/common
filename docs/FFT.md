# Field FFTs

Udon's [`fft` module](../crates/udon/src/fft/mod.rs) provides power-of-two field FFTs
for both Pasta fields. Transforms are always available, including in `no_std`
builds without an allocator. See the
[crate feature documentation](../crates/udon/src/lib.rs) for the reserved `alloc`
feature.

The caller owns all mutable buffers, optional tables, and parallel execution.
A plan borrows tables and keeps the domain constants needed for execution.
Table preparation writes into caller storage; Udon's FFT setup and execution
do not allocate. An executor's resource use belongs to the caller. Arithmetic
is variable-time, with no constant-time guarantee for secret inputs; see the
[field contract](../crates/udon/src/field/mod.rs).

For a worked application migration from the sibling Bento workspace, see the
[Sensei FFT upgrade guide](FFT_UPGRADE.md).

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

Use `forward_prefix` when only the low-degree coefficients are present. It
treats omitted coefficients as zero, and an empty prefix as the zero polynomial.
Its output needs the full domain size and the queried scratch, including for
empty input. The `Plan` documentation defines
buffer lengths, scratch requirements, and errors for each transform method.

`forward_serial` and `inverse_serial` provide the common in-place case without
executor, options, or scratch arguments. `inverse_scaled` selects
`InverseScale::Normalized` or `Unscaled`: the latter returns `n * c[i]` and
**still removes `shift^i`**. `inverse_prefix` declares a natural prefix of
evaluations with an omitted zero suffix. Its support describes evaluation
positions, independently of the polynomial's degree.

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

## Prepared operations and budgets

[`Plan::configure`](../crates/udon/src/fft/operation.rs) fixes a
`TransformRequest` and `Strategy` for repeated execution. The request specifies
direction, full or prefix support, input and output order, inverse scaling, and
input preservation. A prefix requires natural input order. `execute_into`
preserves its separate input; `execute` requires `InputPolicy::Disposable` and
overwrites the full in-place buffer, including a prefix's unused tail.

The same `OperationDescription` drives const sizing and runtime configuration.
Its [executable example](../crates/udon/src/fft/operation.rs) prepares an operation
under a zero-scratch budget and executes with arrays sized by the const query.

`ResourceBudget` caps initialized temporary fields, retained table bytes, and
total tasks. Requirements exclude input/output storage, fixed stack frames, and
executor resources. Borrowed table slices are conservatively counted separately
even if their storage aliases. `OperationRequirements` reports scratch in field
elements, including the size and count of blocked column jobs' partitions.
Configuration and attaching tables to prepared operations reject strategies
that exceed a ceiling before execution.

| Backend | Scratch | Execution |
| --- | --- | --- |
| `InPlace` | Zero | Stage barriers and disjoint butterfly pairs; supports parallel work and radix-2/4/8 codelets |
| `Blocked` | Queried rectangular partitions | Local transforms and column jobs in caller scratch |

`Strategy::serial()` selects the stage backend with radix 2. `Strategy::budgeted`
uses deterministic geometry and `Backend::Auto`, selecting blocked execution
when its geometry, order, codelet, and scratch ceiling permit, otherwise the
stage backend. `Auto` does not measure timings. Low-level geometry remains
available through `ExecutionOptions`; larger codelets are explicit stage-backend
choices. Separate-output initialization can scatter from the input, gather
consecutive destination regions, or gather in bounded tiles.
Forward coset scaling computes a starting power for each region and advances by
recurrence, or borrows a `PowerTable` containing `shift^i` for coefficient degree
`i`.

With the stage backend, natural coefficients to bit-reversed evaluations use
decimation in frequency (DIF). A matching inverse using decimation in time (DIT)
accepts those evaluations directly and returns natural coefficients. Equally
ordered pointwise products preserve this composition. `execute_product_into`
validates a domain-bound factor and writes the pointwise product in the requested
output order. The [module example](../crates/udon/src/fft/mod.rs) demonstrates the
complete product and interpolation pipeline without an intervening scatter.

`execute_batch` transforms a polynomial-major buffer containing consecutive
full-domain polynomials in place. It requires disposable, full-support input.
`batch_requirements(count)` reserves independent scratch partitions and divides
the total task budget between polynomials and their transforms. Empty batches
need no scratch. This layout lets independent small transforms be the parallel
unit while retaining one borrowed set of tables.

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

Prepared stage operations additionally accept
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
independently of the transform's coset shift. Explicit blocked execution
uses `Tables`; attaching an alternate provider to an automatically
selected blocked operation selects the stage backend instead.

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
   using `StoredForm::for_target`.
3. Its [consumer library](../crates/udon/tests/fixtures/fft_embedding/src/lib.rs)
   uses `bento::embed_struct!` with `udon::stored_form!()`, borrows tables directly
   from the embedded record, and executes with stack-owned buffers.

The owner chooses filenames, dimensions, format versions, and integrity checks.
Udon's `TwiddleArtifact` and `ExpansionScaleArtifact` describe mathematical
semantics: version, field modulus, canonical root dimensions and orientation,
Montgomery radix, and applicable shift, layout, and normalization. They are Rust
descriptors, not a serialization format. The fixture's owner-defined header
stores these alongside stage-packed twiddles and pre-normalized residue scales.
Its schema version identifies the owner's record layout separately from Udon's
semantic versions; the consumer checks both before binding the tables.
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

[`ExecutionOptions`](../crates/udon/src/fft/executor.rs) selects a power-of-two
local tile length, columns per cross-tile task, and a task budget. Its const
`requirements(size)` query sizes arrays before a domain or plan exists;
`plan.scratch_requirements(options)` returns the same requirement. Scratch is
initialized field storage, so arrays filled with `Fp::ZERO` or `Fq::ZERO`
suffice. The [module contract](../crates/udon/src/fft/mod.rs) defines scratch
reuse, unchanged buffers on validation errors, and partial results on panic.

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

[`Executor`](../crates/udon/src/fft/executor.rs) supplies a scoped `join` of two
borrowed `FnOnce + Send` jobs. Joins nest as transforms subdivide work, including
inside concurrent residue jobs. A bounded pool that queues child jobs and
blocks its workers waiting for them can deadlock under saturation. The executor
must make progress even when every worker is inside a nested join.

An adapter that delegates to
[`rayon::join`](https://docs.rs/rayon/latest/rayon/fn.join.html) provides suitable
cooperative execution: workers execute available work while waiting for stolen
jobs. `SerialExecutor` also supports nested calls. See the trait's documentation
for the full completion, nesting, and panic contracts. `max_tasks` limits work
partitions without reserving idle workers; a serial executor can still exercise
tiled transforms with the same scratch requirement.

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

[`ResidueLayout` and `ResidueView`](../crates/udon/src/fft/layout.rs) provide
checked index conversion, residue slices, and lookup at rows of a still larger
domain with the same shift. Such lookup rejects rows outside the smaller
domain. Copy helpers convert between natural and residue order into distinct
caller buffers. `CoefficientTiles` instead borrows contiguous coefficient
ranges; its tiles are not residue classes.

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

[`Expansion::configure`](../crates/udon/src/fft/expansion_operation.rs) adds
explicit liveness and persistent ordering. `ExpansionDescription::requirements`
and the configured operation report transform scratch and coefficient workspace
separately; the resource ceiling covers their sum. `ExpansionStrategy` divides
one total task budget across concurrent residues and within-transform work.

| Storage policy | Input | Additional coefficient fields | Scheduling dependency |
| --- | --- | --- | --- |
| `Coefficients` | Immutable coefficient prefix | Zero | All residues independent |
| `ReuseOutput` | Immutable base evaluations | Zero | Residue zero waits until the others finish reading it |
| `CoefficientWorkspace` | Immutable base evaluations | `base_size` | All residues independent after one inverse |
| `DisposableInput` | Mutable base evaluations | Zero | Input becomes coefficient storage; all residues independent after one inverse |

Use `execute_into` for the first two policies, `execute_with_workspace` for a
separate coefficient buffer, and `execute_disposable` to consume the input.
The latter two policies require an explicit `scale: InverseScale` field and return
a [`CoefficientView`](../crates/udon/src/fft/layout.rs) borrowing only the retained
buffer. `Normalized` retains polynomial coefficients `c[i]`; `Unscaled` retains
`n * c[i]`, where `n` is the source base size. Both use increasing degree order
and reduced Montgomery representations. The view's `normalization_factor`
recovers `c[i]`; output and scratch can be reused while the view is live.

Pass the view itself to a coefficient consumer such as `Plan::forward_prefix`
or `Expansion::coefficients` to apply normalization during output initialization.
Passing `view.as_slice()` loses the scale information because ordinary slices
are treated as normalized. The view's executable example shows a smaller
expansion feeding a larger transform without copying or normalizing the retained
buffer. Prepared transforms and expansions with `ExpansionStorage::Coefficients`
provide `execute_coefficients` for this purpose. Prepared transforms require a
forward request with natural input order and the configured full or prefix length.

For the two policies that retain coefficients, scale tables must match the chosen
inverse scale: `Normalized` requires `Coefficients` tables, and `Unscaled`
requires `UnscaledInverse` tables. [`Expansion::configure`](../crates/udon/src/fft/expansion_operation.rs)
rejects mismatches, including for singleton domains. Without residue tables,
either scale is supported. These restrictions concern the retained coefficient
output; consuming a view follows the table rules above. The extra coefficient
buffer removes a scheduling dependency; its latency benefit depends on the
workload and target.

`ExpansionOrder::Residues` uses `ResidueLayout`. `ExpansionOrder::BitReversed`
reverses both the residue blocks and their inner row order. For `r=2^a` residues
and `n=2^b` rows, let `bit_reverse_d(x)` reverse the low `d` bits of `x`:

```text
bit_reverse_(a+b)(s + 2^a*k) = 2^b * bit_reverse_a(s) + bit_reverse_b(k).
```

The resulting vector is already in the input order of a full inverse DIT.
`PreparedExpansion::view` returns a domain-bound view with the correct complete
layout. Its `execute_product_into` validates the factor's domain and order and
writes their pointwise product in that layout.

For bounded-memory consumers, `Expansion::residue(s, order)` borrows a descriptor
that evaluates a coefficient prefix into one reusable base-sized output. Its
`domain` identifies the selected coset; `order` applies only inside that residue.
The caller chooses which residues to retain for rotations or other dependencies.

## Fused class interpolation

[`Class`](../crates/udon/src/fft/interpolation.rs) pairs a plan with a borrowed,
initialized evaluation buffer. It accepts natural or bit-reversed input order.
`scatter` and `scatter_strided` write natural evaluation positions directly to
the requested storage order, checking the requested natural positions before
writing. Unwritten entries retain their prior values; callers own completeness
and accumulation.

`interpolate_classes` interpolates one output class and any number of lift
classes no larger than the output. Classes may have different nonzero coset
shifts. The result is the sum of their coefficient
vectors, with smaller vectors implicitly zero-padded to the output length.
The function's executable example combines residue scatter with a lift on a
different domain.

All class buffers are overwritten with natural-order coefficients. A class
tracks whether interpolation has begun and rejects subsequent interpolation or
scatter with `FftError::InvalidClassState`, including after a transform panic.
To reuse storage, release the class, refill its buffer with evaluations, and
construct a new class. Classes share one scratch buffer in sequence, with
parallel work inside each transform.
`interpolation_scratch` returns the maximum individual requirement, independent
of the number of classes, and validates all class sizes before execution.
The const query `options.interpolation_requirements(output_size, lift_sizes)`
provides the same size and execution checks from domain sizes, so arrays can be
declared before constructing classes. The bound query additionally checks that
every class still contains evaluations.

[`interpolate_classes_parallel`](../crates/udon/src/fft/interpolation_parallel.rs)
uses `InterpolationOptions` to partition a total task budget across classes and
within their inverses. Each concurrent class worker gets a separate caller-owned
scratch partition; an addition pass follows the independent inverses. Its const
and bound queries return the same partition geometry. Lifts retain their own
coefficient vectors, as with sequential fused interpolation.

`interpolate_sum` offers a distinct destructive contract. It combines
equal-domain evaluation vectors before one inverse per distinct domain, mapping
different storage orders during addition. It shares one scratch region and
promises only the output's coefficient sum. All lifts become
`ClassState::Consumed`; individual lift coefficients are unavailable afterward.
`Class::state` distinguishes evaluations, successful coefficient output, and
consumed storage. A panic can leave classes in different phases, and every class
whose evaluations were consumed rejects reuse.

For correctness coverage and performance commands, see the
[testing guide](TESTING.md) and [strategy measurements](FFT_PERFORMANCE.md).
