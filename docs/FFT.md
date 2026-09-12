# Field FFTs

Udon's [`fft` module](../crates/udon/src/fft/mod.rs) provides radix-2 field FFTs
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

## Domains and transform order

[`Domain`](../crates/udon/src/fft/domain.rs) constructs a subgroup with the
canonical Pasta root for its size. Use `Domain::new(k)` for `2^k` elements or
`Domain::for_size(n)` for an element count. The constructor documents supported
orders and target address-space limits. Size one is supported.

`domain.subgroup()` selects shift one. `domain.coset(shift)` accepts any nonzero
shift, including shifts inside the subgroup. For size `n`, canonical `root`,
and coefficients `c[i]`, natural evaluation row `j` is

```text
sum(c[i] * (shift * root^j)^i, i = 0..n), for 0 <= j < n.
```

[`Plan::forward`](../crates/udon/src/fft/transform.rs) replaces coefficients with
these evaluations. `inverse` removes both the shift and the domain-size factor,
returning the original coefficients. Both input and output use natural order.
`forward_into` and `inverse_into` preserve a separate input slice and overwrite
the caller's output. `inverse_bit_reversed` accepts evaluations already placed
at bit-reversed positions and omits the input permutation.

Use `forward_prefix` when only the low-degree coefficients are present. It
treats omitted coefficients as zero, and an empty prefix as the zero polynomial.
Its output still needs the full domain size. The `Plan` documentation defines
buffer lengths, scratch requirements, and errors for each transform method.

The [`reference` module](../crates/udon/src/fft/reference.rs) retains a simple
generic transform through `Twiddle` and `Butterfly`. It takes explicit roots;
its inverse also takes the inverse size. This is useful as an independent
schedule or for a downstream value type. Its trait contracts define the
algebraic laws that a custom implementation must satisfy.

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

Binding tables checks lengths without checking their contents. Call
`Tables::validate` to check a generator or artifact against its domain. It also
rejects unreduced Montgomery entries. Incorrect table contents can cause wrong
results or panics; all APIs remain memory safe. Prepared inverse-finish and
scaling tables depend on the coset shift, even though ordinary twiddles do not.

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

Arbitrary nonzero coset shifts are supported. Shifts of order three, including
`zeta`, use repeating coefficient scales. A generic shift currently requires a
serial progression of field multiplications before the forward transform, even
with prepared twiddles. The [benchmarks](TESTING.md#fft-benchmarks) measure both
`zeta` and the generic shift 7 so this cost is included in comparisons.

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
and a coset of equal or larger size. Their ratio `r` is any supported power of
two, including one. Residue `s` consists of natural rows `s + r*k`, where
`0 <= k < base_size` and `0 <= s < r`.

Output is residue-major: all rows for residue zero, then residue one, and so on.
For natural extended row `j`, its storage index is

```text
(j % r) * base_size + j / r.
```

[`ResidueLayout` and `ResidueView`](../crates/udon/src/fft/layout.rs) provide
checked index conversion, residue slices, and lookup at rows of a still larger
domain with the same shift. Such lookup rejects rows outside the smaller
domain. Copy helpers convert between natural and residue order into distinct
caller buffers. `CoefficientTiles` instead borrows contiguous coefficient
ranges; its tiles are not residue classes.

`Expansion::coefficients` accepts any prefix fitting the base domain. An optional
table holds exactly `extended_size` residue scales; `prepare_scales` fills caller
storage and `validate_scales` checks an existing table.

All expansion operations accept
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
The first output residue holds the unnormalized inverse while the remaining
residues read its coefficients, fusing normalization into their residue scales.
After those residues finish, the first residue is scaled and transformed in
place. The inverse and every residue use `options.transform`. Scratch is reused
between these phases; the requirement accounts for the number of remaining
residues that can run concurrently.

`short_product` expands a nonempty short prefix and multiplies a supplied
residue-major factor into each completed residue. It checks the factor layout;
the caller supplies factor values for the same coset. Its scheduling and scratch
requirements are the same as `coefficients`.
If the product will be interpolated, choose an extended domain larger than its
degree to recover all coefficients.

## Fused class interpolation

[`Class`](../crates/udon/src/fft/interpolation.rs) pairs a plan with a borrowed,
initialized evaluation buffer. It accepts natural or bit-reversed input order.
`scatter` and `scatter_strided` write natural evaluation positions directly to
the requested storage order, checking the requested natural positions before
writing. A supplied bit-reversal table must satisfy `Tables`' content contract;
an invalid entry can panic after partial writes. Unwritten entries retain their
prior values; callers own completeness and accumulation.

`interpolate_classes` interpolates one output class and any number of smaller
lift classes. Each lift must fit in half the output domain, but classes may
have different nonzero coset shifts. The result is the sum of their coefficient
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

For correctness coverage and performance commands, see the
[testing guide](TESTING.md).
