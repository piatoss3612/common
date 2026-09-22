# Pasta curve arithmetic

Udon's [`curve` module](../crates/udon/src/curve/mod.rs) provides Pallas and
Vesta arithmetic. Both have equation `y² = x³ + 5`, generator `(-1, 2)`, and
prime order. The sealed `PastaCurve` trait associates each curve with its
coordinate field and scalar field; the scalar modulus is the group order:

| Curve marker | Coordinate field | Scalar field |
| --- | --- | --- |
| `Pallas` | `Fp` | `Fq` |
| `Vesta` | `Fq` | `Fp` |

All operations are variable-time and provide no constant-time guarantee for
secret inputs, including bases, scalars, and table contents. They work in
`no_std`, require no allocation, and are available without a feature flag. The
optional `sqrt-table-large` field feature also affects point decoding; it
preserves point encodings and storage. See the
[crate docs](../crates/udon/src/lib.rs) for feature definitions.

## Choosing a point representation

| Generic type | Pallas alias | Vesta alias | Representation |
| --- | --- | --- | --- |
| `AffinePoint<C>` | `PallasAffine` | `VestaAffine` | Nonidentity `(x, y)` |
| `Point<C>` | `PallasPoint` | `VestaPoint` | Optional affine point, including identity |
| `ProjectivePoint<C>` | `PallasProjective` | `VestaProjective` | Jacobian `(X, Y, Z)` |

Use `AffinePoint` for a known nonidentity base, `Point` when affine results can
include identity, and `ProjectivePoint` to accumulate arithmetic before
normalizing. `Point::as_affine()` returns `None` for identity. Coordinates are
private and can be borrowed through `coordinates()`. Affine coordinates use
`PastaField<_, Reduced>`; projective coordinates use loose field elements.

`Point::add`, `sub`, and `double` return `ProjectivePoint` without inversion.
The caller decides when affine coordinates are needed and normalizes explicitly.

`AffinePoint::from_xy(x, y)` takes reduced field elements and checks the curve
equation. Reduction is guaranteed by the coordinate types. It rejects `(0, 0)`
and other off-curve coordinates.
`Point::from_xy(x, y)` additionally accepts `(0, 0)` as identity. For constants,
`pallas_affine!(x, y)` and `vesta_affine!(x, y)` enforce the same nonidentity
curve checks during compilation, including in runtime expression positions.
Their inputs may use either reduction state and must be constant expressions
of the correct field:

```rust
use udon::{curve::PallasAffine, fp_hex, pallas_affine};

const BASE: PallasAffine = pallas_affine!(
    fp_hex!("0x40000000000000000000000000000000224698fc094cf91b992d30ed00000000"),
    fp_hex!("0x0000000000000000000000000000000000000000000000000000000000000002"),
);
assert_eq!(BASE, PallasAffine::GENERATOR);
```

`to_projective()` lifts an affine point without inversion; `to_point()`
normalizes a projective point, using one inversion only when `z` is neither
zero nor one. Points with `z = 1` need only coordinate reduction.
Projective equality compares group elements without inversion, even when their
coordinates have different scales. Keep intermediate results projective and
use `add_mixed` when the other operand is already affine:

```rust
use udon::{curve::PallasAffine, field::Fq};

let base = PallasAffine::GENERATOR;
let result = base.mul_projective(&Fq::from_u64(7)).add_mixed(&base);
assert_eq!(result, base.mul_projective(&Fq::from_u64(8)));
assert!(result.sub(&result).is_identity());
```

[`ProjectivePoint::incomplete_double_and_add`](../crates/udon/src/curve/projective.rs)
fuses `A + B` and `A + (A + B)` for projective `A` and affine `B`. It returns
the resulting point and both slope numerators without inversion. Callers can
use [`try_batch_invert_by`](../crates/udon/src/field/batch.rs) on the returned
points' `z` coordinates to recover slopes and affine coordinates together.
The method's docs define the rejected inputs and show slope recovery.

Ordinary scalar multiplication needs no caller preparation or scratch and uses
bounded internal stack storage. The current implementation uses an inversion-free
binary ladder for scalars below `2^64`. For larger scalars and nonidentity bases,
each call prepares eight cached entries with one field inversion before running
the GLV/Eisenstein ladder. The API selects the strategy internally.
Scalars use the [field type's](../crates/udon/src/field/mod.rs) loose
representation, including when read from POD storage. The
[performance report](CURVE_PERFORMANCE.md#ordinary-multiplication) describes
the current algorithm and its measured costs.

### GLV decomposition and the endomorphism

All three point representations provide
[`endomorphism()`](../crates/udon/src/curve/affine.rs): it maps `(x, y)` to
`(zeta * x, y)` using the coordinate field's
[`ZETA`](../crates/udon/src/field/parameters.rs) constant and preserves identity.
On projective points it multiplies `X` by `zeta`, leaving `Y` and `Z` unchanged.
This equals multiplication by the scalar field's `ZETA`, denoted `lambda` below.

GLV decomposition writes one scalar as two smaller signed integers using this
endomorphism. [`glv_decompose::<C>(&scalar)`](../crates/udon/src/curve/glv.rs)
returns `(a, b)` satisfying `scalar = a + lambda * b` modulo the group order,
with both magnitudes strictly below `2^127`. Write `[k] P` for multiplication of
point `P` by integer `k`; then `[scalar] P = [a] P + [b] P.endomorphism()`.
The function's docs include an executable reconstruction example.

## Canonical encodings and stored points

Use `to_bytes()` and `from_bytes()` to exchange canonical 32-byte compressed
points. `Point` supports identity; `AffinePoint::from_bytes()` rejects it.
The [encoding methods](../crates/udon/src/curve/encoding.rs) define the byte
format and rejection rules. These prime-order groups need no additional
subgroup check after decoding.

For direct embedded storage,
[`AffinePoint<C>`](../crates/udon/src/curve/mod.rs) and
[`PreparedAffinePoint<C>`](../crates/udon/src/curve/table_entry.rs) implement
`bento::Pod`. Their type docs define the Montgomery layouts and mathematical
invariants. Cached entries accelerate endomorphism lookups at the cost of
additional storage. Point construction establishes reduced coordinates on the
curve and consistent cached coordinates. Bento preserves these values as bytes;
embedded points are immediately usable without validation or conversion. See
[POD storage](POD.md#storing-affine-points-and-fixed-base-tables) for the
generator and consumer workflow.

## Batch normalization

Use [`batch_normalize`](../crates/udon/src/curve/batch.rs) when several
projective results need affine coordinates. It shares one inversion across
points whose `z` is neither zero nor one, preserving input order and identity
positions. Batches containing only identity or already-affine points need no
inversion. Provide one output point per input point. One field scratch element
per input permits a single shared inversion; smaller scratch works in chunks,
including individual normalization with empty scratch. The function docs include
an executable example and the complete buffer contract.

For example, collect additions and doublings before requesting affine outputs:

```rust
use udon::{curve::{batch_normalize, PallasPoint}, field::Fp};

let base = PallasPoint::GENERATOR;
let projective = [base.add(&base), base.double(), base.sub(&base)];
let mut affine = [PallasPoint::IDENTITY; 3];
let mut scratch = [Fp::ZERO; 3];
batch_normalize(&projective, &mut affine, &mut scratch);
assert_eq!(affine[0], affine[1]);
assert!(affine[2].is_identity());
```

## Fixed-base multiplication

Retain preparation when several scalars act on one nonidentity base. Both
`EisensteinTable<C, E>` and `FixedBaseTable<C, E>` borrow caller-owned entries
and return projective products without allocation or caller scratch. Their
sealed `CurveTableEntry<C>` parameter accepts `AffinePoint<C>` (the default)
or `PreparedAffinePoint<C>` (cached endomorphism coordinates).
Generic code initializes entry buffers through `CurveTableEntry::from_affine`;
the [trait docs](../crates/udon/src/curve/table_entry.rs) define its construction,
coordinate access, and rotation methods. Both table kinds report
entry and scratch lengths through `CurveTableRequirements`.

### Compact tables

`EisensteinTable<C>` stores eight representatives, costing 512 bytes with affine
entries or 768 bytes with cached entries. Signed endomorphism rotations supply
48 possible joint digits; multiplication uses a doubling ladder over the two
GLV halves. Here a rotation applies the endomorphism zero, one, or two times;
each rotation can have either sign. The
[API docs](../crates/udon/src/curve/eisenstein.rs) specify the representative
order and show preparation with cached entries.

`EisensteinTable::<C, E>::REQUIREMENTS` reports eight entries and eight elements
of each scratch type. Preparation shares one inversion across all entries. Use
`prepare(base, entries, projective_scratch, field_scratch)` to fill the table,
or the const `bind(base, entries)` to borrow trusted stored entries.

When one scalar acts on several bases, construct `EisensteinScalar::<C>::new`
once and call each table's `mul_prepared(&scalar)`. The opaque value retains the
GLV decomposition's joint digits without borrowing the scalar or tables.
Ordinary `mul(&scalar)` prepares digits for a single product. Expanded tables
use their own width-specific recoding and take the field scalar directly.

### Compact table batches and same-scalar products

[`EisensteinTableBatch<C, E>`](../crates/udon/src/curve/eisenstein_batch.rs)
borrows a flat slice of consecutive eight-entry tables. Its const
`requirements(number_of_bases)` reports the exact entry count and minimum
projective and field scratch counts. `prepare` accepts either nonidentity base
layout and writes the caller-selected entry layout, allowing preparation to
share inversions across bases. Supply a `TaskBudget` and an `Executor`; serial
callers use `TaskBudget::SERIAL` and `SerialExecutor`. The counts are independent
of the budget, so the same buffers work with either executor.

The returned view borrows only entries. `get(index)` borrows an individual
`EisensteinTable`, while `mul_prepared` multiplies every base by an
`EisensteinScalar`, writing projective products in table order. `mul` also
prepares the scalar. The const `multiplication_scratch(number_of_bases)` query
reports field scratch for these batch methods. The type docs include an
executable example that reuses field scratch after preparation.

`EisensteinScalar::new` also determines whether the scalar permits batched affine
arithmetic. Callers retain the opaque preparation; Udon selects the batch
implementation from that fact and the available scratch. The multiplication
scratch query reports the preferred size. Smaller scratch selects smaller
batches or complete projective arithmetic, and empty scratch remains valid.

Use `bind` for stored entries, following the
[table storage workflow](#preparation-binding-and-stored-formats). A single
compact table's eight entries can also be bound as a one-base batch without
changing the stored format. The
[performance report](CURVE_PERFORMANCE.md#compact-table-batches-and-scalar-reuse)
describes when shared preparation and same-scalar multiplication pay off.

### Expanded tables

`FixedBaseTable<C>` stores shifted multiples to avoid all doublings during
multiplication. Select the curve with `FixedBaseTable<Pallas>` or
`FixedBaseTable<Vesta>`, and optionally specify a prepared entry type.

`FixedBaseTable::prepare_with(description, ...)` prepares an explicit layout
independently of the supplied scratch capacity. `FixedBaseTable::prepare`
optionally selects a layout from the entry and scratch capacities. Both borrow
only the used entry prefix, and `description()` identifies the stored layout.

`FixedBaseDescription { window_bits: w }` describes widths `2..=8`. The two GLV
halves share `ceil(128 / w)` windows, each storing `2^(w - 1)` shifted multiples
of the base. Only width 2 needs an additional entry for the final carry from
signed-digit recoding; the second half applies the endomorphism to its lookups.
The [description docs](../crates/udon/src/curve/fixed_base.rs)
define the entry order and multiples required for binding stored tables.

The const query `description.requirements()` reports the exact stored entry
count and minimum preparation scratch. Binding requires exactly that entry
count; preparation accepts larger buffers and leaves unused tails untouched.
The shorter scratch buffer determines how many whole windows share a batch
normalization. Minimum scratch normalizes one window at a time. With
`table_entries` elements of each scratch type, the entire table shares one
inversion, including the width-2 carry. Intermediate capacities batch as many
whole windows as fit and include the carry in the last batch when space permits.

| Window bits | Entries | Affine bytes | Cached bytes | Projective scratch | Field scratch | Total scratch bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 256 | 16,384 | 24,576 | 8 | 8 | 1,024 |
| 8 | 2,048 | 131,072 | 196,608 | 128 | 128 | 16,384 |

Bytes exclude the base and table handle. Scratch byte counts describe the
minimum allowance: each projective element is 96 bytes and each field element
is 32 bytes. At width 4, increasing total scratch from 1,024 to 32,768 bytes
reduces preparation from 32 inversions to one; at width 8, increasing it from
16,384 to 262,144 bytes reduces 16 inversions to one. Larger windows trade
additional stored multiples for fewer additions during execution.

The executable [preparation example](../crates/udon/src/curve/fixed_base.rs)
shows multiplication and rebinding with the selected description and entries.

To cache endomorphism coordinates, initialize `entries` with
`PreparedAffinePoint::from_affine(&base)` and select
`FixedBaseTable::<Pallas, PreparedAffinePoint<Pallas>>` instead. Preparation uses the
same scratch lengths for either entry type.

### Preparation, binding, and stored formats

Compact and expanded tables, including compact batches, return views borrowing
only the entries, leaving scratch available for other work. Compact preparation
and all table bindings require exact entry lengths; expanded preparation selects
a prefix of the entry capacity. Scratch may be longer, and unused tails remain
untouched. Preparation errors leave all buffers unchanged.

The const `bind` operations attach trusted entries to their base and table
description. They assert lengths and check configuration without inspecting
points, cached coordinates, or multiples. Preparation constructs the entries
according to the [expanded](../crates/udon/src/curve/fixed_base.rs) or
[compact table contract](../crates/udon/src/curve/eisenstein.rs); storing and
embedding them preserves those values exactly.

Artifact schemas, curve identification, table kind, entry representation, window
metadata, and file generation belong to the downstream owner. See
[POD storage](POD.md#storing-affine-points-and-fixed-base-tables) for the format
contract and a complete generator and consumer example.

## Multiscalar multiplication

[`curve::msm`](../crates/udon/src/curve/msm/mod.rs) computes sums of scalar/base
products, returning projective results. Choose the input view to match the
data already owned by the caller:

| Input | Construction | Contract |
| --- | --- | --- |
| Dense | `Input::new(bases, scalars)` | One scalar per base |
| Indexed | `Input::indexed(bases, indices, scalars)` | One `u32` index per scalar; repeated indices contribute separately |
| Dense, prepared scalars | `Input::new_prepared(bases, prepared)` | One prepared scalar per base |
| Indexed, prepared scalars | `Input::indexed_prepared(bases, indices, prepared)` | One index per prepared scalar |

`Bases::Affine`, `Bases::Prepared`, and `Bases::Points` borrow ordinary,
cached, and identity-capable base layouts. `Bases::Compact` and
`Bases::CompactPrepared` borrow `EisensteinTableBatch` in either entry layout,
including tables embedded with Bento. The compact ladder consumes those tables
directly. Construction checks lengths and index bounds without gathering bases.
Each base already carries its point type's invariants.

For prepared MSM over a retained base array, cache each nonidentity base with
`PreparedAffinePoint::from_affine` and borrow the result through
`Bases::Prepared`. Each entry stores the affine coordinates and one cached
endomorphism coordinate (96 bytes per base). The cached array can replace the
original affine array; keeping both uses 160 bytes per base.

Retain `Selection::new(bases)` when every row uses the whole base array, or
`Selection::indexed(bases, indices)` when several scalar rows use the
same mapping. `selection.with_scalars(row)` checks only the length, preserving
index validation independently of the scalar borrow. `with_unsigned(&[u128])`
and `with_signed(&[i128])` avoid Montgomery conversion; signed inputs include
`i128::MIN`. `with_canonical(integers, bits)` checks every integer against both
the scalar modulus and the declared bit bound before execution can write scratch.

For reuse across base sets, allocate initialized `ScalarStorage::ZERO` entries
using `PreparedScalars::<C>::storage_len(terms)`, then call
`PreparedScalars::prepare(scalars, storage, budget, executor)`. Preparation
retains signed GLV components and small-integer classification and releases the
original scalar borrow. Preparation is independent of execution choices.
`cache_len(&plan)` and `cache(&plan, bytes)` optionally retain the recoding
selected by an `MsmPlan`. A plan that cannot reuse a whole-row cache reports
zero cache bytes and leaves preparation unchanged. The same resolved geometry
sizes and consumes the cache, including at algorithm boundaries.
Resolve `MsmPlan::for_input` with the cached handle to size workspace without
duplicating the retained cache; caching does not change an existing plan's counts.
`retained_bytes()` counts the borrowed records and optional cache. This storage
is separate from execution scratch and is not a POD serialization format.

### Sizing and reusing scratch

[`ExecutionOptions`](../crates/udon/src/exec.rs) provides the same resource contract
for MSMs and FFTs: a total task budget and an optional workspace byte ceiling.
The default is serial execution without an additional ceiling. Udon chooses
recoding, window width, accumulation, and chunking from those limits and the
input's scalar and base facts.

For a single operation, `input.requirements(options)` reports preferred typed
scratch. `input.execute(options, executor, scratch)` can select a smaller layout
when actual buffer capacities require it. For reusable or incremental work,
`MsmPlan::for_input(&input, options)` resolves an opaque plan and its fixed
requirements. Reuse requires compatible scalar preparation and base storage,
as specified by the [plan contract](../crates/udon/src/curve/msm/run.rs).
`MsmPlan::new(terms, options)` is conservative when inputs are not yet available.
`MsmPlan::for_produced` additionally takes the maximum source
fragment the provider can lease; Udon chooses the arithmetic subdivisions.

The ceiling counts used arithmetic scratch and retained intermediates. It
excludes input/output, persistent preparation, metadata, unused buffer tails,
and executor resources. A `MemoryLimit` error reports storage at the planner's
stopping point, not a proven global minimum. Planning and scratch errors precede
writes.

Incremental callers use the plan's retained-slot and per-task scratch queries
to provision their provider. Additional retained slots must also fit the plan's
ceiling. The application still accounts for total simultaneous operations,
metadata, queues, alignment, and unused capacity. See the
[run admission protocol](EXECUTION.md#admission-with-a-progress-reservation).

`msm::run::BatchPlan::requirements()` returns counts for six private `Scratch`
slices: `scalars()`, `digits()`, `affine()`, `projective()`, `field()`, and
`indices()`. `Requirements::bytes::<C>()` computes their total with checked
arithmetic. Planning uses the actual inputs, so retained scalar records,
compatible digit caches, and compact tables can reduce the required storage.
Obtain counts from the plan rather than copying formulas.

Construct scratch with `Scratch::new(records, digits, affine, projective, field,
indices)`. Initialize records with `ScalarStorage::ZERO`, affine entries with
the generator, and other entries with zero or identity. Buffers can be reused
without clearing through `scratch.reborrow()`. Execution overwrites every value
it uses and leaves tails beyond the required prefixes untouched. The
[module example](../crates/udon/src/curve/msm/mod.rs) executes two signed scalar
rows with cached bases and one validated selection. The
[workspace example](WORKSPACES.md#owning-an-msm-workspace) retains scratch across
changing inputs, and the
[embedding fixture](../crates/udon/tests/fixtures/curve_embedding) uses fixed
arrays without allocation.

### Grouped jobs and other work

Use `msm::run::BatchPlan` for one or more borrowed `Input` handles on the same
curve. Its `storage_len(input_count, options)` returns job and worker metadata
counts; initialize those slices with `JobStorage::EMPTY` and
`WorkerStorage::EMPTY`, then call `BatchPlan::new`. Planning validates resource
limits before modifying metadata. Use the plan's `requirements()` to size its
execution scratch. Its `temporary_bytes()` reports arithmetic workspace bytes,
excluding metadata and unused tails.

`plan.execute(&mut output, executor, scratch)` writes one output per input in
input order. Jobs share the task budget, and sequential jobs reuse scratch.
Retain the plan to reuse its schedule with new outputs or dirty scratch. A plan
borrows its scalar rows; rebuild it when those rows change. Selection rebinding
and retained scalar preparation remain independent capabilities. The
[execution benchmarks](EXECUTION_PERFORMANCE.md) compare operation geometry and
caller-selected concurrency policies.

Compose fixed-base products or other work with
[`Executor::join`](../crates/udon/src/exec.rs). Choose per-operation budgets and
account for simultaneous scratch as described under
[scoped execution](WORKSPACES.md#scoped-execution), then pass the MSM branch's
budget to its `ExecutionOptions`. Ordinary fixed-base products need no executor.
This uses the same scoped execution contract as FFTs and works inside an
existing pool, including a one-thread pool.

## Validation and performance

See the [testing guide](TESTING.md) for independent arithmetic checks and
[curve benchmarks](TESTING.md#curve-benchmarks) separating setup, binding, and
repeated multiplication. The [performance report](CURVE_PERFORMANCE.md) records
measured latency and storage tradeoffs.
