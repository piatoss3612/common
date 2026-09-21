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
private and can be borrowed through `coordinates()`.

`AffinePoint::from_xy(x, y)` checks reduced field residues before evaluating
the curve equation. It rejects `(0, 0)` and other invalid coordinates.
`Point::from_xy(x, y)` additionally accepts `(0, 0)` as identity. For constants,
`pallas_affine!(x, y)` and `vesta_affine!(x, y)` enforce the same nonidentity
coordinate checks during compilation, including in runtime expression
positions. Their inputs must be constant expressions of the correct field:

```rust
use udon::{curve::PallasAffine, fp_hex, pallas_affine};

const BASE: PallasAffine = pallas_affine!(
    fp_hex!("0x40000000000000000000000000000000224698fc094cf91b992d30ed00000000"),
    fp_hex!("0x0000000000000000000000000000000000000000000000000000000000000002"),
);
assert_eq!(BASE, PallasAffine::GENERATOR);
```

`to_projective()` lifts an affine point without inversion; `to_point()`
normalizes a projective point, using one inversion for a nonidentity result.
Projective equality compares group elements without inversion, even when their
coordinates have different scales. Keep intermediate results projective and
use `add_mixed` when the other operand is already affine:

```rust
use udon::{curve::PallasAffine, field::Fq};

let base = PallasAffine::GENERATOR;
let result = base.mul_projective(&Fq::from_u64(7)).add_mixed(&base);
assert_eq!(result, base.mul_projective(&Fq::from_u64(8)));
assert!(result.to_point().sub(&result.to_point()).is_identity());
```

Ordinary scalar multiplication needs no caller preparation or scratch and uses
bounded internal stack storage. The current implementation uses an inversion-free
binary ladder for scalars below `2^64`. For larger scalars and nonidentity bases,
each call prepares eight cached entries with one field inversion before running
the GLV/Eisenstein ladder. The API selects the strategy internally.
Scalars must satisfy the [field type's](../crates/udon/src/field/mod.rs)
reduced-residue invariant, including when read from POD storage. The
[performance report](CURVE_PERFORMANCE.md#ordinary-multiplication) describes
the current algorithm and its measured costs.

### GLV decomposition and the endomorphism

All three point representations provide
[`endomorphism()`](../crates/udon/src/curve/affine.rs): it maps `(x, y)` to
`(zeta * x, y)` using the coordinate field's cube root of unity and preserves
identity. On projective points it multiplies `X` by `zeta`, leaving `Y` and `Z`
unchanged. This equals multiplication by the scalar field's `zeta()`, denoted
`lambda` below.

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
additional storage. POD checks memory layout; arithmetic and encoding also
require reduced coordinates on the curve and consistent cached coordinates.
Use checked construction or table binding when the producer has not
established these properties. See
[POD storage](POD.md#storing-affine-points-and-fixed-base-tables) for the
generator and consumer workflow.

## Batch normalization

Use [`batch_normalize`](../crates/udon/src/curve/batch.rs) when several
projective results need affine coordinates. It shares one inversion across
nonidentity points and preserves input order and identity positions. Provide
one output point and one field scratch element per input point. The function
docs include an executable example and the complete buffer and error contract.

## Fixed-base multiplication

Retain preparation when several scalars act on one nonidentity base. Both
`EisensteinTable<C, E>` and `FixedBaseTable<C, E>` borrow caller-owned entries
and return projective products without allocation or caller scratch. Their
sealed `CurveTableEntry<C>` parameter accepts `AffinePoint<C>` (the default)
or `PreparedAffinePoint<C>` (cached endomorphism coordinates).
Generic code initializes entry buffers through `CurveTableEntry::from_affine`;
the [trait docs](../crates/udon/src/curve/table_entry.rs) define its construction,
coordinate access, rotation, and cache-checking methods. Both table kinds report
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
or `bind(base, entries)` for checked use of existing storage.

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

For repeated large batches, `scalar.certify_batch()` optionally retains the
check for exceptional affine intermediates. Ordinary scalar preparation leaves
this check unevaluated. Certification preserves complete projective arithmetic
for ladders with exceptional intermediates. See
[`EisensteinScalar::certify_batch`](../crates/udon/src/curve/eisenstein.rs) for the
reuse contract.

Use `bind` or `bind_trusted` for stored entries, following the
[table validation workflow](#preparation-binding-and-stored-formats). A single
compact table's eight entries can also be bound as a one-base batch without
changing the stored format. The
[performance report](CURVE_PERFORMANCE.md#compact-table-batches-and-scalar-reuse)
describes when shared preparation and same-scalar multiplication pay off.

### Expanded tables

`FixedBaseTable<C>` stores shifted multiples to avoid all doublings during
multiplication. Select the curve with `FixedBaseTable<Pallas>` or
`FixedBaseTable<Vesta>`, and optionally specify a prepared entry type.

`FixedBaseDescription { window_bits: w }` accepts widths `2..=8`, with width 4
as the default. The two GLV halves share `ceil(128 / w)` windows, each storing
`2^(w - 1)` shifted multiples of the base. An additional entry handles the final
carry from signed-digit recoding; the second half applies the endomorphism to
its lookups. The [description docs](../crates/udon/src/curve/fixed_base.rs)
define the entry order and multiples required for binding stored tables.

Use the const query `description.requirements()` to size the entry destination
and both scratch buffers. The destination length must match exactly; scratch
can be larger and can be reused after preparation.

| Window bits | Entries | Affine bytes | Cached bytes | Projective scratch | Field scratch | Total scratch bytes |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 257 | 16,448 | 24,672 | 8 | 8 | 1,024 |
| 8 | 2,049 | 131,136 | 196,704 | 128 | 128 | 16,384 |

Bytes exclude the base and table handle. Scratch byte counts describe the
current implementation: each projective element is 96 bytes and each field
element is 32 bytes. Larger windows trade additional stored multiples for
fewer additions during execution.

```rust
use udon::{
    curve::{
        CurveTableRequirements, FixedBaseDescription, PallasAffine,
        FixedBaseTable, Pallas, PallasProjective,
    },
    field::{Fp, Fq},
};

const DESCRIPTION: FixedBaseDescription = FixedBaseDescription { window_bits: 4 };
const REQUIRED: CurveTableRequirements = match DESCRIPTION.requirements() {
    Ok(required) => required,
    Err(_) => panic!("invalid table description"),
};
let base = PallasAffine::GENERATOR;
let mut entries = [base; REQUIRED.table_entries];
let mut projective = [PallasProjective::IDENTITY; REQUIRED.projective_scratch];
let mut field = [Fp::ZERO; REQUIRED.field_scratch];
let table = FixedBaseTable::<Pallas>::prepare(
    DESCRIPTION,
    &base,
    &mut entries,
    &mut projective,
    &mut field,
).unwrap();
let scalar = Fq::from_u64(42);
assert_eq!(table.mul(&scalar), base.mul_projective(&scalar));
```

To cache endomorphism coordinates, initialize `entries` with
`PreparedAffinePoint::from_affine(&base)` and select
`FixedBaseTable::<Pallas, PreparedAffinePoint<Pallas>>` instead. Preparation uses the
same scratch lengths for either entry type.

### Preparation, binding, and stored formats

Compact and expanded tables, including compact batches, return views borrowing
only the entries, leaving scratch available for other work. Entry lengths must
match exactly; scratch may be longer, and preparation leaves unused tails
untouched. Preparation errors leave all buffers unchanged.

Checked `bind` validates each entry against its specified multiple, including
cached coordinates, without scratch or inversion. Use `bind_trusted`
only when the owner has already established the full entry contract in the
[expanded](../crates/udon/src/curve/fixed_base.rs) or
[compact table docs](../crates/udon/src/curve/eisenstein.rs). Incorrect entries
remain memory-safe but can make multiplication panic or return incorrect results.
`validate()` performs the full entry check on an existing view.

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
directly. Construction checks lengths and index bounds without gathering bases;
the producer must establish the mathematical invariants documented on each type.

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
original scalar borrow. It remains usable across chunk sizes, widths, and
backends. Its optional `cache(arithmetic, bytes)` retains recoding as well; pass
`ArithmeticOptions` to it and `cache_len(arithmetic)`. Cache allocation covers
the complete scalar vector independently of batch memory policy. See
[`PreparedScalars::cache`](../crates/udon/src/curve/msm/prepared.rs) for the
geometry and chunking conditions that permit reuse.
`retained_bytes()` counts the borrowed records and optional cache. This storage
is separate from execution scratch and is not a POD serialization format.

### Sizing and reusing scratch

[`ArithmeticOptions`](../crates/udon/src/curve/msm/mod.rs) holds reusable kernel,
accumulation, and staging choices. `with_max_terms_per_pass` caps affine bucket
and temporary table staging; it alone does not bound scalar records or recoding
bytes. `with_chunk_size` caps the terms prepared together. Independent chunk
MSMs reuse temporary buffers, trading repeated collapses for storage independent
of total input size.

`BatchOptions::new(arithmetic)` adds serial execution without a memory ceiling;
`BatchOptions::default()` also uses automatic arithmetic choices. Its
`with_task_budget` and `with_memory_limit` control batch concurrency and adaptive
storage planning. The batch chunk cap defaults to 8,192 terms. A memory ceiling
can reduce pass size, concurrency, or chunk size and change automatic kernel
choices.

`ArithmeticOptions::with_kernel` replaces the complete arithmetic selection.
`Kernel::Auto` selects joint, short-scalar, or nonstreaming Booth arithmetic.
`Kernel::Joint` requires the joint ladder. `Kernel::Booth { width, accumulation }`
requires Booth even for small inputs or short scalars, which can forgo faster
short-scalar arithmetic. `None` chooses a width within that family; supplied
widths must be in `4..=12`. Only `Accumulation::Auto` permits strategy changes.

`Kernel::StreamingBooth { width }` retains projective buckets for all windows
while preparing and recoding one chunk at a time. It avoids repeating each
chunk's weighted collapse, with a larger fixed workspace floor. Each deposit
owns one window's buckets; different windows can execute concurrently. This
choice supports indexed and retained bases. Streaming always uses projective
accumulation. A batch memory ceiling preserves explicit family, streaming mode,
width, and accumulation requirements, and returns `MemoryLimit` when its search
cannot fit them. See the [performance guide](CURVE_PERFORMANCE.md#multiscalar-multiplication)
for tuning evidence and measurement limits.

The ceiling counts required execution buffer prefixes and reserved plan metadata;
it excludes retained preparation and surplus buffer tails. The accounting and
search contract lives on
[`BatchOptions::with_memory_limit`](../crates/udon/src/curve/msm/mod.rs).
The search is not exhaustive. A `MemoryLimit` error reports the storage needed
at its stopping point, which may be the metadata alone; it does not establish
the minimum possible storage. Planning and scratch errors precede writes.

For application-controlled scheduling, `MsmPlan::new` takes `ArithmeticOptions`
and a term grain. Its retained-slot and per-task scratch queries return bounds;
they do not apply a batch concurrency or memory policy. The caller accounts for
all simultaneous scratch bundles, retained intermediates, metadata, and idle
provider capacity. See the
[run admission protocol](EXECUTION.md#admission-with-a-progress-reservation)
for composing those requirements under an application-wide ceiling.

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
execution scratch; metadata counts against the memory ceiling. Its
`temporary_bytes()` includes reserved metadata prefixes.

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
budget to its `BatchOptions`. Ordinary fixed-base products need no executor.
This uses the same scoped execution contract as FFTs and works inside an
existing pool, including a one-thread pool.

## Validation and performance

See the [testing guide](TESTING.md) for independent arithmetic checks and
[curve benchmarks](TESTING.md#curve-benchmarks) separating setup, binding, and
repeated multiplication. The [performance report](CURVE_PERFORMANCE.md) records
measured latency and storage tradeoffs.
