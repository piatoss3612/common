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

### Expanded tables

`FixedBaseTable<C>` stores shifted multiples to avoid all doublings during
multiplication. The `PallasFixedBase` and `VestaFixedBase` aliases select the
curve and accept an optional entry type.

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
        PallasFixedBase, PallasProjective,
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
let table = PallasFixedBase::prepare(
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
`PallasFixedBase::<PreparedAffinePoint<Pallas>>` instead. Preparation uses the
same scratch lengths for either entry type.

### Preparation, binding, and stored formats

Both table types return a view borrowing only the entries, leaving both scratch
buffers available for other work. Entry lengths must match exactly; scratch
may be longer, and preparation leaves unused tails untouched. Preparation
errors leave all buffers unchanged.

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

### Migrating expanded tables

Expanded tables now cover 128-bit halves. Regenerate tables stored with the
older full-scalar layout. Rename `FixedBaseRequirements` to
`CurveTableRequirements` and its `affine_points` field to `table_entries`.
The default width-4 table has 257 entries rather than 513; width 8 has 2,049
rather than 4,097.
There is no compatibility binding for the older layout.

Bump the owner's schema when adopting this layout. Udon's `STORED_FORM`
continues to describe the unchanged field representation.

## Validation and performance

See the [testing guide](TESTING.md) for independent arithmetic checks and
[curve benchmarks](TESTING.md#curve-benchmarks) separating setup, binding, and
repeated multiplication. The [performance report](CURVE_PERFORMANCE.md) records
measured latency and storage tradeoffs.
