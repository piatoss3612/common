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

Ordinary scalar multiplication requires no precomputation or scratch. Scalars
must satisfy the [field type's](../crates/udon/src/field/mod.rs) reduced-residue
invariant, including when read from POD storage.

## Canonical encodings and stored points

Use `to_bytes()` and `from_bytes()` to exchange canonical 32-byte compressed
points. `Point` supports identity; `AffinePoint::from_bytes()` rejects it.
The [encoding methods](../crates/udon/src/curve/encoding.rs) define the byte
format and rejection rules. These prime-order groups need no additional
subgroup check after decoding.

For direct embedded storage, only nonidentity affine points implement
`bento::Pod`. The [point type](../crates/udon/src/curve/mod.rs) defines the
64-byte Montgomery layout and mathematical invariants. POD checks memory
layout; arithmetic and encoding also require reduced coordinates on the curve.
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

Use `FixedBaseTable<C>` when many scalars act on one nonidentity base. The
`PallasFixedBase` and `VestaFixedBase` aliases select the curve. A table borrows
caller-owned affine entries; multiplication returns a projective point and
uses no doubling, scratch, or allocation.

`FixedBaseDescription { window_bits: w }` accepts widths `2..=8`, with width 4
as the default. Each window stores shifted multiples of the base, and an
additional entry handles the final carry from signed-digit recoding. The
[description docs](../crates/udon/src/curve/fixed_base.rs) define the entry
order and multiples required for binding stored tables.

Use the const query `description.requirements()` to size the affine destination
and both scratch buffers. The destination length must match exactly; scratch
can be larger and can be reused after preparation.

| Window bits | Affine entries | Table bytes | Projective scratch | Field scratch | Total scratch bytes |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 513 | 32,832 | 8 | 8 | 1,024 |
| 8 | 4,097 | 262,208 | 128 | 128 | 16,384 |

Bytes exclude the base and table handle. Scratch byte counts describe the
current implementation: each projective element is 96 bytes and each field
element is 32 bytes. Larger windows trade additional stored multiples for
fewer additions during execution.

```rust
use udon::{
    curve::{
        FixedBaseDescription, FixedBaseRequirements, PallasAffine,
        PallasFixedBase, PallasProjective,
    },
    field::{Fp, Fq},
};

const DESCRIPTION: FixedBaseDescription = FixedBaseDescription { window_bits: 4 };
const REQUIRED: FixedBaseRequirements = match DESCRIPTION.requirements() {
    Ok(required) => required,
    Err(_) => panic!("invalid table description"),
};
let base = PallasAffine::GENERATOR;
let mut entries = [base; REQUIRED.affine_points];
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

Preparation returns a table that borrows the affine entries, leaving both
scratch buffers available for other work. Errors leave all buffers unchanged.

For stored tables, `bind(description, base, entries)` validates every entry
against its specified multiple without scratch or inversion. Use `bind_trusted`
only when the owner has already established the full entry contract in the
[table docs](../crates/udon/src/curve/fixed_base.rs). Incorrect entries remain
memory-safe but can make multiplication panic or return incorrect results.
`validate()` performs the full entry check on an existing view.

Artifact schemas, curve identification, window metadata, and file generation
belong to the downstream owner. Udon's `STORED_FORM` continues to describe the
field representation only. See
[POD storage](POD.md#storing-affine-points-and-fixed-base-tables) for a complete
generator and consumer example.

See the [testing guide](TESTING.md) for independent arithmetic checks and
[curve benchmarks](TESTING.md#curve-benchmarks) separating setup, binding, and
repeated multiplication.
