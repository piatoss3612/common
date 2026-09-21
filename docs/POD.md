# POD storage and embedding

The [`bento`](../crates/bento/src/lib.rs) facade lets artifact generators write
records as bytes and consumers embed those files as typed static data. Byte
views borrow existing values; embedding supplies aligned storage without
allocation or runtime initialization.

POD means plain old data. The [`Pod` contract][pod-contract] specifies which
representations can be shared safely as both values and bytes.

## Defining a record

Define stored types in the crate that owns the artifact format. Apply `repr(C)`
or `repr(transparent)` and derive `Clone`, `Copy`, and [`bento::Pod`][pod-derive]:

```rust
#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
struct Record {
    low: u16,
    high: u16,
    value: u32,
}
```

The [`Pod` contract][pod-contract] defines supported layouts and the obligations
of handwritten unsafe implementations. Generic records and marker fields are
supported; see the [derive documentation][pod-derive] for their requirements.

`T: Pod` alone does not establish a usable layout. A generic struct can have
some instantiations suitable for storage and others that are only used as
ordinary values. The byte-view and embedding APIs force compile-time evaluation
of [`Pod::ASSERT_LAYOUT`][pod-contract], even for empty slices or zero-length
arrays. Callers of these APIs need no separate assertion. An explicit assertion
can check a particular format without converting a value:

```rust
const _: () = <Record as bento::Pod>::ASSERT_LAYOUT;
```

New unsafe storage consumers must follow the [`Pod` safety
requirements][pod-contract] for evaluating layout assertions as part of the
conversion.

## Writing and embedding files

Generators pass the byte views from [`bytes_of`][byte-views] or
[`bytes_of_slice`][byte-views] to their file-writing API. These views contain the
exact little-endian stored representation, without a header or length prefix.

Consumers use [`embed_struct!`][embedding] or [`embed_array!`][embedding] to
declare typed statics using the same stored types:

```rust
bento::embed_struct! {
    static RECORD: Record = "record.bin";
}

bento::embed_array! {
    pub static RECORDS: [Record; 16] = concat!(env!("OUT_DIR"), "/records.bin");
}
```

These declarations expose `&'static Record` and `&'static [Record; 16]`,
respectively. Attributes and visibility apply to the declared static. Literal
paths are relative to the source file containing the invocation; path
expressions follow the [embedding macros' path conventions][embedding]. File
length must match the requested type exactly.

The [embedding example](../crates/bento/examples/embed.rs) includes a small
[record file](../crates/bento/examples/data/record.bin) containing bytes `01`
through `08`. It verifies both record and array views against values with the
same stored representation:

```console
cargo run --release -p zakura-bento --example embed
```

Use [`AlignedBytes`][byte-views] when the bytes are already available. Its typed
views enforce the same layout and length checks.

## Storing field elements

Udon's [`Fp` and `Fq`][field-storage] implement `Pod`. A generator constructs
field values normally and writes their existing Montgomery representation.
Consumers embed those same types and use them directly in arithmetic, including
when they are fields of a larger record. The field types' storage contract
requires reduced Montgomery residues for the correct modulus; embedding checks
layout and length but does not validate those residues. Canonical protocol bytes
from `to_bytes()` encode a different representation and must not be embedded as
field storage bytes.

Give the artifact's crate both `udon` and `bento` dependencies under
`[dependencies]` and `[build-dependencies]` so its build script and consumer use
the same types, following the
[dependency naming guide](CRATES.md#dependencies-and-names).
For example, a downstream `build.rs` can write an `Fp` array:

```rust
use std::{env, fs, path::PathBuf};
use udon::{STORED_FORM, field::Fp};

fn main() {
    println!("cargo::rerun-if-changed=build.rs");
    let directory = PathBuf::from(env::var_os("OUT_DIR").unwrap());
    let values = [0, 1, 7, u64::MAX].map(|n| Fp::from_u64(n).square());
    fs::write(
        directory.join(format!("fp-values-{STORED_FORM}.bin")),
        bento::bytes_of_slice(&values),
    )
    .unwrap();
}
```

The consumer borrows those fields from aligned static storage:

```rust
use udon::field::Fp;

bento::embed_array! {
    static VALUES: [Fp; 4] =
        concat!(env!("OUT_DIR"), "/fp-values-", udon::stored_form!(), ".bin");
}

fn main() {
    assert_eq!(VALUES[2], Fp::from_u64(49));
    assert_eq!(VALUES[2].sqrt().unwrap().square(), VALUES[2]);
}
```

The [`STORED_FORM` constant][stored-forms] defines the descriptor's representation
and scope. Build scripts name files with this constant; consumers obtain the
matching string literal through `stored_form!`. The artifact owner must still
define the record schema and distinguish the two field moduli.

Generators and consumers may choose different `sqrt-table-large` configurations:
the feature preserves the stored field representation. See the [performance
report](FIELD_PERFORMANCE.md#optional-larger-square-root-tables) for table
tradeoffs.

The [field embedding test](../crates/udon/tests/embedding.rs) runs a complete
build script and consumer with both table configurations and checks that the
generated artifacts are byte-identical across them. It embeds a shared record
containing arrays of both fields and a separate array of `Fp`:

```console
cargo test --release --locked -p zakura-udon --test embedding -- --ignored
```

## Storing affine points and fixed-base tables

Udon's [`AffinePoint`](../crates/udon/src/curve/mod.rs) and
[`PreparedAffinePoint`](../crates/udon/src/curve/table_entry.rs) implement `Pod`
for both Pallas and Vesta. Their type docs define the coordinate layouts and
mathematical invariants, including the cached endomorphism coordinate in
`PreparedAffinePoint`. The 32-byte compressed protocol encoding is a different
representation. Identity-capable `Point` and Jacobian `ProjectivePoint` do not
implement `Pod`.

Every stored coordinate bit pattern is memory-safe, but point arithmetic and
encoding assume reduced coordinates satisfying the curve equation. Validate
individual points by passing the values from `coordinates()` to
`AffinePoint::from_xy` before use when their producer has not established these
properties. POD layout checks do not validate mathematical contents.

For repeated multiplication, prepare `FixedBaseTable<C, E>` or
`EisensteinTable<C, E>` entries into caller-owned storage and write the resulting
slice or an enclosing record through Bento POD. Both accept `AffinePoint<C>`
(the default) or `PreparedAffinePoint<C>` entries. Embed the same types in the
consumer and use the table's `bind` method to check its mathematical contents,
including cached coordinates. `bind_trusted` is available when the owner has
already established the specified multiples and caches; it checks only the
base, length, and expanded table description when present. The
[expanded](../crates/udon/src/curve/fixed_base.rs) and
[compact](../crates/udon/src/curve/eisenstein.rs) API docs define entry order;
the [curve guide](CURVES.md#fixed-base-multiplication) shows preparation and
storage costs.

`STORED_FORM` identifies the field representation.
The owner must separately identify the curve, base, compact or expanded table
kind, affine or cached entry representation, window width when present, and
record schema.

The [curve embedding fixture](../crates/udon/tests/fixtures/curve_embedding)
shares its record definition between generator and `no_std` consumer. It
demonstrates both table kinds and entry types, writing them through Bento POD
and multiplying directly from embedded storage. The
[testing guide](TESTING.md#generated-artifacts) describes its feature coverage
and damaged-artifact checks. Run it with:

```console
cargo test --release --locked -p zakura-udon --test curve_embedding -- --ignored
```

## Format ownership

Generator and consumer must agree on the stored type definitions, representation
attributes, and any semantic invariants. Layout validation cannot detect a file
generated for a different type of the same size, check canonical field residues,
or establish curve membership. Those checks belong to the artifact's owner.

These format requirements are separate from the [`Pod` contract][pod-contract],
which requires memory safety for every bit pattern admitted by the safe storage
APIs. A trusted generator cannot satisfy that obligation on behalf of arbitrary
callers. Udon's [field storage contract][field-storage] describes the distinction
for field residues.

POD storage uses little-endian bytes and validates primitive size and alignment
on the target. Unsupported endianness or layouts fail when their layout
assertions are evaluated; unrelated crate functionality remains available.

Ordinary facade dependency aliases are discovered automatically. Use
`#[pod(crate = path)]` for build-only dependencies, direct-core consumers,
indirect re-exports, or ambiguous arrangements; see the
[macro path conventions](MACROS.md#support-paths).

## Validation

Follow the [testing guide](TESTING.md) for compiler, Miri, and portability
checks. Use full builds to establish deferred layout failures, and generate
files through the storage API when testing the artifact workflow. The
[embedding round trip](../crates/bento/tests/embedding.rs) is one such consumer.

[pod-derive]: ../crates/bento/src/lib.rs
[pod-contract]: ../crates/bento-core/src/pod/mod.rs
[byte-views]: ../crates/bento-core/src/pod/storage.rs
[embedding]: ../crates/bento-core/src/pod/macros.rs
[field-storage]: ../crates/udon/src/field/mod.rs
[stored-forms]: ../crates/udon/src/stored_form.rs
