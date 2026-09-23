# POD storage and embedding

The [`bento`](../crates/bento/src/lib.rs) facade lets artifact generators write
already-constructed values as bytes and consumers embed those files as typed
static data. The stored bytes are trusted representations of the same types.
Their invariants hold by construction. Byte views and embedding preserve the
exact representation, with no runtime validation, reduction, or initialization.

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
length must match the requested type exactly; the compiler checks this even
when an `AlignedBytes` view is requested from runtime code.

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
when they are fields of a larger record. `Fp` and `Fq` default to the `Loose`
representation, whose Montgomery limbs are below twice the modulus. `Fp<Reduced>`
and `Fq<Reduced>` have limbs below the modulus. Both states implement `Pod` and
have the same four-limb layout. Storage preserves the state and limbs: a loose
value stays loose, and a reduced value stays reduced. Canonical protocol bytes
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
    let values = [0, 1, 7, u64::MAX].map(|n| <Fp>::from_u64(n).square());
    fs::write(
        directory.join(format!("fp-values-{STORED_FORM}.bin")),
        bento::bytes_of_slice(&values),
    )
    .unwrap();
}
```

The consumer borrows those fields from aligned static storage:

```rust
use udon::field::{Fp, Reduced};

bento::embed_array! {
    static VALUES: [Fp; 4] =
        concat!(env!("OUT_DIR"), "/fp-values-", udon::stored_form!(), ".bin");
}

fn main() {
    let value = VALUES[2].reduce();
    assert_eq!(value, Fp::<Reduced>::from_u64(49));
    assert_eq!(value.sqrt().unwrap().square().reduce(), value);
}
```

The [`STORED_FORM` constant][stored-forms] defines the descriptor's representation
and scope. Build scripts name files with this constant; consumers obtain the
matching string literal through `stored_form!`. The artifact owner must still
define the record schema and distinguish the field modulus and reduction state.
If a consumer needs reduced values immediately, the generator stores reduced
values and the record uses `Fp<Reduced>` or `Fq<Reduced>`. Calling `reduce()` is
an explicit arithmetic operation, never an embedding step.

Generators and consumers may choose different `sqrt-table-large` configurations:
the feature preserves the stored field representation. See the [performance
report](FIELD_PERFORMANCE.md#optional-larger-square-root-tables) for table
tradeoffs.

The [field embedding test](../crates/udon/tests/field/embedding.rs) runs a complete
build script and consumer with both table configurations and checks that the
generated artifacts are byte-identical across them. It embeds both fields in
both states, including representatives between the modulus and twice the
modulus, and checks exact preservation of their limbs:

```console
cargo test --release --locked -p zakura-udon --test field embedding:: -- --ignored
```

## Storing affine points and fixed-base tables

Udon's [`AffinePoint`](../crates/udon/src/curve/pasta/mod.rs) and
[`PreparedAffinePoint`](../crates/udon/src/curve/pasta/table_entry.rs) implement `Pod`
for both Pallas and Vesta. Their type docs define the coordinate layouts and
mathematical invariants, including the cached endomorphism coordinate in
`PreparedAffinePoint`. The 32-byte compressed protocol encoding is a different
representation. Identity-capable `Point` and Jacobian `ProjectivePoint` do not
implement `Pod`.

Affine coordinates use `Reduced` field elements. Constructing a point with
`AffinePoint::from_xy` checks the curve equation; constructing a prepared point
computes its cached coordinate. Writing these values preserves those invariants.
Embedded points are ready for arithmetic without a constructor or validation
pass at runtime.

For repeated multiplication, prepare `FixedBaseTable<C, E>` or
`EisensteinTable<C, E>` entries into caller-owned storage and write the resulting
slice or an enclosing record through Bento POD. Both accept `AffinePoint<C>`
(the default) or `PreparedAffinePoint<C>` entries. Embed the same types in the
consumer and use the table's `const bind` method to attach the stored entries
to their base and description. Binding checks shape and configuration and
borrows the entries directly; it performs no point arithmetic or content scan. The
[expanded](../crates/udon/src/curve/pasta/fixed_base.rs) and
[compact](../crates/udon/src/curve/pasta/eisenstein.rs) API docs define entry order;
the [curve guide](CURVES.md#fixed-base-multiplication) shows preparation and
storage costs.

`STORED_FORM` identifies the field representation.
The owner must separately identify the curve, base, compact or expanded table
kind, affine or cached entry representation, window width when present, and
record schema.

The [curve embedding fixture](../crates/udon/tests/curve/fixtures/embedding)
shares its record definition between generator and `no_std` consumer. It
demonstrates both table kinds and entry types, writing them through Bento POD
and multiplying directly from embedded storage. The
[testing guide](TESTING.md#generated-artifacts) describes its feature coverage
and artifact layout checks. Run it with:

```console
cargo test --release --locked -p zakura-udon --test curve embedding:: -- --ignored
```

## Format ownership

Generator and consumer agree on the stored type definitions, representation
attributes, and type parameters. The artifact is the exact byte representation
of values constructed with those types. A matching file length alone cannot
identify its type or schema; that association belongs to the artifact format.
There is no separate content-validation stage.

The [`Pod` safety contract][pod-contract] additionally requires memory safety for
every bit pattern admitted by its safe byte-view APIs. This is a Rust safety
requirement on implementations, independent of the trusted artifact workflow.

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
[field-storage]: ../crates/udon/src/field/pasta/mod.rs
[stored-forms]: ../crates/udon/src/field/pasta/stored_form.rs
