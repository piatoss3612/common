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

When implementing a new unsafe storage consumer, force this validation with
`const { T::ASSERT_LAYOUT };` before relying on the
[`Pod` guarantees][pod-contract].

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

## Format ownership

Generator and consumer must agree on the stored type definitions, representation
attributes, and any semantic invariants. Layout validation cannot detect a file
generated for a different type of the same size, check canonical field residues,
or establish curve membership. Those checks belong to the artifact's owner.

Every bit pattern admitted by the safe byte-conversion APIs must nevertheless
be memory-safe for every safe operation on the resulting type. A generator
cannot establish that obligation for arbitrary bytes supplied by another safe
caller. If arithmetic needs stronger invariants, keep the stored representation
separate from the arithmetic type and establish those invariants through checked
conversion or a narrowly scoped trusted construction path. The complete unsafe
implementation obligations belong to the [`Pod` contract][pod-contract].

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
