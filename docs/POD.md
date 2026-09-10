# POD storage and embedding

The [`bento`](../crates/bento/src/lib.rs) facade lets artifact generators write
records as bytes and consumers embed those files as typed static data. Byte
views borrow existing values; embedding supplies aligned storage without
allocation or runtime initialization.

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

Layout validation happens through [`Pod::ASSERT_LAYOUT`][pod-contract]. A
generic struct can have some instantiations suitable for storage and others
that are only used as ordinary values. Byte conversions validate the layout;
an explicit assertion can check a particular format without converting a value:

```rust
const _: () = <Record as bento::Pod>::ASSERT_LAYOUT;
```

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

POD storage uses little-endian bytes and validates primitive size and alignment
on the target. Unsupported endianness or layouts fail when a storage operation
is instantiated; they do not prevent using unrelated crate functionality.

Dependency aliases follow the [workspace conventions](CRATES.md#procedural-macros)
and are resolved automatically. A crate that reaches support through another
facade can use `#[pod(crate = path)]`; see the [derive documentation][pod-derive]
for the required re-exports.

## Validation

Run the [workspace checks](../README.md#testing), including the release test
suite. `cargo check` alone can miss layout failures deferred until code
generation. The suite covers:

- [Parsing and expansion](../crates/bento-macros/src/derive/pod/tests.rs).
- [Core storage unit tests](../crates/bento-core/src/pod/storage/tests.rs) for
  byte views, primitive boundaries, and runtime length checks.
- [Derived records and embedded data](../crates/bento/tests/pod.rs), including
  alignment boundaries, generic records, and zero-sized types.
- [Generated artifact round trips](../crates/bento/tests/embedding.rs), using a
  build script to write files before compiling and running their consumer.
- [Compiler failures](../crates/bento/tests/pod_compile.rs) for padding,
  unsupported fields and representations, excessive alignment, and file length.
- [Separate Cargo consumers](../crates/bento/tests/consumers.rs) for
  `no_std`, dependency aliases, direct core dependencies, and facade re-exports.

The [testing guide](TESTING.md) defines the roles of these tests and the
conventions for compiler builds and fixtures. The embedding example also runs
as part of the workspace test suite.

[pod-derive]: ../crates/bento/src/lib.rs
[pod-contract]: ../crates/bento-core/src/pod/mod.rs
[byte-views]: ../crates/bento-core/src/pod/storage.rs
[embedding]: ../crates/bento-core/src/pod/macros.rs
