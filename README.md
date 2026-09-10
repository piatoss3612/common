# (no name yet)

This is an experimental stack of Rust crates for Zakura Common.

* This repository is one virtual Cargo workspace; every crate lives under
  [`crates/`](crates/).
* The current implementation provides an addition-chain macro and its support
  trait. Storage utilities and field and curve arithmetic remain scaffolded.
* All target crates (`bento`, `bento-core`, and `udon`) currently use `no_std`.
* See [the crate development guide](docs/CRATES.md) for dependency naming,
  workspace inheritance, macro path resolution, and publication conventions.
* See [the documentation guide](docs/DOCUMENTATION.md) when writing or reviewing
  documentation and code comments.

### `zakura-bento`

The [`bento`](crates/bento/src/lib.rs) crate provides compile-time support for
cryptographic arithmetic. Its `addition_chain!` macro scales a value by a fixed
positive integer using operations supplied by the value's type.

The planned scope also includes:

- **POD storage and embedding.** Shared traits, utilities, and macros will let
  crates write and embed cryptographic artifacts without redundant copies or
  runtime initialization. The crates that define stored types will own their
  representations and semantic invariants.
- **Compile-time arithmetic.** Reference arithmetic will derive constants needed
  to define fields and curves in [`udon`](crates/udon/src/lib.rs). Downstream
  artifact generators will use the built [`udon`](crates/udon/src/lib.rs) crate
  for runtime arithmetic.
- **Procedural macros.** Additional macros will generate checked POD
  implementations and code needed to define field and group operations.
  Artifact-specific generation will belong downstream with the data's owner.

The [`bento`](crates/bento/src/lib.rs) crate is a facade over
[`bento-macros`](docs/MACROS.md) and
[`bento-core`](crates/bento-core/src/lib.rs).

### `zakura-udon`

[`udon`](crates/udon/src/lib.rs) is reserved for Pasta field and curve arithmetic.
The planned implementation will adapt Zakura Common's fork of `pasta_curves`
and provide the traits and utilities needed downstream for Tachyon. The crate
does not yet expose arithmetic APIs.

## Testing

Run the full test suite (every workspace crate, all features) with optimizations
enabled:

```console
cargo test --release --workspace --all-features
```

The pinned toolchain ([`rust-toolchain.toml`](rust-toolchain.toml), the crates'
MSRV) is also used to check formatting, lints, and documentation across the
workspace:

```console
cargo fmt --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo doc --locked --workspace --all-features --no-deps
```
