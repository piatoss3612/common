# (no name yet)

This is an experimental stack of Rust crates for Zakura Common.

* This repository is one virtual Cargo workspace; every crate lives under `crates/`.
* This is mostly scaffolded.
* All _target-compiled_ crates (`bento`, `bento-core`, and `udon`) use `no_std` currently.
* See [the crate development guide](crates/README.md) for dependency naming, workspace inheritance, macro path resolution, and publication conventions.
* See [the documentation guide](docs/DOCUMENTATION.md) when writing or reviewing documentation and code comments.

### `zakura-bento` ([crates.io](https://crates.io/crates/zakura-bento), [docs.rs](https://docs.rs/zakura-bento))

The [`bento`](crates/bento/) crate provides the shared POD storage contract and the compile-time support needed to define `udon`.

- **POD storage and embedding.** Shared traits, utilities, and macros let crates throughout the stack write and embed cryptographic artifacts without redundant copies or runtime initialization. Downstream crates can depend directly on `bento` for this contract; the crates that define stored types own their representations and semantic invariants.
- **Compile-time arithmetic.** Minimal reference arithmetic derives constants needed to define fields and curves in `udon`. Runtime field and group algorithms live in `udon`; downstream artifact generators use the built `udon` crate for arithmetic.
- **Procedural macros.** Macros generate checked POD implementations and code needed to define field and group operations. Artifact-specific generation belongs downstream with the crate that owns the data.

The `bento` crate itself is a facade over the `bento-macros` and `bento-core` crates.

### `zakura-udon` ([crates.io](https://crates.io/crates/zakura-udon), [docs.rs](https://docs.rs/zakura-udon))

[`udon`](crates/udon/) provides an implementation of the Pasta curves and much of the highly optimized arithmetic needed to interact with them and their finite fields. It is adapted from Zakura Common's fork of `pasta_curves` and subsumes the traits and utilities needed downstream for Tachyon.

## Testing

Run the full test suite (every workspace crate, all features) with optimizations enabled:

```console
cargo test --release --workspace --all-features
```

The pinned toolchain (`rust-toolchain.toml`, the crates' MSRV) is also used to check formatting, lints, and documentation across the workspace:

```console
cargo fmt --check
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
cargo doc --locked --workspace --all-features --no-deps
```
