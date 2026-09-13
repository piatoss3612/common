# (no name yet)

This is an experimental stack of Rust crates for Zakura.

* This repository is one virtual Cargo workspace; every crate lives under
  [`crates/`](crates/).
* The current implementation provides reference integer and Montgomery
  arithmetic, addition chains, checked POD storage and embedding, and runtime
  Pasta field and curve arithmetic and field FFTs.
* All target crates (`bento`, `bento-core`, and `udon`) currently use `no_std`.
* See [the crate development guide](docs/CRATES.md) for dependency naming,
  workspace inheritance, macro path resolution, and publication conventions.
* See [the documentation guide](docs/DOCUMENTATION.md) when writing or reviewing
  documentation and code comments.

## `zakura-bento`

The [`bento`](crates/bento/src/lib.rs) crate provides compile-time support for
cryptographic arithmetic. Its `addition_chain!` macro scales a value by a fixed
positive integer using operations supplied by the value's type. Its
[POD storage APIs](docs/POD.md) let generators write records as bytes and
consumers embed those files as typed static data.

The [`const_arithmetic`](crates/bento/src/const_arithmetic/mod.rs) module
derives field and curve parameters with integer and Montgomery arithmetic
macros. Each macro evaluates its arguments and result inside `const { ... }`,
so runtime arguments are rejected. The reference arithmetic operates on public
parameters and provides no constant-time guarantee.
The [field constants example](crates/bento/examples/field_constants.rs)
derives a root of unity and its inverse from a modulus and generator:

```console
cargo run --release --locked -p zakura-bento --example field_constants
```

Additional procedural macros will generate code needed to define field and
group operations. Downstream artifact generators will use the built
[`udon`](crates/udon/src/lib.rs) crate for runtime arithmetic; artifact formats
and their generation belong with the data's owner.

The [`bento`](crates/bento/src/lib.rs) crate is a facade over
[`bento-macros`](docs/MACROS.md) and
[`bento-core`](crates/bento-core/src/lib.rs).

## `zakura-udon`

[`udon`](crates/udon/src/lib.rs) provides the two Pasta prime fields, `Fp` and
`Fq`, with Montgomery arithmetic, canonical encodings, inversion, square roots,
and product sums. These operations require no allocation. Arithmetic is
variable-time and provides no constant-time guarantee for secret inputs.
Field parameters and fixed exponentiation schedules use `bento` at compile time.
Fields implement `bento::Pod`, so downstream build scripts can generate them
with Udon and embed their Montgomery representations for direct runtime use;
see the [field storage example](docs/POD.md#storing-field-elements).
The optional `sqrt-table-large` feature selects larger square-root tables; see
the [performance report](docs/FIELD_PERFORMANCE.md#optional-larger-square-root-tables)
for latency, build-time, and storage tradeoffs.

The [`fft` module](crates/udon/src/fft/mod.rs) provides power-of-two transforms,
cosets, residue expansion, and fused class interpolation for both fields.
Callers own all tables, buffers, scratch, and parallel execution; Udon's FFT
setup and execution do not allocate or require a feature flag. See the
[FFT guide](docs/FFT.md) for layouts, scratch requirements, and downstream table
generation with Bento POD, and the [crate docs](crates/udon/src/lib.rs) for
feature definitions.

The [`curve` module](crates/udon/src/curve/mod.rs) provides Pallas and Vesta
points, canonical encodings, GLV/Eisenstein scalar multiplication, batch
normalization, and borrowed compact and expanded fixed-base tables. Scalar
preparation can be reused across compact tables. The
[`msm` module](crates/udon/src/curve/msm/mod.rs) sums dense or indexed inputs
with caller-owned scratch and execution.
Nonidentity `AffinePoint` and cached `PreparedAffinePoint` entries implement
`bento::Pod` for direct storage. All curve operations are variable-time and
require no allocation. See the [curve guide](docs/CURVES.md) for point
representations, caller-owned preparation buffers, and table validation, and the
[performance report](docs/CURVE_PERFORMANCE.md) for timing and storage tradeoffs.

Additional traits and utilities needed downstream for Tachyon remain planned.

## Testing

See the [testing guide](docs/TESTING.md) for test roles, fixture organization,
and executable examples.

Run the default test suite (every workspace crate, all features) with optimizations
enabled:

```console
cargo test --release --locked --workspace --all-features
```

Udon's slower compiler and artifact consumer tests are ignored by default.
Run them explicitly with `--ignored` as described in the
[testing guide](docs/TESTING.md#slow-consumer-tests); CI runs them separately.

The pinned toolchain ([`rust-toolchain.toml`](rust-toolchain.toml), the crates'
MSRV) is also used to check formatting, lints, and documentation across the
workspace:

```console
python3 ci/check-format
cargo clippy --workspace --all-targets --all-features --locked -- -D warnings
RUSTDOCFLAGS="-D warnings" cargo doc --locked --workspace --all-features --no-deps
```

The [CI workflow](.github/workflows/ci.yml) defines these gates plus focused
Miri checks, field, curve, and FFT tests without optimizations, and cross-target
checks. The format script also checks standalone Rust fixtures that Cargo does
not discover.
