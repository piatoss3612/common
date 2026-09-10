# `bento-macros`

This crate implements procedural macros exposed through the
[`bento`](../crates/bento/src/lib.rs) facade. Consumers should use the macros through
[`bento`](../crates/bento/src/lib.rs).

See the [crate development guide](CRATES.md) for workspace structure and
dependency conventions. Implementation details are documented alongside the
code; build them with `cargo doc --document-private-items`.

## Background

Macros run on the build host, while their output must compile for the caller's
target. Shared support interfaces and reference arithmetic belong in
[`bento-core`](../crates/bento-core/src/lib.rs); parsing and token generation belong
in `bento-macros`.

## Design

Entry points in the [crate root](../crates/bento-macros/src/lib.rs) parse input,
resolve dependency paths, and invoke
[`macro_body`](../crates/bento-macros/src/helpers.rs). Expansion uses
`proc_macro2::TokenStream` and `syn::Result` so it can be tested outside the
compiler's procedural macro context.

The implementation is organized into:

- [`derive`](../crates/bento-macros/src/derive/mod.rs): derive macro conventions
  and the checked `Pod` implementation.
- [`proc`](../crates/bento-macros/src/proc/mod.rs): function-like macro parsing and
  expansion.
- [`helpers`](../crates/bento-macros/src/helpers.rs): shared error reporting for
  entry points.
- [`path_resolution`](../crates/bento-macros/src/path_resolution.rs): caller
  dependency lookup for generated library paths.

## Authoring conventions

Follow the module conventions in [`derive`](../crates/bento-macros/src/derive/mod.rs)
or [`proc`](../crates/bento-macros/src/proc/mod.rs) when adding a macro. Report
invalid input with `syn::Error`; reserve panics for internal invariants. In
generated code, interpolate the supplied
[`BentoCorePath`](../crates/bento-macros/src/path_resolution.rs) for library items
and use absolute `::core` paths for standard types to support `no_std` callers.

Document and explicitly re-export each macro from
[`bento`](../crates/bento/src/lib.rs). Test parsing and expansion in the
implementation module, and use the facade dev-dependency for tests of generated
behavior.
Expansions that reference library items also need separate Cargo consumer tests
to exercise dependency resolution.

## Addition chains

`addition_chain!` provides a working example:

- [Public documentation and examples](../crates/bento/src/lib.rs).
- [Parsing and expansion](../crates/bento-macros/src/proc/addition_chain/mod.rs) and
  [chain planning](../crates/bento-macros/src/proc/addition_chain/schedule.rs).
- [Expansion tests](../crates/bento-macros/src/proc/addition_chain/tests.rs),
  [behavioral tests](../crates/bento-macros/tests/addition_chain.rs), and
  [Cargo consumer tests](../crates/bento-macros/tests/consumers.rs).

## POD storage

The [`Pod` derive](../crates/bento-macros/src/derive/pod/mod.rs) validates a
struct's representation and generates field bounds and recursive layout
assertions. It resolves dependencies through
[`BentoCorePath`](../crates/bento-macros/src/path_resolution.rs), with an optional
`#[pod(crate = path)]` override for support reached through another facade.
Assertions remain associated with the concrete type so generic records can be
validated when used for storage.

The declarative [embedding macros](../crates/bento-core/src/pod/macros.rs) live
in core and use `$crate` paths. Their initializers borrow aligned bytes for
static typed views.

See the [POD guide](POD.md) for usage, format ownership, and validation coverage.
In particular, compiler tests must perform full builds: metadata-only checks
can miss deferred layout assertion failures.
