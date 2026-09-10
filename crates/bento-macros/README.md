# `bento-macros`

This crate implements procedural macros exposed through the
[`bento`](../bento/src/lib.rs) facade. Consumers should use the macros through
[`bento`](../bento/src/lib.rs).

See the [crate development guide](../README.md) for workspace structure and
dependency conventions. Implementation details are documented alongside the
code; build them with `cargo doc --document-private-items`.

## Background

Macros run on the build host, while their output must compile for the caller's
target. Shared support interfaces and reference arithmetic belong in
[`bento-core`](../bento-core/src/lib.rs); parsing and token generation belong here.

## Design

Entry points in the [crate root](src/lib.rs) parse input, resolve dependency
paths, and invoke [`macro_body`](src/helpers.rs). Expansion uses
`proc_macro2::TokenStream` and `syn::Result` so it can be tested outside the
compiler's procedural macro context.

The implementation is organized into:

- [`derive`](src/derive/mod.rs): conventions and space for derive macro
  implementations.
- [`proc`](src/proc/mod.rs): function-like macro parsing and expansion.
- [`helpers`](src/helpers.rs): shared error reporting for entry points.
- [`path_resolution`](src/path_resolution.rs): caller dependency lookup for
  generated library paths.

## Authoring conventions

Follow the module conventions in [`derive`](src/derive/mod.rs) or
[`proc`](src/proc/mod.rs) when adding a macro. Report invalid input with
`syn::Error`; reserve panics for internal invariants. In generated code,
interpolate the supplied [`BentoCorePath`](src/path_resolution.rs) for library
items and use absolute `::core` paths for standard types to support `no_std`
callers.

Document and explicitly re-export each macro from
[`bento`](../bento/src/lib.rs). Test parsing and expansion in the implementation
module, and use the facade dev-dependency for tests of generated behavior.
Expansions that reference library items also need separate Cargo consumer tests
to exercise dependency resolution.

## Addition chains

`addition_chain!` provides a working example:

- [Public documentation and examples](../bento/src/lib.rs).
- [Parsing and expansion](src/proc/addition_chain/mod.rs) and
  [chain planning](src/proc/addition_chain/schedule.rs).
- [Expansion tests](src/proc/addition_chain/tests.rs),
  [behavioral tests](tests/addition_chain.rs), and
  [Cargo consumer tests](tests/consumers.rs).
