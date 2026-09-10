# bento-macros

Implementation crate for procedural macros exposed through the `bento` facade.
Consumers should use the macros through `bento`.

Authoring conventions live in the [crate documentation](src/lib.rs), with
implementation details documented alongside the code. See the
[crate development guide](../README.md) for workspace structure and dependency
conventions.

`addition_chain!` provides a working example:

- [Public documentation and examples](../bento/src/lib.rs).
- [Parsing and expansion](src/proc/addition_chain/mod.rs) and
  [chain planning](src/proc/addition_chain/schedule.rs).
- [Expansion tests](src/proc/addition_chain/tests.rs),
  [behavioral tests](tests/addition_chain.rs), and
  [Cargo consumer tests](tests/consumers.rs).
