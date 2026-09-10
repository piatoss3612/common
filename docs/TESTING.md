# Testing

Tests should make clear which behavior they establish and why that behavior is
checked at that layer. Run the [workspace checks](../README.md#testing) before
finishing a change.

## Test roles and placement

All integration tests for the `bento` stack live in the
[`bento` facade's `tests/` directory](../crates/bento/tests/), including compiler
diagnostics, dependency resolution, and artifact generation. These tests verify
the assembled library and its consumer behavior. Keep implementation unit tests
in `bento-core` and `bento-macros` beside the code they exercise.

- **Unit tests** live beside the implementation under `#[cfg(test)]`. Test
  parsing, algorithms, validation, and expansion without invoking a downstream
  compiler. Storage operations that need only core types belong in
  [`bento-core`](../crates/bento-core/src/pod/storage/tests.rs); macro parsing and
  token generation belong in
  [`bento-macros`](../crates/bento-macros/src/derive/pod/tests.rs).
- **Public API integration tests** exercise derived types, re-exports,
  and storage together through the facade. Keep detailed edge cases here when
  they need several components to work together.
- **Compiler integration tests** build separate Cargo consumers when the result
  depends on dependency lookup, type checking, or constant evaluation. The
  [Cargo consumer tests](../crates/bento/tests/consumers.rs) cover dependency
  arrangements and diagnostics. The
  [POD compiler tests](../crates/bento/tests/pod_compile.rs) cover rejected
  representations and storage operations.
- **Examples** live in the public crate's `examples/` directory. Each should
  demonstrate a complete, representative use and assert its result, serving as
  both usage documentation and an integration check. Set `test = true` and
  `harness = false` in its `[[example]]` manifest entry so the ordinary workspace
  test command executes `main`. Keep diagnostic fixtures and exhaustive edge
  cases in tests, where their intent is clearer.
- **Doctests** verify the usage shown in public API documentation. Keep them
  focused on the documented contract and follow the
  [documentation guide](DOCUMENTATION.md).

Prefer the narrowest layer that establishes a property. Expansion comparisons
show what code a macro emits; they do not establish that the code type-checks or
behaves correctly. Add integration coverage for those properties without
repeating every parser case in a consumer build.

## Rust inputs and expansion expectations

Write Rust inputs and expected expansions with `quote!` or `syn::parse_quote!`,
so the code remains readable as Rust. For a complete expansion expectation,
compare token streams through `to_string()` and parse the result as the expected
syntax category. See the [derive snapshot](../crates/bento-macros/src/derive/pod/tests.rs)
and [addition-chain snapshot](../crates/bento-macros/src/proc/addition_chain/tests.rs).

These comparisons discard ordinary source whitespace, but token punctuation
still matters. For example, an interpolated type can leave separate closing
`> >` tokens where handwritten `>>` has joint punctuation. Match the emitted
tokens in the expectation; avoid broad string replacements that could conceal
a change in the expansion.

Use `#[rustfmt::skip]` on individual tests or case-building functions when
formatting quoted Rust obscures the inputs or expectations. Maintain that
layout by hand and keep the surrounding implementation and harness formatted
normally. This also preserves deliberately separate punctuation tokens.

Use ordinary `.rs` fixtures for complete consumer programs and cases involving
source-relative paths. Small parameterized compiler cases can use `quote!` and
be serialized when the harness writes their source files. Keep Rust source out
of string literals unless the literal text itself is under test, as with a
lexer or malformed tokens that `quote!` cannot express. Diagnostic messages,
numeric parser inputs, and generated manifests remain strings.

Format standalone `.rs` fixtures directly with the pinned `rustfmt` and the
workspace edition. `cargo fmt` visits discovered targets and modules; it does
not find every fixture file.

## Consumer builds and artifacts

Group fixtures by feature under `tests/fixtures/`; use a shared directory for
consumers that exercise multiple features. Preserve module and relative file
paths when copying fixtures into generated packages. Name compiler inputs as
test fixtures and build them as library or binary targets, reserving examples
for runnable demonstrations.

Nested Cargo invocations need a separate target directory to avoid the parent
build's lock. Seed their dependency resolution from the workspace lockfile and
run offline. Assert the expected diagnostic and its source location without
pinning the compiler's entire rendered message.

Use full release builds for assertions deferred until code generation;
`cargo check` can miss those failures. Fixed byte fixtures test embedding and
length validation. A generator-to-consumer test must also write the artifact
through the storage API before compiling and running the consumer. The
[embedding integration test](../crates/bento/tests/embedding.rs) does this with a
build script and a shared record definition.
