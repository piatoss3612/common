# Testing

Choose the narrowest test layer that establishes the property under review.
Keep tests with the crate that owns the behavior; integration across crate
boundaries belongs with the public API or consumer that assembles them. Future
arithmetic tests need not live in the support facade.

The [CI workflow](../.github/workflows/ci.yml) defines the required checks and
pins the additional toolchain and targets. The [README](../README.md#testing)
lists the local baseline. Tests run with optimizations so validation must not
depend on debug assertions.

## Test roles

- Unit tests check algorithms, parsers, and local contracts beside their code.
  Use independent references for arithmetic and representation checks, including
  boundary values and inputs wider than native integers. Reference arithmetic
  tests use `num-bigint` as a development dependency for this purpose. Replaying
  output with the same algorithm is not an independent correctness check.
- Public API tests check observable behavior and interactions between components.
  Token snapshots establish expansion structure, not successful compilation or
  runtime semantics.
- Compiler tests establish type, diagnostic, constant-evaluation, and dependency
  contracts in separate consumers. Use full builds for assertions deferred to
  code generation; `cargo check` can miss them. Check the relevant diagnostic and
  source location without pinning the compiler's entire rendered output.
- Examples demonstrate complete uses and assert their results. Configure runnable
  examples with `test = true` and `harness = false` so the suite executes them.
  Doctests verify focused public API examples.

Safety and portability need targeted evidence as well as native tests. CI runs
Miri over storage unit tests and the public storage integration tests; nested
Cargo tests stay in the native suite. The portability test builds `no_std`
libraries for a 32-bit little-endian target and separately checks that big-endian
storage fails while addition chains and constant arithmetic compile. The
arithmetic fixture also asserts computed values during constant evaluation.
The test is ignored in ordinary runs because target libraries must be installed,
and explicitly executed in CI.
These checks do not establish correctness on every target or constant-time
behavior; extend validation when new code introduces new assumptions.

## Fixtures and nested builds

Keep complete Rust consumer programs in `.rs` files under `tests/fixtures/`,
grouped by the behavior they exercise. Preserve relative module and data paths.
Small parameterized inputs and expected expansions can use `quote!` or
`syn::parse_quote!`. Use source strings only when text itself is under test or
when writing the generated source at the compiler boundary.

Compare token expectations without normalizing away meaningful punctuation,
and parse complete expansions as the expected syntax category. A local
`#[rustfmt::skip]` is appropriate when formatting quoted Rust obscures a case;
keep the harness itself formatted. The [format check](../ci/check-format)
includes standalone fixtures that `cargo fmt` does not discover.

Give every nested Cargo test run a unique temporary workspace and target
directory, owned until its processes finish. This prevents concurrent runs from
rewriting each other's manifests, sources, or artifacts and avoids the parent
Cargo lock. Seed resolution from the workspace lockfile and run offline, allowing
Cargo to adapt the seed to the fixture's dependency graph. The parent build must
first fetch any dependencies those consumers require.

Fixed bytes test format interpretation and length checks. A generator-to-consumer
round trip must actually generate the artifact through the writing API before
building its consumer; copying a golden file does not exercise generation.
