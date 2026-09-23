# Testing

Choose the narrowest test layer that establishes the property under review.
Keep tests with the crate that owns the behavior; integration across crate
boundaries belongs with the public API or consumer that assembles them.
The [CI workflow](../.github/workflows/ci.yml) defines required gates and pins
the additional toolchain and targets. The [README](../README.md#testing) lists
formatting, lint, and documentation commands.

Run the release suites with and without the consumer traits, then the runtime
unit tests with debug assertions:

```console
cargo test --release --locked --workspace --all-features
cargo test --release --locked -p zakura-udon --no-default-features
cargo test --locked -p zakura-udon --lib
cargo test --locked -p zakura-udon --lib --all-features
```

Release tests ensure validation does not depend on debug assertions. Debug
runs also exercise loose FFT bounds and stack growth in generated chains.
The default configuration has a separate Clippy gate because workspace lints
enable all features. Slow consumers and timing experiments are ignored by the
ordinary suites and have separate commands below.

## CI enforcement

The arithmetic matrix executes release workspace tests and debug Udon tests
natively on x86-64 and ARM64, each with `traits` disabled and enabled and with
both square-root configurations. Consumer-interface tests stay in their owning
domains and run with `traits`; native kernel tests run in every configuration.
Lints, slow compiler and artifact consumers, benchmark smoke tests, Miri, and
target portability have separate jobs. Cross-target compilation is additional
coverage; it does not replace either native runner.

The stable `ci-required` job runs even when a dependency fails or is skipped and
succeeds only when every mandatory job succeeds. Repository rules must require
this exact GitHub Actions check on `main`; defining the job alone does not block
merges. Keep its `needs` list and explicit success conditions in sync when adding
a mandatory job. A new check must be present on the branches being merged before
enabling its repository rule.

Filtered runs check their completed test summaries in the workflow: every
selected target must report at least one passing test. A target with only ignored
tests fails this check too. Bash's pipeline failure handling preserves Cargo's
exit status while recording output. Job timeouts bound hung tests, and newer runs
cancel superseded runs for the same PR or branch.

## Udon test layout

Unit tests live with the implementation they exercise, including Poseidon
parameters, cycle bindings, encoding bounds, arithmetic, resource limits,
and the FFT/MSM execution protocols. This also applies to assertions that use
only public methods. External suites check behavior that needs a separate
consumer: macro resolution, independent trait implementations, caller-owned
adapters, and artifact generation and embedding. They are grouped by the API
they exercise. Paths below are relative to
`crates/udon/`; each integration suite has a `main.rs` Cargo entry point.

| Location | Responsibility | Selection |
| --- | --- | --- |
| `src/**/tests.rs` and `src/**/tests/` | Arithmetic, private kernels, and internal protocol contracts | `--lib`, optionally with a module filter |
| `tests/field/` | Field constant macros, independent trait implementations, POD, and embedding | `--test field` |
| `tests/curve/` | Curve constant macros, independent trait implementations, POD, and embedding | `--test curve` |
| `tests/fft/` | Caller-owned FFT execution, workspaces, and embedded tables | `--test fft` |
| `tests/msm/` | Caller-owned MSM execution and workspaces | `--test msm` |
| `tests/execution/` | Shared executor contracts and worker-pool selection | `--test execution` |
| `tests/api/` | Compiler checks of public API boundaries and explicit trait opt-in | `--test api` |
| `tests/<domain>/fixtures/` | Programs compiled by their owning domain's tests | Through their owning tests, with `--ignored` |
| `tests/harness/` | Reusable Cargo consumer machinery, executor adapter, and independent field models | Included by the suites that need them |

Keep shared helpers with the implementation that owns their behavior. Pasta
sampling and integer references live in `src/field/pasta/test_support.rs` and
are shared by the arithmetic tests that need them.
Execution buffer adapters and the test worker pool live beside the corresponding
execution implementation as `test_buffers.rs`, `test_pipeline.rs`, or `test_pool.rs`.
They are compiled only for tests; integration tests and benchmarks explicitly
include the helpers that use public APIs. Test suites do not import another
suite's internal helpers. Independent arithmetic references remain separate
from buffer setup and worker adapters.

Private execution filters are `fft::execution::tests`,
`msm::execution::tests`, and `exec::execution::tests`. The arithmetic `execution`
modules contain production plans and kernels; `exec::execution` supplies their shared
task protocol. The
[crate guide](CRATES.md#udon-module-boundaries) explains their ownership.

## Bento test layout

Bento's facade tests exercise macro expansion in callers and the storage APIs
those expansions use. They are grouped by feature, with fixtures inside the
owning suite. Paths below are relative to `crates/bento/`.

| Location | Responsibility | Selection |
| --- | --- | --- |
| `tests/addition_chain/` | Expansion behavior, hygiene, value ownership, and compiler rejections | `--test addition_chain` |
| `tests/const_arithmetic/` | Constant-only inputs and arithmetic precondition rejections | `--test const_arithmetic` |
| `tests/pod/` | Derived records, byte views, compiler contracts, and artifact embedding | `--test pod` |
| `tests/api/` | Dependency aliases, support paths, re-exports, and target portability | `--test api` |
| `tests/<feature>/fixtures/` | Consumer source files and stored data owned by that suite | Through the owning tests |
| `tests/harness/` | Cargo execution, isolated workspaces, and diagnostic checks | Included by the suites that need them |

Each `main.rs` only declares its suite's modules. The dependency matrix owns
its complete consumers under `tests/api/fixtures/dependencies/`, including
their arithmetic and POD modules. It exercises those APIs through different
dependency arrangements; feature-specific rejection cases live with their
feature. Compiler case tables reuse one temporary workspace per matrix.
POD's parameterized record cases stay in its compiler test as Rust tokens.

Reference arithmetic and storage implementation tests stay beside their code
in `bento-core`. Parser and expansion tests stay beside each macro in
`bento-macros`, under the existing `proc/` and `derive/` modules.

## Test roles

- Unit tests check algorithms and local contracts against independent references.
  Field arithmetic uses integer oracles and a conventional Tonelli–Shanks
  reference. Curves use affine `num-bigint` arithmetic, raw integer group orders,
  and integer GLV reconstruction. MSMs use binary ladders that bypass production
  decomposition and recoding. FFTs use direct sums, polynomial evaluation, and
  independent transform schedules, including loose-range and inverse-scale
  boundaries. Comparing two paths that share the same algorithm is insufficient.
- Public API tests check consumer-defined adapters. The
  [execution tests](../crates/udon/tests/execution/main.rs) exercise worker-pool
  selection and nested work; the [FFT](../crates/udon/tests/fft/main.rs) and
  [MSM](../crates/udon/tests/msm/main.rs) suites exercise borrowed execution and
  reusable workspaces. Unit tests cover scheduling and failure boundaries.
  Distinguish setup
  rejection, rejection before one task writes, and failure after earlier writes;
  cancellation and draining do not promise whole-run rollback.
- Compiler consumers establish reachability, type, constant-evaluation, feature,
  and dependency contracts. Use full builds for checks deferred to code
  generation. Check the intended diagnostic and source location without pinning
  the entire rendered error. Macro token snapshots alone do not establish
  compilation or runtime semantics. Aliases and re-exports are part of the API.
- Examples demonstrate complete uses with result assertions. Set `test = true`
  and `harness = false` for runnable examples. Doctests exercise focused public
  examples; build documentation with warnings denied.

Keep operation-specific boundaries beside their tests. Shared deterministic
sampling and integer conversions live in Udon's test-only
[Pasta helpers](../crates/udon/src/field/pasta/test_support.rs). Executor adapters should
also run inside a one-worker pool to expose nested joins that assume an idle
worker. Preserve independent checks of cached rotations, scalar reuse, fused
and unnormalized FFTs, group-valued transforms, and supplied/batched chains
when changing their public entry points.

### Arithmetic properties and dispatch

Field, FFT, and MSM properties live beside their existing arithmetic suites in
`src/field/pasta/tests/properties.rs`, `src/fft/tests/properties.rs`, and
`src/msm/tests/properties.rs`. They compare generated inputs against integer
arithmetic, direct polynomial evaluation, and binary scalar ladders. Cases include
redundant field representatives, full-width and short scalars, repeated and inverse
bases, indexed inputs, optional FFT tables, and reused dirty scratch. The same
comparison functions must reject deliberately corrupted results at a selected
boundary while accepting the real result and neighboring cases.

PRs run 32 generated cases per property in each native/feature configuration.
Run a larger campaign locally by overriding `PROPTEST_CASES`:

```console
PROPTEST_CASES=1024 cargo test --release --locked -p zakura-udon --lib ::properties::
PROPTEST_CASES=1024 cargo test --release --locked -p zakura-udon --lib --all-features ::properties::
```

Proptest shrinks failures and saves replay seeds in `properties.regressions`
beside the owning source file. CI uploads these files on failure. Copy a failing
seed file from the artifact to that same directory, reproduce and fix the failure,
and commit the seed with the fix. For a whole generated run, `PROPTEST_RNG_SEED`
selects a reproducible seed. These checks establish finite-case agreement, not a
proof of arithmetic correctness.

MSM transition tests exercise sizes immediately below, at, and above automatic
window changes and the preparation chunk boundary, including cancellation and
scratch reuse. Test-only counters check actual kernel execution. Generic MSM
tests must reach MSM kernels; generic scalar FFTs must avoid the reference
transform, and batch inversion must retain its expected inversion count. Result
equality alone cannot detect a slower fallback. The counters and faulty results
are compiled only for tests; runtime kernels stay in Udon.

Miri checks storage separately from native nested-Cargo consumers:

```console
cargo +nightly-2026-09-06 miri test --locked -p zakura-bento-core --lib pod::
cargo +nightly-2026-09-06 miri test --locked -p zakura-bento --test pod storage::
cargo +nightly-2026-09-06 miri test --locked -p zakura-udon --test field --test curve pod::
```

Install that toolchain with `miri` and `rust-src` as described in CI. Portability
requires the `thumbv7em-none-eabi` and `s390x-unknown-linux-gnu` target libraries:

```console
cargo test --release --locked -p zakura-bento --test api portability:: -- --ignored
```

This builds `no_std` consumers with both square-root configurations, checks
constant results and independent big-endian storage rejections, and compiles
runtime arithmetic and execution helpers. It does not execute arithmetic on
those targets or establish constant-time behavior.

## Benchmarks

The [benchmarking guide](BENCHMARKING.md) lists the timing suites, smoke-test
commands, fixture checks, and measurement boundaries. Run timing measurements
without concurrent builds or tests.

## Fixtures and nested builds

Keep complete Rust consumer programs in `.rs` files inside a `fixtures/`
directory owned by their suite. Udon keeps field, curve, and FFT artifact
consumers under their domain's `fixtures/embedding/` directory. Cross-domain
compiler consumers live in `tests/api/fixtures/`. Bento also groups its
fixtures under their owning suites, including its dependency matrix under
`tests/api/fixtures/dependencies/`.
Preserve relative module and data paths. Udon's `api/fixtures/boundaries` fixture
keeps successful facade use in `src/pass.rs` and rejections in separate field,
curve, and FFT modules. Run one rejection feature
at a time and check its diagnostic and source file. Keep each nested Cargo
workspace alive across its feature cases so they can reuse compiled artifacts.
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

Udon's compiler and embedding tests use the shared
[consumer utility](../crates/udon/tests/harness/mod.rs) for workspace
isolation and Cargo outcome checks. Keep case tables and artifact-specific
assertions in the individual tests.

### Slow consumer tests

Udon's embedding, curve constant, and public API boundary compiler tests are
marked `#[ignore]`. They create fresh Cargo workspaces and run release builds
across feature and rejection cases, so their compilation cost recurs even when
the parent workspace is already built. Keep arithmetic correctness, encoding,
and in-process storage checks in the default suite.

Run the slow consumers explicitly when changing their fixtures or harness,
constant macros, public trait bounds, storage contracts, or artifact preparation
and binding:

```console
cargo test --release --locked -p zakura-udon \
  --test api --test field --test curve --test fft -- --ignored
```

CI runs this command in a separate job. Each consumer selects its own dependency
feature matrix, so it only needs to run once. Naming these integration targets
also avoids selecting the ignored FFT and inversion timing experiments, which
have their own commands. Target portability checks belong to Bento and run
separately. To run one consumer, select its suite and module, for example:

```console
cargo test --release --locked -p zakura-udon --test field embedding:: -- --ignored
```

### Generated artifacts

A producer/consumer round trip must generate data through the writing API before
building its consumer. Golden bytes alone do not exercise generation. The
[field](../crates/udon/tests/field/fixtures/embedding),
[FFT](../crates/udon/tests/fft/fixtures/embedding), and
[curve](../crates/udon/tests/curve/fixtures/embedding) fixtures own their schemas,
generate through Udon, and write through Bento POD. Consumers borrow embedded
records directly, including `no_std` libraries with stack-owned scratch.

Both feature configurations exercise arithmetic directly on embedded values.
Field fixtures preserve both loose and reduced limbs exactly, including loose
zero represented by the modulus. Truncation must fail during embedding
compilation. Compiler tests enforce the reduction-state API boundaries and
reject raw constructor inputs outside their state's bound.

The curve fixture also prepares both entry layouts, reuses scalar digits, and
executes indexed MSMs from embedded bases. Its structured reference string
stores coefficient and Lagrange bases generated with group-valued FFTs. A direct
group DFT checks the basis conversion; polynomial coefficients and evaluations
must give the same commitment. Preserve this artifact-generation use when
changing generic transform support.
