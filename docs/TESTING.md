# Testing

Choose the narrowest test layer that establishes the property under review.
Keep tests with the crate that owns the behavior; integration across crate
boundaries belongs with the public API or consumer that assembles them.
The [CI workflow](../.github/workflows/ci.yml) defines required gates and pins
the additional toolchain and targets. The [README](../README.md#testing) lists
formatting, lint, and documentation commands.

Run the release suites with both square-root configurations, then the runtime
unit tests with debug assertions:

```console
cargo test --release --locked --workspace --all-features
cargo test --release --locked -p zakura-udon
cargo test --locked -p zakura-udon --lib
cargo test --locked -p zakura-udon --lib --all-features
```

Release tests ensure validation does not depend on debug assertions. Debug
runs also exercise loose FFT bounds and stack growth in generated chains.
The default configuration has a separate Clippy gate because workspace lints
enable all features. Slow consumers and timing experiments are ignored by the
ordinary suites and have separate commands below.

## Test roles

- Unit tests check algorithms and local contracts against independent references.
  Field arithmetic uses integer oracles and a conventional Tonelli–Shanks
  reference. Curves use affine `num-bigint` arithmetic, raw integer group orders,
  and integer GLV reconstruction. MSMs use binary ladders that bypass production
  decomposition and recoding. FFTs use direct sums, polynomial evaluation, and
  independent transform schedules, including loose-range and inverse-scale
  boundaries. Comparing two paths that share the same algorithm is insufficient.
- Public API tests check observable interactions. The
  [workspace tests](../crates/udon/tests/workspaces/main.rs) and
  [execution tests](../crates/udon/tests/execution/main.rs) exercise borrowed
  workers, scratch reuse, scheduling, and failure boundaries. Distinguish setup
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
[`test_support`](../crates/udon/src/test_support.rs). Executor adapters should
also run inside a one-worker pool to expose nested joins that assume an idle
worker. Preserve independent checks of cached rotations, scalar reuse, fused
and unnormalized FFTs, group-valued transforms, and supplied/batched chains
when changing their public entry points.

Miri checks storage separately from native nested-Cargo consumers:

```console
cargo +nightly-2026-09-06 miri test --locked -p zakura-bento-core --lib pod::
cargo +nightly-2026-09-06 miri test --locked -p zakura-bento --test pod
cargo +nightly-2026-09-06 miri test --locked -p zakura-udon --test pod
```

Install that toolchain with `miri` and `rust-src` as described in CI. Portability
requires the `thumbv7em-none-eabi` and `s390x-unknown-linux-gnu` target libraries:

```console
cargo test --release --locked -p zakura-bento --test portability -- --ignored
```

This builds `no_std` consumers with both square-root configurations, checks
constant results and independent big-endian storage rejections, and compiles
runtime arithmetic and execution helpers. It does not execute arithmetic on
those targets or establish constant-time behavior.

## Benchmark execution

Run timing commands sequentially, without concurrent builds or tests. Use name
filters for focused runs. Criterion's test mode executes each case once without
collecting timing samples; CI runs all six suites plus both square-root table
configurations for fields and curves:

```console
cargo bench --locked -p zakura-udon --bench field --bench curve --bench fft --bench fft_strategies --bench msm --bench execution -- --test
cargo bench --locked -p zakura-udon --features sqrt-table-large --bench field --bench curve -- --test
```

Inputs are deterministic, with fixture checks outside timing and optimization
barriers around inputs/results. Timing boundaries differ by workload as noted
below. Criterion writes local output under `target/criterion/` (or
`crates/udon/target/criterion/` when Cargo metadata is unavailable). Keep generated
results out of curated source. Timings measure particular public inputs; they
are not side-channel guarantees.

## Field benchmarks

The [field suite](../crates/udon/benches/field.rs) covers arithmetic, encodings,
reduction, inversion, roots, product sums, and integer helpers in both fields.
Corpora span all limbs and distinguish dependent and independent products,
variable-time exponent shapes, and lengths around dispatch boundaries.
`ProductSum` cases populate fresh accumulators outside timing; formatting reuses
an allocated buffer. Preparation and allocation are outside ordinary execution.

```console
cargo bench --locked -p zakura-udon --bench field
cargo bench --locked -p zakura-udon --bench field -- Fp/inner_product
```

The [field report](FIELD_PERFORMANCE.md) explains measured choices, including
the larger square-root tables and their storage/build-time tradeoffs.

## Curve benchmarks

The [curve suite](../crates/udon/benches/curve.rs) covers both curves, including
complete arithmetic exceptions, encoding rejection, GLV/endomorphism operations,
normalization, and ordinary and retained-table multiplication. Scalar corpora
straddle short/full-width boundaries; table cases distinguish affine and cached
entries, preparation, binding, and multiplication.
Compare the same 32-scalar corpus at equal width or similar storage budgets.
Preparation fills allocated buffers; binding borrows existing entries; table
multiplication includes recoding. Invalid table entries at the beginning and
end distinguish early rejection from full scans.

```console
cargo bench --locked -p zakura-udon --bench curve
cargo bench --locked -p zakura-udon --bench curve -- Pallas/fixed_base
cargo bench --locked -p zakura-udon --bench curve -- Pallas/batch_normalize
```

See the [curve guide](CURVES.md#fixed-base-multiplication) for binding and
sizing contracts and the [performance report](CURVE_PERFORMANCE.md) for evidence.

## MSM and compact-table batch benchmarks

The [MSM suite](../crates/udon/benches/msm.rs) compares dense and indexed inputs,
base layouts, full-width and short scalars, compact-table batches, and grouped
jobs. The `msm_corpus` cases add dense 96/128-bit scalars, sparse scalars,
repeated/inverse bases, and cancellation. `ipa` groups two equal indexed jobs;
`commitments` groups unequal dense jobs. Serial and four-worker cases compare
unrestricted workspace with a 64 KiB workspace ceiling.

`warm` reuses buffers. `cold` touches each 64-byte interval of a 64 MiB eviction
buffer before execution, excluding that work from timing. This creates repeatable
cache pressure without guaranteeing hardware eviction. `prepare_scalars` times
preparation; `reused` retains the prepared handle and sizes its execution scratch
accordingly. Compact `mul_prepared` and `mul_same_scalar` prepare scalar digits
before timing; `mul` prepares them for each table inside timing.

Fixture checks use independent scalar inner products over known generator
multiples. Allocation, pool entry, and input length/index validation are outside
timing. Execution scratch checks, recoding, initialization, scheduling, and
arithmetic are inside. Compact preparation takes constructed affine bases.

```console
cargo bench --locked -p zakura-udon --bench msm
cargo bench --locked -p zakura-udon --bench msm -- pallas/msm/dense/affine/full
cargo bench --locked -p zakura-udon --bench msm -- pallas/msm_batch
cargo bench --locked -p zakura-udon --bench msm -- pallas/msm_corpus
cargo bench --locked -p zakura-udon --bench msm -- pallas/eisenstein
```

### Internal MSM experiments

```console
cargo test --release --locked -p zakura-udon --lib compare_batch_inversion_endpoints -- --ignored --nocapture
cargo test --release --locked -p zakura-udon --lib curve::msm::experiments::native_controls -- --ignored --nocapture
cargo test --release --locked -p zakura-udon --lib curve::msm::experiments::phases -- --ignored --nocapture
```

The inversion experiment compares schedules on identical nonzero inputs,
alternating each vector with its inverse. Native controls compare the six-field
reducer, production fused inverse recovery, and a two-pass `mul_sub` expression
candidate. The expression candidate does not fuse inverse recovery. Histograms
and result checks occur outside timing. Phase measurements isolate preparation,
reduction (including resets), window kernels, collapse, and recombination;
workspace figures are not an additive peak. The
[MSM performance guide](CURVE_PERFORMANCE.md#multiscalar-multiplication) describes
the rationale and limits of these comparisons.

## FFT benchmarks

The [FFT suite](../crates/udon/benches/fft.rs) covers subgroups and cosets in both
fields, transforms, prefixes, expansion, and fused interpolation. `into` and
`copy_in_place` include initialization. Expansion comparisons give algorithms
equivalent coefficient prefixes and factors. `native` retains each output
layout; compare natural-order methods with `residues/natural` when the consumer
needs natural order, including its timed conversion.

The [strategy suite](../crates/udon/benches/fft_strategies.rs) compares task budgets,
workspace ceilings, output orders, retained twiddles/powers, expansion
storage/normalization, batches, and class interpolation. Udon selects the
implementation under those constraints. Interpolation plans are bound outside
timing. Both suites measure table preparation into allocated buffers separately.
In-place cases restore inputs outside timing; separate-output cases include
initialization.
Persistent pool creation and entry are outside timing. Task allowances cover
both outer jobs and inner transforms; scratch counts exclude input/output and
executor storage. Callers supply the executor; Rayon is a development dependency.

```console
cargo bench --locked -p zakura-udon --bench fft -- Fp/fft/16384
cargo bench --locked -p zakura-udon --bench fft -- Fp/expansion/16384/generic_7
cargo bench --locked -p zakura-udon --bench fft_strategies -- Fp/strategies/2048/generic_7/tasks_1
cargo bench --locked -p zakura-udon --bench fft_strategies -- Fp/expansion_prefixes/tasks_1
cargo test --release --locked -p zakura-udon compare_fft_butterfly_candidates -- --ignored --nocapture
```

The ignored kernel experiment compares range corrections and interleaved
products over zero, boundary, and random loose inputs. Its ordinary correctness
check runs by default. Test-only candidates retain the `[0,2p)` range and do not
select production defaults. See the [strategy report](FFT_PERFORMANCE.md) for
measurement limits and the [execution report](EXECUTION_PERFORMANCE.md) for
isolated and mixed-operation timing boundaries.

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

Udon's compiler and embedding tests use the shared
[consumer utility](../crates/udon/tests/support/consumer.rs) for workspace
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
  --test embedding --test fft_embedding \
  --test curve_constants --test curve_embedding --test api_boundaries -- --ignored
```

CI runs this command in a separate step. Each consumer selects its own dependency
feature matrix, so it only needs to run once. Naming these integration targets
also avoids selecting the ignored FFT and inversion timing experiments, which
have their own commands. Target portability checks belong to Bento and run
separately. To run one consumer, retain just its `--test` argument and
`-- --ignored`.

### Generated artifacts

A producer/consumer round trip must generate data through the writing API before
building its consumer. Golden bytes alone do not exercise generation. The
[field](../crates/udon/tests/fixtures/embedding),
[FFT](../crates/udon/tests/fixtures/fft_embedding), and
[curve](../crates/udon/tests/fixtures/curve_embedding) fixtures own their schemas,
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
