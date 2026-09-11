# Testing

Choose the narrowest test layer that establishes the property under review.
Keep tests with the crate that owns the behavior; integration across crate
boundaries belongs with the public API or consumer that assembles them. Future
arithmetic tests need not live in the support facade.

The [CI workflow](../.github/workflows/ci.yml) defines the required checks and
pins the additional toolchain and targets. The [README](../README.md#testing)
lists the local baseline. The main suite runs with optimizations so validation
must not depend on debug assertions. CI also runs the runtime field unit tests
without optimizations to exercise internal bounds assertions and catch stack
growth in generated addition chains:

```console
cargo test --locked -p zakura-udon --lib
```

## Test roles

- Unit tests check algorithms, parsers, and local contracts beside their code.
  Use independent references for arithmetic and representation checks, including
  boundary values and inputs wider than native integers. Reference arithmetic
  and runtime field tests use `num-bigint` as a development dependency for this
  purpose. Replaying output with the same algorithm is not an independent
  correctness check.
- Public API tests check observable behavior and interactions between components.
  Token snapshots establish expansion structure, not successful compilation or
  runtime semantics.
- Compiler tests establish type, diagnostic, constant-evaluation, and dependency
  contracts in separate consumers. Use full builds for assertions deferred to
  code generation; `cargo check` can miss them. Check the relevant diagnostic and
  source location without pinning the compiler's entire rendered output.
  Reference-arithmetic consumers verify that every facade macro rejects runtime
  arguments, that direct functions and contexts are unavailable, and that
  constant inputs work through dependency aliases and re-exports.
- Examples demonstrate complete uses and assert their results. Configure runnable
  examples with `test = true` and `harness = false` so the suite executes them.
  Doctests verify focused public API examples.

Safety and portability need targeted evidence as well as native tests. CI runs
Miri over storage unit tests and the public storage integration tests; nested
Cargo tests stay in the native suite. The portability test builds `no_std`
libraries for a 32-bit little-endian target and separately checks that big-endian
storage fails while addition chains, constant arithmetic, and runtime Pasta
field operations compile. The arithmetic fixture also asserts computed values
during constant evaluation; cross-target runtime field operations are built but
not executed.
The test is ignored in ordinary runs because target libraries must be installed,
and explicitly executed in CI.
These checks do not establish correctness on every target or constant-time
behavior; extend validation when new code introduces new assumptions.

## Field benchmarks

The [`udon` Criterion suite](../crates/udon/benches/field.rs) measures the
nontrivial public field operations for both `Fp` and `Fq`, including arithmetic,
encoding and reduction, roots and inverses, and product accumulation. It also
measures the `CanonicalUint` integer helpers. Constant accessors and plain
copies are omitted.

```console
cargo bench --locked -p zakura-udon --bench field
```

Filter by benchmark name to focus a run, or use Criterion's test mode to execute
every case once without collecting timing samples:

```console
cargo bench --locked -p zakura-udon --bench field -- Fp/inner_product
cargo bench --locked -p zakura-udon --bench field -- --test
```

Inputs are deterministic and prepared before timing. Ordinary arithmetic uses
operands spanning all four limbs; variable-time operations have separate cases
for different inputs, exponent shapes, and lengths. Byte reduction covers its
short, wide, and arbitrary-width paths. Inner products include sizes around the
32- and 64-term dispatch thresholds and report throughput in products per second.
The 128-value corpus measures dependent and independent multiplication/squaring,
and varied inversion and square-root inputs. Byte reduction reports bytes per
second. See the [field performance report](FIELD_PERFORMANCE.md) for measured
optimization choices and their limits.

The `ProductSum` method benchmarks prepare a fresh populated accumulator outside
each timed iteration. Debug formatting reuses a preallocated output buffer.
Inputs and results pass through optimization barriers. Timings describe these
particular inputs and do not establish a constant-time guarantee. Criterion
stores results and HTML reports under `target/criterion/`, falling back to
`crates/udon/target/criterion/` when Cargo metadata is unavailable. These
measurements are separate from the correctness suite.

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
