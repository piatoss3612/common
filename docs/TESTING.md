# Testing

Choose the narrowest test layer that establishes the property under review.
Keep tests with the crate that owns the behavior; integration across crate
boundaries belongs with the public API or consumer that assembles them. Future
arithmetic tests need not live in the support facade.

The [CI workflow](../.github/workflows/ci.yml) defines the required checks and
pins the additional toolchain and targets. The [README](../README.md#testing)
lists the local baseline. The main suite runs with optimizations so validation
must not depend on debug assertions. CI also runs the runtime field and FFT unit
tests without optimizations to exercise internal bounds assertions, including
the loose FFT butterflies, and catch stack growth in generated addition chains:

```console
cargo test --locked -p zakura-udon --lib
cargo test --locked -p zakura-udon --lib --all-features
```

CI also runs the release Udon tests with both default and all features to
exercise both [square-root configurations](../crates/udon/src/lib.rs), and
lints the default configuration separately because workspace lints enable
all features. Tests compare field operations against independent integer
arithmetic and square roots against a conventional Tonelli–Shanks reference.
Checks of the larger tables cover every stored power and all 256 subgroup
hash inputs, and table construction rejects colliding hash multipliers and
unreduced entries.

FFT tests cover both fields against direct polynomial evaluation and an
independently scheduled reference FFT, all subsets of optional tables, coset
shifts, short prefixes, tiled execution, residue layouts, expansion, and fused
class interpolation. Both inverse input orders and the separate-output inverse
run in these unit tests, including the debug CI lane. Tests include domain sizes
from 2,048 to 16,384 elements and a 65,536-element reference comparison and round
trip in each field. Algebraic checks cover all 32 nontrivial root orders. They
also cover scoped parallel execution, scratch reuse and rejection before
mutation, and normalization on unwind. Expansion checks cover caller execution
within every residue, combined concurrency across and within residues, and
partitioned scratch with one, two, and eight residues. Const sizing queries cover
valid configurations, invalid sizes and options, and target storage limits. The
portability and embedding consumers use these queries for table and scratch
array lengths. Private butterflies that retain unreduced Montgomery residues
are checked against integer arithmetic around `p` and `2p`, where `p` is the
field modulus, with reduced twiddles near zero and `p`.
Inverse-size division is checked against integer arithmetic for every exponent
from 0 through 32, using canonical and loose inputs around modulus and limb
boundaries, including carries into a fifth numerator limb. Panic checks cover
partially normalized regions and every join boundary in tiled inverse execution.

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
Miri over storage unit tests and the public Bento and Udon storage integration
tests, including field arrays and nested records; nested Cargo tests stay in
the native suite. The portability test builds `no_std` libraries for a 32-bit
little-endian target and separately checks that big-endian storage fails while
addition chains, constant arithmetic, and runtime Pasta field and FFT operations
compile with either square-root configuration. The arithmetic fixture also
asserts computed values during constant evaluation; cross-target runtime field
and FFT operations are built but not executed. The test is ignored in
ordinary runs because target libraries must be installed, and explicitly
executed in CI. These checks do not establish correctness on every target or
constant-time behavior; extend validation when new code introduces new
assumptions.

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
cargo bench --locked -p zakura-udon --bench field --features sqrt-table-large -- --test
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

## FFT benchmarks

The [FFT Criterion suite](../crates/udon/benches/fft.rs) measures both fields at
2,048, 16,384, and 1,048,576 elements on subgroups and cosets with shifts `zeta`
and 7. Transform cases compare the reference, computed powers, direction-specific
twiddle tables, and inverse finish tables.
`into` and `copy_in_place` both include output initialization. Prefix cases cover
zero, one, five, and one-eighth of the domain's coefficients.

Expansion uses a 2,048-element base with one, two, or eight residues. The
`compare` cases give dense zero-padding, `forward_prefix`, and residue expansion
the same coefficient prefixes and, for products, equivalent factor values.
`native` leaves each algorithm's output in its own layout. Dense and prefix
outputs are already natural order; compare them with `residues/natural` when a
consumer needs natural output, including that case's timed layout conversion.
Short-product comparisons include pointwise multiplication in every method.
Other cases measure expansion from evaluations with and without residue scales,
and fused versus separate class interpolation.

```console
cargo bench --locked -p zakura-udon --bench fft
cargo bench --locked -p zakura-udon --bench fft -- Fp/fft/16384
cargo bench --locked -p zakura-udon --bench fft -- Fp/expansion/16384/generic_7
cargo bench --locked -p zakura-udon --bench fft -- --test
```

The [prepared-strategy suite](../crates/udon/benches/fft_strategies.rs) compares
in-place and blocked backends; scatter, gather, and blocked initialization;
radix-2/4/8 codelets producing bit-reversed output; dense, local stage-packed,
full packed, and strided tables; ordinary bound `Plan` tables in both root
orientations; and forward coset powers. It also compares short coefficient
expansions and fused products, including the ten-coefficient instance case,
expansion storage and normalization policies in both output orders,
polynomial-major batches, and sequential, parallel, or destructive class sums.
Both fields run with one task and a persistent four-worker Rayon pool.
Preparation of twiddles and scale tables is timed separately.

```console
cargo bench --locked -p zakura-udon --bench fft_strategies -- Fp/strategies/2048/generic_7/tasks_1
cargo bench --locked -p zakura-udon --bench fft_strategies -- Fp/expansion_strategies/tasks_4
cargo bench --locked -p zakura-udon --bench fft_strategies -- Fp/expansion_prefixes/tasks_1
cargo bench --locked -p zakura-udon --bench fft_strategies -- --test
cargo test --release --locked -p zakura-udon compare_fft_butterfly_candidates -- --ignored --nocapture
```

The ignored kernel experiment compares branching and masked range corrections
and interleaving two independent products, over zero, boundary, and random
loose inputs. Its ordinary boundary test always runs. Candidates retain four
limbs in `[0,2p)`, where `p` is the field modulus, and do not change runtime
defaults. See the [performance report](FFT_PERFORMANCE.md) for measurement
conditions and limits.

Both suites prepare tables separately into allocated destinations. In-place
kernel cases clone inputs outside timing; output cases reuse buffers and include
all initialization. Reported table and scratch bytes exclude input/output
buffers and executor resources. In the `fft` suite, whole-transform execution
needs no scratch. Its tiled cases use 1,024-element tiles at size 2,048
and 2,048-element tiles at larger sizes, with up to 128 columns per task.

That suite's serial whole-transform and tiled controls measure kernel and
scheduling costs. Persistent Rayon pools with one, two, and four workers measure
parallel scaling; pool creation and entry are outside the timed loop. Expansion
compares execution across residues, within residues, and both, with the product
of the two task budgets bounded by the worker count. The tiled serial control
uses a budget of four. Interpolation remains a serial comparison. Rayon is only
a benchmark dependency; callers still supply Udon's executor.

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
The [FFT embedding consumer](../crates/udon/tests/fixtures/fft_embedding) owns its
record schema and build script. It generates both fields' tables through Udon,
writes them through Bento POD, then runs transforms directly from the embedded
records in a `no_std` library with stack-owned buffers. The harness runs with
no Udon features and with `alloc,sqrt-table-large`.
It also injects damage after generation: a truncated record must fail in
`embed_struct!` during compilation, while unreduced field entries, incorrect
residue scales, unsupported schema or twiddle-kind metadata, wrong normalization
metadata, and corrupted packed powers must reach the embedded consumer and fail
its explicit validation before operations use the damaged data.
