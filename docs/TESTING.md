# Testing

Choose the narrowest test layer that establishes the property under review.
Keep tests with the crate that owns the behavior; integration across crate
boundaries belongs with the public API or consumer that assembles them. Future
arithmetic tests need not live in the support facade.

The [CI workflow](../.github/workflows/ci.yml) defines the required checks and
pins the additional toolchain and targets. The [README](../README.md#testing)
lists the local baseline. The main suite runs with optimizations so validation
must not depend on debug assertions. CI also runs the runtime field, curve, and
FFT unit tests without optimizations to exercise internal bounds assertions,
including the loose FFT butterflies, and catch stack growth in generated
addition chains:

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

Curve tests compare both Pasta groups against an independent affine reference
using `num-bigint`, including full-width scalars, identity and inverse cases,
different Jacobian scalings, and canonical encoding rejection. The group-order
check walks the raw integer order instead of reducing it to a zero scalar.
[GLV](CURVES.md#glv-decomposition-and-the-endomorphism) checks compare lattice
relations, fixed-point rounding, signed bounds, and reconstruction against
integer arithmetic, including coefficient-rounding boundaries. Digit checks
reconstruct all 48 signed rotations and extreme halves; expanded-window checks
include partial windows and synthetic final carries.
Fixed-base checks cover both entry layouts and every supported window width,
comparing every table entry with the independent reference. Compact tables
also check all eight representatives. Batch and preparation tests cover
scratch reuse, untouched tails, and rejection before mutation. Full compiler
builds check both constant point macros through an aliased dependency and
re-exports, and reject invalid coordinates, runtime
arguments, wrong fields, wrong curves, and wrong scalar types.

MSM tests compare dense and repeated-index inputs in all three base layouts
against binary-ladder sums that bypass GLV and MSM recoding. Both curves cover
dispatch and digit-chunk boundaries, zero and full-width scalars, final signed
carries, bucket doubling and cancellation, odd survivors, and passes as short
as one term. Grouped execution checks varied budgets, unequal jobs, nested
single-worker execution, and dirty scratch reuse after either scheduling phase
unwinds.

Compact-table batches check shared preparation, retained scalar digits, and
same-scalar products against individual multiplication. Synthetic digit
schedules exercise exact modular exception detection and its projective fallback.

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

Test executor adapters from inside their worker pool, including a pool with one
worker, to expose nested joins that depend on an idle worker. The
[execution tests](../crates/udon/src/exec/tests.rs) also exercise borrowed results
and cleanup after a branch panics. Nested helper tests track the combined task
allowances of active callbacks, independently of how many threads run them.

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

Udon's unit tests share deterministic sampling and integer conversions in
[`test_support`](../crates/udon/src/test_support.rs), compiled only with
`cfg(test)`. Keep operation-specific boundary cases and reference algorithms
beside their tests.

Safety and portability need targeted evidence as well as native tests. CI runs
Miri over storage unit tests and the public Bento and Udon storage integration
tests, including field arrays, affine and cached points, and nested records;
nested Cargo tests stay in the native suite. The portability test builds
`no_std` libraries for a 32-bit little-endian target and separately checks that
big-endian storage fails while addition chains, constant arithmetic, shared
execution helpers, and
runtime Pasta field, curve, and FFT operations compile with either square-root
configuration. The fixture also asserts computed values during constant
evaluation; runtime field, curve, and FFT operations and execution helpers are
built but not executed on those targets. The test is ignored in ordinary runs
because target libraries must be installed, and explicitly executed in CI.
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

## Curve benchmarks

The [curve Criterion suite](../crates/udon/benches/curve.rs) measures meaningful
runtime operations for both Pallas and Vesta:

- Projective addition, mixed addition, subtraction, negation, doubling,
  normalization, and equality, including identity, equal points at different
  Jacobian scales, and inverse pairs. `point` cases measure affine-result
  arithmetic, including its normalization cost.
- Curve-equation evaluation and checked coordinate construction, with valid,
  off-curve, unreduced, and identity inputs. Compressed encoding and decoding
  cover both point types, both signs, identity, and malformed encodings that
  fail canonical-coordinate or square-root checks.
- Ordinary scalar multiplication from all three point representations, with
  zero, one, small, sparse high-bit, dense low-limb, dense full-limb, and
  minus-one scalars, plus identity bases. Sparse and dense 64- and 65-bit inputs
  straddle the ordinary multiplication dispatch threshold. Scalars `lambda`,
  `-lambda`, and `1 +/- lambda`, where `lambda` is the scalar field's cube root
  of unity, exercise short GLV decompositions despite full-width encodings.
  A 32-scalar corpus compares all three point representations and retained
  tables on the same base. A separate 32-point corpus varies encoding and
  decoding inputs.
- Standalone GLV decomposition uses the same scalar shapes and corpus, with
  reconstruction and magnitude checks outside timing. The inputs cover all
  four sign combinations of nonzero halves. Endomorphisms cover all three
  point representations, including identity and nontrivial projective scaling;
  cached affine entry construction measures the cost of preparing its extra
  coordinate. Public table-entry rotations cover both representations and both
  nonzero rotations. Cache checks cover valid, inconsistent, and unreduced
  cached coordinates with valid affine coordinates.
- Batch normalization at 1, 2, 3, 8, 64, and 1,024 points, including mixed and
  all-identity batches. `individual` controls normalize the same inputs one at
  a time into the same output layout. Throughput counts all input positions,
  including identities.
- Table preparation, checked binding, trusted binding, explicit validation,
  and multiplication with both affine and cached entries. `fixed_base` and
  `fixed_base_cached` cover every supported window width (`2..=8`);
  `eisenstein` and `eisenstein_cached` cover the eight-entry compact tables.
  The same scalar corpus supports comparisons at equal width and similar
  storage, such as affine width 4 versus cached width 3. Checked binding and
  validation also cover wrong multiples, unreduced entries, and inconsistent
  caches at the first and last entries, exposing early rejection and full scans.
  The last expanded entry checks carry-slot validation. Width-2 multiplication
  also covers final carries with positive and negative second GLV halves. See the
  [curve guide](CURVES.md#fixed-base-multiplication) for table and scratch costs.

Constant accessors and storage sizing, plain representation copies, derived
affine equality, the constant cache check for uncached entries, and debug
formatting are omitted. Table multiplication includes internal recoding, which
has no separate benchmark in this suite. Rejection of invalid descriptions and
buffer lengths remains in the correctness suite.

```console
cargo bench --locked -p zakura-udon --bench curve
cargo bench --locked -p zakura-udon --bench curve -- Pallas/fixed_base
cargo bench --locked -p zakura-udon --bench curve -- Pallas/glv_decompose
cargo bench --locked -p zakura-udon --bench curve -- Pallas/endomorphism
cargo bench --locked -p zakura-udon --bench curve -- Pallas/prepared_affine
cargo bench --locked -p zakura-udon --bench curve -- Pallas/table_entry
cargo bench --locked -p zakura-udon --bench curve -- Pallas/encoding
cargo bench --locked -p zakura-udon --bench curve -- Pallas/batch_normalize
cargo bench --locked -p zakura-udon --bench curve -- --test
cargo bench --locked -p zakura-udon --bench curve --features sqrt-table-large -- --test
```

Inputs are deterministic and fixture checks run before timing. Setup and
allocation occur outside timed execution; preparation measures filling existing
buffers, and binding borrows existing entries. Batch and corpus cases report
points, scalar decompositions, or scalar multiplications per second. Inputs and
results pass through optimization barriers. Original `operations`, dense batch,
and width-4/8 fixed-base benchmark names are preserved for existing Criterion
baselines.
Use name filters for focused timing runs or `--test` to exercise every case
once; CI runs test mode with both square-root configurations. These measurements
describe the chosen inputs, not a constant-time guarantee.

## MSM and compact-table batch benchmarks

The [MSM Criterion suite](../crates/udon/benches/msm.rs) measures both curves with
preallocated outputs and scratch. Compact batches compare preparation, ordinary
per-table multiplication, reused scalar digits, and shared same-scalar ladders
in both entry layouts, at 1, 8, 32, 64, 128, and 512 bases. MSM cases cover dense
and indexed access in all three base layouts with full-width scalars. Short
scalars use affine bases. Sizes straddle dispatch boundaries and extend to
4,096 terms.

The `msm_corpus` groups add random 96- and 128-bit scalars, two-bit scalars within
128 bits and with a bit above position 128, repeated equal bases, alternating
inverse bases with random scalars, and exact pair cancellation. Sizes include
both sides of the 32- and 128-term dispatch boundaries. `warm` reuses input
and working buffers. `cold` touches every 64 bytes of a 64 MiB eviction buffer
before each measured execution at 128 and 1,024 terms; eviction time is excluded.
This establishes repeatable cache pressure, not a hardware guarantee that every
input is absent from cache. `prepare_scalars` times preparation separately;
`reused` retains a `PreparedScalars` handle and includes execution with its
reduced scratch counts.

Grouped `ipa` cases use two equal indexed jobs; `commitments` uses four unequal
dense jobs. Each runs serially and inside a persistent four-worker Rayon pool,
with no pass cap and with a 512-term cap. Benchmark names identify the curve,
operation, base layout, scalar corpus, execution policy, cap, and size. MSM
names also identify dense or indexed access, either explicitly or through the
grouped fixture's name. For example:

```text
pallas/eisenstein/mul_prepared/full/serial/cap_all/64
pallas/msm/indexed/affine/full/serial/cap_all/128
pallas/msm_batch/ipa/cached/full/rayon4/cap_512/1024
```

Fixture checks compare MSMs with scalar inner products over known generator
multiples before timing. Allocation and pool entry are outside timing. MSM
input construction, including length and index validation, is also outside
timing; execution's scratch checks, recoding, initialization, scheduling, and
arithmetic are inside. Compact preparation includes base validation. The
`mul_prepared` and `mul_same_scalar` cases prepare scalar digits before timing;
`mul` prepares them for each table during timing.

```console
cargo bench --locked -p zakura-udon --bench msm
cargo bench --locked -p zakura-udon --bench msm -- pallas/msm/dense/affine/full
cargo bench --locked -p zakura-udon --bench msm -- pallas/msm_batch
cargo bench --locked -p zakura-udon --bench msm -- pallas/eisenstein
cargo bench --locked -p zakura-udon --bench msm -- pallas/msm_corpus
cargo bench --locked -p zakura-udon --bench msm -- --test
```

CI runs every case once in test mode. The
[curve performance report](CURVE_PERFORMANCE.md) records dispatch experiments,
scratch tradeoffs, and the limits of the local measurements.

An ignored Criterion experiment retains the former batch-inversion schedule
for a same-process comparison with the endpoint optimization in both fields:

```console
cargo test --release --locked -p zakura-udon --lib compare_batch_inversion_endpoints -- --ignored --nocapture
```

It checks equivalent outputs before timing and alternates each input vector
with its inverse, using the same operands in both schedules. Run timing
experiments without concurrent builds or tests.

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
uses a budget of four. Interpolation remains a serial comparison. Rayon is a
development dependency for benchmarks and execution tests; callers still supply
Udon's executor.

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

Udon's field, FFT, and curve embedding tests and curve constant compiler tests
are marked `#[ignore]`. They create fresh Cargo workspaces and run release builds
across feature and rejection cases, so their compilation cost recurs even when
the parent workspace is already built. Keep arithmetic correctness, encoding,
and in-process storage checks in the default suite.

Run the slow consumers explicitly when changing their fixtures or harness,
constant macros, storage contracts, or artifact preparation and validation:

```console
cargo test --release --locked -p zakura-udon \
  --test embedding --test fft_embedding \
  --test curve_constants --test curve_embedding -- --ignored
```

CI runs this command in a separate step. Each consumer selects its own dependency
feature matrix, so it only needs to run once. Naming these integration targets
also avoids selecting the ignored FFT and inversion timing experiments, which
have their own commands. Target portability checks belong to Bento and run
separately. To run one consumer, retain just its `--test` argument and
`-- --ignored`.

### Generated artifacts

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

The [curve embedding consumer](../crates/udon/tests/fixtures/curve_embedding)
also owns its record schema and generator. It prepares both curves' compact and
expanded tables with affine and cached entries through Udon, writes them through
Bento POD, and checks multiplication from borrowed embedded records against
ordinary multiplication. It also reuses prepared scalar digits, borrows compact
table batches, and executes indexed MSMs directly from affine and cached
embedded entries with const-sized stack scratch and capped passes. Both feature
configurations run the consumer and damage cases. Truncation must fail during
compilation; obsolete schema, incorrect curve, table-kind, entry-layout, or
window metadata, unreduced or off-curve entries, reordered multiples, wrong
bases, inconsistent caches, and incorrect final carry entries must fail
validation before multiplication.
