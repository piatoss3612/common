# Benchmarking

Timing workloads and their measurement boundaries. See the
[testing guide](TESTING.md) for correctness suites and compiler consumers.

## Benchmark execution

Run timing commands sequentially, without concurrent builds or tests. Use name
filters for focused runs. Criterion's test mode executes each case once without
collecting timing samples; CI runs all six suites plus both square-root table
configurations for fields and curves:

```console
cargo bench --locked -p zakura-udon --features traits --bench field --bench curve --bench fft --bench fft_strategies --bench msm --bench execution -- --test
cargo bench --locked -p zakura-udon --features traits,sqrt-table-large --bench field --bench curve -- --test
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
It requires `traits` because it also measures the generic batch-inversion API.
Corpora span all limbs and distinguish dependent and independent products,
variable-time exponent shapes, and lengths around dispatch boundaries.
`ProductSum` cases populate fresh accumulators outside timing; formatting reuses
an allocated buffer. Ordinary arithmetic excludes input preparation and allocation.

Polynomial comparisons distinguish retained preparation from `prepare_*` cases
that include it. The weighted-fold `powers_and` cases include challenge-power
generation. Per-call validation and result writes are timed; buffer allocation,
input restoration, and fixture checks are excluded. Polynomial `bind` cases
borrow trusted powers or weights with constant work, as specified by the
[evaluation](../crates/udon/src/polynomial/evaluation.rs) and
[interpolation](../crates/udon/src/polynomial/interpolation.rs) contracts.

```console
cargo bench --locked -p zakura-udon --features traits --bench field
cargo bench --locked -p zakura-udon --features traits --bench field -- Fp/inner_product
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
multiplication includes recoding. Table binding checks dimensions and
configuration but trusts the stored entries; it does not time an entry scan.
Expanded preparation compares one-window, four-window, and full-table scratch
allowances at an explicitly selected width.

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

The dense, indexed, corpus, and grouped MSM cases use independent scalar inner
products over known generator multiples for fixture checks. These cases exclude
allocation, pool entry, and input length/index validation from timing. Execution
scratch checks, recoding, initialization, scheduling, and arithmetic are inside.
Compact preparation takes constructed affine bases.

Structured-row comparisons distinguish retained preparation, preparation plus
execution, and caller-known support. Per-row conversion, aggregation, and
validation are timed; known-support cases exclude support discovery, while
`scan_execute` includes scanning and compaction. Independent binary
ladders check these results before timing. Retained preparation and compacted
inputs are outside the execution workspace ceiling. The `shared_scalars` cases
compare matrix execution with separate MSMs that also reuse prepared scalars.

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
cargo test --release --locked -p zakura-udon --lib msm::experiments::native_controls -- --ignored --nocapture
cargo test --release --locked -p zakura-udon --lib msm::experiments::phases -- --ignored --nocapture
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
fields, transforms, prefixes, expansion, and fused interpolation. It enables
`traits` for its consumer batch-inversion comparisons. `into` and
`copy_in_place` include initialization. Expansion comparisons give algorithms
equivalent coefficient prefixes and factors. `native` retains each output
layout; compare natural-order methods with `residues/natural` when the consumer
needs natural order, including its timed conversion.

The `constant_prefix` FFT controls include dense input materialization and, for
extension, conversion from residue order to natural order. Retained-sample cases
exclude preparation; `prepare_and_extend` includes it. The `vanishing` cases
compare forward-transform finishing with pointwise division and interpolation
into the same coefficient pieces; `finish` alone excludes the transform.
The `group_fft` cases include root powers and scalar preparation on each call;
`basis_conversion` also includes output normalization. Their buffer allocation,
input restoration, and fixture checks precede timing.

The [strategy suite](../crates/udon/benches/fft_strategies.rs) compares task budgets,
workspace ceilings, output orders, retained twiddles and expansion scales,
storage/normalization, batches, and class interpolation. Udon selects the
implementation under those constraints. Interpolation plans are bound outside
timing. Both suites measure table preparation into allocated buffers separately.
In-place cases restore inputs outside timing; separate-output cases include
initialization.
Persistent pool creation and entry are outside timing. Task allowances cover
both outer jobs and inner transforms; scratch counts exclude input/output and
executor storage. Callers supply the executor; Rayon is a development dependency.

```console
cargo bench --locked -p zakura-udon --features traits --bench fft -- Fp/fft/16384
cargo bench --locked -p zakura-udon --features traits --bench fft -- Fp/expansion/16384/zeta
cargo bench --locked -p zakura-udon --bench fft_strategies -- Fp/strategies/2048/zeta/tasks_1
cargo bench --locked -p zakura-udon --bench fft_strategies -- Fp/expansion_prefixes/tasks_1
cargo test --release --locked -p zakura-udon compare_fft_butterfly_candidates -- --ignored --nocapture
```

The ignored kernel experiment compares range corrections and interleaved
products over zero, boundary, and random loose inputs. Its ordinary correctness
check runs by default. Test-only candidates retain the `[0,2p)` range and do not
select production defaults. See the [strategy report](FFT_PERFORMANCE.md) for
measurement limits and the [execution report](EXECUTION_PERFORMANCE.md) for
isolated and mixed-operation timing boundaries.
