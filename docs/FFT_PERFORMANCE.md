# FFT strategy performance

FFT strategy selection depends on the caller's memory, ordering, and concurrency
constraints. These September 12, 2026 measurements compare the explicit choices
in the earlier version of the
[strategy suite](../crates/udon/benches/fft_strategies.rs).
They do not select runtime defaults. See the [FFT guide](FFT.md) for operation
contracts and resource queries.

The original strategy comparisons were collected before the removal of
experimental backends and table representations. This report retains
measurements for the remaining strategies. The
[upgrade refinements](#measured-upgrade-refinements) record a later comparison
against revision `d0c00ac`.

The current suite uses the `fft::run` plans, whose drivers and scratch counts
differ from the measured versions. It also constructs interpolation plans
outside timing. The tables below retain the historical results; use the
commands below to measure the current implementation.

## Method

Measurements ran sequentially on `aarch64-apple-darwin`, with Rust 1.91.0,
LLVM 21.1.2, the pinned dependencies, and the default Udon features. No builds
or tests ran concurrently with timing. Criterion used ten samples, 300 ms of
warmup, and a one-second requested measurement period; slow cases can require
longer. Tables below report elapsed-time estimates. Brackets, where shown,
give Criterion's 95% confidence intervals.

Inputs contain deterministic canonical field values. Single-transform cases
preserve their input and reuse an allocated destination. Timing includes
initialization, coset scaling, permutations required by the selected order,
and scratch writes. Allocation, domain construction, operation configuration,
and table preparation are outside execution timing. Preparation is measured
separately into allocated buffers.

Parallel cases use a persistent four-worker Rayon pool, entered outside the
timed loop. Expansion, batch, and class inputs are restored outside timing.
The reported class timings include binding restored buffers to class descriptors.
The total task budget includes both outer jobs and inner transforms. Blocked
geometry uses 1,024-element local transforms and 32 columns per job. Reported
table and temporary bytes exclude input/output buffers, fixed stack frames,
and executor resources. Each Pasta field occupies 32 bytes.

The suite also covers both fields, subgroup and order-three shifts, lengths
2,048, 16,384, and 1,048,576, and one or four tasks. The measurements below
cover selected cases, not every combination. Run the corresponding current cases
with:

```console
cargo bench --locked -p zakura-udon --bench fft_strategies -- 'Fp/(strategies/2048/generic_7/tasks_1|expansion_strategies/tasks_4|class_strategies/tasks_4|strategy_preparation)'
cargo bench --locked -p zakura-udon --bench fft_strategies -- 'Fp/(strategies/1048576/generic_7/tasks_4/(columns_(false|true)|DIF)|batch_strategies/tasks_4)|Fq/strategies/2048/generic_7/tasks_1'
```

Criterion retains local samples under `target/criterion`. A downstream tuner
can measure candidate descriptions under its own limits and persist a selected
description. A mathematically valid table does not identify the fastest strategy
on another machine. These results do not establish x86-64 performance or
constant-time behavior.

## Measured upgrade refinements

Two changes reduce repeated work: prepared stages reuse validated `Plan`
twiddles, and bit-reversed expansion prunes stages for short coefficient
prefixes. In baseline revision `d0c00ac`, prepared stages revalidated twiddles
on every stage, and bit-reversed expansion always ran full DIF transforms.
These September 12, 2026 comparisons compile the same benchmark cases against
both implementations. The platform and timing boundaries follow the method
above. Values in this section are sample means in microseconds.

Prepared stages now adapt already-bound twiddle slices without rescanning
their entries. Imported contents are still checked by `Tables::bind`; trusted
binding keeps its existing caller obligations. All cases below borrow one
ordinary forward half-table, including inverse transforms that reconstruct
the opposite powers. Inputs and outputs use natural order on the zeta coset.

| Field | Size | Direction | Tasks | Before, µs | After, µs | Less time |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Fp | 2,048 | Forward | 1 | 317.2 | 201.1 | 36.6% |
| Fp | 2,048 | Inverse | 1 | 331.2 | 215.8 | 34.8% |
| Fp | 16,384 | Forward | 1 | 3,901.1 | 2,666.5 | 31.6% |
| Fq | 2,048 | Forward | 1 | 331.6 | 202.3 | 39.0% |
| Fq | 2,048 | Forward | 4 | 357.5 | 65.6 | 81.7% |
| Fq | 2,048 | Inverse | 4 | 359.5 | 68.2 | 81.0% |

The larger parallel improvement removes validation work that was repeated in
each stage's calling task before its butterflies could run concurrently. The
2,048-element cases retain 32,768 table bytes; the 16,384-element case retains
262,144. None requires scratch fields. The change introduces no new storage.

Short bit-reversed expansions now broadcast the scaled coefficient prefix,
skip the initial zero-only DIT stages, and permute each completed residue into
its requested order. Terminal stores read factors in that order before the
permutation. The following cases expand ten coefficients from a 2,048-element
base to the 16,384-element zeta coset and multiply a nonconstant factor.
Timing includes coefficient initialization, scaling, the local permutations,
and the fused factor product. There is no base inverse in these cases.

| Field | Tasks | Twiddles | Before, µs | After, µs | Less time |
| --- | ---: | --- | ---: | ---: | ---: |
| Fp | 1 | Computed | 1,979.2 | 1,138.8 | 42.5% |
| Fp | 1 | Forward table | 2,335.4 | 738.0 | 68.4% |
| Fp | 4 | Computed | 541.4 | 305.6 | 43.6% |
| Fp | 4 | Forward table | 635.4 | 202.2 | 68.2% |
| Fq | 1 | Computed | 2,025.1 | 1,172.4 | 42.1% |
| Fq | 1 | Forward table | 2,362.6 | 759.9 | 67.8% |
| Fq | 4 | Computed | 545.6 | 310.0 | 43.2% |
| Fq | 4 | Forward table | 637.4 | 208.5 | 67.3% |

Table cases benefit from both changes and retain 32,768 bytes. All cases
require zero scratch fields and produce 524,288 output bytes; the factor
also occupies 524,288 bytes. No expansion-scale table is retained.

The pruning cutoff is `base_size / 16`, saving at least four initial stages
for nonempty input. At the 128-coefficient boundary, table-free products took
7–8% less time across these fields and task budgets. Trying a 256-coefficient
cutoff did not consistently improve the table-free product, so longer prefixes
retain DIF. Consecutive before/after control runs of table-free inverses and
256- or 2,048-coefficient expansions changed by −2.1% to +0.2%. Longer runs
showed timing drift; small differences do not establish additional wins.

The suite covers constants, the pruning boundary, and full coefficients in
both expansion orders. To compare revisions, use the same benchmark harness
on the baseline implementation and save its samples, then repeat on the changed
implementation with `--baseline fft_upgrade_before`:

```console
cargo bench --locked -p zakura-udon --bench fft_strategies -- \
  '(Fp|Fq)/strategies/2048/zeta/tasks_[14]/plan_twiddles|expansion_prefixes' \
  --save-baseline fft_upgrade_before
```

These kernel and expansion measurements exclude the cost of a consuming
application. Measure complete pipelines with their buffer ownership, allocation,
and row access patterns before choosing a strategy.

## Transform backends and initialization

For 2,048 Fp values on the coset with shift 7, one task, natural input and output,
and no retained tables, times are in microseconds:

| Backend | Temporary bytes | Forward | Normalized inverse |
| --- | ---: | ---: | ---: |
| In-place stages | 0 | 312.04 [310.92, 314.02] | 312.87 [310.82, 315.10] |
| Blocked | 2,048 | 325.20 [323.46, 327.19] | 316.42 [314.92, 317.84] |

These are separate-input operations even for the in-place stage backend: the
backend name describes its working transform.

Scatter, gather, and blocked initialization took 312.51, 316.34, and 319.30 µs
respectively for the same forward stage transform. These are complete transform
times, not isolated copy timings. Scatter visits source coefficients in order;
gather initialization can partition destination writes and seed scaling at
chunk boundaries. This serial result does not determine their parallel ranking.

For 1,048,576 Fp values, shift 7, four workers, natural input and output, and
no tables, times are in milliseconds. Each input or output occupies 32 MiB.

| Backend | Temporary bytes | Forward | Normalized inverse |
| --- | ---: | ---: | ---: |
| In-place stages | 0 | 181.60 [180.83, 182.38] | 188.02 [179.85, 196.00] |
| Blocked | 4,194,304 | 194.02 [184.97, 204.15] | 168.18 [159.52, 177.20] |

Large-transform intervals are wider than the small serial controls. This run
does not establish one winner for both directions. The blocked candidate
includes its microtiled column gathers; these measurements do not isolate the
effect of that gather change from the rest of the backend.

## Twiddle storage and coefficient powers

The following 2,048-element Fp cases use the stage backend, one task, shift 7,
natural boundaries, and zero temporary fields. Times are in microseconds.
All inverse cases borrow the same forward-oriented table, reconstructing inverse
powers by the root identity; they do not retain a separate inverse table.

| Retained strategy | Bytes | Forward | Normalized inverse |
| --- | ---: | ---: | ---: |
| Recurrence | 0 | 312.04 | 312.87 |
| Dense half-table | 32,768 | 202.40 | 225.44 |
| Packed stages through size 256 | 8,160 | 234.15 | 252.36 |
| All packed stages | 65,504 | 202.07 | 229.18 |
| Strided dense table of size 4,096 | 65,536 | 204.15 | 235.24 |

Dense and fully packed forward intervals overlap: [200.64, 204.49] and
[199.77, 205.09] µs. Full packing did not establish a benefit over the smaller
dense half-table here. Local packing retains fewer bytes and speeds up these
cases relative to recurrence, with less improvement than full packing.
The strided case charges the complete retained larger table, even though the
operation only reads a subset.

Prepared coefficient powers alone took 301.00 [300.26, 302.54] µs and retained
65,536 bytes. This case still regenerates twiddles. Removing coset power
recurrences and removing butterfly twiddle recurrences are separate choices.

Preparation of a forward table for size 1,048,576 gave:

| Table | Retained bytes | Preparation |
| --- | ---: | ---: |
| Dense | 16,777,216 | 6.093 ms |
| All packed stages | 33,554,400 | 12.059 ms |

These preparation times exclude artifact serialization, validation of imported
contents, and compiler work.

## Expansion storage, normalization, and order

These Fp operations interpolate 2,048 base subgroup evaluations and evaluate
eight residues on the size-16,384 coset with shift 7, using four workers.
Each result occupies 524,288 bytes. Every case here needs 8,192 scratch bytes;
the coefficient-workspace policy additionally needs 65,536 bytes. Disposable
input instead consumes the existing 65,536-byte input buffer. Times are in
microseconds, with identical input restoration excluded from each case.

| Scale table | Output order | Reuse output | Coefficient workspace | Disposable input |
| --- | --- | ---: | ---: | ---: |
| None | Residues | 1,244.3 | 905.4 | 904.9 |
| None | Full bit reversal | 1,129.4 | 814.4 | 820.0 |
| Ordinary | Residues | 1,182.2 | 873.5 | 871.0 |
| Ordinary | Full bit reversal | 1,169.0 | 852.6 | 863.3 |
| Pre-normalized | Residues | 1,214.6 | 839.8 | 839.7 |
| Pre-normalized | Full bit reversal | 1,143.4 | 840.8 | 842.0 |

Both table forms retain 524,288 bytes. Ordinary entries contain
`(shift * extended_root^residue)^i`, where `extended_root` is the extended
domain's canonical root, `residue` ranges from zero through seven, and `i` is
the coefficient degree, `0 <= i < base_size`. Pre-normalized entries include
the factor `1 / base_size` and consume an unscaled inverse. Preparation took
220.98 and 219.52 µs respectively. A size-16,384 coefficient power sequence
took 217.69 µs.

Separate coefficient storage removes the residue-zero dependency and improved
these cases. Pre-normalized tables improved the residue-order workspace case,
but did not improve every ordering and storage combination. For example, the
table-free bit-reversed workspace case took [811.96, 817.27] µs, compared with
[838.82, 842.96] µs using pre-normalized scales. Retaining a table is not an
unconditional execution win.

The two output orders serve different consumers. Full bit reversal places
residue blocks and their inner outputs directly in the order consumed by a
full inverse accepting bit-reversed input. The comparison includes producing
each declared layout; it does not charge either layout for converting to the
other. Domain-bound views
carry that distinction through pointwise products.

## Batches and class interpolation

Polynomial-major batches of size-2,048 Fp subgroup forward transforms use one
total four-task budget. The time is for the entire batch, in microseconds;
each polynomial occupies 65,536 bytes. Inputs are restored outside timing.

| Backend | One polynomial | Four polynomials | Eight polynomials | Temporary bytes, one / four / eight |
| --- | ---: | ---: | ---: | ---: |
| In-place stages | 166.79 | 704.74 | 1,495.3 | 0 / 0 / 0 |
| Blocked | 317.38 | 607.12 | 1,408.8 | 8,192 / 8,192 / 8,192 |

The batch scheduler spends concurrency on separate polynomials before inner
transforms. Batching exposes that scheduling choice, but does not guarantee
lower time per polynomial. The blocked samples were noisy: the four-polynomial
interval was [588.48, 646.93] µs.

Four equal-domain size-2,048 Fp classes, one output plus three lifts, gave the
following four-worker results. All cases retained no tables and required
8,192 bytes of scratch with this geometry.

| Operation | Time, µs | Observable lift contents |
| --- | ---: | --- |
| Sequential classes, fused finish | 609.11 [597.71, 624.97] | Each lift's coefficients |
| Parallel classes | 311.39 [310.95, 312.48] | Each lift's coefficients |
| Destructive sum | 174.29 [172.24, 176.93] | Consumed |

The destructive case adds equal-domain evaluations before one inverse. Its
lower arithmetic cost follows from the weaker storage contract; it is not a
replacement for callers that need each interpolated lift. Mixed domains need
separate inverses for distinct groups. Scratch equality in this example follows
from dividing the total task budget among classes; other geometries and class
sizes can require different amounts.

## Codelets and field-kernel experiments

For the 2,048-element Fp forward transform with bit-reversed output, one task,
shift 7, no tables, and no scratch, radix-2, radix-4, and radix-8 took 284.91,
272.98, and 261.18 µs. This compares the same output order. A natural-order
transform has a different permutation contract and is not an equivalent timing
control for this table. Radix choices remain explicit.

The corresponding small Fq times were 299.25, 285.57, and 271.41 µs.
Small natural-order Fq forward/inverse pairs took 318.31/318.25 µs for stages
and 336.52/329.67 µs for blocked execution.
Later Fq table measurements showed substantial variation during this run;
they are not used to rank table strategies across fields.

For size 1,048,576 Fp transforms with four workers and the same bit-reversed
output contract, the corresponding times were 161.94 [155.11, 168.48],
155.73 [148.83, 162.13], and 161.99 [161.57, 162.38] ms. Their intervals overlap;
the small serial radix ranking did not establish a large parallel winner.

The [codelet schedules](../crates/udon/src/fft/stages.rs) have one const step
representation, replayed by a test interpreter and checked against direct
evaluation. Production code expands the steps into fixed-index Rust operations.
Optimized AArch64 assembly contains no remaining `codelet_step` calls. Stage
direction and inverse normalization produce separate specializations, while
twiddle families dispatch at stage boundaries. Terminal
factor and order handling still appear where applicable. This inspection does
not establish that every backend or butterfly is branch-free, or that all
intermediates remain in registers. An order-four Pasta root still requires real
field multiplication.

The [isolated kernel experiment](../crates/udon/src/field/fft/experiments.rs)
compares two independent butterflies in the existing four-limb `[0,2p)` range.
Here `p` is the field modulus. It reports the median of five passes over 1,024
input pairs repeated 256 times, in nanoseconds per pair. A repeated run after
compilation gave:

| Field/input | Branching | Interleaved | Masked | Masked and interleaved |
| --- | ---: | ---: | ---: | ---: |
| Fp / zero | 28.86 | 20.51 | 26.73 | 23.34 |
| Fp / random loose | 24.46 | 20.70 | 27.16 | 23.80 |
| Fp / upper boundary | 23.96 | 19.98 | 27.26 | 23.51 |
| Fq / zero | 23.53 | 19.97 | 27.31 | 23.66 |
| Fq / random loose | 24.33 | 20.46 | 28.12 | 24.09 |
| Fq / upper boundary | 23.90 | 20.23 | 27.26 | 23.35 |

The first Fp zero measurement varied substantially between runs, so it should
not be used to estimate a stable speedup. Interleaving helped these isolated
candidates across distributions. Assembly also shows an unrolled interleaved
pair, a loop in the sequential pair, and out-of-line masked corrections; the
timing includes those compiler decisions. The compiler generated conditional
branches inside the masked correction itself, so source masks do not establish
branchless machine code. These candidates remain test-only and do not change
the production range correction or claim a whole-transform speedup.

Run the experiment and emit the benchmark assembly with:

```console
cargo test --release --locked -p zakura-udon compare_fft_butterfly_candidates -- --ignored --nocapture
cargo rustc --release --locked -p zakura-udon --bench fft_strategies -- --emit=asm
```

The emitted `.s` file is under `target/release/deps`. On macOS, inspect the
release test executable with `xcrun llvm-objdump --demangle --disassemble` to
include the test-only candidates. Correctness coverage separately checks both
fields against direct evaluation, independent inverse transforms, loose-range
integer bounds, artifact validation, resource limits, and canonical recovery
when an executor panics.
