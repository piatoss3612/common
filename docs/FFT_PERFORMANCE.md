# FFT strategy performance

FFT strategy selection depends on memory, ordering, and concurrency constraints.
The [strategy suite](../crates/udon/benches/fft_strategies.rs) measures Udon's
choices under supplied constraints. See the [FFT guide](FFT.md) for
operation contracts and resource queries.

The tables below were collected on September 12, 2026. They retain comparisons
between internal strategies, but the current `fft::run` drivers, interpolation
plan binding, and scratch requirements differ from the measured versions.
Use these results as evidence of tradeoffs and the commands below for current
timings. Obtain temporary storage counts from the current plans.

## Method

Measurements ran sequentially on `aarch64-apple-darwin`, with Rust 1.91.0,
LLVM 21.1.2, the pinned dependencies, and the default Udon features. The CPU
model and exact strategy-comparison revision were not recorded. No builds or
tests ran concurrently with timing. Criterion used ten samples, 300 ms of
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
The recorded class timings include binding restored buffers to class descriptors;
the current suite constructs interpolation plans outside timing.
The total task budget includes both outer jobs and inner transforms. Blocked
geometry uses 1,024-element local transforms and 32 columns per job. Reported
table bytes exclude input/output buffers, fixed stack frames, and executor
resources. Each Pasta field occupies 32 bytes.

The suite also covers both fields, subgroup and order-three shifts, lengths
2,048, 16,384, and 1,048,576, and one or four tasks. The measurements below
cover selected cases, not every combination. Run the corresponding current cases
with:

```console
cargo bench --locked -p zakura-udon --bench fft_strategies -- 'Fp/(strategies/2048/generic_7/tasks_1|expansion_strategies/tasks_4|class_strategies/tasks_4|strategy_preparation)'
cargo bench --locked -p zakura-udon --bench fft_strategies -- 'Fp/(strategies/1048576/generic_7/tasks_4|batch_strategies/tasks_4)|Fq/strategies/2048/generic_7/tasks_1'
```

Criterion retains local samples under `target/criterion`. Measure the memory,
concurrency, and persistent-table tradeoffs in the consuming application.
Udon owns kernel and decomposition choices within those constraints. These
results do not establish x86-64 performance or
constant-time behavior.

## Avoiding repeated work

Prepared stages reuse validated `Transform` twiddles without rescanning entries.
Imported contents are checked by `Tables::bind`; trusted binding retains its
caller obligations. A forward-oriented half-table can also serve inverse
transforms by reconstructing opposite powers.

Short bit-reversed expansions broadcast the scaled coefficient prefix, skip
zero-only DIT stages, and permute the result into the requested order. Terminal
products read factors in that declared order. The `base_size / 16` cutoff saves
at least four stages for nonempty input; longer prefixes retain DIF. The cutoff
was selected using September 12 comparisons based on `d0c00ac`: extending it to
`base_size / 8` did not consistently improve table-free products. It remains a
heuristic, not a portable crossover guarantee. Measure boundary cases with:

```console
cargo bench --locked -p zakura-udon --bench fft_strategies -- expansion_prefixes
```

These kernel measurements exclude the consuming application. Include buffer
ownership, allocation, and row access patterns when choosing a pipeline.

## Transform backends and initialization

For 2,048 Fp values on the coset with shift 7, one task, natural input and output,
and no retained tables, times are in microseconds:

| Backend | Forward | Normalized inverse |
| --- | ---: | ---: |
| In-place stages | 312.04 [310.92, 314.02] | 312.87 [310.82, 315.10] |
| Blocked | 325.20 [323.46, 327.19] | 316.42 [314.92, 317.84] |

These are separate-input operations even for the in-place stage backend: the
backend name describes its working transform.

Scatter, gather, and blocked initialization took 312.51, 316.34, and 319.30 µs
respectively for the same forward stage transform. These are complete transform
times, not isolated copy timings. Scatter visits source coefficients in order;
gather initialization can partition destination writes and seed scaling at
chunk boundaries. This serial result does not determine their parallel ranking.

For 1,048,576 Fp values, shift 7, four workers, natural input and output, and
no tables, times are in milliseconds. Each input or output occupies 32 MiB.

| Backend | Forward | Normalized inverse |
| --- | ---: | ---: |
| In-place stages | 181.60 [180.83, 182.38] | 188.02 [179.85, 196.00] |
| Blocked | 194.02 [184.97, 204.15] | 168.18 [159.52, 177.20] |

Large-transform intervals are wider than the small serial controls. This run
does not establish one winner for both directions. The blocked candidate
includes its microtiled column gathers; these measurements do not isolate the
gather cost from the rest of the backend.

## Twiddle storage and coefficient powers

The following 2,048-element Fp cases use the stage backend, one task, shift 7,
and natural boundaries. Times are in microseconds.
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
Each result occupies 524,288 bytes. The coefficient-workspace policy retains
a separate 65,536-byte coefficient buffer; disposable input uses the existing
input buffer for coefficients. These are data-buffer sizes, excluding transform
scratch. Times are in microseconds, with identical input restoration excluded.

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

| Backend | One polynomial | Four polynomials | Eight polynomials |
| --- | ---: | ---: | ---: |
| In-place stages | 166.79 | 704.74 | 1,495.3 |
| Blocked | 317.38 | 607.12 | 1,408.8 |

The batch scheduler spends concurrency on separate polynomials before inner
transforms. Batching exposes that scheduling choice, but does not guarantee
lower time per polynomial. The blocked samples were noisy: the four-polynomial
interval was [588.48, 646.93] µs.

Four equal-domain size-2,048 Fp classes, one output plus three lifts, gave the
following four-worker results. None of these cases retained tables.

| Operation | Time, µs | Observable lift contents |
| --- | ---: | --- |
| Sequential classes, fused finish | 609.11 [597.71, 624.97] | Each lift's coefficients |
| Parallel classes | 311.39 [310.95, 312.48] | Each lift's coefficients |
| Destructive sum | 174.29 [172.24, 176.93] | Consumed |

The destructive case adds equal-domain evaluations before one inverse. Its
lower arithmetic cost follows from the weaker storage contract; it is not a
replacement for callers that need each interpolated lift. Mixed domains need
separate inverses for distinct groups. Scratch depends on the class sizes,
geometry, and division of the total task budget.

## Codelets and field-kernel experiments

For the 2,048-element Fp forward transform with bit-reversed output, one task,
shift 7, and no tables, radix-2, radix-4, and radix-8 took 284.91,
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
The measured AArch64 build contained no remaining `codelet_step` calls. Stage
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
candidates across distributions. The measured assembly showed an unrolled interleaved
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
integer bounds, artifact validation, scratch requirements, and canonical recovery
when an executor panics.
