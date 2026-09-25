# FFT strategy performance

FFT strategy selection depends on memory, ordering, and concurrency constraints.
The [strategy suite](../crates/udon/benches/fft_strategies.rs) measures Udon's
choices under supplied constraints. See the [FFT guide](FFT.md) for
operation contracts and resource queries.

The tables below were collected on September 12, 2026. They retain comparisons
between internal strategies, but the current `fft::run` drivers and scratch
requirements differ from the measured versions.
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
timed loop. Batch inputs are restored outside timing.
The total task budget includes both outer jobs and inner transforms. Blocked
geometry uses 1,024-element local transforms and 32 columns per job. Reported
table bytes exclude input/output buffers, fixed stack frames, and executor
resources. Each Pasta field occupies 32 bytes.

The suite also covers both fields, subgroup and order-three shifts, lengths
2,048, 16,384, and 1,048,576, and one or four tasks. The measurements below
cover selected cases, not every combination. Run selected current cases with:

```console
cargo bench --locked -p zakura-udon --bench fft_strategies -- 'Fp/(strategies/2048/zeta/tasks_1|expansion_strategies/tasks_4|class_strategies/tasks_4|strategy_preparation)'
cargo bench --locked -p zakura-udon --bench fft_strategies -- 'Fp/(strategies/1048576/zeta/tasks_4|batch_strategies/tasks_4)|Fq/strategies/2048/zeta/tasks_1'
```

Criterion retains local samples under `target/criterion`. Measure the memory,
concurrency, and persistent-table tradeoffs in the consuming application.
Udon owns kernel and decomposition choices within those constraints. These
results do not establish x86-64 performance or
constant-time behavior.

## Avoiding repeated work

Prepared stages reuse borrowed twiddles.
[`Tables::bind`](../crates/udon/src/fft/tables.rs) checks lengths and trusts
the caller to supply the documented entries; it does
not validate their mathematical contents. A canonical forward-root
[`TwiddleTable`](../crates/udon/src/fft/powers.rs) serves both transform directions.
Other table families have their own
[binding contracts](FFT.md#optional-tables-and-downstream-storage).

Short bit-reversed expansions broadcast the scaled coefficient prefix, skip
zero-only DIT stages, and permute the result into the requested order. Terminal
products read factors in that declared order. The `base_size / 16` cutoff saves
at least four stages for nonempty input; longer prefixes retain DIF. It is a
heuristic, not a portable crossover guarantee. Measure boundary cases with:

```console
cargo bench --locked -p zakura-udon --bench fft_strategies -- expansion_prefixes
```

These kernel measurements exclude the consuming application. Include buffer
ownership, allocation, and row access patterns when choosing a pipeline.

## Twiddle preparation

Preparation of a forward table for size 1,048,576 gave:

| Table | Retained bytes | Preparation |
| --- | ---: | ---: |
| Dense | 16,777,216 | 6.093 ms |
| All packed stages | 33,554,400 | 12.059 ms |

These preparation times exclude artifact serialization and compiler work.

## Batches

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

## Codelets and field-kernel experiments

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

The [isolated kernel experiment](../crates/udon/src/field/butterfly/experiments.rs)
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
branchless machine code. These measurements alone did not establish a
whole-transform speedup or change the production range correction. The
[specialization report](OPTIMIZATION_PERFORMANCE.md#fft-execution-and-preparation)
records the complete-transform comparisons supporting production paired
butterflies with distinct twiddles; masked corrections remain test-only.

Run the experiment and emit the benchmark assembly with:

```console
cargo test --release --locked -p zakura-udon compare_fft_butterfly_candidates -- --ignored --nocapture
cargo rustc --release --locked -p zakura-udon --bench fft_strategies -- --emit=asm
```

The emitted `.s` file is under `target/release/deps`. On macOS, inspect the
release test executable with `xcrun llvm-objdump --demangle --disassemble` to
include the test-only candidates. Correctness coverage separately checks both
fields against direct evaluation, independent inverse transforms, loose-range
integer bounds, embedded artifacts, scratch requirements, and canonical recovery
when an executor panics.
