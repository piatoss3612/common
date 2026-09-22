# Curve multiplication performance

Choose retained tables, scalar preparation, and MSM geometry according to the
reuse and memory budget of the application. The [curve guide](CURVES.md)
explains the workflows; API rustdoc owns their sizing and validation contracts.
This report records tuning evidence and how to repeat the relevant measurements.

## Method

The retained-table and compact-batch measurements below were collected on
September 12, 2026 with Rust 1.91.0, LLVM 21.1.2, `aarch64-apple-darwin`, and
default features. They measured a working GLV implementation based on revision
`96aead50d8fff5c24561e4fa096249bb2cabd4fd`; its exact candidate revision and CPU
model were not recorded. Subsequent arithmetic and scheduling changes mean
these timings are evidence of storage/reuse tradeoffs, not current latency
estimates. The MSM comparisons below identify their separate provenance.

The [curve harness](../crates/udon/benches/curve.rs) measures a deterministic
corpus of 32 full-width scalars acting on the same base. Preparation fills one
already allocated table. The table measurements used 20 samples, one second of
warmup, and one second of measurement. Allocation and fixture checks were
outside timing; inputs and outputs passed through optimization barriers.
Timing runs were sequential without concurrent builds or tests. No x86-64
runtime measurements were available, and none of these timings establishes
constant-time behavior.

Measure the current implementation with:

```console
cargo bench --locked -p zakura-udon --bench curve -- 'scalar_mul/corpus' --save-baseline candidate --sample-size 20 --measurement-time 1 --warm-up-time 1
cargo bench --locked -p zakura-udon --bench curve -- '(fixed_base(_cached)?/w(3|4|7|8)|eisenstein(_cached)?)/(prepare|mul/corpus)$' --save-baseline candidate --sample-size 20 --measurement-time 1 --warm-up-time 1
```

## Ordinary multiplication

GLV uses an eight-entry cached table on the stack, normalizes its representatives
with one inversion, and runs a joint Eisenstein doubling ladder over two signed
halves. Preparing from projective coordinates avoids a separate base inversion.
Scalars fitting `u64` retain the binary ladder.

The current setup's local arrays hold eight projective points (768 bytes), eight
cached affine entries (768 bytes), and eight field scratch elements (256 bytes).
These capacities total 1,792 bytes before other locals, temporaries, and called
functions; they are not a stack usage bound. Compiler inlining, register allocation,
and the target determine the actual stack requirement. Neither one-shot method
offers a strategy override or an inversion-free path for large scalars.

## Retained tables

Times below are microseconds. Preparation covers one table; multiplication
covers 32 products. Values are Criterion point estimates. Storage counts only
entries, excluding the base, handle, and temporary preparation scratch.

| Table | Entry bytes | Pallas prepare | Pallas multiply | Vesta prepare | Vesta multiply |
| --- | ---: | ---: | ---: | ---: | ---: |
| Compact, affine | 512 | 2.30 | 468.71 | 2.31 | 469.45 |
| Compact, cached | 768 | 2.38 | 464.91 | 2.38 | 463.36 |
| Expanded width 4, affine | 16,448 | 83.95 | 223.06 | 86.12 | 227.61 |
| Expanded width 4, cached | 24,672 | 87.54 | 220.78 | 88.37 | 222.69 |
| Expanded width 8, affine | 131,136 | 474.15 | 118.54 | 482.70 | 119.84 |
| Expanded width 8, cached | 196,704 | 499.01 | 116.99 | 500.21 | 117.18 |

Compact tables save repeated preparation with a small retained footprint.
Expanded tables avoid scheduled doublings and achieve lower multiplication
latency at the cost of more storage and setup. Cached entries add 50% storage
at a fixed width and gave only about 1–2% lower multiplication times here;
the compact intervals overlap. Affine entries remain the default.

Similar storage budgets favor larger affine windows in these measurements:

| Comparison | Affine bytes / cached bytes | Pallas multiply, affine / cached | Vesta multiply, affine / cached |
| --- | ---: | ---: | ---: |
| Affine width 4 / cached width 3 | 16,448 / 16,608 | 223.06 / 276.54 | 227.61 / 280.45 |
| Affine width 8 / cached width 7 | 131,136 / 116,832 | 118.54 / 134.99 | 119.84 / 135.68 |

## Compact-table batches and scalar reuse

The [MSM harness](../crates/udon/benches/msm.rs) separately measures preparing
many compact tables and multiplying their bases by one full-width scalar. It
uses the same toolchain and target as above, default features, 20 samples,
0.3 seconds of warmup, and one second of measurement. Tables and scratch are
allocated before timing; preparation includes base validation. These cases vary
the base and hold the scalar fixed. Times are Criterion point estimates in
microseconds for the whole batch; entries use the affine layout.

| Bases | Pallas prepare | Vesta prepare |
| --- | ---: | ---: |
| 8 | 7.11 | 7.28 |
| 32 | 22.00 | 22.28 |
| 128 | 81.25 | 82.42 |
| 512 | 325.61 | 329.38 |

At eight bases, preparation switches to four shared affine inversion phases.
The 512-base case costs about 0.64 microseconds per table. Retained entries
still occupy 512 bytes per base, and preparation needs four field elements
(128 bytes) per base with no projective scratch at these sizes.

`EisensteinScalar` retains the joint digits computed from GLV decomposition. On
the 512-base fixture, ordinary per-table `mul` took 8,691.61 / 9,084.08
microseconds for Pallas / Vesta; retaining digits reduced this to
8,468.16 / 8,813.51, about 3%. Scalar preparation occurs before timing for both
individual `mul_prepared` and batch `mul_same_scalar` cases. The shared
same-scalar ladder saves further work:

| Bases | Pallas individual, prepared | Pallas batch | Vesta individual, prepared | Vesta batch |
| --- | ---: | ---: | ---: | ---: |
| 64 | 896.21 | 819.83 | 917.09 | 873.68 |
| 128 | 1,871.52 | 1,643.10 | 1,985.77 | 1,709.36 |
| 512 | 8,468.16 | 7,144.43 | 8,813.51 | 7,420.51 |

The affine ladder did not consistently beat individual prepared products at
32 bases, so its current threshold is 64 bases per partition. It uses five
field elements (160 bytes) of scratch per base and one inversion per column,
including fused `2P + D` columns. The exact exceptional-schedule check is
included in batch timings. Below the threshold, or for an exceptional schedule,
the batch reuses scalar digits with complete projective arithmetic. These
measurements cover one scalar schedule and do not establish the same savings
for every scalar or executor budget.

## Multiscalar multiplication

The [MSM harness](../crates/udon/benches/msm.rs) separates full-width, short,
sparse, repeated-base, cancellation, and reused-scalar workloads. Fixture checks
use scalar inner products over known generator multiples before timing.
Length/index validation, allocation, and pool entry are outside timing;
execution scratch checks, recoding, initialization, and arithmetic are included.
See the [testing guide](TESTING.md#msm-and-compact-table-batch-benchmarks) for
cache-pressure cases and preparation timing boundaries.

Udon chooses kernels, recoding widths, and accumulation from input size, scalar
shape, available scratch, and the task budget. Measure the operation under its
intended workspace ceiling and concurrency; a task allowance is not a worker count.

### Arithmetic and bucket reduction

The [nonzero inversion helper](../crates/udon/src/field/batch.rs) uses two
independent multiplication lanes. Seeding each lane directly and omitting its
unused final update gives `3(n - 1)` field multiplications for `n >= 2`
denominators, excluding the inversion. A singleton uses field inversion
alone. The [endpoint experiment](TESTING.md#internal-msm-experiments) compares
schedules on identical operands; a fixed work saving need not materially
improve a large batch.

[Projective addition](../crates/udon/src/curve/projective.rs) uses unscaled
differences and fused `mul_sub_product`; doubling uses half-scaled Jacobian
coordinates. The local comments derive those formulas. Equal, inverse, and
identity branches remain complete. Independent affine integer references and
field-halving boundary checks establish their arithmetic behavior separately
from timings.

The [affine bucket reducer](../crates/udon/src/curve/msm/buckets.rs) recovers
inverses while adding and compacting point pairs, sharing an inversion across
one level. Its chord attempt falls back to complete reduction on a zero
product before changing points or lengths. The fallback also preserves odd
survivors when every pair cancels. Pair workspace is two field elements
(64 bytes); this is not a whole-MSM storage bound.

The following reducer and width measurements used working candidates based on
`57b76e7` on September 13, 2026: Apple M4 Max, `aarch64-apple-darwin`, Rust
1.91.0, LLVM 21.1.2. The temporary harness and raw samples were not retained;
these summaries are historical evidence, not fresh measurements of every
current default. Values are median microseconds over three session medians.
Reducer sessions used seven samples of at least 30 ms, rotating candidate order
and including point/length resets. At occupancy 17, Pallas/Vesta pairs were:

| Points | Prefix inversion | Complete fused recovery | Chord attempt with fallback |
| ---: | ---: | ---: | ---: |
| 128 | 10.813 / 10.810 | 10.694 / 10.699 | 10.322 / 10.357 |
| 1,024 | 69.403 / 69.133 | 68.255 / 67.970 | 65.936 / 65.658 |
| 8,192 | 550.489 / 578.325 | 536.986 / 533.446 | 507.259 / 520.251 |

These ordinary pairs support fused recovery; they do not measure frequent
exception fallback. The retained [native controls](TESTING.md#internal-msm-experiments)
compare production reduction with alternative schedules and expressions, and
check results before timing.

### Width, accumulation, and memory

The width sweep used indexed cached bases, eight deterministic pseudorandom
254-bit scalar rows, preallocated scratch, and persistent Rayon pools. Such
scalars exercise full-width arithmetic but are not uniform field samples.
Seven samples per session lasted at least 10 ms; policy order rotated between
samples and sessions. Scalar preparation, recoding, and execution were timed;
base preparation, binding, allocation, pool entry, and independent result checks
were excluded. At 4,096 terms, Pallas/Vesta times were:

| Workers | Width ten, µs | Width eleven, µs |
| ---: | ---: | ---: |
| 1 | 11,085 / 11,071 | 11,631 / 11,653 |
| 4 | 3,542 / 3,541 | 3,091 / 3,084 |
| 16 | 1,560 / 1,560 | 1,269 / 1,260 |

Width eleven uses twelve windows instead of thirteen, with 1,024 buckets per
workspace instead of 512. Twelve tasks divide evenly across four workers;
fewer windows also reduce input passes. The sweep does not isolate those
contributions or establish every size/budget crossover. Current
[plan geometry](../crates/udon/src/curve/msm/recode.rs) selects width eleven for
automatic Booth plans at grains of at least 4,096 with more than one task.
Serial plans retain width ten at that boundary. Preparation and execution use
the same resolved geometry; callers do not select a width.
The [execution report](EXECUTION_PERFORMANCE.md) compares stage and blocked FFTs
and synchronous and bounded MSM execution under selected worker counts.

Projective buckets retain sums across short passes without repeatedly inverting
sparse affine levels. Automatic accumulation uses them below a 128-term pass;
a single small pass can still favor affine. Hybrid accumulation remains an
internal experimental alternative. These are replaceable heuristics, not universal occupancy
thresholds. Scalar shape can also favor short arithmetic, so the `short` corpus
of coefficients `137 * i` cannot represent every dense 128-bit workload.

The workspace ceiling covers temporary arithmetic storage and retained
intermediates. Obtain requirements from the resolved plan and use the
[memory contract](CURVES.md#sizing-and-reusing-scratch) for its accounting scope.
Streaming Booth retains all windows' projective buckets while recoding
one chunk at a time. It avoids repeated chunk collapses at the cost of a larger
fixed bucket floor. Application-wide accounting also includes idle scratch,
retained preparation, and metadata outside one operation's ceiling.

Retaining scalar records or digits moves preparation outside execution; it does
not make their storage free. Compare one-shot, prepare-and-execute, and repeated
execution lifecycles separately. For repeated indices, caller-side coalescing
may reduce arithmetic, but a fair comparison includes sorting, gathering, and
scalar summation. Neither reuse nor coalescing benefits every workload.

Run selected current cases with:

```console
cargo bench --locked -p zakura-udon --bench msm -- 'msm/dense/affine/(full|short)' --save-baseline msm --sample-size 20 --measurement-time 1 --warm-up-time 0.3
cargo bench --locked -p zakura-udon --bench msm -- 'msm_batch/' --save-baseline msm --sample-size 20 --measurement-time 1 --warm-up-time 0.3
cargo bench --locked -p zakura-udon --bench msm -- 'eisenstein/(prepare_batch|mul|mul_prepared|mul_same_scalar)/' --save-baseline msm --sample-size 20 --measurement-time 1 --warm-up-time 0.3
```

Unchanged controls in the recorded development runs moved by several percent,
occasionally around 10%. Build comparable binaries before timing, run them
sequentially, and repeat controls when interpreting small differences. Measure
complete application lifecycles before selecting a memory or concurrency policy.
