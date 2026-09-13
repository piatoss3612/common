# Curve multiplication performance

The September 12, 2026 GLV implementation reduces ordinary full-width scalar
multiplication latency by about half on the measured target. Expanded tables
use approximately half the former storage and preparation time, with a modest
increase in multiplication latency at the same window width. The
[curve guide](CURVES.md) describes the APIs, storage contracts, and migration.

## Method

Measurements used Rust 1.91.0, LLVM 21.1.2, and `aarch64-apple-darwin` with the
default features. The baseline is Udon revision
`96aead50d8fff5c24561e4fa096249bb2cabd4fd`. The
[Criterion harness](../crates/udon/benches/curve.rs) preserves the ordinary and
affine expanded-table benchmark names and inputs across the comparison.
Each run used 20 samples, one second of warmup, and one second of measurement.
Timing runs were sequential, without concurrent builds or tests.

Multiplication times cover the entire deterministic corpus of 32 full-width
scalars acting on the same base. Preparation times cover one table filled into
existing buffers. Inputs and outputs pass through optimization barriers;
allocation and fixture checks are outside timing. These are local
microbenchmarks, not application speedups or constant-time measurements. No
x86-64 runtime measurements were available.

Local Criterion snapshots use `before-glv` and `glv`; the ordinary post-change
comparison is in `new`. Repeat with each implementation and separate snapshots:

```console
cargo bench --locked -p zakura-udon --bench curve -- 'scalar_mul/corpus' --save-baseline candidate --sample-size 20 --measurement-time 1 --warm-up-time 1
cargo bench --locked -p zakura-udon --bench curve -- '(fixed_base(_cached)?/w(3|4|7|8)|eisenstein(_cached)?)/(prepare|mul/corpus)$' --save-baseline candidate --sample-size 20 --measurement-time 1 --warm-up-time 1
```

## Ordinary multiplication

Times are microseconds for 32 products, with 95% confidence intervals.

| Curve | Input | Before | GLV |
| --- | --- | ---: | ---: |
| Pallas | Affine | 1,027.80 [1,017.22, 1,035.22] | 508.26 [507.25, 509.53] |
| Pallas | Projective | 1,193.61 [1,185.35, 1,203.53] | 509.22 [507.70, 510.65] |
| Vesta | Affine | 1,014.09 [1,006.36, 1,021.06] | 509.57 [507.77, 511.26] |
| Vesta | Projective | 1,188.25 [1,181.79, 1,197.24] | 510.24 [508.22, 511.91] |

GLV uses an eight-entry cached table on the stack, normalizes its representatives
with one inversion, and runs a joint Eisenstein doubling ladder over two signed
halves. Preparing from projective coordinates avoids a separate base inversion.
The measured cost is about 15.9 microseconds per product. Scalars fitting `u64`
retain the binary ladder; this corpus does not measure those inputs.

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
Expanded tables avoid doublings and achieve substantially lower multiplication
latency at the cost of more storage and setup. Cached entries add 50% storage
at a fixed width and gave only about 1–2% lower multiplication times here;
the compact intervals overlap. Affine entries remain the default.

Similar storage budgets favor larger affine windows in these measurements:

| Comparison | Affine bytes / cached bytes | Pallas multiply, affine / cached | Vesta multiply, affine / cached |
| --- | ---: | ---: | ---: |
| Affine width 4 / cached width 3 | 16,448 / 16,608 | 223.06 / 276.54 | 227.61 / 280.45 |
| Affine width 8 / cached width 7 | 131,136 / 116,832 | 118.54 / 134.99 | 119.84 / 135.68 |

The old full-scalar affine layout had the following costs:

| Width | Entry bytes | Pallas prepare | Pallas multiply | Vesta prepare | Vesta multiply |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 32,832 | 172.46 | 208.07 | 163.18 | 210.19 |
| 8 | 262,208 | 907.51 | 109.39 | 921.41 | 110.55 |

Sharing shifted multiples between the GLV halves approximately halves entries
and preparation work. At the same width, the number of additions stays similar,
while decomposition and endomorphism lookups add work: affine multiplication
was about 7–9% slower here. This change makes the storage and preparation tradeoff
explicit; it does not establish a speedup for every retained-table workload.

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

The MSM harness uses deterministic full-width scalars and generator
multiples spanning all four field limbs. Its `short` corpus uses `137 * i`,
representing small public coefficients. Each time below covers one dense sum
with affine bases and `SerialExecutor`, including recoding, scratch
initialization, and scheduling. Input construction, including length and index
validation, and allocation are outside timing. Execution's scratch checks are
included. The sample settings match the compact batch measurements above;
values are Criterion point estimates in microseconds.

| Terms | Pallas full | Vesta full | Pallas short | Vesta short |
| --- | ---: | ---: | ---: | ---: |
| 31 | 196.55 | 199.15 | 23.61 | 23.91 |
| 32 | 203.47 | 205.18 | 21.73 | 21.97 |
| 64 | 396.45 | 399.46 | 37.56 | 37.80 |
| 128 | 783.55 | 792.88 | 75.48 | 75.75 |
| 256 | 1,176.21 | 1,190.20 | 122.08 | 122.08 |
| 1,024 | 3,668.88 | 3,780.77 | 341.21 | 349.08 |
| 4,096 | 13,338.72 | 13,528.70 | 1,247.32 | 1,323.64 |

Full-width sums below 128 terms use joint Eisenstein Strauss, sharing table
preparation and doubling steps across terms. From 128 terms, signed width-eight
Booth windows group digits of the two GLV halves into buckets for summation.
An intermediate signed width-four tier was slower on the measured full-width
inputs, supporting the direct transition to Booth. For sums below 128 terms
whose scalars all fit 128 bits, bit interleaving avoids table preparation; from
32 terms, four-bit projective buckets improve that short-scalar path. The
`short` timings above cover coefficients of the form `137 * i`, a small subset
of 128-bit scalars. The Booth reducer skips empty high buckets.

A separate local comparison checked identical compressed results against the
prototype in `../bento` at revision
`fe69bfb0a080524f6f06d36aee74ee5d07626d35`. Both implementations received the
same scalars and cached bases through their public serial APIs. Three timed
repetitions alternated implementation order after warmup; each repetition ran
for at least 250 milliseconds. The medians below are microseconds. Udon reused
scratch, while the prototype's one-shot API allocated internally, so these are
API cost comparisons rather than isolated arithmetic-kernel comparisons.

| Terms | Pallas Udon / prototype | Vesta Udon / prototype |
| --- | ---: | ---: |
| 31 | 183.95 / 182.67 | 184.04 / 184.09 |
| 32 | 188.58 / 427.73 | 190.65 / 417.31 |
| 64 | 367.87 / 641.44 | 370.85 / 624.62 |
| 128 | 742.85 / 1,028.35 | 742.04 / 1,040.64 |
| 255 | 1,114.00 / 1,814.91 | 1,089.94 / 1,782.31 |
| 256 | 1,110.27 / 1,135.78 | 1,131.29 / 1,087.99 |
| 1,024 | 3,436.64 / 3,379.21 | 3,468.27 / 3,422.16 |
| 4,096 | 12,533.41 / 12,310.77 | 12,544.48 / 12,412.38 |

The 32–255 term cases improved by roughly 1.4–2.3 times. Larger cases stayed
within a few percent, with small regressions as well as improvements.

### Grouped execution and working storage

The grouped fixtures use cached bases. `ipa` contains two indexed sums of `n`
terms; `commitments` contains dense sums of `n`, `n/2`, `n/4`, and 17 terms.
A persistent Rayon pool supplies the four-worker executor. Pool entry is
outside timing, while both scoped execution phases are included. These
uncapped batch times are microseconds for all results:

| Jobs | n | Pallas serial | Pallas four workers | Vesta serial | Vesta four workers |
| --- | ---: | ---: | ---: | ---: | ---: |
| ipa | 128 | 1,537.27 | 473.94 | 1,582.19 | 471.29 |
| ipa | 1,024 | 7,158.25 | 2,031.93 | 7,189.49 | 2,016.87 |
| commitments | 128 | 1,567.96 | 785.41 | 1,559.98 | 780.36 |
| commitments | 1,024 | 7,192.93 | 2,980.99 | 7,229.43 | 2,928.17 |

The scheduler shares reusable working storage across jobs and gives concurrent
partitions exclusive slices. A pass cap bounds staged terms while retaining
affine bucket survivors between passes. These byte counts sum the five scratch
slices on the measured 64-bit target, excluding borrowed inputs, outputs,
handles, and executor resources. Counts are identical for both curves and
should be obtained through the sizing APIs.

| Input | Task budget | Uncapped scratch bytes | Cap 512 scratch bytes |
| --- | ---: | ---: | ---: |
| One 1,024-term sum | 1 | 404,576 | 236,640 |
| One 4,096-term sum | 1 | 1,516,640 | 341,088 |
| One 4,096-term sum | 4 | 5,643,872 | 941,664 |
| Two 1,024-term ipa sums | 4 | 1,545,408 | 873,664 |

For the last row, capping passes reduced scratch by 43% and changed measured
four-worker latency from 2,031.93 to 2,104.91 microseconds for Pallas and from
2,016.87 to 2,098.02 for Vesta, about 4%. Recoded digits still scale with all
terms, and sizing is independent of scalar values. Concurrent windows need
more working storage than serial execution; a task budget is an allowance,
not a requirement to create that many threads.

Run selected cases with the stable names described in the
[testing guide](TESTING.md#msm-and-compact-table-batch-benchmarks):

```console
cargo bench --locked -p zakura-udon --bench msm -- 'msm/dense/affine/(full|short)' --save-baseline msm --sample-size 20 --measurement-time 1 --warm-up-time 0.3
cargo bench --locked -p zakura-udon --bench msm -- 'msm_batch/' --save-baseline msm --sample-size 20 --measurement-time 1 --warm-up-time 0.3
cargo bench --locked -p zakura-udon --bench msm -- 'eisenstein/(prepare_batch|mul|mul_prepared|mul_same_scalar)/' --save-baseline msm --sample-size 20 --measurement-time 1 --warm-up-time 0.3
```

These local measurements guide internal dispatch and caller memory choices
within the [measurement limits](#method) above.
