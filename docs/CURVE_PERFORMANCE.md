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
