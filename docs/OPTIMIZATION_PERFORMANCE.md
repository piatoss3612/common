# Arithmetic specialization measurements

This September 25, 2026 pass evaluated ten opportunities to replace composed
operations or connect existing specialized helpers to production callers.
Nine produced useful improvements. A fused field multiply/add prototype was
slower and remains an experiment. The runtime changes preserve allocation-free,
`no_std`, variable-time arithmetic and caller-owned execution policy.

## Method and scope

The starting revision was `37ebedbbcc8f6e8bc8c4df0f97e3266e6c50f937`.
Measurements ran sequentially on `aarch64-apple-darwin`, Rust 1.91.0, without
concurrent builds or test runs. No x86-64 runtime measurements were available.
Results describe these inputs and this machine, not application-wide gains.

Existing Criterion suites supplied complete-operation comparisons. Targeted
runs used 20 or 25 samples, 0.2 seconds of warmup, 0.3 or 0.5 seconds of
measurement, and 1,000 resamples. Criterion comparisons use its mean estimates
and 95% confidence intervals. Short runs warrant caution about small changes.
Saved local baselines are named `optimizations-before` under `target/criterion`;
logs are under `target/optimizations-again`. These generated files are not part
of the repository.

Ignored experiments compare implementations in one binary. Their reported
medians are descriptive measurements, not confidence intervals. Arithmetic
uses seven passes over prepared inputs; prefix timings use seven batches of
at least 15 ms; owned addition chains use nine passes over 1,024 rows. The
square-root comparison alternates implementation order for 15 passes. The MSM
writer comparison uses seven passes inside an already entered four-worker
pool. Allocation, fixture checks, scalar decomposition, and pool construction
are outside those writer timings.

## Decisions

Numbers below retain the audit's ordering. Reductions refer to elapsed time.

| Finding | Decision | Evidence and scope |
| --- | --- | --- |
| 1. Field `mul_add` / `mul_sub` | Keep composed implementation | A wide product with the addend inserted before Montgomery reduction took about 11.5–11.6 ns versus 8.9–10.1 ns for the existing operations. |
| 2. Paired FFT butterflies | Use pairs in applicable stages | Complete transforms with dense retained twiddles improved by about 7–19% at 2,048 and 16,384 elements in both fields. |
| 3. Large-table square-root success flag | Use the flag | The post-exponentiation step improved from 410.3 to 397.2 ns in Fp and 409.7 to 396.0 ns in Fq. Complete square roots were within measurement noise. |
| 4. Identity factors in FFT tasks | Skip known identities | Applying a zeta twist to 4,096 values improved from 35.9 to 25.4 µs in natural order and 41.1 to 29.8 µs in bit-reversed order for Fp; Fq was similar. |
| 5. Sparse FFT residue powers | Advance a progression for sufficiently large regions | For 4,096 destinations and a 256-coefficient prefix, arbitrary-residue initialization improved from 30.7 to 6.4 µs in Fp. Complete expansion comparisons for findings 4 and 5 improved by 2–6%. |
| 6. FFT table generation | Reuse `fill_powers` | Twiddle/finish preparation improved by 28–31%; expansion-scale preparation improved by about 30%. Execution with prebuilt tables does not include these savings. |
| 7. Small field multiples | Specialize 3, 4, and 8 | Direct modulus-shape folding takes about 1.5–1.6 ns, versus 3.2–5.3 ns for compositions. Complete curve doubling showed only a small change, near measurement noise. |
| 8. Iterator product sums | Specialize short inputs | Empty/singleton sums improved by 47–69%; lengths 2–4 by 4–13%. The 1,024-term controls were unchanged. Four-lane batching was rejected. |
| 9. Owned addition-chain result | Move the result | A 181-chain over a 64-word owned vector improved from 318.6 to 283.5 ns; 1,024-word vectors from 1,286.3 to 1,179.2 ns. No field-arithmetic speedup is claimed. |
| 10. Retained MSM digit caches | Expose `cache_parallel` | The existing writer achieved about 3.5–3.9× speedup with four workers at 1,024–16,384 records for Booth and joint recoding. |

### FFT execution and preparation

The paired butterfly computes two independent twiddle products before their
DIT corrections, or after the DIF sums and differences. It accepts distinct
twiddles and retains the scalar remainder. Terminal stages keep their existing
normalization and output fusion.

Representative complete-transform means, in microseconds:

| Field / size | Dense forward, before → after | Dense inverse, before → after |
| --- | ---: | ---: |
| Fp / 2,048 | 163.0 → 149.8 | 207.5 → 168.3 |
| Fq / 2,048 | 160.0 → 149.3 | 196.1 → 171.0 |
| Fp / 16,384 | 2,372.8 → 2,065.3 | 2,874.6 → 2,331.0 |
| Fq / 16,384 | 2,367.1 → 2,078.5 | 2,868.7 → 2,322.8 |

Computed-twiddle controls were generally within about 2%. These are serial
zeta-coset strategy cases; they do not imply the same gain for every layout,
task budget, or retained-table choice.

Prefix initialization skips an identity common scale and known identity coset
phases. Arbitrary residues reuse `BitReversedPowers` when a region contains at
least 32 complete coefficient repeats; smaller regions retain direct powers.
Progression starts at the region's actual offset and advances across padding.
Retained tables continue to supply their stored factors, including normalization.

An initial whole-expansion comparison suggested a slowdown while unrelated
controls also moved. A follow-up held the other changes fixed and replaced only
`fft/run.rs` with its starting implementation for the control build. Against
that `prefix-control` baseline, the final means were:

| Field / prefix | Control, µs | Optimized, µs | Time change |
| --- | ---: | ---: | ---: |
| Fp / 10 | 878.8 | 856.0 | -2.6% |
| Fp / 128 | 1,638.3 | 1,568.2 | -4.3% |
| Fp / 256 | 1,885.1 | 1,775.2 | -5.8% |
| Fq / 10 | 877.9 | 853.6 | -2.8% |
| Fq / 128 | 1,630.5 | 1,569.6 | -3.7% |
| Fq / 256 | 1,866.8 | 1,825.1 | -2.2% |

These `expansion_prefixes/tasks_1/computed/Residues` cases produce 16,384
outputs without a pointwise product. All six Criterion change intervals
excluded zero. The isolated prefix experiment additionally covers zeta and
arbitrary residues, short regions, and prefixes up to 1,024 coefficients.

For 16,384-point table preparation, Fp forward twiddles improved from 86.6 to
61.1 µs; inverse finish factors from 173.6 to 122.5 µs. Fq improved from 86.4
to 61.2 µs and 171.6 to 122.5 µs respectively. Expansion-scale preparation
uses 2,048-element rows in a 16,384-element extended domain; both coefficient
and unscaled-inverse normalizations improved from about 175–177 to 123 µs.
Generated loose Montgomery limbs need not be byte-identical to the old
recurrence, but field values and representation bounds are preserved.

### Field arithmetic and owned values

Small multiples use `p = 2^254 + c`. For input `x < 2p` and multiplier at most
eight, folding `q = floor(k*x / 2^254)` leaves a value between `-p` and `p`;
one addition of `p` repairs a borrow. Independent integer tests cover quotient
boundaries, borrow cases, and the largest loose representatives in both fields.
The alternative `x + x.half()` for three-halves was slower than the specialized
`x.triple().half()` and was not adopted. Multiplication by four/eight has no
current production call sites; those gains apply to direct API users.

The rejected multiply/add prototype forms `ab + cR` before Montgomery
reduction. Preserving and correcting its wider result costs more than the
existing loose CIOS multiplication followed by addition. The test-only
prototype and independent integer checks remain available for future compiler
or architecture comparisons.

Iterator sums dispatch using actual entries, without trusting `size_hint()`.
At most four initial products omit overflow folding; subsequent accumulation
keeps the unrestricted overflow handling. A four-lane candidate cost about
47 ns for four terms versus the original 33 ns, and regressed long sums by
about 6%; the retained implementation avoids that setup and merge cost.

The large square-root table already recovers the exponent parity needed to
distinguish a square from its fixed nonsquare multiple. Consuming that flag
removes one square and equality check. Its isolated correction step improves
about 3%; the unchanged exponentiation dominates the approximately 3 µs full
operation, whose corpus comparison did not establish a significant change.

Moving an addition chain's final owned local removes one clone and drop. The
owned-vector benchmark includes chain operations and output destruction, with
input allocation outside timing. Scalar one particularly benefits because it
now returns the input directly: 64 words improved from 39.6 to 13.4 ns and
1,024 words from 142.8 to 12.0 ns. Udon's `Copy` field wrapper could already
have its clone removed by the compiler.

### Explicit MSM cache preparation

`PreparedScalars::cache_parallel` accepts the caller's executor and task budget.
The existing `cache` method uses the same implementation with a serial budget.
The parallel writer partitions at packed-chunk boundaries; fewer than 1,024
records remain serial. No global pool or runtime allocation is introduced.

For 16,384 Pallas records, Booth width-eight writing improved from 484.0 to
125.8 µs; joint writing from 6,890.9 to 1,778.5 µs. Vesta improved from 483.9
to 125.4 µs and 6,898.6 to 1,789.4 µs. The 512-record controls stayed serial
and were neutral. These timings measure internal cache writers used by the
public methods; they exclude decomposition and MSM execution. Ordinary MSM
task preparation already supported parallel execution.

Tests exercise both curves around the dispatch boundary, one/four-worker
executors, exact digit equivalence, unused tails, insufficient storage before
writes, and consumption through the public cached-input API.

## Validation

The following checks passed:

- Release tests for the full workspace with all features, and Udon with default
  features; debug Udon library tests in both feature configurations.
- Formatting, workspace Clippy with all targets and features, default-feature
  Udon Clippy, and documentation with warnings denied.
- Ignored public API, curve-constant, and field/curve/FFT embedding consumer tests.
- Portability builds for `thumbv7em-none-eabi` and `s390x-unknown-linux-gnu`,
  including both square-root table configurations and expected embedding
  rejections on big-endian targets.
- Benchmark smoke tests for field, curve, FFT, FFT strategies, MSM, and execution;
  field and curve smoke tests also ran with `sqrt-table-large`.

## Reproduction

Run timing commands separately, with no concurrent builds or tests. The
same-binary experiments retain reference implementations:

```console
cargo test --release --locked -p zakura-udon --lib compare_arithmetic_specializations -- --ignored --nocapture
cargo test --release --locked -p zakura-udon --lib compare_prefix_initialization -- --ignored --nocapture
cargo test --release --locked -p zakura-udon --features sqrt-table-large --lib compare_sqrt_success_flag -- --ignored --nocapture
cargo test --release --locked -p zakura-udon --lib compare_digit_cache_preparation -- --ignored --nocapture
cargo test --release --locked -p zakura-bento --test addition_chain owned_result_timing -- --ignored --nocapture
```

For before/after Criterion comparisons, run the same filter and arguments on
both revisions, saving the first with `--save-baseline before` and comparing
the second with `--baseline before`. Relevant filters are:

| Suite | Filter |
| --- | --- |
| `field` | `(Fp|Fq)/inner_product/sum_of_product_pairs` |
| `field` with `sqrt-table-large` | `(Fp|Fq)/corpus/sqrt_` |
| `curve` | `(Pallas|Vesta)/operations/double` |
| `fft` | `(Fp|Fq)/fft_setup/(2048|16384)/zeta/` |
| `fft_strategies` | `(Fp|Fq)/strategies/(2048|16384)/zeta/tasks_1` |
| `fft_strategies` | `(Fp|Fq)/strategy_preparation/expansion` |
| `fft_strategies` | `(Fp|Fq)/expansion_prefixes/tasks_1/computed/Residues/prefix_(10|128|256)/product_false` |

For example:

```console
cargo bench --locked -p zakura-udon --bench field -- '(Fp|Fq)/inner_product/sum_of_product_pairs' --sample-size 25 --warm-up-time 0.2 --measurement-time 0.5 --nresamples 1000 --noplot --save-baseline before
```

Use the [testing guide](TESTING.md) for arithmetic, compiler-consumer,
portability, and benchmark smoke checks. Correctness does not depend on timing
assertions or architecture-specific performance thresholds.
