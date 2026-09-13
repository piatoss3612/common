# Historical MSM arithmetic measurements: September 12, 2026

This historical report describes the API and implementation at `70c00ed`.
Its preparation layout and some benchmark timing boundaries have since changed.
See the [current MSM report](MSM_REVIEW_PERFORMANCE.md) for the September 13
remediation; do not compare these numbers as if they measured that rewrite.

The September 12, 2026 changes address inversion endpoint work, projective
formulas, reusable MSM scalar preparation, and missing benchmark distributions.
The measured gains are strongest in small denominator batches, projective
arithmetic, small dense 128-bit MSMs, and workloads that reuse scalar vectors.
Large random MSMs benefit much less.

## Measurement method

Measurements used Rust 1.91.0, LLVM 21.1.2, `aarch64-apple-darwin`, and default
features. The original implementation is revision
`eb1ce31d1ce9d02245f95efa21c6ac61282c9ec0`. Criterion binaries were retained for
each implementation so comparisons did not require a build between timing
runs. Builds and tests did not run concurrently with measurements.

The [curve harness](../crates/udon/benches/curve.rs) used 30 samples, 0.5
seconds of warmup, and two seconds of measurement. The [MSM
corpus](../crates/udon/benches/msm.rs) used 20 samples, 0.3 seconds of warmup,
and one second of measurement for the final comparisons. Inputs, allocation, and
fixture checks are outside timing. MSM execution includes scratch checks,
initialization, scheduling, and recoding unless explicitly labeled reused. All
MSM results here use affine bases and `SerialExecutor`, without a pass cap.

Tables report means with 95% bootstrap confidence intervals. These intervals
describe sampling within a run; they do not capture changes in machine state
between binaries. Controls with unchanged algorithms moved by roughly 1–5%
across the dispatch runs, so small effects are not strong evidence. These are
local microbenchmarks, without x86-64 or application measurements.

## Inversion endpoints

The MSM denominator helper keeps its two independent multiplication lanes.
Seeding each lane directly and omitting its final unused update removes six
field multiplications per nonempty call: `3(n - 1)` instead of `3n + 3`, where
`n` is the denominator count and both counts exclude the inversion itself.
A singleton calls field inversion directly. This change concerns the nonzero
denominator helper; the public identity-preserving normalization helper has a
separate schedule.

The [retained control](../crates/udon/src/curve/tests/batch_inversion.rs)
compares both schedules in the same process on the same nonzero inputs,
alternating each vector with its inverse. It uses 30 samples, 0.3 seconds of
warmup, and one second of measurement. Times below are nanoseconds; the complete
experiment also covers 2, 3, 32, and 128 denominators.

| Field | Denominators | Former schedule | Endpoint schedule |
| --- | ---: | ---: | ---: |
| Fp | 1 | 582.91 [579.25, 586.28] | 525.10 [521.95, 527.82] |
| Fp | 8 | 770.08 [767.17, 772.87] | 722.65 [718.57, 726.21] |
| Fp | 1,024 | 27,215.85 [27,069.20, 27,343.81] | 27,380.25 [27,249.67, 27,477.35] |
| Fq | 1 | 586.46 [584.60, 588.15] | 523.02 [518.12, 526.88] |
| Fq | 8 | 775.30 [772.16, 777.97] | 727.45 [724.32, 730.26] |
| Fq | 1,024 | 27,283.05 [27,175.43, 27,384.63] | 27,406.14 [27,309.84, 27,493.21] |

The 1–8 element cases improve by about 6–11%; 32 elements improve by about 3%.
At 128–1,024 elements the difference is about 1% or less. The fixed work saving
does not establish a material speedup for every large bucket batch.

## Projective formulas

General and mixed addition use unscaled differences and the existing fused
`mul_sub_product` kernel. Doubling uses half-scaled Jacobian coordinates and
halves the stored Montgomery residue with a conditional modulus addition and
shift. Equal, inverse, and identity branches remain complete.

Addition was measured after changing addition and inversion, while retaining the
former doubling formula. Doubling candidates share the new addition and
inversion implementations. Times are nanoseconds.

| Curve | Operation | Original | Selected formula |
| --- | --- | ---: | ---: |
| Pallas | `add` | 153.23 [152.94, 153.51] | 140.42 [139.43, 141.48] |
| Pallas | `add_mixed` | 107.02 [106.77, 107.25] | 96.70 [96.50, 96.92] |
| Pallas | `double` | 73.98 [73.80, 74.16] | 66.63 [66.52, 66.73] |
| Vesta | `add` | 154.81 [154.42, 155.18] | 141.69 [141.29, 142.12] |
| Vesta | `add_mixed` | 107.87 [107.62, 108.12] | 97.47 [97.30, 97.67] |
| Vesta | `double` | 75.07 [74.84, 75.31] | 67.28 [67.16, 67.41] |

Unscaled addition improves these kernels by about 8–10%. Against the unchanged
doubling control in the same implementation stage, half-scaled doubling improves
by 8–9%. An alternative with two field multiplications and five squarings was
also measured. For input Jacobian coordinates `(X, Y, Z)`, it computes
`D = 2 * ((X + Y²)² - X² - Y⁴)` while retaining the former doubling formula's
other terms. It took 72.01 ns [71.64, 72.45] on Pallas and 72.88 ns
[72.68, 73.08] on Vesta, slower than the retained half-scaled formula.

Full-width affine scalar multiplication, including its table preparation,
changed from 16.10 to 15.21 microseconds on Pallas and 16.40 to 15.58 on Vesta.
The addition and inversion changes contribute to those combined results; they
should not be attributed entirely to doubling.

Independent BigUint affine references cover full-width points at unrelated
nonunit Jacobian scales, including equal and inverse inputs. Field tests check
halving against integer arithmetic, including stored residues near zero and the
modulus. Canonical coordinate encodings remain unchanged.

## Reusing scalar vectors

The [curve guide](CURVES.md#multiscalar-multiplication) explains how to retain
scalar preparation with `msm::PreparedScalars`. Reused execution excludes
recoding and scalar-dependent dispatch and needs no digit scratch. Ordinary
execution performs preparation each time, then shares the dispatch result across
arithmetic partitions.

The following measurements use the same implementation for ordinary execution,
preparation alone, and execution with retained scalars. Times are microseconds.
Preparation is a one-time cost; the reused column excludes it.

| Curve | Corpus | Terms | Ordinary | Prepare once | Reused |
| --- | --- | ---: | ---: | ---: | ---: |
| Pallas | full | 32 | 171.22 [170.62, 171.86] | 13.13 [13.11, 13.16] | 159.42 [158.95, 159.90] |
| Pallas | full | 1,024 | 3,446.81 [3,424.24, 3,469.10] | 44.77 [44.50, 45.00] | 3,389.09 [3,374.86, 3,402.19] |
| Pallas | cancellation | 32 | 1.34 [1.34, 1.35] | 0.39 [0.38, 0.39] | 0.96 [0.95, 0.96] |
| Pallas | cancellation | 1,024 | 78.42 [78.18, 78.66] | 39.74 [39.59, 39.90] | 38.39 [38.32, 38.46] |
| Vesta | full | 32 | 177.04 [176.30, 177.77] | 13.30 [13.25, 13.36] | 164.10 [163.55, 164.66] |
| Vesta | full | 1,024 | 3,551.95 [3,531.94, 3,571.84] | 45.75 [45.53, 45.96] | 3,478.62 [3,463.19, 3,492.96] |
| Vesta | cancellation | 32 | 1.38 [1.38, 1.39] | 0.40 [0.39, 0.40] | 0.98 [0.98, 0.98] |
| Vesta | cancellation | 1,024 | 78.51 [78.39, 78.64] | 40.13 [39.94, 40.31] | 38.41 [38.36, 38.45] |

Full-width 32-term execution improves by about 7% when scalars are retained. At
1,024 random terms the observed gain is only about 2%, near the scale of
between-run variation. Exact cancellation makes scalar preparation a larger
fraction of runtime: retaining it saves about 29% at 32 terms and 51% at 1,024.
These exceptional inputs are useful cost controls, not representative random
MSMs. These measurements do not establish a benefit from separate preparation
for a scalar vector used just once.

## Broader corpus and dispatch

The [testing guide](TESTING.md#msm-and-compact-table-batch-benchmarks) defines
the scalar and base distributions and the `warm` and `cold` timing boundaries.
The `cold` cases apply cache pressure without guaranteeing hardware-cold inputs;
their times can be similar to warm times.

The former policy sent every scalar vector fitting 128 bits to bit interleaving
or four-bit buckets. For dense random 128-bit vectors, joint Eisenstein
preparation amortizes at eight terms in these measurements. The retained change
uses it for dense inputs at 8–32 terms. The guards in
[`short_bits`](../crates/udon/src/curve/msm/recode.rs) preserve existing paths
for sparse scalars, moderate bit lengths, singletons, and larger sums.

These are policy ablations with the new arithmetic and preparation API on both
sides. Times are microseconds for ordinary execution on random 128-bit scalars.

| Curve | Terms | Former short policy | Joint candidate |
| --- | ---: | ---: | ---: |
| Pallas | 8 | 56.31 [56.15, 56.47] | 51.18 [50.99, 51.36] |
| Pallas | 16 | 109.78 [109.23, 110.19] | 87.97 [87.69, 88.23] |
| Pallas | 31 | 207.37 [206.04, 208.34] | 162.11 [161.66, 162.54] |
| Pallas | 32 | 190.69 [190.18, 191.23] | 167.87 [167.47, 168.26] |
| Pallas | 47 | 238.26 [237.46, 239.00] | 242.12 [241.49, 242.75] |
| Vesta | 8 | 58.07 [57.90, 58.24] | 52.04 [51.89, 52.19] |
| Vesta | 16 | 112.79 [112.44, 113.15] | 88.88 [88.53, 89.18] |
| Vesta | 31 | 212.23 [211.42, 213.01] | 164.13 [163.47, 164.80] |
| Vesta | 32 | 194.53 [193.69, 195.37] | 168.62 [167.64, 169.31] |
| Vesta | 47 | 241.89 [241.18, 242.60] | 242.89 [242.10, 243.62] |

The retained 8–32 term cases improve by 9–23% in this repeat; an earlier run
measured 16–26% at 31–32 terms. At 47 terms, an initial 3% gain changed to a
0–2% regression in the repeat, so that candidate was rejected. A broader
candidate that routed every scalar wider than 64 bits to joint arithmetic
regressed Pallas 64- and 127-term random 128-bit sums by about 7% and 28%. Those
sizes keep their four-bit buckets.

Random 96-bit, sparse 128-bit, and singleton controls retain the same arithmetic
paths. In the final dispatch experiment they moved by about 1–5%, alongside the
shift in random 128-bit timings. These controls support retaining only the
larger, repeated gains; they do not prove that all unchanged cases have
identical latency. The new corpus also exposes the extra cost of sparse 128-bit
inputs at the existing 32-term bucket transition. Optimizing that separate
policy remains outside this change.

## Complete MSM comparison

Times are microseconds for ordinary execution, including recoding. The original
binary was remeasured after the final implementation to check the earlier
baseline against changes in machine state.

| Curve | Corpus | Terms | Cache | Original | Follow-up |
| --- | --- | ---: | --- | ---: | ---: |
| Pallas | full | 32 | warm | 184.84 [184.12, 185.70] | 171.22 [170.62, 171.86] |
| Pallas | full | 128 | warm | 730.82 [728.41, 733.44] | 691.84 [688.17, 696.54] |
| Pallas | full | 1,024 | warm | 3,456.29 [3,441.59, 3,471.56] | 3,446.81 [3,424.24, 3,469.10] |
| Pallas | sparse_high | 1,024 | warm | 3,174.41 [3,154.71, 3,193.41] | 3,122.88 [3,106.40, 3,139.82] |
| Pallas | equal | 1,024 | warm | 2,784.17 [2,765.12, 2,803.73] | 2,769.03 [2,749.02, 2,790.11] |
| Pallas | inverse | 1,024 | warm | 2,811.22 [2,789.90, 2,832.00] | 2,736.47 [2,718.39, 2,756.12] |
| Pallas | inverse | 1,024 | cold | 2,781.58 [2,765.31, 2,795.61] | 2,769.14 [2,752.64, 2,784.49] |
| Pallas | cancellation | 1,024 | warm | 77.59 [77.28, 77.88] | 78.42 [78.18, 78.66] |
| Vesta | full | 32 | warm | 188.46 [187.43, 189.33] | 177.04 [176.30, 177.77] |
| Vesta | full | 128 | warm | 741.52 [737.43, 746.35] | 712.06 [707.10, 717.80] |
| Vesta | full | 1,024 | warm | 3,523.11 [3,507.50, 3,539.86] | 3,551.95 [3,531.94, 3,571.84] |
| Vesta | sparse_high | 1,024 | warm | 3,211.10 [3,193.28, 3,228.80] | 3,213.37 [3,194.63, 3,232.32] |
| Vesta | equal | 1,024 | warm | 2,780.88 [2,762.91, 2,799.41] | 2,734.34 [2,714.63, 2,756.11] |
| Vesta | inverse | 1,024 | warm | 2,806.25 [2,791.92, 2,819.09] | 2,805.27 [2,775.84, 2,833.41] |
| Vesta | inverse | 1,024 | cold | 2,785.55 [2,762.99, 2,808.45] | 2,764.53 [2,744.45, 2,784.93] |
| Vesta | cancellation | 1,024 | warm | 77.55 [77.32, 77.76] | 78.51 [78.39, 78.64] |

The 32-term full-width sums improve by 6–7%, and 128-term sums by 4–5%. At 1,024
terms, differences are mostly small and mixed. Ordinary cancellation is about 1%
slower in this comparison; its large scalar-reuse gain comes from skipping
preparation on subsequent executions. Equal/inverse and cache-pressure controls
do not establish a broad speedup for large sums.

## Reproduction

The stable cases and timing boundaries are described in the
[testing guide](TESTING.md#msm-and-compact-table-batch-benchmarks). Run timing
experiments sequentially. Criterion stores means, intervals, and samples under
`target/criterion`. The reported comparison used local binaries and logs under
`target/curve-review`; these artifacts are not part of the repository.

```console
cargo test --release --locked -p zakura-udon --lib compare_batch_inversion_endpoints -- --ignored --nocapture
cargo bench --locked -p zakura-udon --bench curve -- '(Pallas|Vesta)/operations/(add|add_mixed|double|mul_affine)$' --sample-size 30 --warm-up-time 0.5 --measurement-time 2 --save-baseline candidate
cargo bench --locked -p zakura-udon --bench msm -- 'msm_corpus/(full|cancellation)/(warm|prepare_scalars|reused)/(32|1024)$' --sample-size 20 --warm-up-time 0.3 --measurement-time 1 --save-baseline candidate
cargo bench --locked -p zakura-udon --bench msm -- 'msm_corpus/(random128|sparse128)/warm/(8|16|31|32|47|48|64|127)$' --sample-size 20 --warm-up-time 0.3 --measurement-time 1 --save-baseline candidate
cargo bench --locked -p zakura-udon --bench msm -- 'msm_corpus/(sparse_high|equal|inverse)/(warm|cold)/1024$' --sample-size 20 --warm-up-time 0.3 --measurement-time 1 --save-baseline candidate
```

The original MSM comparison backported the corpus to the original revision,
omitting the unavailable prepared-scalar cases. The dispatch control uses the
new arithmetic and preparation API but always chooses the short path when all
scalars fit 128 bits. This isolates dispatch from the arithmetic improvements.
The scalar-reuse and inversion comparisons require no alternate build.

For correctness, documentation, and benchmark smoke checks, use the [testing
guide](TESTING.md) and [CI gates](../.github/workflows/ci.yml).
