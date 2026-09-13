# MSM review remediation: September 13, 2026

This report covers MSM storage, preparation, scheduling, and measurements,
including the original remediation and subsequent policy follow-ups.
The runtime remains allocation-free and retains caller-owned initialized typed
scratch and scoped execution. The unpublished
MSM API changes deliberately; see [migration](#migration).

The results support smaller affine pair staging, resource-aware planning, and
several workload-specific choices. Historical arithmetic measurements from
September 12 are preserved [separately](MSM_ARITHMETIC_PERFORMANCE.md);
they describe `70c00ed`, not this rewrite.

## Method and provenance

Measurements used an Apple M4 Max, `aarch64-apple-darwin`, Rust 1.91.0, and LLVM
21.1.2. The original remediation started at `70c00ed`; the candidate was a
working-tree change. A temporary development harness used an isolated Cargo
package. The harness and its raw result artifacts are no longer retained in
this repository; the tables below preserve the summarized measurements.

Both curves use deterministic pseudorandom 254-bit coefficients and bases with
separately generated known generator multiples. These coefficients exercise
full-width arithmetic but are not uniform samples of the scalar field.
Every alternative is checked outside timing against an independent scalar inner
product over those known multiples. Results cover zero and one as controls,
powers of two through 32,768, and neighbors of 8, 32, 64, 128, and 256. The
1,048,576-term case measures sizing only. No x86 machine was available.

Tables below report median microseconds. The full-width MSM measurements use
seven samples of at least 30 ms. Exploratory corpus and policy runs use three
of at least 10 ms; final lifecycle and grouped rechecks use five of at least
20 ms. Each sample repeats the operation until its duration is met. Samples
describe within-run spread; these are not confidence intervals. Fixture creation,
pool entry, allocation, and result checks are excluded unless the case explicitly
names that lifecycle. Builds and tests were kept separate from timing runs.
Unchanged controls moved by several percent, occasionally around 10%; small
rankings are not portable crossover guarantees.

The later reducer, parallel-width, and digit-extraction results below used
working-tree candidates based on `57b76e7` on the same host and toolchain.
Those tables report the median of three session medians. Reducer sessions used
seven samples of at least 30 ms, rotating candidate order between sessions and
including point and length resets. Width and extraction sessions used seven
samples of at least 10 ms, rotating policy order between samples and sessions.
They executed indexed MSMs over cached `PreparedAffinePoint` bases, cycling
eight scalar rows with preallocated scratch inside persistent Rayon pools of
1, 4, or 16 workers. Scalar preparation, recoding when applicable, and execution
were timed; base preparation, binding, allocation, pool entry, and independent
integer-inner-product/binary-ladder checks were outside timing. These follow-ups
are separate from the original measurements, not fresh timings of every current
default. Their coverage limits are stated alongside each policy.

## Full-width MSM measurements

Times are microseconds for the original remediation's complete MSM execution.

| Terms | Workers | Pallas | Vesta |
| ---: | ---: | ---: | ---: |
| 32 | 1 | 175 | 182 |
| 128 | 1 | 610 | 630 |
| 1,024 | 1 | 3,651 | 3,715 |
| 32,768 | 1 | 80,541 | 80,835 |
| 32 | 4 | 182 | 183 |
| 128 | 4 | 218 | 218 |
| 1,024 | 4 | 1,058 | 1,057 |
| 32,768 | 4 | 22,691 | 22,697 |

The singleton convenience call measured 15.59 / 16.64 us versus 15.35 / 16.38 us
for direct scalar multiplication. That small difference does not justify a
separate singleton implementation in this change.

## Selected arithmetic and memory policies

Width-eight Booth recoding now emits sixteen data windows, with no dead carry
window. Central curve-specific GLV bounds establish that the final carry is
zero. The writer overwrites all digit bytes and the reader visits contiguous
rows once per intersecting chunk. Wider supported windows use two-byte digits.

### Affine reduction

Production [`reduce`](../crates/udon/src/curve/msm/buckets.rs) uses fused inverse
recovery: it retains denominators and suffix products in two independent lanes,
shares one inversion across the level, and recovers each inverse while adding
and compacting its point pair. It first tries chord denominators for every pair.
A zero product detects doubling or cancellation and retries the level through
the complete two-pass `reduce_level::<C, false>` before changing any points or
lengths. That fallback uses denominator and inversion-prefix storage; its
compaction pass runs even when every pair cancels, preserving odd survivors and
updating lengths. In either path, pair `j` reads positions `2j` and `2j+1` before
writing at most position `j`. Pair workspace remains two field elements: 64
bytes on this target versus the original 200 bytes, a 68% reduction. This is a
pair-workspace claim, not a whole-MSM memory reduction.

The original native comparison measured the six-field reducer, the then-current
two-pass replacement, and a `mul_sub` expression candidate in one binary. At
128 points with occupancy 17, Pallas medians were 10.576, 10.621, and 10.710 us;
the repeated original was 10.579 us. At 8,192 points with occupancy 17, Vesta
medians were 536.7, 540.3, and 542.0 us, with 532.7 us for the repeated original.
Those results supported the two-field storage reduction without a timing claim.
Here `mul_sub` means combining multiplication and subtraction for the output
y-coordinate; it does not mean fusing inverse recovery. That expression did not
win consistently and remains a test-only control through `reduce_with`.

The later reducer comparison held the two-field workspace fixed and measured
prefix inversion, complete fused recovery, and fused recovery with the chord
attempt and complete fallback now used in production. At occupancy 17, the
Pallas/Vesta pairs were:

| Points | Prefix (us) | Complete fused (us) | Chord attempt with fallback (us) |
| ---: | ---: | ---: | ---: |
| 128 | 10.813 / 10.810 | 10.694 / 10.699 | 10.322 / 10.357 |
| 1,024 | 69.403 / 69.133 | 68.255 / 67.970 | 65.936 / 65.658 |
| 8,192 | 550.489 / 578.325 | 536.986 / 533.446 | 507.259 / 520.251 |

These ordinary-pair measurements support the production recovery schedule; they
do not measure the cost of repeatedly taking its exception fallback. The
six-field original and `mul_sub` variant remain native test controls; the
two-pass complete reducer remains production fallback code.

### Width selection

Full-width complete chunks below 128 terms retain the joint ladder. Larger
chunks use the following defaults in
[`Geometry::for_len`](../crates/udon/src/curve/msm/recode.rs):

| Effective chunk terms | Task budget 1–3 | Task budget at least 4 |
| --- | ---: | ---: |
| 128–191 | 6 | 6 |
| 192–511 | 7 | 7 |
| 512–4,095 | 8 | 8 |
| 4,096–32,767 | 10 | 11 |
| At least 32,768 | 11 | 11 |

Explicit widths and forced joint tables override this rule; scalar shape can
select short arithmetic. The selected width follows the effective chunk size,
including retained scalar input, and the task budget supplied to that geometry
decision. A budget is a concurrency allowance, not a guarantee of active workers.
These are replaceable internal defaults. The original serial width sweep
measured the following Pallas/Vesta pairs:

| Terms | Chosen width | Chosen time (us) | Width-eight control (us) |
| --- | ---: | ---: | ---: |
| 128 | 6 | 608 / 612 | about 740 / 740 |
| 256 | 7 | 1,038 / 1,063 | 1,071 / 1,095 |
| 1,024 | 8 | 3,616 / 3,744 | same candidate |
| 4,096 | 10 | 12,002 / 12,171 | 13,206 / 13,317 |
| 8,192 | 10 | 22,398 / 22,614 | 25,520 / 26,063 |
| 32,768 | 11 | 79,604 / 80,609 | 101,179 / 102,462 |

At 2,048 terms width ten was only about 2% faster; width eight retains fewer
bytes. In that serial sweep, width eleven's small gain at 8,192 and 16,384 did
not justify its bucket floor. The budget-dependent rule now uses eleven
throughout 4,096–32,767 terms when the task budget is at least four.

The later parallel-width sweep, using fused recovery and packed digit recoding,
measured the following Pallas/Vesta pairs at 4,096 terms:

| Workers | Width ten (us) | Width eleven (us) |
| ---: | ---: | ---: |
| 1 | 11,085 / 11,071 | 11,631 / 11,653 |
| 4 | 3,542 / 3,541 | 3,091 / 3,084 |
| 16 | 1,560 / 1,560 | 1,269 / 1,260 |

Width eleven uses twelve window tasks instead of thirteen, at a cost of 1,024
buckets per workspace instead of 512. The scheduling rationale is that twelve
tasks divide evenly across four workers, while fewer windows reduce input
passes and can reduce scheduling overhead at larger budgets. These benefits can
outweigh the larger bucket floor; the sweep does not isolate their individual
contributions. The follow-up sampled budgets 1, 4, and 16 and lengths 4,096 and
32,768, without a new sweep at 8,192 or 16,384. The intervening range and other
budgets are policy extrapolations, not individually measured crossovers. Widths
4 through 12 and all three accumulation backends remain forceable for
application-specific measurements.

### Accumulation and storage

Projective buckets win for tiny effective passes: unlike an affine tree, they
retain bucket sums across passes without repeatedly inverting sparse levels.
At 1,024 terms and pass 128, affine width eight measured about 4.2 ms serially,
against roughly 4.5 ms projective; the hybrid was about 4.3 ms. At passes 1, 2,
and 17 the projective alternative won. Hybrid remains an explicit candidate.
The boundary sweep also favored projective at pass 64 for repeated passes over
larger inputs. Auto therefore uses projective below pass 128. A single 64-term
pass can favor affine; the rule is a conservative default, not a universal
occupancy crossover.

`with_memory_limit` accounts for scalar records, digits, arithmetic buffers,
indices, window results, and reusable plan metadata. It reduces pass size,
concurrency, and chunk size and can select projective buckets. Its nonexhaustive
search reports an error before writes if it finds no fitting layout. See the
[memory contract](CURVES.md#sizing-and-reusing-scratch) for accounting and error
semantics. The figures below count typed buffer capacity, not process RSS.

Sizing 1,048,576 terms requires no input allocation or execution:

| Policy | Temporary bytes | MiB |
| --- | ---: | ---: |
| Serial, uncapped | 318,958,208 | 304.18 |
| Budget 12, uncapped | 2,535,640,192 | 2,418.17 |
| Serial, pass 512 | 117,729,920 | 112.28 |
| Budget 12, pass 512 | 120,900,736 | 115.30 |
| Either budget, 32 KiB ceiling | 19,008 | 0.018 |

Uncapped execution still grows with input length and concurrency. Retained
records and width-eleven digits also make a pass cap alone larger than the old
34-byte-per-term digit-only preparation. Use a byte ceiling or chunk size when
bounded temporary storage is required; a pass cap is insufficient. The tiny
ceiling trades additional chunk collapses for the much smaller workspace.

Completing independent chunks bounds every temporary buffer but repeats
collapses. `with_streaming_buckets` retains all windows' projective buckets and
recodes one chunk at a time, avoiding those repeated collapses. In the isolated
8,192-term Pallas experiment, width eight with chunks of 256 took 34.69 ms with
retained windows versus 49.27 ms with complete projective chunks. Its prototype
workspace was 221,184 versus 38,400 bytes. These are different memory budgets;
the public implementation additionally accounts for its reserved intermediate
prefix. Streaming is an explicit serial bucket policy, not the default under
every ceiling. Width and chunk choices let the caller control its fixed floor.

## Preparation, reuse, and scheduling

Raw execution converts and classifies each coefficient once per chunk, retaining
signed small-integer metadata and GLV components. Scalar shape applies at every
input length. Zero, one, Boolean, small negative, and sparse bounded-integer rows
use short arithmetic. For complete rows, dense bounded inputs of at least 512
terms and 32 useful bits use Booth when their average population exceeds eight
set bits per coefficient.
This retains the cheap paths for sparse rows while avoiding a serial four-bit
ladder on large dense rows. Joint ladders use their actual highest occupied
column and skip table preparation for zero scalars and identity bases.

The bounded-row sweep at 8,192 Pallas terms measured the previous short policy at
7.87 / 15.74 / 31.62 ms for signed 32 / 64 / 128-bit rows. The selected policy
recheck measured 3.83 / 6.56 / 17.63 ms serially and 3.59 / 4.68 / 6.89 ms with
three workers. These are separate runs; the large changes exceed the observed
control drift. At 128 terms the dense 128-bit short kernel still wins serially,
so the default retains it. Forced Booth can improve parallel execution there.

### Digit extraction and parallel recoding

For a complete nonstreaming chunk without compatible cached digits,
[`prefer_direct`](../crates/udon/src/curve/msm/recode.rs) extracts Booth digits
inside each window when the chunk has at least 512 and fewer than 4,096 terms,
its geometry is `Booth(8)`, and its task budget is at least 16. A compatible
retained digit cache takes precedence. Direct extraction avoids the separate
recoding joins and digit writes, but rereads the larger scalar records in the
window kernels. Sizing still reserves digit space, so this policy makes no
scratch-reduction claim. Streaming continues to use packed digits.

The later width-eight extraction comparison measured complete execution in
separate packed-recoding and direct-extraction builds, both using fused inverse
recovery. Packed recoding used the parallel writer where eligible:

| Terms | Workers | Packed (Pallas / Vesta us) | Direct (Pallas / Vesta us) |
| ---: | ---: | ---: | ---: |
| 1,024 | 4 | 930 / 925 | 940 / 936 |
| 1,024 | 16 | 653 / 655 | 609 / 610 |
| 4,096 | 16 | 1,573 / 1,537 | 1,551 / 1,602 |

The 1,024-term result supports direct extraction with sixteen workers; four
workers favored packed recoding. At 4,096, the ranking differed between curves
and direct Pallas session medians ranged from 1,550 to 1,727 us, so the default
keeps packed digits there. The 512-term lower bound was not sampled, nor were
budgets between 4 and 16 or above 16. Those bounds are internal defaults inferred
from the sampled workloads, not measured crossover points.

When execution needs to write digits, `write_parallel` uses the caller's
executor for at least 1,024 records with a nonserial budget and nonzero digit
stride. It splits at 256-record packed-chunk boundaries, preserving the serial
layout and completing writes before window execution. Smaller inputs avoid
another round of executor joins; short geometry writes no digits. This writer
was enabled in the packed follow-up measurements, but they did not isolate
serial versus parallel recoding or sweep the 1,024-record boundary. That cutoff
is an unmeasured scheduling default intended to amortize joins, not an
established speedup threshold. Compatible cached digits and direct extraction
bypass this writer during complete-chunk execution.

### Retained preparation and grouped scheduling

`Selection` retains checked index mapping independently of a scalar borrow.
Rebinding a field scalar row checks only its length; canonical integer binding
also validates values. `PreparedScalars` retains opaque records across bases,
widths, and chunk policies; optional cached digits are reused only for compatible
geometry in a single, nonstreaming chunk. Its retention is reported separately
from execution scratch. Embedded ordinary or cached compact tables bind directly
as MSM bases.
`EisensteinScalar::certify_batch` optionally retains the adjacent compact batch
ladder's check for exceptional affine intermediates, including the decision to
use complete projective arithmetic.

Grouped scheduling splits contiguous ranges by estimated work and assigns each
worker its own maximum scratch across sequential jobs. Serial groups reuse
maximum digit storage rather than summing all rows. Independent output folds
run within their job's worker. `ExecutionPlan` retains job and worker metadata
for repeat execution; no scalar-sized task objects or dynamic queue are needed.
The weight is intentionally coarse; it does not model every cache or occupancy
effect. The workload harness compared this policy with serial calls and
caller-scheduled independent MSMs in the same fixed-size pool.

The Pallas lifecycle recheck separates retained data from temporary buffers:

| Case | 32 terms (us) | 1,024 terms (us) | 32,768 terms (us) |
| --- | ---: | ---: | ---: |
| Raw execution | 176 | 3,636 | 79,285 |
| Scalar preparation alone | 0.96 | 30.4 | 978 |
| Recoding cache alone | 12.1 | 29.5 | 834 |
| Execution with prepared records | 176 | 3,512 | 79,031 |
| Execution with cached scalar digits | 163 | 3,592 | 77,065 |
| Temporary endomorphism cache plus execution | 176 | 3,473 | 76,905 |

At 1,024 terms, raw scratch is 321,536 bytes. Prepared execution uses 256,000
temporary bytes plus 65,536 retained bytes; cached digits reduce temporary bytes
to 223,232 and increase retention to 98,304. Cache timing does not establish a
consistent win over prepared records at this size. At 32,768 terms, scalar
records occupy 2 MiB and width-eleven digits another 1.5 MiB. These are optional
retention costs, not memory removed from the process.

Retained ordinary compact tables at 32 terms execute in 142 us, with 20.8 us to
prepare their 16 KiB table. Cached compact entries execute in 135 us with a larger
table. At 32,768 terms, forcing the compact joint ladder takes about 207 ms;
letting the same table-backed input use Booth takes about 79 ms. Table availability
therefore does not force joint execution. Optional batch certification costs
about 1.5 us and showed no consistent complete-ladder speedup; its benefit is
retaining the exceptional-intermediate check when that scalar is reused.

Temporarily caching each 256-term chunk took 4.84 ms versus 5.00 ms for ordinary
chunks at 1,024 terms, and 160.6 versus 165.2 ms at 32,768. It adds 24 KiB to the
90,400-byte ordinary chunk workspace. These modest local gains do not make it a
default. At 32,768 terms, explicit streaming width eight takes 137.9 ms with
222,720 temporary bytes; complete execution under a 32 KiB ceiling takes 222.1 ms
with 19,008 bytes, and under 8 KiB takes 382.6 ms with 6,656 bytes. They offer
different time/storage tradeoffs.

The corrected grouped scheduler preserves a dominant job's internal worker
budget when neighboring jobs are too small to justify a separate range. Odd
budgets divide work in proportion to the available workers. With three workers,
the 1,024-term heterogeneous family measured 1.73 ms with a reused plan, versus
3.91 ms for serial calls and 3.58 ms for caller-scheduled serial MSMs. The
8,192-term family measured 9.40, 22.91, and 22.41 ms respectively. A group of 256
two-term jobs measured 2.36 ms planned versus 6.51 ms serial and 2.55 ms
caller-scheduled. The 8,192-term shrinking IPA family measured 45.75 ms planned,
102.08 ms serial, and 84.31 ms caller-scheduled. The ordinary batch wrapper
stayed close to retained-plan execution; metadata reuse mostly removes setup.

## Caller-side candidates

Computing two separate GLV-half MSMs and applying the endomorphism at the end
did not win: at 8,192 Pallas terms it took 27.97 ms versus 23.06 ms serially,
and 10.85 versus 9.22 ms with three workers. It also required two workspaces.
The production path continues to deposit both halves into each window.

Partitioning the mostly-small fixture (one full-width exception per 64 terms)
did win. At 8,192 terms, including classification, gathering, and validation,
the caller-side partition took 1.46 ms versus 2.72 ms for uniform execution;
with three workers it took 0.95 versus 1.49 ms. That lifecycle used 2,923,552
temporary bytes versus 2,620,128 for the uniform serial case. The experiment
remains explicit because it requires additional gathering buffers and a scan
whose value depends on the distribution.

Sorted index coalescing also pays for strong repetition. At 1,024 terms selecting
eight bases from a 32,768-point pool, sorting and accumulating coefficients plus
execution took 66.5 us and 91,336 temporary bytes, versus 3,598 us and 321,536
bytes for ordinary indexed execution. At 8,192 terms it took 165 us versus
22,195 us. A skewed selection improved from 20.95 to 2.17 ms, while the
nonrepeating cold-pool selection stayed around 23 ms. Clearing a coefficient
array for the entire pool was much less effective at small sizes and required
over 11 MB. Both approaches were measured as caller-side experiments; indexed
binding continues to preserve the supplied terms without implicit sorting.

## Phase accounting

The isolated width-eight Pallas run at 1,024 terms measured canonicalization at
5.07 us and GLV rounding from canonical values at 10.86 us. Counting window zero
took 1.06 us, scatter 12.03 us, weighted collapse 31.44 us, and final recombination
10.95 us. All sixteen window kernels in that original run took 3,531 us. These
are separate experiments, not additive subdivisions of one measured execution.

That first window scanned 2,048 half-term digits, deposited 2,039 points, and
staged `[980, 501, 249, 129, 51, 1]` denominators across six levels. Individual
level timings were `[67.10, 36.49, 18.96, 11.06, 5.80, 2.67]` us, including
restoring points and lengths before each iteration. Vesta staged
`[990, 495, 250, 122, 56]` in the corresponding fixture. Counters are collected
outside timed loops.

## Finding disposition and validation

| Review concern | Disposition |
| --- | --- |
| Dead carry and duplicated geometry | Central GLV bounds and one geometry for sizing, cache, recoding, and execution; sixteen width-eight windows |
| Excess affine staging | Two-field fused inverse recovery in production, complete two-pass exception fallback; six-field original and `mul_sub` controls in tests |
| Repeated scalar classification | One conversion/preparation per chunk; retained shape; signed/unsigned/canonical bindings with checked bounds |
| Full-input scratch multiplied by concurrency | Explicit ceiling, bounded complete chunks, optional retained window buckets, checked byte counts |
| Mandatory width-eight affine path | Forced widths 4–12 and affine/projective/hybrid backends; budget-dependent widths with sampled measurements and extrapolation limits |
| Digit preparation overhead | Direct extraction for medium width-eight chunks with large budgets; parallel packed recoding; measured cases and unmeasured boundaries distinguished above |
| Coupled preparation lifetimes | Rebindable selection, reusable records and optional recoding, compact basis binding, reusable plans |
| Heterogeneous scheduling and serial tail | Weighted ranges, per-worker maxima, independent folds, retained offsets |
| Missing optimization invariants | Direct BigUint reducer/collapse oracle, production layouts and bounds, forced modes, planner partitions, cache and input contracts |
| Public implementation layout | Private option/count/scratch fields with constructors and accessors; typed initialized storage preserved |
| Broader FFT convention changes | Deferred as outside MSM scope |

Production layout tests cover both curves, signs and GLV halves, widths 4–12,
all final-chunk lengths, partial row visits, high actual decompositions and
conservative bounds. Reducer cases include empty buckets, gaps, doubling,
inverses, odd survivors, and cancellation-only levels. Planner tests exercise
empty jobs, budgets including 3/5/17/65, pass 1/2/17 and larger caps, metadata
coverage, disjoint scratch splitting, output ordering, memory floors, overflow,
retained preparation, and dirty buffers. Forced-backend differentials include
incompatible caches, repeated indices, identity bases, and signed extremes.
Reusable plans and preparation recover after an injected scoped executor panic.
The real no-allocator embedding and cross-target consumers exercise bounded
execution and retained table/selection reuse.

The original remediation passed the following validation with the pinned
toolchain; these counts do not describe a rerun of the later policy changes:

- The release workspace suite with all features, including doctests, and Udon's
  default-feature release suite.
- Udon's debug library suites with default and all features: 116 and 119 tests
  passed respectively, with four opt-in experiments ignored in each suite.
- All four opt-in Udon compiler/artifact consumer tests. The curve consumer
  exercises retained preparation, selections, and plans without an allocator.
- Bento's portability consumer, including 32-bit `thumbv7em-none-eabi` builds
  and `s390x-unknown-linux-gnu` builds and expected embedding rejections.
- Every MSM Criterion case in test mode, `ci/check-format`, workspace Clippy
  with all targets/features, default-feature Udon Clippy, and workspace rustdoc
  with warnings denied.

The native control and phase experiments also passed their result checks.
Cross-target consumers were compiled, not executed. Miri was not run for this
change; the typed storage implementation introduces no new unsafe code.

## Migration

- Replace `ExecutionOptions` literals with `ExecutionOptions::SERIAL` and
  builders. There is no default byte ceiling. The pass cap alone is not a total
  memory bound; use `with_memory_limit` or `with_chunk_size` for that requirement.
- Replace public requirement fields with `scalars()`, `digits()`, `affine()`,
  `projective()`, `field()`, and `indices()`; use `bytes::<C>()` for checked totals.
  Use `input.requirements(options)` for retained scalars and compact tables;
  the const length-only query sizes unprepared scalars with ordinary bases.
- Construct six-slice `Scratch::new` with a new initialized `ScalarStorage::ZERO`
  buffer before the digit and arithmetic slices. Use `reborrow()` for reuse.
- Prepare scalars into `ScalarStorage` entries. The returned handle releases the
  original coefficient borrow. Call `cache_len` and `cache` only when retaining
  a particular recoding is useful; reuse conditions are documented on
  [`PreparedScalars::cache`](../crates/udon/src/curve/msm/prepared.rs).
- Retain `Selection` for repeated indexed rows and `ExecutionPlan` for immutable
  input batches. Initialize plan metadata with `JobStorage::EMPTY` and
  `WorkerStorage::EMPTY`. Its `temporary_bytes()` includes metadata; ordinary
  single-input execution needs no metadata buffer.
- Existing compact POD tables and field/point formats are unchanged. Bind them
  through `EisensteinTableBatch` and select `Bases::Compact` or
  `Bases::CompactPrepared`. Scalar records and plans are ephemeral, not POD.

See the [curve guide](CURVES.md#multiscalar-multiplication) for contracts and an
executable static-buffer example, and the
[testing guide](TESTING.md#internal-msm-experiments) for the retained internal
controls and phase experiments. Measurements in this report span implementation
stages; they are not measurements of one identical working-tree snapshot.
