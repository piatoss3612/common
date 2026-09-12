# Field arithmetic performance

These measurements record the September 11, 2026 optimization pass. Arithmetic
identities, input bounds, and performance were checked separately. The changes
use the existing dependencies and preserve the target crates' `no_std` and
variable-time contracts.

The initial measurements predate the Bento facade redesign and removal of
runtime hex parsing. The [optional larger-table
comparison](#optional-larger-square-root-tables) was measured after those
changes; the earlier comparisons have not been repeated for them.

## Method

The before implementation is Udon revision
`ca7c809bbc0108a011313e3750161c1d131c4e3c`. Both implementations used the same
[benchmark source](../crates/udon/benches/field.rs) and the workspace's existing
Criterion dependency and lockfile changes. Measurements ran sequentially on
`aarch64-apple-darwin`, using Rust 1.91.0 and LLVM 21.1.2, without concurrent
builds or test runs.

The final comparison uses 30 samples, 0.1 seconds of warmup, 0.5 seconds of
measurement, and Criterion's 95% confidence intervals. The corpus contains 128
deterministic values and reports elapsed time for the entire batch. Operands
are prepared outside timing. Independent multiplication and squaring return
all 128 outputs; dependent cases carry each result into the next operation.
These are microbenchmarks of particular inputs, not application speedups or
constant-time measurements. No x86-64 runtime measurements were available.

The acceptance threshold was a practical 2% change, considered alongside
confidence intervals and repeated runs. Equivalent candidates favored simpler
code. The saved local results are under `target/criterion`, with final snapshots
named `pasta-final-before` and `pasta-final`. To repeat a comparison, use the
same harness with each implementation and distinct snapshot names:

```console
cargo bench --locked -p zakura-udon --bench field -- 'corpus/|encoding/from_wide_bytes_reduced|inner_product/sum_of_products' --sample-size 30 --warm-up-time 0.1 --measurement-time 0.5 --save-baseline candidate
```

## Runtime results

All times are in nanoseconds. Brackets give 95% confidence intervals for the mean.

The runtime hex measurements are historical: field hex construction now uses
only the compile-time `fp_hex!` and `fq_hex!` macros, and the runtime benchmark
has been removed.

| Operation | Field | Before | After | Time change |
| --- | --- | ---: | ---: | ---: |
| 64-byte reduction | Fp | 28.71 [28.60, 28.81] | 15.14 [15.08, 15.20] | -47.3% |
| 64-byte reduction | Fq | 28.78 [28.69, 28.88] | 14.87 [14.83, 14.92] | -48.3% |
| 1,024-byte reduction | Fp | 2,211.30 [2,203.85, 2,219.16] | 604.52 [601.03, 609.81] | -72.7% |
| 1,024-byte reduction | Fq | 2,219.36 [2,209.79, 2,229.22] | 604.49 [602.22, 607.34] | -72.8% |
| Runtime hex conversion | Fp | 2,714.61 [2,705.20, 2,723.40] | 125.12 [124.45, 125.78] | -95.4% |
| Runtime hex conversion | Fq | 2,722.35 [2,708.61, 2,739.01] | 124.20 [123.44, 124.95] | -95.4% |
| Root of order 2^16 | Fp | 206.07 [205.43, 206.78] | 1.66 [1.65, 1.67] | -99.2% |
| Root of order 2^16 | Fq | 207.07 [206.13, 208.34] | 1.65 [1.65, 1.66] | -99.2% |
| 3-term array product | Fp | 29.80 [29.68, 29.91] | 22.73 [22.66, 22.81] | -23.7% |
| 3-term array product | Fq | 29.54 [29.42, 29.66] | 22.91 [22.85, 22.97] | -22.4% |
| 32-term array product | Fp | 190.00 [189.38, 190.66] | 142.60 [142.02, 143.20] | -24.9% |
| 32-term array product | Fq | 190.41 [189.69, 191.19] | 143.62 [142.83, 144.43] | -24.6% |
| 1,024-term array product | Fp | 5,582.83 [5,562.55, 5,603.10] | 4,043.16 [4,009.64, 4,076.84] | -27.6% |
| 1,024-term array product | Fq | 5,589.00 [5,564.39, 5,619.24] | 4,083.88 [4,016.98, 4,190.90] | -26.9% |
| a*b - c*d | Fp | 16.57 [16.52, 16.64] | 15.75 [15.69, 15.81] | -4.9% |
| a*b - c*d | Fq | 16.58 [16.51, 16.65] | 15.95 [15.90, 16.00] | -3.8% |
| a*b - 2*c*d | Fp | 17.27 [17.22, 17.34] | 16.41 [16.36, 16.45] | -5.0% |
| a*b - 2*c*d | Fq | 17.24 [17.16, 17.32] | 16.58 [16.51, 16.65] | -3.8% |

The following table times batches of 128 operations.

All times are in microseconds. Brackets give 95% confidence intervals for the mean.

| Operation | Field | Before | After | Time change |
| --- | --- | ---: | ---: | ---: |
| Dependent multiplication | Fp | 1.69 [1.69, 1.70] | 1.70 [1.69, 1.70] | +0.4% |
| Dependent multiplication | Fq | 1.69 [1.69, 1.70] | 1.70 [1.69, 1.71] | +0.3% |
| Independent multiplication | Fp | 1.30 [1.29, 1.30] | 1.30 [1.29, 1.31] | +0.3% |
| Independent multiplication | Fq | 1.29 [1.29, 1.30] | 1.30 [1.29, 1.30] | +0.5% |
| Dependent squaring | Fp | 1.70 [1.70, 1.71] | 1.65 [1.65, 1.66] | -2.7% |
| Dependent squaring | Fq | 1.70 [1.69, 1.70] | 1.66 [1.65, 1.66] | -2.3% |
| Independent squaring | Fp | 1.16 [1.15, 1.16] | 1.12 [1.11, 1.12] | -3.5% |
| Independent squaring | Fq | 1.16 [1.15, 1.16] | 1.12 [1.11, 1.12] | -3.3% |
| Square roots of squares | Fp | 1,063.56 [1,061.66, 1,065.57] | 796.05 [793.29, 798.64] | -25.2% |
| Square roots of squares | Fq | 1,052.81 [1,050.82, 1,054.90] | 797.45 [792.13, 804.60] | -24.3% |
| Square roots of nonsquares | Fp | 625.13 [623.20, 627.19] | 406.74 [405.62, 407.95] | -34.9% |
| Square roots of nonsquares | Fq | 623.23 [621.40, 624.85] | 406.93 [405.24, 409.35] | -34.7% |

Multiplication remained within noise. Squaring improved by approximately 2–4%
in these batches; the larger square-root gains also include batched squaring
and root-ladder lookups. Checked 1,024-term slice products improved by 29% in
both fields, matching the array dispatch improvement.

## Implementation choices

The measured implementation shared one checked `MontgomeryContext` across each
field's parameters. The current [parameter derivation](../crates/udon/src/field/parameters.rs)
uses Bento's compile-time macros, sharing setup within the root tables.
Wide decoding combines two raw products with one REDC, with a compile-time
check of `R2 + R3 < p`. Arbitrary-width decoding uses 32-byte Horner digits;
its separate numerator bound is also checked. Runtime hex conversion reused
`R2` in the measured implementation. Product differences add `pR` only after
a negative subtraction, so one final correction suffices.

Forward and inverse root ladders replace repeated root construction and supply
Tonelli–Shanks corrections directly. Their combined raw payload is 4,224 bytes
for both fields, stored directly as field elements. Larger tables are available
through `sqrt-table-large`; see the [optional table
measurements](#optional-larger-square-root-tables). The generic small-field
square-root algorithm remains a test oracle.

REDC cancels the low half before adding the high half once. The private batched
squaring hook retains raw intermediates until the end of a run. Compile-time
checks verify the integer bound recurrence for every run length through 256,
including the final fused product; longer runs normalize between batches.
Public field values remain reduced. This uses a finite bound check because
both Pasta primes are slightly above `2^254`; treating arbitrary residues below
`2p` as closed under lazy squaring would be incorrect.

The chain planner prefers smaller prepared tables on arithmetic-cost ties.
Planned and supplied chains share one operation graph and compact, unrolled,
and batched emitters. Supplied schedules undergo exact integer replay, and
the narrow `tonelli_shanks(modulus, two_adicity)` input derives the exponent
without duplicating a literal. The new trait hooks have defaults for existing
implementations. Udon uses planned chains with batched emission.

The following exploratory runs timed 128 square roots with the root ladders
already enabled. Entries are means in microseconds; these runs used the same
30-sample settings as the final comparison.

| Chain / emission | Fp squares | Fq squares | Fp nonsquares | Fq nonsquares |
| --- | ---: | ---: | ---: | ---: |
| Supplied / compact | 963.3 | 952.4 | 571.7 | 572.4 |
| Supplied / unrolled | 979.7 | 972.4 | 597.1 | 594.6 |
| Supplied / batched | 789.9 | 783.6 | 400.8 | 403.6 |
| Planned / batched | 791.7 | 785.9 | 405.0 | 403.4 |

The supplied chains use 223 squarings plus 23 or 24 multiplications,
versus 222 plus 26 for the planner. Their measured benefit was below 2%, so
Udon retains generated schedules. Batched emission clearly beat both compact
and fully unrolled emission with canonical intermediates.

Inversion retains its original canonical-input initialization. The proposed
stored-input seed eliminated one conversion but changed the divstep trajectory.
Initial 128-value corpus measurements improved by 3.8% for Fp and 1.6% for Fq;
a longer repeat (50 samples, 0.5-second warmup, two-second measurement) measured
1.0% improvement for Fp and 2.9% regression for Fq, with overlapping confidence
intervals. This did not establish a repeatable win across both fields, so the
seed change was discarded.

A fuller signed-62 coefficient implementation with batched even steps and
shrinking active length reduced inversion of one to about 186 ns but took
approximately 658–757 ns on dense fixtures, versus 530–550 ns before the
changes. That implementation was also discarded.

Contiguous arrays and slices share dispatch. Zero through three terms use
specialized paths; the exact three-versus-four REDC cutoff is checked at compile
time. Fresh slice accumulators exploit the physical slice-length bound while
the public `ProductSum` continues folding overflow for unrestricted additions
and merges. On AArch64, blocks of 32 use independent Comba columns. This
outperformed four accumulator lanes at large lengths: 1,024-term arrays took
about 4.0 µs rather than 5.5–5.6 µs. Other architectures retain the four-lane
path at 64 terms; cross-compilation does not establish its speed.

## Coverage and remaining work

- **Evaluated and rejected:** direct stored-input inversion, the fuller
  signed-62 inverter, and using supplied chains in Udon. The measured reasons
  appear above. Support for verified supplied chains remains available in
  Bento. These results concern the tested implementations and host, not every
  possible implementation of those techniques.
- **Implemented after the initial pass:** the [optional larger square-root
  tables](#optional-larger-square-root-tables).
- **Unimplemented API suggestion:** a non-panicking
  `try_from_montgomery_limbs` returning `Option`. The existing const constructor
  still rejects unreduced limbs by panicking. For the separate workflow of
  embedding generated field values, see the [field storage
  guide](POD.md#storing-field-elements).
- **Further arithmetic candidates:** interleaved lazy squaring on x86-64,
  batched runtime exponentiation, and direct multiplication by inverse powers
  of two remain unimplemented and unbenchmarked. An AArch64 assembly backend
  is also unimplemented; Udon retains its existing prohibition on unsafe code.
  Low-half REDC, bounded lazy square runs, and AArch64 column accumulation use
  safe Rust as described above.
- **Further planner and validation work:** the planner now breaks arithmetic
  cost ties by prepared-table size; it does not search using peak liveness or
  caller-supplied squaring/multiplication weights. x86-64 runtime measurements,
  detailed assembly analysis, and quantitative debug stack measurements remain
  outstanding. Local word kernels stay private in Udon, and Bento's higher-level
  reference arithmetic remains independent.

## Build cost and size

With external dependencies already built, forcing all four local crates to
recompile in the release profile took a median 2.690 seconds before and 1.513
seconds after. Three measurements of each ranged from 2.684–2.801 seconds
before and 1.512–1.614 seconds after. Each used
`cargo build --release --locked -p zakura-udon`, after touching each local
crate's `src/lib.rs`. This measures local-crate recompilation, not a cold Cargo
registry or a clean build of every dependency.

A standalone executable exercising both fields was compiled with
`rustc --edition=2024 -C opt-level=3 -C panic=abort`, linking each release Udon
library. `size -m` reported:

| Mach-O section | Before (bytes) | After (bytes) |
| --- | ---: | ---: |
| `__TEXT,__text` | 309,452 | 260,828 |
| `__TEXT,__const` | 13,824 | 20,240 |
| `__DATA_CONST,__const` | 10,056 | 10,104 |

This probe includes Rust's standard-library code and linker choices. It shows
the linked tradeoff for these call sites, not an intrinsic size of either field
type. The source below uses the current API:

```rust
use std::hint::black_box;
use zakura_udon::field::{PallasBase, PallasScalar, PastaField, PrimeModulus, ProductSum};

fn exercise<M: PrimeModulus>(input: &[u8; 64]) -> [u8; 32] {
    let a = PastaField::<M>::from_wide_bytes_reduced(black_box(input));
    let b = a.square();
    let inverse = a.invert().unwrap_or(PastaField::ZERO);
    let root = b.sqrt().unwrap_or(PastaField::ZERO);
    let u = PastaField::<M>::root_of_unity(black_box(16)).unwrap();
    let v = PastaField::<M>::root_of_unity_inverse(black_box(16)).unwrap();
    let c = PastaField::sum_of_products(black_box(&[a; 64]), black_box(&[b; 64]));
    let d = PastaField::from_bytes_reduced(black_box(&[0xa7; 129]));
    let mut sum = ProductSum::new();
    sum.add_product(&inverse, &root);
    sum.add_term(&u);
    sum.finish().add(&c).add(&d).mul_sub_product(&v, &a, &b).to_bytes()
}

fn main() {
    let input = black_box([0xa7; 64]);
    black_box(exercise::<PallasBase>(&input));
    black_box(exercise::<PallasScalar>(&input));
}
```

## Optional larger square-root tables

The `sqrt-table-large` feature selects a table-assisted algorithm. The [crate
feature documentation](../crates/udon/src/lib.rs) describes configuration and the
[square-root implementation](../crates/udon/src/field/sqrt/large.rs) explains the
exponent recovery and subgroup lookup. Both algorithms remain variable-time;
the [public square-root contract](../crates/udon/src/field/sqrt.rs) leaves the
choice of root unspecified.

Each field's larger table contains 897 field elements and 1,024 hash bytes:
29,728 bytes, or 59,456 bytes (58.06 KiB) for both fields. These are additional to
the 4,224-byte small ladders, which remain available for root-of-unity lookups.
These payload sizes describe the table definitions; linked section sizes also
depend on which fields and operations an executable uses.

The [parameter derivation](../crates/udon/src/field/parameters.rs) builds field
entries in immutable statics at compile time. No table conversion, initialization,
or allocation is charged to the runtime measurements. The feature preserves the
[stored field representation](POD.md#storing-field-elements).

### Runtime measurements

This comparison records the optional-table implementation from the September 11,
2026 pass on `aarch64-apple-darwin`, Rust 1.91.0 and LLVM 21.1.2. Runs were
sequential with no concurrent builds or tests: 50 samples, 0.5 seconds of warmup,
and two seconds of measurement. Longer runs followed a noisier initial 30-sample
comparison.
Local Criterion snapshots are `sqrt-small-final` and `sqrt-large-final`.

These times cover batches of 128 operations, in microseconds. Brackets give
95% confidence intervals for the mean.

| Operation | Field | Default small | Large | Time change |
| --- | --- | ---: | ---: | ---: |
| Square roots of squares | Fp | 730.46 [728.38, 732.79] | 363.55 [362.81, 364.31] | -50.2% |
| Square roots of squares | Fq | 726.44 [725.08, 727.81] | 368.36 [366.96, 369.91] | -49.3% |
| Square roots of nonsquares | Fp | 372.75 [371.92, 373.59] | 365.70 [364.44, 367.15] | -1.9% |
| Square roots of nonsquares | Fq | 374.39 [373.38, 375.60] | 367.68 [366.95, 368.41] | -1.8% |

The larger algorithm roughly halves the time for this square corpus. Nonsquare
changes are below the 2% practical threshold. It does more fixed work even for
one: individual `sqrt(1)` calls rose from 2.518 to 2.826 microseconds for Fp
(+12.2%) and 2.517 to 2.859 microseconds for Fq (+13.6%). Zero keeps the shared
early return and measured about 1.8 nanoseconds in both configurations. These
are repeated calls into one field's tables; applications with competing cache
pressure, different inputs, or other architectures can have different results.

```console
cargo bench --locked -p zakura-udon --bench field -- 'corpus/sqrt_|/sqrt/(one|zero)' --sample-size 50 --warm-up-time 0.5 --measurement-time 2 --save-baseline sqrt-small-final
cargo bench --locked -p zakura-udon --bench field --features sqrt-table-large -- 'corpus/sqrt_|/sqrt/(one|zero)' --sample-size 50 --warm-up-time 0.5 --measurement-time 2 --save-baseline sqrt-large-final
```

### Build cost and linked storage

With dependencies already built, recompiling only Udon in release mode took a
median 1.206 seconds with the default tables and 2.105 seconds with the larger
tables. Three alternating trials ranged from 1.203–1.207 seconds and
2.103–2.122 seconds, respectively. Each used
`cargo rustc --release --locked -p zakura-udon --lib`, adding
`--features sqrt-table-large` for the larger configuration and a unique
`-- -C metadata=sqrt-build-cost-MODE-TRIAL` to force recompilation. This includes
Cargo overhead and excludes rebuilding dependencies; it is not a cold-build
measurement.

The standalone probe above was rebuilt and run against both configurations'
release libraries from this comparison, with the same `rustc` options.
`size -m` reported:

| Mach-O section | Default small (bytes) | Large (bytes) |
| --- | ---: | ---: |
| `__TEXT,__text` | 260,748 | 271,972 |
| `__TEXT,__const` | 18,128 | 77,536 |
| `__DATA_CONST,__const` | 10,104 | 10,040 |

`nm -nm` placed both fields' root ladders and larger tables in `__TEXT,__const`.
`otool -l` reported `__TEXT` permissions `0x5` (read and execute, no write).
The table symbols occupy 2,112 bytes per field for the ladders and 29,728 bytes
per field for the larger tables. The writable `__DATA` segment's section totals
were unchanged. This confirms static read-only residency in this linked probe;
the section differences also include code generation and linker decisions,
beyond the table payload itself.

## Correctness and portability

The release workspace suite, debug Udon unit tests, Criterion smoke run,
format checks, Clippy with warnings denied, and warning-free rustdoc build all
passed. The ignored portability test also passed, compiling `no_std` consumers
for `thumbv7em-none-eabi` and `s390x-unknown-linux-gnu` and checking the expected
big-endian storage rejection. Cross-target field code was built, not executed.

New checks include independent integer oracles for context arithmetic and
constants, wide-decoder boundary halves, exact lazy REDC intermediates at every
supported length, raw maximal product sums, and square-root table agreement on
exhaustive small fields. Existing forced-overflow accumulator tests remain.
Macro tests cover symbolic replay, invalid schedules, non-`Copy` values, hook
dispatch, value lifetimes, caller-name collisions, and caller control flow.
Debug tests exercise the generated field chains and normalization across long-run
boundaries; no debug stack-size claim is inferred from successful execution.

The optional-table comparison also passed release and debug tests, benchmark
smoke runs, and the same cross-target checks in both configurations, plus Miri
checks of field POD storage. The [testing guide](TESTING.md) describes the table
oracles, embedding consumers, and compiler checks that support these results.
