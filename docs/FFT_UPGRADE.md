# Upgrading Sensei's FFT pipelines

This guide maps the FFT workloads in the sibling `../bento` workspace to this
repository's Udon rewrite. Preserve Sensei's polynomial representations and
degree reductions while replacing its transform machinery. Much of Sensei's
existing optimization already has a direct counterpart here.

The inspection used Bento commit
`055d0801cfd320efeda0609ad337a4025a6f37fc` and Udon commit
`d0c00accbb33354444e8b9cf76e9468ae9a425e6`. Sibling source links assume those
workspaces remain beside each other. This is a migration design based on source
inspection; it does not report a completed port or measured Sensei speedups.
Use the [FFT guide](FFT.md) and linked API definitions for full contracts.

## Start with the application

Sensei uses FFTs to keep polynomials in two useful forms: coefficients for
commitments and openings, and coset evaluations for row-wise quotient work.
The quotient fold also produces several degree classes, which are interpolated
and added in coefficient form. FFT choices must preserve both consumers.

The current [dimensions](../../bento/crates/sensei-shared/src/dimensions.rs)
are `N = 2^11 = 2048` and an extended domain of `8N = 16384`. Production prover
polynomials use `Fp` (`PallasBase` in the sibling). Generic helpers and tests
also cover `Fq`; that does not make both fields equally important to the runtime
FFT workload. Let `g = Fp::zeta()` and let `w_m` denote the canonical root for a
domain of size `m`. Natural coset row `j` evaluates at `g * w_m^j`.

| Application operation | Existing implementation | Rewrite entry point |
| --- | --- | --- |
| Base evaluations to retained coefficients and an extended coset | `owned_lagrange_to_forms`: normalized inverse, then residue transforms | Prepared `Expansion` with `DisposableInput { scale: Normalized }` |
| Coefficients to full or reduced coset data | `base_coefficients_to_extended_coset_tiles`, `base_coefficients_to_class_2n_coset_tiles` | `Expansion::coefficients`, with extended size `8N` or `2N` |
| Base evaluations to coset data when coefficients are temporary | `base_lagrange_to_extended_coset_tiles` | `Expansion::evaluations`, or a prepared storage policy |
| Prepared public-instance contribution | `short_coefficients_times_factor_coset_tiles` | `Expansion::short_product` with a domain-bound factor |
| Quotient degree classes to coefficient pieces | `ClassedTileBuilder` and `interpolate_classed_quotient` | `Class::scatter_strided`, then `interpolate_classes` or `interpolate_classes_parallel` |
| Simple field FFTs in artifact generation and tests | Old `udon::fft::{transform, inverse_transform}` | `fft::reference`, or `Plan` where independence from the optimized runtime is unnecessary |
| Point-valued inverse FFT for Lagrange SRS | Generic old FFT over `VestaProjective` | Downstream `fft::reference::Butterfly` adapter; the optimized field `Plan` does not cover this |

The main sources are Sensei's
[polynomial machinery](../../bento/crates/sensei/src/native/polynomial/mod.rs),
[prover forms](../../bento/crates/sensei/src/native/prover/mod.rs),
[quotient folding](../../bento/crates/sensei/src/native/prover/quotient.rs), and
[target preparation](../../bento/crates/sensei/src/native/target/mod.rs).
Several simple dense FFT wrappers in the polynomial module are test-only.
The runtime already has specialized kernels of its own; changing the old
`udon::fft` import alone would leave most of this workload untouched.

Those kernels call the old field module's public
[`LooseFftSlice`](../../bento/crates/udon/src/field/loose_fft.rs) for butterfly
rounds. The rewrite keeps its loose arithmetic inside the implementation.
Replace the surrounding transform operations with FFT APIs instead of trying
to port the low-level field calls individually.

## Preserve these contracts before choosing a backend

An FFT buffer needs more meaning than its element type and length. For every
producer and consumer, record the field, ordered domain, storage order,
coefficient or evaluation support, normalization, and ownership after the call.
These distinctions determine which new entry point is appropriate.

### Coefficients and evaluations have different support

`forward_prefix` and coefficient expansion omit high-degree coefficients.
`inverse_prefix` omits a suffix of evaluation positions. A vector with ten
nonzero evaluations usually interpolates to a polynomial with `N` coefficients.
Its coefficient prefix is not ten elements long.

Sensei's prepared instance path exploits more structure: it represents the
instance polynomial as `q(X) * S(X)`, where `q` has ten coefficients and
`S(X) = (X^N - 1) / H(X)` is precomputed. Here `H` is the monic polynomial
vanishing on the ten supported base rows. Preserve that factorization and use
`short_product(q, factor_view, ...)`. Replacing it with a
generic sparse inverse and a dense expansion would lose the short-polynomial
forward path. The retained coefficients for commitments are still those of the
full instance polynomial, not just `q`.

### Residue tiles are evaluations, coefficient tiles are degree ranges

Sensei's full coset tiles already match `ExpansionOrder::Residues`. Residue `s`
contains natural extended rows `s + 8*k`, with `0 <= k < N`. Thus:

```text
natural extended row j -> storage (j % 8) * N + j / 8
```

Rotations by a base-domain row move `j` by eight and stay in the same residue
tile. This is useful locality for quotient folding. Use `EvaluationView` for
checked domain and layout metadata, or `ResidueView` for layout-only access.
Neither binding verifies which polynomial produced the values.

The `2N` product form evaluates only full-domain rows divisible by four.
It uses the same shift and the nested root `w_2N = w_8N^4`. In its own domain,
row `j / 4` maps to:

```text
full row j, with j % 4 == 0 -> ((j / 4) % 2) * N + (j / 4) / 2
```

`EvaluationView::get_extended_row(j, full_coset)` checks this relationship.
Preserve the reduced product form selected by `owned_lagrange_to_product_forms`;
expanding every permutation product to `8N` would discard an existing saving.

Global bit-reversed order is a separate layout. For full-domain row `s + 8*k`:

```text
reverse_14(s + 8*k) = N * reverse_3(s) + reverse_11(k)
```

Both the residue number and its inner row are reversed. Setting
`ExpansionOrder::BitReversed` and continuing to use Sensei's old tile indexer
would read the wrong rows. That order is attractive for an immediate pointwise
product followed by a full inverse, but requires a separate evaluation of the
quotient fold's rotation access costs. Residue order preserves short-prefix
pruning. Bit-reversed expansion also prunes prefixes of at most `N / 16`
coefficients, using DIT followed by a local permutation; longer prefixes use
the full DIF schedule. The factor multiply remains fused into the terminal
stores in either case. See the [measured changes](FFT_PERFORMANCE.md#measured-upgrade-refinements)
for execution costs, including the permutation.

After interpolation, `CoefficientTiles` borrows consecutive degree ranges
`[0, N)`, `[N, 2N)`, and so on. These are the quotient pieces used by commitments;
they no longer denote coset residues.

### Inverse scaling belongs to the result contract

`Plan::inverse` returns normalized coefficients and removes the coset shift.
Do not follow it with the old separate size normalization or inverse-zeta
distribution. `InverseScale::Unscaled` still removes the shift but returns
`m * c[i]` for a source domain of size `m`.

A retained `CoefficientView` carries that scale. New coefficient consumers can
fold it into their initialization if passed the view itself. Passing
`view.as_slice()` discards the metadata and declares ordinary coefficients.
Sensei's commitments, polynomial evaluation, and synthetic division currently
consume ordinary coefficients. Use `Normalized` for those retained buffers
unless all downstream consumers are deliberately adapted.

## Migrate buffer ownership explicitly

### The ordinary prover form

`owned_lagrange_to_forms` already consumes its base-evaluation allocation and
retains the coefficients. `DisposableInput { scale: Normalized }` expresses
exactly that lifetime. It also lets all residues read coefficients from the
retained allocation after the inverse, without a second coefficient copy.

The following small example checks that lifetime, output layout, and agreement
with an independent reference schedule. Production uses log sizes 11 and 14.

```rust
use zakura_udon::{
    fft::{
        Domain, Expansion, ExpansionOrder, ExpansionStorage, ExpansionStrategy, InverseScale, Plan,
        SerialExecutor, reference,
    },
    field::Fp,
};

fn main() {
    let base = Domain::new(3).unwrap();
    let extended = Domain::new(6).unwrap().coset(Fp::zeta()).unwrap();
    let base_plan = Plan::without_tables(base.subgroup());
    let expansion = Expansion::new(base_plan, extended, None).unwrap();
    let operation = expansion
        .configure(
            ExpansionOrder::Residues,
            ExpansionStorage::DisposableInput {
                scale: InverseScale::Normalized,
            },
            ExpansionStrategy::serial(),
        )
        .unwrap();

    let coefficients: [Fp; 8] = core::array::from_fn(|i| Fp::from_u64(i as u64 + 1));
    let mut owned_base = coefficients;
    reference::transform(&mut owned_base, &base.root());
    let mut coset_values = [Fp::ZERO; 64];
    assert_eq!(operation.requirements().scratch_fields, 0);
    assert_eq!(operation.requirements().coefficient_fields, 0);
    let retained = operation
        .execute_disposable(&mut owned_base, &mut coset_values, &SerialExecutor, &mut [])
        .unwrap();
    assert_eq!(retained.as_slice(), &coefficients);

    let mut expected = [Fp::ZERO; 64];
    let mut power = Fp::ONE;
    for (slot, coefficient) in expected.iter_mut().zip(coefficients) {
        *slot = coefficient.mul(&power);
        power = power.mul(&extended.shift());
    }
    reference::transform(&mut expected, &extended.domain().root());
    let view = operation.view(&coset_values).unwrap();
    for (row, value) in expected.iter().enumerate() {
        assert_eq!(view.get(row), Some(value));
    }
}
```

The rewrite borrows contiguous output storage. Sensei's `CosetTiles` instead
owns an array of separately allocated `Owned<[Fp; N]>`. Choose the ownership
transition at the application boundary:

- Use one contiguous allocation per expanded polynomial and borrow its residue
  slices. This suits prepared target columns, which currently expand into
  separate tiles and immediately flatten them into `CosetColumns`.
- Keep separate tile owners and use `Expansion::residue(s, InputOrder::Natural)`
  to fill each base-sized allocation. Existing owned worker jobs can each run
  one such descriptor with `SerialExecutor`. Keep the coefficient allocation
  in a shared owner when the jobs must outlive a local borrow.

The second choice preserves the current tile interface, but the application
still schedules residues and manages their owners. It does not use the prepared
whole-expansion storage policy in the example. In either design, avoid copying
an already retained coefficient buffer solely to satisfy a task closure.

For `short_product`, both the factor view and the output need contiguous
storage. Prepare the support factor in that form once, rather than flattening
it during every proof. The selected-residue API exposes coefficient expansion
without a factor parameter; keeping separate tile owners on that path also
keeps the separate factor multiplication unless the application composes a
different transform operation.

When base evaluations must survive, choose between `ReuseOutput` and
`CoefficientWorkspace`. `ReuseOutput` saves a coefficient allocation but uses
the first output residue as temporary coefficients, delaying that residue.
`CoefficientWorkspace` uses `N` additional fields and removes this dependency.
For callers that already surrender the input, `DisposableInput` removes the
dependency without that extra allocation.

### The quotient classes

Keep the four domain sizes `8N`, `4N`, `2N`, and `N`, all shifted by `g`.
The fold has already applied its vanishing-polynomial factors and challenge
weights. Interpolation should just sum the four coefficient vectors, padding
smaller vectors with zeros. It should not repeat the quotient division.

The old builder receives a window identified by residue `s` and inner offset
`k`. Preserve its natural-position scatter mapping:

| Class size | Windows contributing to it | Natural start | Stride |
| --- | --- | --- | --- |
| `8N` | All residues | `s + 8*k` | 8 |
| `4N` | Even `s` | `s/2 + 4*k` | 4 |
| `2N` | `s = 0, 4` | `s/4 + 2*k` | 2 |
| `N` | `s = 0` | `k` | 1 |

Allocate initialized contiguous buffers, construct a `Class` for each, and
call `scatter_strided(start, stride, values)`. To reproduce the old builder's
orders, use `InputOrder::BitReversed` for the three larger classes and
`InputOrder::Natural` for `N`. The constructor describes existing contents;
it does not permute a natural vector. Scatter accepts natural positions and
performs the mapping as it writes.

Retain an application-level completeness check. The old builder checks a total
filled count before interpolation, although that alone cannot detect overlapping
windows. `Class` checks ranges and phase, but neither counts writes nor detects
missing or duplicate rows. Repeated scatter overwrites; it does not accumulate.
Zero initialization makes unwritten entries zero, which can hide a lost window.

For execution, compare these two schedules:

- `interpolate_classes` reuses one scratch region across classes and fuses
  addition of the smaller coefficient vectors into the output's inverse finish.
- `interpolate_classes_parallel` runs independent class inverses under a total
  task budget, then adds their coefficients. It partitions scratch for
  concurrent inverses and uses a separate addition pass, exposing parallelism
  between classes.

The old builder launches local tiles from all classes together before its
cross-tile combines. The first new function therefore does not reproduce its
schedule just because it reproduces its mathematics. Benchmark both schedules
with the actual four classes and concurrent prover work.

`interpolate_sum` can combine equal ordered domains before one inverse, at the
cost of consuming individual class results. Sensei only needs the final sum,
but its four domains have different sizes. This API does not turn the current
four inverses into one. Reconsider it if future code accumulates several buffers
on the same domain; the current fold already aggregates by degree class.

There is also an output-ownership decision. The old interpolation returns eight
independent owned coefficient tiles that become quotient `SharedPoly` values
and survive for commitments and openings. New interpolation leaves one
contiguous `8N` output. `CoefficientTiles` only lends slices; it cannot produce
eight independent owners. Either give `SharedPoly` an owning slab plus a range,
or explicitly copy the pieces and measure the cost. Avoid introducing a flatten
copy before interpolation followed by another split copy after it.

`Class` itself does not remove the temporary row chunks produced by
`fold_classed_rows`, nor provide concurrent shared scatter. Eliminating those
chunks would require changing the fold's buffer partitioning or output
interface. Once interpolation begins, class state also prevents scattering or
interpolating again, including after a panic. Refill the storage and construct
new classes for another proof.

## Integrate the executor separately from the arithmetic

Sensei's [worker pool](../../bento/crates/sensei/src/native/worker/mod.rs) already
makes cooperative progress while waiting for queued work. The mismatch is its
owned, `'static` job interface versus the new
[`fft::Executor`](../crates/udon/src/fft/executor.rs), which requires a scoped
`join` of borrowed `FnOnce + Send` closures. The existing implementation of the
old `udon::exec::Executor` is not an implementation of this new trait.

Start by running serial Udon operations inside existing outer prover jobs, or
one serial residue operation per existing tile job. This permits correctness
comparison before changing the worker lifetime model. Sensei already exposes
parallelism across columns and product forms, so serial work inside each job
does not necessarily make the proof serial.

Then add a scoped adapter if measurements justify inner parallelism. It must
complete both closures and release their captures before returning or
unwinding, and support nested joins at a one-worker budget. Preserve Sensei's
allocation-context and replay behavior. Its `scoped_map` starts additional
scoped OS threads; invoking that mechanism at every recursive FFT join would
not provide a cheap persistent-pool adapter.

Divide the available budget across simultaneously active application jobs.
Direct `ExpansionOptions` multiply the residue task limit by the per-transform
task limit. Prepared `ExpansionStrategy` divides one total budget between those
levels, but it does not coordinate budgets across separate prover operations.
Serial inner transforms with parallel residues are a useful zero-scratch case.
Setting only `max_tasks` on `ExecutionOptions::serial()` keeps a whole-transform
tile and does not enable the direct tiled transform's inner parallelism.

Allocate scratch from the operation's query and reuse it only after its jobs
complete. Concurrent calls need disjoint mutable scratch. Resource ceilings
exclude inputs, outputs, executor allocations, and fixed stack storage. For
scale, the four class buffers alone contain `15N` fields, or 960 KiB here.
The existing tile geometry of 2048 rows and 128 combine columns, with four
tasks, currently asks a full `8N` direct transform for another 128 KiB of
scratch. These figures describe payloads, not a process-memory budget.

## Replace metadata by mathematical meaning

The old `FftMetadata` retains bit-reversal indices, several twiddle families,
and extended residue scales. Start with table-free plans for correctness, then
carry over only tables needed by the selected operations. The new
[table contracts](../crates/udon/src/fft/tables.rs) distinguish ordinary inverse
roots from a normalized inverse finish.

For a size `m` plan with shift `g`, the relevant correspondence is:

| Old metadata | New meaning |
| --- | --- |
| `forward_twiddles[i] = w_m^i` | `Tables::forward` |
| `inverse_twiddles[i] = w_m^-i` | `Tables::inverse` |
| `normalized_inverse_twiddles[i] = m^-1 * w_m^-i` | `Tables::inverse_finish` on a subgroup plan |
| `normalized_inverse_coset_twiddles[i] = m^-1 * g^-i * w_m^-i` | `Tables::inverse_finish` on the corresponding coset plan |
| Three-entry inverse-coset scale cycle | Omit the table; the domain already recognizes order-three shifts |
| `bit_reversed` index array | No retained index table is required |
| Extended `residue_scales[(s,i)] = (g * w_8N^s)^i` | `ExpansionScales` with `Coefficients` normalization |

`Tables::inverse_scales`, if supplied, holds `m/2` entries
`m^-1 * g^-i`, not the old three-entry cycle. Query lengths through
`TableRequirements`. Every table family is independently optional. A normalized
finish table cannot substitute for ordinary inverse twiddles.

Use `TablesMut::prepare` for native generation or `Tables::bind` once when
importing old entries after adapting their field representation. Binding checks
every entry; repeating it on every proof would introduce linear setup work.
`BoundTables::for_coset` reuses ordinary twiddles for a different shift on the
same subgroup, dropping shift-dependent tables. It does not change the size.

An `8N` expansion scale table has 16384 fields, or 512 KiB. For a `2N` expansion,
the matching old blocks are residues zero and four of that table, not its first
two blocks. The new handle requires the declared expansion's own contiguous
layout. Prepare a separate table, gather the correct blocks during setup, or
generate powers without a table.

`UnscaledInverse` expansion scales additionally contain `N^-1`. They can fold
base inverse normalization into residue initialization, but cannot be obtained
by relabeling the old ordinary scale table. Retained `Normalized` coefficients
require ordinary `Coefficients` tables; retained `Unscaled` coefficients require
`UnscaledInverse` tables. Both retention choices work without scale tables.
Passing normalized coefficients to an expansion with an `UnscaledInverse` table
regenerates ordinary powers instead of using that table.

Prepared stage transforms additionally support dense or stage-packed
`TwiddleTable` providers and radix-4/8 codelets. A larger canonical twiddle
table can serve smaller transforms, and either root orientation can serve both
directions. These are possible storage or kernel experiments, but `Expansion`
and `Class` currently accept ordinary `Plan` tables and do not expose these
prepared-stage controls. Attaching one shared alternate provider to every
high-level pipeline is not currently a supported configuration. Decomposing the
pipeline into prepared transforms would need its own justification.

## What can improve, and what is already optimized

The most promising changes visible from source concern storage and scheduling.
Their end-to-end benefit remains to be measured.

| Candidate | Existing cost or constraint | Suggested experiment |
| --- | --- | --- |
| Share retained coefficients with expansion jobs | `residue_transforms` copies coefficients into a new shared buffer for owned jobs | Disposable-input expansion, or reuse an existing shared coefficient owner |
| Write prepared columns directly into final storage | Target preparation expands into tiles, then `into_flat` copies them | Contiguous residue-major output |
| Reuse class scratch instead of job-local gather buffers | The old tiled inverse uses mutex-backed tile ownership and temporary combine buffers | Queried scratch plus scoped disjoint work; include owner changes in the measurement |
| Fuse the short-instance factor multiply | The old short transform multiplies its factor after the FFT | `short_product` fuses factor application into the forward finish |
| Match class concurrency to outer prover concurrency | Old local-tile waves span all classes; new fused interpolation sequences classes | Compare serial-class fusion with class-parallel interpolation under the same total budget |
| Reduce retained metadata | Four cached `Fp` sizes retain several overlapping power families and index arrays | Selective tables or table-free plans; separately test setup and warm execution |
| Fold normalization into expansion when coefficients are temporary | The old base-Lagrange helper uses unscaled coefficients and applies an extra factor during expansion | Pre-normalized expansion scales, with the appropriate coefficient scale contract |

Do not count these existing behaviors as new savings:

- Full coset expansion already runs eight `N`-point transforms and avoids
  cross-residue combines of a zero-padded `8N` transform.
- The short-instance helper already broadcasts a ten-coefficient prefix into
  a width-16 schedule and skips seven of the eleven base transform stages.
  Prefix pruning must survive the migration.
- Quotient folding already scatters larger classes directly into bit-reversed
  positions. Their inverses do not need another input permutation.
- The old inverse finish already fuses normalization, zeta unscaling, and the
  lifted coefficient sums. The order-three shift already has a short scale
  cycle.
- Product forms already use a smaller `2N` representation where the fold needs
  only those rows.

Other capabilities are conditional, rather than missing optimizations in this
application. Arbitrary shifts and mixed-shift classes broaden the API but are
not needed for Sensei's fixed zeta domains. `inverse_prefix` is not a substitute
for the prepared instance factorization. Polynomial-major batches need matching
contiguous storage and full-support transforms; Sensei currently owns and
schedules columns independently. Global bit-reversed expansion suits a different
consumer access pattern. Alternate codelets, tables, and blocked geometry need
measurement; `Auto` is a deterministic selection policy, not a benchmark tuner.

Use the [FFT performance report](FFT_PERFORMANCE.md) to choose experiments, not
to predict Sensei speedups. Its results show that more tables, batching, and
larger codelets do not win uniformly, and document revision-specific limits.
Measure complete prover stages, allocation counts and bytes, peak live storage,
first-use preparation, and warm execution at fixed worker budgets.

## Keep artifact and point FFTs in scope

The sibling's [artifact generators](../../bento/crates/sensei-artifacts/src)
are the important users of the old generic Udon FFT:

- Polynomial coefficient and instance-basis generation use base inverse FFTs.
- Coset generation performs a base inverse, zero padding, zeta scaling, and a
  full extended transform. Its serialized evaluations are in natural row order.
- [SRS generation](../../bento/crates/sensei-artifacts/src/points/srs.rs) performs
  an inverse FFT over Vesta projective points, then batch normalizes them.
  Its twiddle field is Vesta's scalar field `Fp`.

Preserve the natural-row artifact format unless deliberately updating its
schema, consumers, and pinned digests. An optimized generator using `Expansion`
must convert residue order at the serialization boundary. Keeping the simple
reference schedule in generators or test oracles also helps avoid validating
the runtime with the same optimized algorithm.

The [old generic FFT](../../bento/crates/udon/src/fft/mod.rs) and its
[Pasta adapters](../../bento/crates/udon/src/fft/pasta.rs) provide the model for
the new `udon::fft::reference`, with `Twiddle` and `Butterfly` traits. The rewrite
has no curve implementation and its fast `Plan` is restricted to its two Pasta
field types. The old point FFT therefore
needs a downstream value adapter with matching scalar arithmetic, potentially
a newtype to satisfy Rust's orphan rules. Retaining the old generic point FFT
temporarily is another independent migration step. Field FFT availability does
not imply that the whole old Udon crate can yet be replaced: curve, executor,
and ownership dependencies extend beyond this guide.

The rewritten FFTs have no `fft` feature gate and need no allocator. Downstream
buffers, worker pools, and cached tables still have their own resource costs.
If moving metadata into generated artifacts, use the
[FFT embedding example](../crates/udon/tests/fixtures/fft_embedding) and keep
field representation, table semantics, and the owner's file format explicit.
Existing runtime table generation is a design choice to benchmark, not evidence
that compile-time embedding was accidentally overlooked.

## A practical port and review sequence

1. **Pin the semantics.** Compare canonical bytes of roots, inverse roots, size
   inverses, and zeta between versions, including sizes `N`, `2N`, `4N`, and
   `8N`. A transform round trip alone cannot detect a consistently changed row
   convention. Keep field conversion explicit while both crate versions exist;
   matching type names do not establish compatible Rust or POD representations.
2. **Port leaf operations under the existing scheduler.** Use table-free serial
   plans first, preserve ordinary coefficients, and compare coefficient
   expansion, base-evaluation expansion, and the ten-coefficient factor product
   with independent existing oracles. Check full and reduced layouts by logical
   row, including rotations and rejection of rows absent from `2N`.
3. **Port the four quotient classes.** Preserve the scatter mapping and verify
   every intended row is filled once. Compare the sum with four independent
   inverse transforms and coefficient addition. Check the eight coefficient
   pieces and the commitments/openings consuming them, not just a sum digest.
4. **Resolve ownership and scheduling.** Choose contiguous owners or per-residue
   jobs, then implement any scoped executor adapter. Exercise nested joins,
   one-worker execution, panic completion, and allocation replay. Query scratch
   for each active configuration and account for simultaneous prover calls.
5. **Restore and tune tables selectively.** Validate imported entries during
   setup. Measure normalized versus temporary unscaled paths, storage policies,
   class schedules, and table choices without changing their mathematical
   outputs. Include the costs of any flattening, splitting, and factor-layout
   conversion in timings.
6. **Validate the application and artifacts.** Run Sensei's existing fixed-seed
   proof, prepared-instance, target derivation, and artifact consistency tests.
   Preserve established canonical outputs and golden digests; investigate
   differences before changing expectations. Keep point-SRS checks independent
   of the field runtime migration.

Introduce the API to the porting agent in this order: polynomial purpose,
domain and layout, support and normalization, ownership, execution, then tables
and backend tuning. The hardest migration errors arise when an untyped slice
silently changes meaning. Make those meanings explicit at Sensei's existing
boundaries, and the new FFT machinery can replace the specialized kernels
without losing the application choices that made them useful.
