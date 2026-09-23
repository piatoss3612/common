# Weighted group sums

[Algebra reference](../ALGEBRA.md). Names here belong to
[`curve::msm`](../../crates/udon/src/curve/msm/mod.rs). An MSM computes one
point `Q=sum_i [a_i]G_i`, with coefficients in the selected curve's scalar
field. Look first for structure in the coefficients, the base mapping,
or a result already available from an earlier calculation.

### `Bases`

`Bases` describes the storage of `G_i`: `Affine` for nonidentity points,
`Points` when identity entries are allowed, `Prepared` for cached affine
entries, and `Compact` or `CompactPrepared` for compact Eisenstein table
batches. Choose the variant matching the representation the application
already retains. Every variant describes the same kind of linear
combination; `len` and `is_empty` count bases, including one base per
compact table rather than one per table entry. Expanded fixed-base
tables have their own `FixedBaseTable::sum` API.

### `Selection`

`Selection::new` fixes the dense base order;
`Selection::indexed` fixes a validated mapping `m(i)` so later rows
compute `sum_i [a_i]G_(m(i))`. Use a selection when the base mapping is
stable across changing coefficients. Indexed inputs can address sparse,
reordered, or repeated bases without gathering point storage. Repeated
indices remain separate additive terms; indexing alone does not coalesce
them. `len` and `is_empty` describe the term list, not the base bank.

`with_scalars` binds a field row and `with_prepared_scalars` binds retained
scalar preparation. These borrow the new row independently of the original
selection construction, so successive rounds can reuse a mapping without
keeping old scalar rows alive. A different support pattern needs a
different mapping or explicit zero coefficients.

### `with_unsigned`, `with_signed`, and `with_canonical`

These `Selection` methods preserve integer facts that would be hidden by
first converting every coefficient to an unrestricted field element.
`with_unsigned` takes `u128`; `with_signed` takes `i128`, including its
minimum, and negative coefficients subtract the corresponding point
multiple. Use them for counts, bounded weights, and signed differences
whose meaning is naturally integer-valued.

`with_canonical` takes `CanonicalUint` values and an explicit bit bound.
It checks both the bound and that each integer is below the curve's
scalar modulus; it neither truncates bits nor reduces invalid integers.
A zero-bit bound accepts only zero. Choose modular field arithmetic
instead when wrapped differences are intended, or the corresponding
`PreparedScalars` constructors when the row will recur.

### `Selection::with_nonzero_scalars`

This extracts the support of a coefficient row: discard terms with
`a_i=0`, copy the remaining scalars and base indices to caller buffers,
and return an ordinary `Input`. Use it when a dense field calculation
has produced a sparse row but the point bank should stay where it is.
Both reduction states are accepted, and equivalent loose zero
representations are recognized. Nonzero coefficients of identity bases
and repeated indices are retained; their simplifications are separate.

The returned input's `selection()` can retain the chosen mapping after
the original row is gone. Reuse that support only when later rows are
known to vanish at the omitted positions. If support is already known,
construct an indexed selection directly. See the
[support conversion](../../crates/udon/src/curve/msm/support.rs).

### `Input`

`Input::new` and `indexed` bind raw field scalars to dense or indexed
bases; `new_prepared` and `indexed_prepared` bind retained scalar
records. They are conveniences for selecting bases and binding one row.
`execute` returns the weighted group sum, including identity for empty
or canceling input. Use `selection`, `len`, and `is_empty` to retain or
inspect the term mapping; `requirements` describes execution storage.

Choose `run::MsmPlan` when the same compatible operation shape needs a
resolved plan, or `MsmRun` when parts of the sum should enter an
application scheduler. An `Input` describes the sum as supplied. It
does not promise to discover constant regions, equal bases, or a
better algebraic basis; the following APIs express those facts.

### `PreparedScalars` and `ScalarStorage`

[`PreparedScalars`](../../crates/udon/src/curve/msm/prepared.rs) retains
classification and GLV decomposition for a whole coefficient vector,
releasing the original row borrow. Use `prepare` for field scalars,
or `unsigned`, `signed`, and `canonical` for the corresponding integer
forms. `storage_len` sizes caller-owned `ScalarStorage`, initialized
with `ScalarStorage::ZERO`; `len`, `is_empty`, and `retained_bytes`
describe the resulting preparation. Reuse it with different base sets
or concurrent executions with separate scratch.

`cache_len(plan)` and `cache(plan, storage)` optionally retain the
resolved plan's digit recoding as well. An unsupported cache returns
the same usable scalar preparation. Existing plans keep their original
requirements; use `MsmPlan::for_input` on the returned handle to plan
with the retained cache. This is preparation for one **scalar vector**;
`EisensteinScalar` instead prepares one scalar for compact table
multiplication, and `Selection` retains the bases while scalars vary.

### `SharedScalarInput`

[`SharedScalarInput::new`](../../crates/udon/src/curve/msm/matrix.rs)
binds a matrix of bases to one `PreparedScalars` vector. Output `j` is
`Q_j=sum_i [a_i]G_(j*output_stride+i*term_stride)`. Use it for several
linear maps sharing coefficients, such as commitments to the same
polynomial in several bases. Row-major storage uses strides `(terms,1)`;
term-major storage uses `(1,outputs)`. Strides count bases even when
each base is represented by a compact table. Padding, overlapping
immutable rows, and zero strides are allowed.

`outputs` and `terms` describe the matrix shape; `requirements` sizes
scratch, and `execute` writes one projective sum per output. Zero
terms give identity outputs. This differs from
`EisensteinTableBatch::mul`, which applies one scalar to each base,
and `run::BatchPlan`, whose independent MSM inputs need not share
scalars or lengths. A shared scalar vector is the defining invariant.

### `CoalescingPlan` and `CoalescingKey`

[`CoalescingPlan::prepare`](../../crates/udon/src/curve/msm/coalesce.rs)
groups identity-capable points by equality up to sign, using caller
storage initialized with `CoalescingKey::EMPTY`. `with_scalars`
then applies `[a]P+[b]P=[a+b]P` and
`[a]P+[b](-P)=[a-b]P`, discards identity contributions and groups
whose coefficient cancels, and returns an ordinary MSM input over the
remaining representatives. Use it when point values, rather than
their storage indices, repeat across a stable base row.

`len` and `is_empty` describe the original terms; `groups` bounds the
nonidentity groups before coefficient cancellation. The output buffers
hold the transformed bases and scalars, so executing the result does
not require retaining the plan. Grouping only recognizes equal or
opposite points; it does not solve for other scalar relations.

### `IndexedCoalescingPlan`

`IndexedCoalescingPlan::prepare` instead groups repeated indices into
a retained base bank. `with_scalars` sums their coefficients modulo
the scalar order and omits zero groups, returning an indexed input;
`len`, `is_empty`, and `groups` describe its original terms and index
groups. Use it for repeated references to compact or cached bases
that should remain in their existing storage.

Two different indices remain different groups even if their points
are equal, opposite, or identity. Choose `CoalescingPlan` when that
point-level equivalence is the useful fact, and ordinary
`Selection::indexed` when each repeated reference should simply stay
as a separate term. The index grouping does not reorder coefficients
without carrying their corresponding indices.

### `BasisSum`

[`BasisSum::prepare`](../../crates/udon/src/curve/msm/sum.rs) retains
`S=sum_i G_i` for one ordered region. For coefficients equal to `c`
except for differences `delta_j` at indices `i_j`, use
`Q=[c]S+sum_j [delta_j]G_(i_j)`. `sum` exposes `S`, `original` the
region, and `corrections` binds the second term. Repeated bases and
indices contribute with their multiplicity. Choose one `BasisSum`
per constant region and add the region results when the row has
several unrelated baselines.

`tail_corrections` accepts a `field::ConstantPrefix`, subtracts its
constant from each **actual tail value**, and binds the corresponding
suffix bases. The returned input contains only corrections: add
`[c]S` and any explicit extra terms yourself. A wholly constant row
needs only `[c]S`; with a full tail, the baseline cancels out of the
combined result. An empty or canceling region simply has `S=O`.

### Update an existing linear combination

If `Q=sum_i [a_i]G_i` is already known and only selected coefficients
change to `b_i`, compute
`Q'=Q+sum_changed [(b_i-a_i)]G_i`. Use an indexed `Input` or
`BasisSum::corrections` for those differences. They are additive
changes, not replacement coefficients. The prior result must belong
to the same ordered basis and include exactly the baseline terms
being updated; no API infers an application's omitted or extra terms.

### `SuffixBasis`

[`SuffixBasis::prepare`](../../crates/udon/src/curve/msm/suffix.rs)
retains `H_i=sum_(j>=i)G_j`. The identity
`sum_i [a_i]G_i = [a_0]H_0 + sum_(i>0)[a_i-a_(i-1)]H_i`
turns a piecewise constant row into its change points. Use
`with_scalars` for arbitrary field coefficients, including decreases
and modular wraparound. For example, `(a,a,b,b)` becomes
`(a,0,b-a,0)` over the suffix basis. `suffix`, `len`, and `is_empty`
expose the prepared basis, which no longer borrows the original bases.

`with_monotone_unsigned` validates a nondecreasing `u128` row and
keeps its nonnegative integer differences as `u128`. Use it when that
integer ordering is known, rather than applying canonical integer
ordering to wrapped field differences. Both bindings return ordinary
MSM inputs and leave zero differences in the row; explicit support
extraction can compact a field difference row afterward. Choose
`BasisSum` for a baseline with sparse exceptions and `SuffixBasis`
when transitions between adjacent runs are the useful structure.

### Known relations between bases

If the application knows `G_i=[b_i]H`, distributivity reduces its MSM
to `[sum_i a_i*b_i]H`, using a field product sum and one scalar
multiplication. In particular,
`[a]P+[b]phi(P)=[a+lambda*b]P`, with `lambda` the scalar-field
`ZETA`. Use only relations established by the construction of the
bases; neither equality coalescing nor GLV decomposition discovers
arbitrary discrete-log relations. When separate products are needed,
this reduction would discard required outputs.

### `Requirements` and `Scratch`

`Requirements` reports typed counts through `scalars`, `digits`,
`affine`, `projective`, `field`, and `indices`; `bytes` combines their
arithmetic storage sizes. `Scratch::new` borrows the corresponding
initialized buffers, and `reborrow` permits successive uses of the
same workspace. Use the requirements for the selected input or plan:
retained preparation can change what execution needs. These buffers
carry temporary arithmetic, not additional terms in the sum, and
persistent bases, scalar preparation, outputs, and scheduler metadata
have separate lifetimes. See [execution](EXECUTION.md) for plans,
batches, and incremental runs.
