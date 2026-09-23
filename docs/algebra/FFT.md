# Evaluation domains and polynomial transforms

[Algebra reference](../ALGEBRA.md). Names belong to
[`fft`](../../crates/udon/src/fft/mod.rs) unless qualified. For a domain
of size `n`, root `w`, and shift `s`, write `x_j=s*w^j`. Natural
coefficient index `i` means degree `i`; natural evaluation index `j`
means the point `x_j`. These two meanings remain distinct even when
the arrays have the same length. See the [FFT guide](../FFT.md) for
buffer and table contracts.

## Domains, transforms, and views

### `Domain` and `CosetDomain`

`Domain::new(k)` selects the canonical subgroup of size `2^k`, while
`for_size(n)` takes the nonzero power-of-two size directly. Use
`log_size`, `size`, `root`, `inverse_root`, and `size_inverse` to
carry its dimensions and scalar constants into related arithmetic.
The roots are nested: if `N=r*n`, then `w_N^r=w_n`. This is the
relationship behind subdomain lookups and residue expansion. Size
one is valid; available sizes also depend on field and address-space
limits.

`subgroup()` selects shift one and `coset()` selects `ZETA`.
`CosetDomain` exposes `domain`, `size`, `shift`, and `inverse_shift`;
`same_domain` compares the ordered evaluation points. Use the coset
when the evaluation points must avoid roots of `X^n-1`: an order-three
shift is outside every radix-two subgroup. These constructors do not
accept arbitrary shifts. `VanishingDivision` has a separate, more
general shift contract.

### `Transform::forward`

`Transform::new(domain)` gives a reusable transform descriptor without
requiring tables. `forward` replaces `n` natural-order coefficients
with `f(x_j)=sum_i c_i*x_j^i`, in natural evaluation order. Use it
when many values on this canonical grid are wanted or polynomial
arithmetic will be performed pointwise. Use `polynomial::evaluate`
for a single arbitrary point. `domain()` retains which ordered
points the output describes.

### `Transform::inverse`

`inverse` maps natural-order evaluations back to the unique polynomial
of degree below `n`, dividing by `n` and removing `s^i` from coefficient
`i`. Use it when evaluations must feed coefficient arithmetic, a
coefficient-basis commitment, or evaluation on another domain. If the
samples came from a higher-degree polynomial, the result is its
remainder modulo `X^n-s^n`, not all of its original coefficients.

For example, pointwise multiplication of two evaluation rows followed
by inverse interpolation gives their polynomial product modulo that
vanishing polynomial. To recover the ordinary product, choose a domain
whose size exceeds the product degree. The inverse itself cannot infer
whether that bound holds from its sample values.

### `TransformRequest`, `Transform::execute`, and `Direction`

`TransformRequest::new(Direction::Forward)` or `Inverse` starts with
full, natural-order, in-place semantics. Use `Transform::execute` when
the calculation has a shorter declared support, separate input, another
physical order, or a deliberately unscaled inverse. Its fields
`direction`, `support`, `input_order`, `output_order`, `inverse_scale`,
and `input_storage` describe mathematical and ownership facts rather
than implementation choices. `scratch_requirements` queries the direct
transform's preferred scratch; direct calls can adapt to bounded or
empty scratch, while a resolved `run::FftPlan` fixes its workspace.

### `InputSupport`

`Full` supplies all domain-sized inputs. `Prefix(k)` asserts that
natural positions `k..n` are zero and permits omitting or overwriting
them. In a forward transform this is a low-degree coefficient bound.
In an inverse transform it is **evaluation support**: the interpolated
polynomial vanishes at the omitted nodes and can still have degree
`n-1`. Use the latter for a known zero evaluation suffix, not for
missing or unknown samples. Empty prefixes represent zero; prefixes
require natural input order.

### `ElementOrder` and `InputStorage`

`ElementOrder::Natural` follows degree or node index, while
`BitReversed` permutes the low `log2(n)` index bits. The polynomial
is unchanged when the mapping is carried with the data. Use a forward
output order that the next pointwise operation and inverse can consume;
there is no mathematical requirement to scatter into natural order
between compatible stages. `InputStorage::Preserve` reads a separate
input, while `InPlace` consumes the values bank. Ownership does not
change whether those inputs mean coefficients or evaluations.

### `InverseScale` and `CoefficientView`

`InverseScale::Normalized` produces `c_i`; `Unscaled` produces
`n*c_i`, still removing the coset twist. Use unscaled coefficients
when their common factor can be carried into a following linear
operation. [`CoefficientView`](../../crates/udon/src/fft/layout.rs)
keeps that mathematical factor attached to retained expansion
coefficients: `normalization_factor()` recovers `c_i`, using the
**source** base size even when the next transform is larger.

`CoefficientView::normalized` and conversions from ordinary slices
assert a factor of one. `as_slice` exposes stored values and `scale`
describes their interpretation. Pass the view itself to consumers
that accept it; passing only its slice loses the normalization fact.
For raw-slice plans, carry it explicitly with `FftPlan::with_input_scale`
or `ExpansionPlan::with_coefficient_scale`. Field `reduce()` does
not remove this factor.

### `EvaluationLayout` and `EvaluationView`

`EvaluationLayout` describes `Natural`, `BitReversed`, or
`Residues(ResidueLayout)` storage, and `index` maps a natural row.
`EvaluationView::bind` combines that mapping with a `CosetDomain`
and a field slice. Use it when samples cross API boundaries so equal
lengths do not silently stand in for equal points. `domain`, `layout`,
and `as_slice` expose the declared interpretation; `get` retrieves a
natural row independently of storage order. Binding validates shape,
not the polynomial claimed to have produced the values.

`get_extended_row` recognizes a smaller domain inside a larger one
only with the same shift and nested roots. If the size ratio is `r`,
the smaller domain occupies larger natural rows divisible by `r`.
Use this for matching evaluations at common points; it does not
interpolate missing larger-domain rows. `ResidueLayout` alone cannot
check the field or shift, so the view provides the fuller statement.

### Domain rotations

For `y_j=f(x_j)` at `x_j=s*w^j`, the polynomial `g(X)=f(w^k*X)`
has samples `g(x_j)=y_((j+k) mod n)`. This is a cyclic permutation
of natural node indices on either a subgroup or a coset. Use
`EvaluationView::get((j+k) % n)` to read a rotated sample through
its layout; rotating a raw bit-reversed or residue-major slice
does not generally perform this permutation.

Use these lookups when an expression needs shifted evaluation
rows. No inverse or forward transform is needed, and a consumer
can read the permuted rows without materializing another array.
If coefficients are needed instead, coefficient `i` is multiplied
by `w^(k*i)`.

### `EvaluationView::multiply_into`

This multiplies evaluation rows pointwise after requiring the same
domain and layout. Use it for samples of `f*g`, constraint products,
or a factor already tabulated on the exact target points. The output
keeps that layout. If coefficients of the full product are needed,
establish `deg(f)+deg(g)<n` before interpolating; otherwise the
interpolant is the product modulo `X^n-s^n`. Domain equality alone
does not establish this degree bound.

### `ResidueLayout`

For `N=r*n`, residue `a` contains natural rows `a+r*k` and occupies
one contiguous block of `n` elements. `ResidueLayout::new(N,r)`
records this mapping: natural row `j` is stored at
`(j % r)*n+j/r`. Use `index` and `natural_row` to move between
logical and physical indices, `size`, `residues`, and `rows` for the
dimensions, and `copy_from_natural` / `copy_to_natural` for explicit
copies. These are evaluation residues, not coefficient-degree blocks
or polynomial remainders.

`index_at_extended_row` applies the nested-domain index relationship
using sizes alone; the caller must establish the same shift and
canonical root family. Use `EvaluationView::get_extended_row` when
those domain checks should accompany the lookup. Slice ranges of
ordinary coefficients already represent degree intervals and do not
need a residue mapping.

## Extension and structured evaluations

### `Expansion` from coefficients or evaluations

[`Expansion::new`](../../crates/udon/src/fft/expansion.rs) combines a
base **subgroup** transform of size `n` with a subgroup or `ZETA`
coset of size `N=r*n`. `coefficients` evaluates a preserved prefix
of at most `n` coefficients on the larger domain, treating missing
coefficients as zero. `evaluations` instead interpolates the full
base-subgroup row before extending the same degree-below-`n`
polynomial. Use the form matching the representation already available;
for coset-base samples, inverse-transform first or use the specialized
constant-prefix extension.

Direct outputs are residue-major; `layout` describes them and `view`
binds them for natural-row access or factor multiplication.
`coefficient_scratch` and `evaluation_scratch` query the corresponding
direct operations. Extension adds evaluations, not information about
an unknown higher-degree polynomial: input samples only select their
bounded-degree interpolant. The ratio can be one, so changing between
subgroup and coset samples is also an extension operation.

### `Expansion::short_product`

`short_product` evaluates a nonempty short coefficient prefix `q` and
multiplies it by an `EvaluationView` of a factor `S` on the extended
domain, giving samples of `p=q*S`. Use it when a large polynomial
has a known factor and only its smaller quotient varies. The factor
must have the exact extended domain and direct residue layout. A
subsequent interpolation recovers the full product only if its degree
is below the extended size.

One source of such a factor is sparse evaluation support. If a
degree-below-`n` polynomial vanishes at every base-subgroup node except
a nonempty set `T` of size `t`, let `H=product_(x in T)(X-x)` and
`S=(X^n-1)/H`. Then `p=q*S` with `deg(q)<t`. At the selected nodes,
`q(x)=p(x)/S(x)` because `S(x) != 0`; arbitrary-node interpolation
can recover the short `q`. Retain samples of `S` when `T` recurs.
An inverse `Prefix(t)` request alone only says which samples vanish;
it does not extract this quotient.

### `Expansion::residue` and `Residue`

`residue(a, order)` selects the evaluations at
`g*w_N^a*w_n^k`, where `g` is the target shift.
`Residue::coefficients` computes just these `n` outputs from the
preserved coefficient prefix; `scratch_requirements` describes its
direct workspace. Use it when the consumer can process one residue
at a time, or only selected residues are needed. The requested
`ElementOrder` applies inside that block, while `a` remains the
natural residue number. This permits streaming a polynomial product
or reduction without retaining every target sample at once.

### `ExpansionScales` and `ExpansionScaleNormalization`

These retain the substitution factors for the residue polynomial
`f(g*w_N^a*X)`. In `Coefficients` convention, entry `a*n+i` is
`(g*w_N^a)^i`; in `UnscaledInverse` convention it includes `1/n`.
Use `requirements` and `prepare` to construct this common data,
or `bind` for trusted stored entries, then attach it with
`Expansion::new` or `with_scales`. `domain`, `base_size`,
`normalization`, and `as_slice` expose its meaning.

The table convention describes its entries; it does not select the
inverse transform's scale. Expansion combines the table factor with
the actual coefficient normalization, including a `CoefficientView`
from another base size. Binding or attachment checks dimensions and
domain compatibility, not whether the powers were generated correctly.
See [scale tables](../../crates/udon/src/fft/expansion_scales.rs).

### Constant-prefix interpolation

`CosetDomain::interpolate_constant_prefix` accepts natural samples
described by `field::ConstantPrefix` and returns ascending normalized
coefficients directly. The identity is
`f(X)=c+sum_(i in tail)(y_i-c)*L_i(X)`, because the domain's
Lagrange basis satisfies `sum_i L_i=1`. Use it when only a tail
differs from a repeated baseline; an empty tail is just the constant
polynomial. The tail contains actual `y_i`, not differences.

This describes an evaluation pattern, not a coefficient prefix.
Both subgroup and `ZETA` coset nodes are supported, including removal
of their shift. `ConstantPrefix::write_values` followed by an ordinary
inverse is the dense expression of the same operation. The specialized
API exposes the baseline-plus-corrections identity without first
materializing the repeated prefix.

### `ConstantPrefixExpansion`

[`ConstantPrefixExpansion::prepare`](../../crates/udon/src/fft/constant_prefix.rs)
retains samples of the base domain's `L_0` on an equal or larger
target domain; cyclic index shifts supply the other `L_i`. `evaluate`
then applies `f=c+sum_tail (y_i-c)*L_i` directly on the target.
Use it when the domains repeat but tail length, values, or baseline
vary. Base and target may independently be subgroups or `ZETA`
cosets; the output is in **natural** target order.

`bind` trusts imported `L_0` samples, while `base`, `extended`, and
`samples` expose the retained interpretation. This operation avoids
requiring coefficient input from the caller. Ordinary `Expansion`
instead accepts arbitrary coefficient prefixes or full subgroup
samples and writes residue-major output; do not exchange the two
output arrays without their layout descriptions.

## Basis queries and division

### `CosetDomain::evaluate_lagrange`

This evaluates a selected natural-index range of basis polynomials
`L_i` at a query `z`. For distinct domain nodes, `L_i(x_j)` is one
when `i=j` and zero otherwise. Use the resulting weights for a
selected-node selector `sum_selected L_i(z)`, or obtain a polynomial
value by the dot product `f(z)=sum_i y_i*L_i(z)`. A range denotes
node indices, without wrapping or implicit bit reversal; split a
wrapped range explicitly.

For `z` outside the domain,
`L_i(z)=(z^n-s^n)/(n*s^n) / (z/x_i-1)`. The API handles exact
node hits by the Kronecker-delta rule, avoiding the apparent `0/0`
in that formula. It also handles singleton and empty ranges.
Choose `polynomial::InterpolationPlan` when the nodes are arbitrary;
these domain queries need no separately retained barycentric weights.

### `prepare_lagrange` and `LagrangeCompletion`

`CosetDomain::prepare_lagrange` writes the denominators
`z/x_i-1` and returns a `LagrangeCompletion` containing the remaining
scale or exact-node case. Use it when several basis queries or
other arithmetic operations can share denominator inversion.
`value_count` gives the relevant prefix; invert that prefix without
scaling, preserving zeros, then `complete` replaces it with `L_i(z)`.
Exact-node and singleton cases prepare zero placeholders so they
can participate in the same zero-preserving batch. Completion
trusts that the supplied inverses belong to its preparation.

### Opening quotients from evaluations

Suppose `y_i=f(x_i)` on the size-`n` domain `x_i=s*w^i`, with
`deg(f)<n`, and `z` is outside that domain. Compute
`v=f(z)=sum_i y_i*L_i(z)`. Then `q(X)=(f(X)-v)/(X-z)` is a
polynomial of degree below `n`, and its samples are
`q(x_i)=(y_i-v)/(x_i-z)`.

Use [batch inversion](FIELDS.md#batch-inversion) to obtain
`d_i=1/(x_i-z)`. If the Lagrange weights are not already retained,
these same inverses give `L_i(z)=-C*x_i*d_i`, where
`C=(z^n-s^n)/(n*s^n)`. Form `C` from the domain's inverse size
and inverse shift. A [product sum](FIELDS.md#dot-products-and-productsum)
computes `v`, then multiplication by `d_i` produces each quotient
sample. Both stages share one inversion batch; retain the inverses
and weights when the domain and query repeat.

With a [Lagrange commitment basis](#reference-transforms-and-group-bases)
`H_i` for this same domain, an MSM `sum_i [q(x_i)]H_i` gives the
quotient commitment directly. Use this when only that commitment
is needed: it avoids an inverse FFT and a quotient coefficient
buffer. Pair samples and bases by natural node index, accounting
for any physical evaluation layout.

Divisibility follows from using the actual value `v=f(z)`; an
unverified claimed value does not establish it. If `z=x_j`, the
exceptional sample is `q(x_j)=f'(z)`. Zero-preserving inversion
does not supply that derivative. Handle it separately, or use
[linear division](POLYNOMIALS.md#divide_linear_in_place) when
coefficients are available.

### `VanishingDivision`

[`VanishingDivision::new`](../../crates/udon/src/fft/vanishing.rs)
describes division by `X^n-1` on `N` points `g*w_N^j`, with
`n=piece_size` dividing `N`. It accepts any nonzero shift `g` with
`g^N != 1`, so none of the divisors vanish. Use `write_pieces` when
the desired result is coefficient blocks: first perform the
**unscaled forward subgroup** transform of the numerator's evaluation vector,
then pass that completed output and its `ElementOrder` to the
finish. The finish supplies inverse indexing, size normalization,
shift removal, and division. An inverse or coset transform is not
the required intermediate, despite the coefficient result.

The result is `h` satisfying `(X^n-1)*h=a` modulo `X^N-g^N`, with
`deg(h)<N`. It is the ordinary quotient if the numerator's degree
is below `N` and it is divisible by `X^n-1`. `domain`, `shift`,
`piece_size`, and `piece_count` expose the descriptor; piece `j`
holds coefficients of degrees `j*n..(j+1)*n`. Requesting fewer
pieces discards high coefficients without testing them, so a
truncated exact quotient needs an additional degree bound.

### `prepare_factors` and `VanishingFactors`

`VanishingDivision::prepare_factors` retains the inverses of
`(g*w_N^j)^n-1`, which repeat every `N/n` natural rows.
`VanishingFactors::divide_in_place` then turns numerator samples
into divided samples in natural or bit-reversed order. Use it
when the next consumer still works in evaluation space, such as
another product or a pointwise combination. `division` identifies
the plan and `as_slice` exposes one factor period.

Subsequent interpolation with the correct shift recovers the same
modular quotient as `write_pieces`; for an arbitrary shift, a
subgroup inverse needs the corresponding coefficient untwist.
The application still establishes divisibility and degree bounds.
For a generic coefficient divisor or a remainder that must be
checked, use `polynomial::divide_monic_in_place` instead.

## Retained transform data and plans

### `TableRequirements`, `TablesMut`, and `Tables`

`TableRequirements::for_size` or `for_domain` supplies `twiddles`,
the count for each optional table. `TablesMut::prepare` fills
selected `forward`, `inverse`, and `inverse_finish` slices and
returns a `Transform`; `Tables::bind` borrows trusted stored
versions. Use them when the powers defining a recurring transform
are retained by the application or its artifact generator. Omitting
a table preserves the same transform through computed powers.

`forward` and `inverse` hold ordinary root powers and can be shared
between subgroup and coset transforms of the same size.
`inverse_finish` also contains normalization and shift information,
so it must match the domain. Its precise fused formula belongs to
the [table definition](../../crates/udon/src/fft/tables.rs); it is
not simply an arbitrary row of inverse powers. Binding checks
lengths and trusts the generated contents.

### `TwiddleDescription`, `TwiddleStorage`, and `TwiddleTable`

`TwiddleDescription { size, storage }` identifies canonical subgroup
powers. `Dense` stores consecutive powers; `StagePacked` stores
the powers used by each radix-two stage. `requirements` sizes the
chosen representation, and `TwiddleTable::prepare` or trusted
`bind` creates a handle, exposed by `description` and `as_slice`.
Use `FftPlan::with_twiddles` to attach it to transform plans.

Unlike a shift-specific finish table, this data describes subgroup
twiddles for either direction and either coset. Nested roots let
a larger table serve a smaller transform, while a smaller table
covers local stages of a larger one. The table's representation
is independent of `ElementOrder`: stage-packed powers do not mean
that polynomial values are stage-packed.

### `run::FftPlan`

[`FftPlan::new`](../../crates/udon/src/fft/run.rs) resolves a
`TransformRequest` together with a domain, tables, `StorageLayout`,
and execution constraints. Use it when compatible transforms recur
or must be scheduled incrementally through `FftRun`. `size`, `tile`,
`fragments`, and `retained_fields` describe its resolved storage;
execution requires that workspace even if another direct transform
could use less. `with_twiddles` selects a retained power provider.

`with_input_scale(c)` multiplies forward coefficients by a common
factor before evaluation. Use it to carry a retained inverse's
normalization or compute samples of `c*f` without a separate
coefficient pass. It is a forward-only operation; applying a
coefficient scale and changing the evaluation domain are independent
mathematical choices.

### `FftPlan::execute` and its optional factor

`execute` performs the resolved transform and optionally multiplies
each output by a supplied factor in the output's physical order.
For a forward transform, this composes evaluation with a pointwise
product; the application must pair factors with the same ordered
domain. A raw factor slice carries no `EvaluationView` checks.
Equal order and degree bounds still govern later interpolation.

For an inverse transform, the factor weights **coefficients** after
inverse scaling. For example, normalized natural coefficients
multiplied by `alpha^i` describe `f(alpha*X)`, and multiplying
coefficient `i` by the field image of `i` describes `X*f'(X)`.
These are coefficientwise operations, not polynomial convolution.
Reorder the factor consistently for bit-reversed output.
When `alpha=w^k` and only domain samples are needed, use
[domain rotations](#domain-rotations) directly.

### `FftPlan::execute_batch`

Use `execute_batch` when several independent polynomials need the
same full in-place transform. Each consecutive `size()`-element
block is a separate coefficient or evaluation row; `batch_fields`
sizes the batch workspace. The batch retains separate results and
does not fold the polynomials together. Choose class interpolation
when the desired result is their coefficient sum, or the ordinary
`execute` method when input prefixes or preservation are needed.

### `ExpansionStorage` and `run::ExpansionPlan`

`ExpansionPlan::new` fixes extension semantics and input liveness.
Choose `Coefficients` for existing coefficient input, `ReuseOutput`
for preserved base evaluations whose temporary coefficients can
occupy output block zero, `CoefficientWorkspace { scale }` to
retain coefficients separately, or `DisposableInput { scale }`
to overwrite evaluations with retained coefficients. The latter
two select `InverseScale` explicitly. These choices compute the
same extended polynomial but keep different useful intermediates.

Use `execute` for preserved input or `execute_disposable` to consume
the input; a retained coefficient buffer is returned as a
`CoefficientView`. An optional factor multiplies output evaluations.
`with_coefficient_scale` carries a scale for the `Coefficients`
mode. `residues`, `base_size`, and `tile` describe the result's
geometry; `coefficient_fields`, `snapshot_fields`, and
`scratch_fields` distinguish retained coefficients, per-slot
snapshots, and synchronous scratch. See the
[plan](../../crates/udon/src/fft/run/expansion.rs) for exact ownership.

### `ExpansionOrder`

`Residues` keeps naturally numbered residue blocks with natural
inner rows. `BitReversed` reverses both the residue number and
inner row bits, giving full-domain bit-reversed evaluations.
Use the latter when the next full inverse accepts that order
directly; choose residue order when consumers process individual
residue slices. Bind the result to the corresponding
`EvaluationLayout` and supply any execution factor in that same
order. The selected order changes positions, not evaluation points.

### `run::InterpolationPlan` for a sum of classes

[`run::InterpolationPlan::new`](../../crates/udon/src/fft/run/interpolation.rs)
accepts classes described by `(Transform, ElementOrder)` pairs.
`execute` interpolates each class and leaves `sum_i f_i` as
normalized natural coefficients in class zero, implicitly extending
smaller coefficient vectors by zero. Use it when polynomial terms
arrive on different subgroup or coset domains and only their sum
needs the output domain's coefficient extent. Class zero must be
at least as large as every other class; `snapshot_fields` sizes
each class's scratch.

With `consume=false`, lifts retain their individual coefficients;
with `consume=true`, their values may be consumed, allowing
equal-domain evaluations to add before a shared inverse by
linearity. Different-domain raw samples cannot be added that way.
Weights can be applied to class values beforehand since
interpolation is linear. This type differs from
`polynomial::InterpolationPlan`, which interpolates one arbitrary
set of point/value pairs and does not sum FFT classes.

### `StorageLayout`, `ClassState`, `FftError`, and `LagrangeError`

`StorageLayout::Contiguous` or `Fragments { length, whole_bank }`
describes which physical regions a provider can lend to a plan.
It is independent of natural, reversed, or residue **index order**.
Use `ClassState::Evaluations`, `Coefficients`, and `Consumed` to
interpret incremental interpolation's published state; consumed
storage has no promised polynomial value. See [execution](EXECUTION.md)
for the requests that make those transitions.

`FftError` reports unsupported sizes, invalid prefixes, incompatible
layouts or class states, and execution or workspace constraints.
`LagrangeError` reports invalid natural-index ranges or short
buffers. Neither error type proves a declared polynomial's degree,
that imported table entries are correct, or that divided samples
come from an exact polynomial quotient.

## Reference transforms on other value types

### Reference transforms and group bases

[`reference::transform`](../../crates/udon/src/fft/reference.rs)
computes `v'_j=sum_i w^(i*j)*v_i` on a nonempty power-of-two
slice, using an explicit root satisfying the requirements below.
`inverse_transform` takes the inverse root and inverse size. Use these when the
values are curve points, when supplying a custom value type,
or when an independent transform schedule is useful. Field values
use their own field; curve points use their curve's **scalar**
field. The caller establishes the roots' algebraic properties.

For a coefficient commitment `C=sum_i [c_i]G_i` and subgroup
evaluations `y_j=f(w^j)`, apply the inverse group transform to
`G_i` to obtain `H_j=(1/n)*sum_i [w^(-i*j)]G_i`. Then
`C=sum_j [y_j]H_j`. This moves interpolation into a reusable
Lagrange commitment basis. For coset evaluations at `s*w^j`,
first scale coefficient base `G_i` by `s^(-i)`, then apply the
inverse group transform. Batch-normalize the resulting bases;
they may include identity points for an arbitrary input basis.

### `reference::Twiddle` and `reference::Butterfly`

Implement `Twiddle` for scalars with `ONE`, `multiply`, and
`square` obeying commutative-ring multiplication, and `Butterfly`
for values with `scaled`, `add`, and `negated` forming an additive
commutative group with a compatible scalar action. This lets the
same Fourier linear map act on field elements, curve points, or
another module over the scalar ring. Scalar multiplication must
distribute and compose correctly; cloning must preserve values.

The transform additionally needs a principal root of order `n`:
besides exact order, its nontrivial character sums must vanish.
For `n>1`, also require `w^(n/2)=-1`, which makes subtraction
implement the second output of each radix-two butterfly. A root
of exact order `n` in a field satisfies these conditions.

When `n` is not invertible in a general ring, the character sums
alone do not imply the required half power. For example, in
`F_2[e]/(e^3)`, `w=1+e` has order four and vanishing nontrivial
character sums, but `w^2=1+e^2` differs from `-1=1`. This root
does not meet the transform's contract. An inverse transform also
needs invertible length. These laws are not verified by trait
bounds or runtime checks. Use the specialized Pasta `Transform`
when its field and domain model already fit.
