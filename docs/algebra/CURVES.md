# Points, scalar products, and retained bases

[Algebra reference](../ALGEBRA.md). Names here belong to
[`curve`](../../crates/udon/src/curve/mod.rs), except the crate-root
construction macros. Write `[k]P` for multiplication of a point by a
scalar and `O` for the identity. [MSMs](MSM.md) combine many independently
weighted points; the [curve guide](../CURVES.md) covers storage contracts.

### `Pallas`, `Vesta`, and `PastaCurve`

Both curves have equation `y^2=x^3+5`. Pallas coordinates are in `Fp`
and its scalars in `Fq`; Vesta exchanges those roles. The sealed
`PastaCurve` trait names `Base` and `Scalar` for generic code. Use its
scalar field for group coefficients, roots in a group FFT, and GLV
decomposition. A coordinate field element with the same integer
encoding is not automatically the same group scalar. Valid affine
points are already in these prime-order groups.

### `AffinePoint` and the `PallasAffine` / `VestaAffine` aliases

`AffinePoint` represents a **nonidentity** point with reduced coordinates.
Use it when a base is known to be nonzero, for mixed addition, reusable
tables, or POD storage. `from_xy` checks the curve equation;
`coordinates` borrows `(x,y)`; `to_point` includes it in the type that
also represents `O`; `to_projective` starts Jacobian arithmetic with
denominator one. `GENERATOR` is the specified point `(-1,2)`.
Operations that can cancel to `O` necessarily return another point type.

### `Point` and the `PallasPoint` / `VestaPoint` aliases

Use `Point` for an affine result that may be `O`, including normalized
MSM outputs and bases with identity entries. `IDENTITY` and `Default`
are `O`; `GENERATOR` is the same nonidentity generator as above.
`is_identity`, `as_affine`, and `coordinates` expose the distinction
without inventing coordinates for infinity. `from_xy` recognizes the
special pair `(0,0)` as identity and otherwise validates the curve
equation; `to_projective` enters the arithmetic representation.

### `ProjectivePoint` and the `PallasProjective` / `VestaProjective` aliases

Use projective points for chains of group operations before an affine
consumer needs coordinates. Jacobian coordinates `(X,Y,Z)` represent
`(X/Z^2,Y/Z^3)` when `Z != 0`; `Z=0` represents `O`. `coordinates`
exposes that representation, not ordinary affine coordinates.
`from_affine`, `from_point`, `IDENTITY`, `GENERATOR`, and `Default`
enter the same group with different input shapes; `is_identity` tests
the group identity. Equality compares group points, so different
coordinate triples can compare equal.

`to_point` normalizes one result and retains the possibility of identity.
Use `batch_normalize` for several results with the same downstream
affine requirement. Projective coordinates are working data rather
than the nonidentity POD representation used by retained tables.

### `add`, `sub`, `double`, and `add_mixed`

`Point` and `ProjectivePoint` provide complete addition, subtraction,
and doubling, with projective results. Use them when equality,
cancellation, or identity can occur: `P+(-P)=O` and `P+P=[2]P` are
ordinary cases. `ProjectivePoint::add_mixed` expresses the same addition
with a known nonidentity affine right-hand operand. Choose that form
when the application already holds a retained affine base.

### `neg`

Negation preserves each point type and changes the sign of the group
element; affine `(x,y)` becomes `(x,-y)`. Use it to absorb negative
scalars into bases or to turn a subtraction into an addition. When many
terms contain repeated opposite bases, `msm::CoalescingPlan` combines
their coefficients using `[a]P+[b](-P)=[a-b]P`.

### `endomorphism`

For `lambda=ZETA` in the curve's scalar field, the endomorphism satisfies
`phi(P)=[lambda]P`; in affine coordinates it maps
`(x,y)` to `(ZETA*x,y)` using the **base** field's `ZETA`.
It preserves the point representation, including identity, and
`phi^3=id`, so `P+phi(P)+phi^2(P)=O`. Use it when a scalar expression
contains powers of `lambda`, or when rearranging a known endomorphism
orbit. The base and scalar roots have the paired orientation supplied
by Udon; an arbitrary cube root substitution can reverse that relation.

### `mul_projective` and `ProjectivePoint::mul`

These compute one `[k]P` for an ordinary field scalar, returning a
projective point. Use `mul_projective` from affine or identity-capable
affine input, and `mul` while already in projective form. Zero scalar
or identity input gives `O`. If the base repeats, a compact
`EisensteinTable` or expanded `FixedBaseTable` retains base-specific
structure. If the desired result is `sum [k_i]P_i`, use an MSM rather
than retaining a vector of individual products unless that vector is
also needed.

### `to_bytes` and `from_bytes`

`Point` and `AffinePoint` encode a point with the canonical `x` integer
and a bit selecting the parity of `y`. Use this representation across
serialization boundaries; decoding validates the encoding and curve
point. `Point` supports the all-zero identity encoding, while
`AffinePoint` rejects it because its type promises nonidentity.
Normalize projective results before encoding. Direct POD storage of a
constructed affine point serves a different purpose: preserving typed
coordinates for immediate arithmetic in a trusted artifact.

### `pallas_affine!` and `vesta_affine!`

These crate-root macros construct nonidentity affine constants from two
constant field expressions, checking `y^2=x^3+5` at compile time. Use
them for fixed protocol or test points whose coordinates are already
known, often written with `fp_hex!` or `fq_hex!`. They reject `(0,0)`,
the wrong field, invalid points, and runtime inputs. Use `from_xy` when
coordinates are only available at runtime, or `Point::IDENTITY` for
the group identity.

### `batch_normalize`

[`batch_normalize`](../../crates/udon/src/curve/batch.rs) converts a row of
projective points to identity-capable affine `Point`s in the same order.
Use it at a shared boundary after independent scalar products, a group
FFT, or a sequence of MSM outputs. Identity entries remain identity.
The output still distinguishes them; extract `AffinePoint` only where
nonidentity has been established. Inversion scratch can be bounded or
empty, so the algebra does not require a separate full-size inverse row.

### `incomplete_double_and_add` and `IncompleteDoubleAndAdd`

`ProjectivePoint::incomplete_double_and_add(B)` computes
`A+(A+B)=[2]A+B` together with the numerators of its two chord slopes.
Use it when a downstream calculation needs those slopes as well as
the point, for example to express affine addition relations. In the
returned `IncompleteDoubleAndAdd`, `point` has Jacobian denominator
`Z`, and both `slope_numerators` divide by that same `Z`. One inverse
can therefore recover both slopes and normalize the returned point.

It returns `None` for identity input or exceptional equal-x chord cases
at either addition; it does not substitute tangent formulas. Use
complete `double` followed by `add_mixed` when only the group result is
needed for all inputs, or explicitly handle the exceptional equations
if the slopes are part of the desired result. See the
[operation](../../crates/udon/src/curve/projective.rs).

### `glv_decompose`

[`glv_decompose`](../../crates/udon/src/curve/glv.rs) expresses a scalar as
`k=k_1+lambda*k_2` modulo the group order, with signed `i128` components
of magnitude below `2^127`. It follows that
`[k]P=[k_1]P+[k_2]phi(P)`. Use the decomposition when composing an
algorithm that explicitly handles signed components and endomorphism
bases. Ordinary scalar multiplication and MSMs already know this
structure. The equation is modular; the two components are not the
low and high halves of the canonical integer encoding of `k`.

### `PreparedAffinePoint` and `CurveTableEntry`

`PreparedAffinePoint::from_affine` retains a point together with its
rotated x-coordinate; `to_affine` recovers the same point. Use it for
stored bases that are repeatedly consumed through endomorphism-based
arithmetic. Both it and `AffinePoint` implement the sealed
`CurveTableEntry` interface: `from_affine` constructs an entry, `affine`
gets its point, and `rotated(r)` selects `phi^r(P)` for `r=0,1,2`.
This is a choice of retained representation, not a change of group
element or an expanded fixed-base multiplication table. Both entry
types support trusted POD storage.

### `EisensteinScalar`

`EisensteinScalar::new(k)` retains the scalar's curve-specific
endomorphism decomposition and compact-table recoding. Use it when
one scalar is applied to several compact tables, via `mul_prepared`.
The object describes the scalar and borrows no input row or base.
It differs from `msm::PreparedScalars`, which describes an entire
coefficient vector for weighted sums.

### `EisensteinTable`

A compact [`EisensteinTable`](../../crates/udon/src/curve/eisenstein.rs)
retains eight representatives of `aP+b*phi(P)` for one nonidentity
base. Endomorphism rotations and signs provide the digit points used
to reconstruct `[k]P`. Use `prepare` for a base that will recur, then
`mul` for ordinary scalars or `mul_prepared` for a reusable
`EisensteinScalar`. `REQUIREMENTS` describes preparation storage;
`base`, `as_slice`, and `as_array` expose the bound data.

`bind` borrows an already generated table and trusts its entries to
match the claimed base and ordering. This is suitable for stored
artifacts. The compact table retains an endomorphism digit set; the
expanded `FixedBaseTable` instead retains shifted multiples for scalar
windows. Both compute the same `[k]P` and support zero scalars.

### `EisensteinTableBatch`

[`EisensteinTableBatch`](../../crates/udon/src/curve/eisenstein_batch.rs)
retains compact tables for bases `P_i`. Its `mul(k)` or
`mul_prepared(prepared_k)` writes the **vector** `([k]P_i)_i`.
Use it for scaling a basis or producing individual products needed by
later work. If only their sum is wanted, distributivity gives
`sum [k]P_i=[k](sum P_i)`; if each coefficient differs, bind the batch
as `msm::Bases::Compact` or `CompactPrepared` instead.

`requirements` and `prepare` establish the table storage; `bind`
borrows trusted complete groups of eight entries. `len` and `is_empty`
count bases, `get` selects one compact table, and `as_slice` exposes
the flat storage. `multiplication_scratch` describes the preferred
shared scratch, while execution accepts bounded or empty inversion
scratch. The batch shares the scalar preparation, not a sum result.

### `FixedBaseDescription` and `FixedBaseTable`

An expanded [`FixedBaseTable`](../../crates/udon/src/curve/fixed_base.rs)
retains shifted scalar-window multiples of one nonidentity base.
`FixedBaseDescription { window_bits }` fixes that geometry, and
`requirements` sizes its preparation. Use `prepare_with` when that
description must be reproducible, or `prepare` when caller capacities
may determine it. Then `mul(k)` computes `[k]P`; `description`, `base`,
and `as_slice` expose the retained meaning.

`bind` checks the table geometry and trusts its mathematical contents.
Choose this form when the application owns expanded tables, including
stored artifacts; choose the compact table when its endomorphism
digit representation is what the application retains. A description
is not a different scalar domain, and neither kind of table changes
the point represented by a product.

### `FixedBaseTable::sum`

`sum(tables, scalars, ...)` computes `sum [k_i]P_i` using the supplied
expanded tables, including tables with different window widths,
repeated bases, opposite bases, and empty or canceling sums. Use it
when those expanded tables already describe the bases of the desired
linear combination. `sum_scratch_len` sizes its affine and field
scratch. General `msm::Input` instead accepts point rows, cached
points, or compact tables; it has no expanded-table `Bases` variant.

### `CurveTableRequirements` and `CurveError`

`CurveTableRequirements` names counts of `table_entries`,
`projective_scratch`, and `field_scratch`; these are typed element
counts for constructing retained arithmetic data. `CurveError`
separates invalid window or scalar descriptions, matrix and index
errors, size overflow, insufficient scratch, and workspace limits.
They help establish an operation's declared shape. In contrast,
ordinary cancellation to `O` is a valid group result, and invalid
point decoding or an exceptional incomplete addition uses `Option`.
