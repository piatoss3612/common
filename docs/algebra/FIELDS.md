# Fields and rational expressions

[Algebra reference](../ALGEBRA.md). Names in this chapter are in
[`field`](../../crates/udon/src/field/mod.rs) unless qualified. Write `p` for
the selected field modulus; equations are in that field unless they explicitly
describe integers or storage.

## Field choice and representation

### `Fp`, `Fq`, `PastaField`, and `PrimeModulus`

Use `Fp` for Pallas coordinates and Vesta scalars, and `Fq` for Vesta
coordinates and Pallas scalars. The cycle swaps the roles of the two fields;
it does not identify their arithmetic. `PastaField<M, S>` makes the modulus and
representation state generic. The sealed `PrimeModulus` implementations
`PallasBase` and `PallasScalar` select these fields; their `MODULUS` constants
are ordinary little-endian integer limbs. Use the curve's associated `Base`
and `Scalar` types when writing curve-generic algebra.

### `Loose`, `Reduced`, `ReductionState`, `reduce`, and `into_loose`

Both states describe the same field. Arithmetic returns `Loose`, whose stored
Montgomery residue can lie in `[0, 2p)`; `reduce()` selects the representative
in `[0, p)`. Use that boundary for equality, ordering, or square-root APIs,
then continue arithmetic normally. `into_loose()` relaxes the bound without
changing the stored limbs. The sealed `ReductionState` trait lets generic
consumers accept either state without inventing another representation.

Reduced equality is field equality. Reduced ordering compares the canonical
integer representatives, so it is useful for sorting or deterministic choices,
but inequalities are not preserved by modular addition. Reduction is also
independent of polynomial normalization: reducing `n*a` does not divide by
`n`. See the [representation contract](../../crates/udon/src/field/pasta/representation.rs).

### `ZERO`, `ONE`, `from_u64`, `from_i64`, `is_zero`, and `is_one`

Use `ZERO` and `ONE` for additive and multiplicative identities, `from_u64`
for a small nonnegative integer, and `from_i64` for its signed field image.
Negative inputs mean additive inverses modulo `p`, including `i64::MIN`.
`Default` is zero. `is_zero()` and `is_one()` test the field value directly
even for loose representations; they avoid treating two stored forms of the
same residue as different algebraic cases.

### Canonical encodings: `from_canonical_uint`, `from_bytes`, and their inverses

Use `from_canonical_uint` or `from_bytes` when an input must name exactly one
field element: they reject ordinary integers at least `p`. Their inverses,
`to_canonical_uint` and `to_bytes`, remove Montgomery encoding and choose the
canonical residue. Bytes are 32-byte little-endian encodings. These methods
fit serialization and equality across implementations, where accepting both
`x` and `x+p` would introduce multiple encodings of the same value.

### Modular input: `from_uint_reduced`, `from_bytes_reduced`, and `from_wide_bytes_reduced`

Use these when the intended operation is the ring map `integer -> integer mod
p`. `from_uint_reduced` accepts a full 256-bit integer;
`from_bytes_reduced` accepts an arbitrary-length little-endian integer, with
the empty slice representing zero; `from_wide_bytes_reduced` is the 64-byte
case. Choose checked canonical decoding instead when out-of-range inputs must
be rejected. Modular reduction alone does not promise a uniformly distributed
field sample. The [encoding definitions](../../crates/udon/src/field/encoding.rs)
specify both families.

### `montgomery_limbs` and `from_montgomery_limbs`

These expose or construct the stored integer representing `x*2^256 mod p`,
with the bound determined by `Loose` or `Reduced`. Use them for trusted
representation-aware storage or interoperability with the same Montgomery
convention. The constructor checks the state's bound; it does not interpret
the limbs as an ordinary integer to embed in the field. Use canonical
conversion for ordinary integers, and [POD storage](../POD.md#storing-field-elements)
when retaining already constructed elements.

### `CanonicalUint`

[`CanonicalUint`](../../crates/udon/src/field/pasta/uint.rs) is an ordinary unsigned
256-bit integer, without a modulus. `from_limbs`/`limbs` and
`from_le_bytes`/`to_le_bytes` preserve that integer. Use it when scalar bit
bounds, digits, or integer encodings matter before field arithmetic;
`from_canonical_uint` can then check that it belongs to a chosen field. It is
also the input to canonical bounded-scalar MSMs.

`bit`, `highest_set_bit`, `window`, `bit_slice`, and `shr` expose binary
decompositions; `fits_in_bits` checks a claimed bound, `power_of_two` constructs
one bit, and `checked_add_u128` performs integer addition with overflow
reporting. These are integer operations: shifting a canonical residue right
is generally different from multiplication by `TWO_INVERSE` in the field.

### `is_odd`

`is_odd()` tests the canonical integer residue's low bit. Use it to choose
between `r` and `-r`, for example when a protocol specifies a root sign, or
to interpret compressed curve encodings. It does not inspect the parity of
Montgomery limbs, and a square-root routine's choice of root should not be
assumed to have a particular parity.

### `fp_hex!`, `fq_hex!`, `stored_form!`, and `STORED_FORM`

The crate-root `fp_hex!` and `fq_hex!` macros turn a fixed-width canonical
hexadecimal literal into a field constant, checking it at compile time. Use
them for mathematical constants whose source notation is an ordinary residue,
rather than writing Montgomery limbs. The syntax is `0x` followed by exactly
64 hexadecimal digits; out-of-range values are rejected.

[`stored_form!` and `STORED_FORM`](../../crates/udon/src/field/pasta/stored_form.rs)
identify Udon's `mont-u64x4` storage convention for generated artifacts.
They do not identify a modulus, curve, reduction state, table layout, or
schema version. A field element's `bento::Pod` implementation preserves its
constructed representation; the artifact schema must preserve those other
facts. See [POD storage](../POD.md).

## Expressions and roots

### `add`, `sub`, and `neg`

Use these for field addition and additive inverses. They express modular
algebra, so `a.sub(&b)` is defined even when canonical integer `a < b`.
Negate coefficients to turn a difference of products or weighted sums into
the corresponding sum; for example, put `-c` into a dot product to subtract
`c*d`. Reduce only when a consumer requires the reduced state.

### `mul` and `square`

Use `mul` for independent factors and `square` when both factors are the same
field value. Recognizing squares keeps identities such as
`(a+b)^2-a^2-b^2 = 2ab` explicit. Multiplication distributes over all the
linear operations in this guide, but multiplication of coefficient arrays
is convolution, whereas multiplication of evaluations is pointwise.

### `double`, `triple`, `mul_by_4`, and `mul_by_8`

These multiply a field element by the named small integer. Use them for
fixed coefficients in curve formulas, finite differences, and polynomial
identities instead of constructing an unrelated field multiplier. They do
not change the representation's mathematical interpretation; for example,
`double()` still returns a field value, not an integer with an extra bit.

### `mul_add` and `mul_sub`

Use `a.mul_add(&b, &c)` for `a*b+c` and `mul_sub` for `a*b-c` when one product
feeds a linear update. Horner evaluation `v <- v*z+a_i`, affine recurrences,
and accumulation of a scaled correction all have this shape. For a whole
dot product, retain the sum-of-products structure instead of expressing it
as many independent updates.

### `mul_sub_product` and `mul_sub_double_product`

These express `a*b-c*d` and `a*b-2*c*d`. The first is a two-by-two
determinant and a cross-multiplied ratio comparison: `a/b = c/d` implies
`a*d-c*b = 0`, and the converse requires both denominators nonzero. The
second fits quadratic identities with a doubled cross term. Use them when
the intended result is a difference of products; neither routine gives a
meaning to division by zero.

### Dot products and `ProductSum`

`sum_of_products` computes an array dot product, `sum_of_products_slice`
accepts equal-length slices, and `sum_of_product_pairs` accepts an iterator
of paired factors. Choose the form that preserves the application's data
organization: the iterator form fits gathered, strided, or filtered terms
without assembling parallel vectors. Empty sums are zero. These APIs apply
to inner products, coefficient evaluation against powers, and sums of
products arising from constraint equations.

[`ProductSum`](../../crates/udon/src/field/products.rs) keeps that expression
open across calls. Start with `new`, feed `add_product`, `add_square`, and
`add_term`, combine independent accumulators with `merge`, then obtain the
field result with `finish`. This expresses quadratic forms with linear
offsets or a rational sum whose inverse denominators arrive from a visitor.
Negated factors encode subtraction; merging preserves the sum rather than
the order in which terms arrived.

### `pow_u64`

Use `pow_u64(e)` for a runtime nonnegative exponent fitting `u64`, with
exponent zero giving one. A fixed power such as two has `square`, a power
of two as a *scalar factor* has `power_of_two_inverse` when inverted, and
field inversion has `invert`. These are different operations: `x^(2^k)`
is repeated squaring, while `2^(-k)*x` is scalar multiplication.

### `invert`

`invert()` returns `Some(x^-1)` exactly for nonzero `x`. Use it when one
denominator is needed, or keep it outside a repeated expression such as
`sum a_i/d = (sum a_i)/d`. For distinct denominators, use the batch family;
for a root of a ratio, use `sqrt_ratio`. Handling the `None` case is part of
the algebra, since fields do not assign an inverse to zero.

### `sqrt`

On a reduced input, `sqrt()` returns a root `r` satisfying `r^2=x` when one
exists, including `r=0` for zero. Use it when failure should mean that the
input is not a square. Either sign is valid; choose a canonical sign
explicitly with `is_odd` if the application needs one. Do not rely on the
same root sign across square-root feature configurations.

### `sqrt_alt` and `SQRT_NONSQUARE`

`sqrt_alt()` returns `(true, r)` with `r^2=x` for a square, or `(false, r)`
with `r^2=nu*x` otherwise, where `nu=SQRT_NONSQUARE` is a fixed nonsquare.
For nonzero `x`, exactly one of these branches is square. Use this when
both algebraic branches are useful, such as a construction that can retain
which nonsquare multiplier was used. Zero returns the square branch. This
API supplies a square root and a branch flag; it is not by itself a
complete map from bytes to curve points.

### `sqrt_ratio`

For reduced `u` and `v` with `v != 0`, `u.sqrt_ratio(&v)` gives `(true, r)`
with `r^2*v=u`, or `(false, r)` with `r^2*v=nu*u` for the same fixed
nonsquare `nu`. Use it when the sought root is of a rational expression;
the equations let subsequent code keep the denominator instead of forming
a field quotient first. A nonsquare branch is useful only if its adjusted
equation is acceptable to the application.

The zero cases have explicit conventions: zero numerator returns
`(true, 0)`, including `0/0`; nonzero numerator with zero denominator
returns `(false, 0)`. Thus a true flag alone does not establish that a
quotient was defined, and the nonzero-over-zero case satisfies neither
root equation. See the [root contracts](../../crates/udon/src/field/pasta/sqrt/mod.rs).

### `TWO_INVERSE` and `power_of_two_inverse`

These provide `1/2` and `1/2^k` in the field. Use them for averaging,
recovering even/odd polynomial components, or normalizing a size-`2^k`
transform. The exponent of `power_of_two_inverse` is not bounded by the
field's two-adicity: division by a nonzero power of two is meaningful even
when a transform of that size is unavailable.

### `root_of_unity` and `root_of_unity_inverse`

These return a primitive root of order `2^k` and its inverse for `k <= 32`;
`k=0` gives one. Use them for radix-two evaluation domains, periodic
characters, and the reference field or group FFT. The choices are nested:
squaring a root halves its order and gives the canonical root of the
smaller domain. `fft::Domain` packages these facts with the size and its
inverse; use it when the consumer is Udon's field FFT API.

### `DELTA`, `ZETA`, and `ZETA_INVERSE`

`DELTA = 5^(2^32)` generates the odd-order subgroup of the field's
multiplicative group. Use it when a construction needs that subgroup,
rather than a radix-two root. `ZETA` is a primitive cube root and
`ZETA_INVERSE` is its inverse: `ZETA^3=1` and `1+ZETA+ZETA^2=0`.
The order-three shift supplies a coset disjoint from radix-two subgroups,
and the paired base/scalar choices give the curve endomorphism relation.
See [field parameters](../../crates/udon/src/field/pasta/parameters.rs).

## Denominators and structured rows

### Batch inversion

`batch_invert` replaces every nonzero entry by its inverse, preserving
zeros. `batch_invert_scaled` instead produces `c/x` on nonzero entries,
again preserving zeros even when `c=0`. Use the scaled form when a common
numerator can be kept outside the denominator preparation. The `_groups`
forms, `batch_invert_groups` and `batch_invert_groups_scaled`, apply the
same operation to several disjoint slices without flattening them; a
shared scale applies to every group. Inversion scratch may be bounded or
empty.

This is a useful meeting point for unrelated algebra: collect interpolation
weights, Lagrange-query denominators, and ordinary field denominators into
one batch, then finish each operation separately. Prepared completion
descriptors require **unscaled** inverses in the original order. Zero
preservation is a convention for batching, not a field identity `1/0=0`;
callers that require nonzero denominators must establish that condition.

### `try_batch_invert_by` and `try_batch_invert_scaled_by`

Use these when denominators live inside immutable records and their inverses
should feed a visitor, rather than replace a field slice. A stable getter
selects each denominator; the visitor receives its original index, record,
and inverse, or common-scale inverse. For example, a visitor can add
`numerator*inverse` to `ProductSum` and avoid retaining a quotient row.
Visitation order is unspecified, so reductions should not depend on it.

These functions reject a zero denominator before any visitor call, reporting
`BatchInversionError::ZeroDenominator`; the scaled version does so even for
zero scale. A visitor error can occur after earlier visitor effects. This
strict behavior distinguishes them from zero-preserving in-place batching.
See [batch inversion](../../crates/udon/src/field/pasta/batch.rs).

### `fraction_prefixes` and `fraction_prefixes_in_place`

Use these for every partial value of a multiplicative recurrence:
`z_0=initial`, `z_(i+1)=z_i*n_i*inv_or_zero(d_i)`. With nonzero
denominators, this is `z_k=initial*product_(i<k)(n_i/d_i)`, the shape of
grand-product accumulators and telescoping ratios. The output includes
`z_0`, so `n` fractions produce `n+1` values; the in-place form consumes
numerators stored in a buffer with that extra slot.

A zero numerator or denominator makes the next prefix and the remaining
suffix zero. In particular, factors cannot be canceled across a zero
denominator, even if a symbolic rational expression would have a removable
singularity. Use this API only when that convention matches the operation;
use strict denominator validation otherwise.
[`FractionPrefixError`](../../crates/udon/src/field/pasta/fractions.rs) reports shape
and workspace problems, not an undefined-denominator error.

### `ConstantPrefix`

[`ConstantPrefix::new(length, constant, tail)`](../../crates/udon/src/field/pasta/constant_prefix.rs)
describes a row whose first `length-tail.len()` entries equal `constant`,
followed by the tail's **actual values**. Its algebraic interpretation is
`row = constant*ones + tail_corrections`, where each correction is
`tail_i-constant`. Use it to share this structure between constant-region
MSMs and polynomial interpolation or extension; these consumers form the
differences themselves.

`len`, `is_empty`, `prefix_len`, `constant`, and `tail` expose that description;
`write_values` materializes it for a consumer needing a dense row. An empty
tail is wholly constant and a full-length tail is an arbitrary row.
`ConstantPrefixError` rejects an oversized tail or undersized output. A tail
of zeros is not the same as an empty tail unless the constant is also zero.
