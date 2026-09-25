# Coefficients, values, and quotients

[Algebra reference](../ALGEBRA.md). Names here belong to
[`polynomial`](../../crates/udon/src/polynomial/mod.rs). A slice
`[a_0, ..., a_(n-1)]` represents `f(X)=sum a_i*X^i`. Its length is a
coefficient extent, not a promise that the last coefficient is nonzero;
the empty slice represents zero. This chapter covers arbitrary coefficient
vectors and small point sets; [FFTs](FFT.md) cover radix-two domains.

### `fold_weighted`

[`fold_weighted`](../../crates/udon/src/polynomial/fold.rs) writes
`h(X)=sum w_i*f_i(X)`, implicitly extending shorter coefficient vectors by
zero. Use it for challenge-weighted combinations, constraint aggregation,
or subtraction using a negative weight. Supply one weight per input,
including empty inputs; the API neither derives challenge powers nor
assigns a protocol meaning to them.

Linearity gives `h(z)=sum w_i*f_i(z)` and, for a fixed commitment basis,
`commit(h)=sum w_i*commit(f_i)`. Thus use a field product sum or an MSM
instead if only the resulting value or commitment is needed and the
individual results already exist. The returned extent is the largest
input length, including zero-weight inputs and trailing zeros; it does
not certify the degree after cancellation.

### `evaluate`

[`evaluate(coefficients, point)`](../../crates/udon/src/polynomial/evaluation.rs)
uses Horner's identity `f(z)=a_0+z*(a_1+z*(...))`. Use it when coefficients
are available and one value is wanted, without preparing powers or an
evaluation domain. Empty input gives zero. If many polynomials share the
query point, `EvaluationPlan` exposes that common structure; if a polynomial
is already represented by values at arbitrary nodes, use the interpolation
plan's `evaluate` instead of treating those values as coefficients.

### `EvaluationPlan`

`EvaluationPlan::prepare(point, powers)` retains `z, z^2, ...` so each later
evaluation is a dot product plus the constant coefficient. Use `evaluate`
for one coefficient row or `evaluate_many` for several rows at that same
point. The rows may have different lengths. `power_count(n)` supplies the
number of nonconstant powers needed for `n` coefficients; `point` and
`powers` expose the retained data.

Use `bind` when those powers already exist, preserving their order and
point yourself: binding does not verify their mathematical contents.
This plan fixes a **query point** while coefficient vectors vary. A
polynomial `InterpolationPlan` instead fixes the **sample nodes** and
accepts different query points and sample values. Neither is an FFT plan.

### `divide_linear_in_place`

[`divide_linear_in_place`](../../crates/udon/src/polynomial/division.rs)
computes `f(X)=(X-z)*q(X)+r`, where `r=f(z)`. The returned split separates
the remainder prefix from the quotient suffix in the same buffer: for
nonempty input the first element is `r` and the remainder are ascending
coefficients of `q`; empty input has split zero. Use it when both a value
and the quotient `(f(X)-f(z))/(X-z)` are needed, such as a polynomial
opening witness. Nonzero remainder is a valid result, not division failure.

If domain evaluations and a Lagrange commitment basis already exist, an
[opening quotient from evaluations](FFT.md#opening-quotients-from-evaluations)
can produce the commitment without first recovering coefficients.

Repeatedly divide the quotient suffix to expand around `z`:
`f(X)=r_0+r_1*(X-z)+r_2*(X-z)^2+...`. The successive remainders are
Taylor coefficients, without needing to compute derivatives or divide
by factorials. Distinct successive divisors likewise let an application
remove known linear factors while retaining the remainder at each step.

### `divide_monic_in_place`

`divide_monic_in_place` generalizes this to `f=D*q+r` for a nonempty
divisor `D` whose last supplied coefficient is one. For divisor degree
`d` and input length `n`, the returned split is `min(n,d)`: the prefix
holds `r`, the suffix holds `q`, and no trailing zeros are trimmed.
Use it for divisibility checks, reduction modulo a polynomial, and
quotients by a product of known factors. A short dividend is entirely
remainder, and division by the constant polynomial one leaves all
coefficients in the quotient.

The API requires monicity, not just a nonzero leading term. If
`D=b*D_monic` with `b != 0`, dividing by `D_monic` gives a quotient that
must then be divided by `b` to be the quotient for `D`. Unlike division
of FFT samples, this operation retains a remainder and therefore lets
the caller test exact divisibility. Use `divide_linear_in_place` for
the particular divisor `X-z`.

### `vanishing_polynomial`

[`vanishing_polynomial`](../../crates/udon/src/polynomial/vanishing.rs)
constructs `Z(X)=product_i (X-x_i)` as `n+1` monic coefficients. Use it
to encode a set of required zeros, build a divisor, or construct the
complement factor of a polynomial supported on selected domain nodes.
Repeated roots are retained with multiplicity; an empty list gives one.
This differs from interpolation, which requires distinct nodes to
recover a polynomial from ordinary values alone.

On a size-`n` subgroup, the full vanishing polynomial is `X^n-1`; on
the coset with shift `s`, it is `X^n-s^n`. Use that known structure
when forming an expression or selecting `fft::VanishingDivision`,
rather than inferring that every explicit root list requires a dense
generic divisor. An arbitrary subset still has the product form above.

### `InterpolationPlan` for arbitrary nodes

[`InterpolationPlan::prepare`](../../crates/udon/src/polynomial/interpolation.rs)
retains distinct nodes `x_i` and weights
`w_i=1/product_(j!=i)(x_i-x_j)`. For values `y_i`, the unique interpolant
of degree below the node count is
`f(X)=sum_i y_i*w_i*product_(j!=i)(X-x_j)`. Use `interpolate` when later
work needs its coefficients, or `evaluate` when only its value at a
query point is needed. Exact node queries return the corresponding
value; empty data represents zero and a singleton represents a constant.

Reuse the plan when nodes stay fixed while values or queries change.
`points` and `weights` expose the retained data; `bind` trusts supplied
weights and distinctness rather than recomputing them. This is an
arbitrary-node interpolation API. For the canonical power-of-two nodes,
an inverse FFT produces coefficients, and `CosetDomain::evaluate_lagrange`
produces selected basis values for dotting against samples.

### `prepare_denominators` and `InterpolationPreparation::complete`

If several operations need inverses, call
`InterpolationPlan::prepare_denominators` to write
`d_i=product_(j!=i)(x_i-x_j)` without inverting them. It validates
distinctness and returns an `InterpolationPreparation` tied to the nodes.
Invert these entries together with other groups, **without a scale**, then
call `complete` with their inverses to obtain the same interpolation plan.
Completion checks size but trusts the pairing and values; it cannot infer
which batch produced them. This separates denominator algebra from the
application's choice of inversion batch.

### Compose a small multipoint quotient

Suppose distinct nodes have claimed values `y_i`, and `r` is their
interpolant. Form `f-r` with `fold_weighted`, construct
`Z=product_i(X-x_i)` with `vanishing_polynomial`, and divide by `Z`.
The remainder vanishes exactly when `f(x_i)=y_i` for every node; then
the quotient is `(f-r)/Z`. Keeping the returned remainder distinguishes
a valid quotient construction from an unchecked claim of divisibility.
For a single node, linear division already provides the necessary value
and quotient together.

### Polynomial errors

`FoldError`, `EvaluationError`, and `VanishingError` report incompatible
counts, insufficient storage, or size overflow. `MonicDivisionError`
rejects an empty or nonmonic divisor; it does not reject a nonzero
remainder. `InterpolationError` additionally rejects repeated nodes
and mismatched value counts. Those duplicate checks compare field
values, including equivalent loose representatives. Treat these
errors as failures to establish the intended operation's inputs,
separately from successful operations whose algebraic result is zero.
