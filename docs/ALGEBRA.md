# Choosing Udon APIs by algebra

Use this reference while reading an expression, polynomial pipeline, or group
calculation: identify the mathematical result that the code needs, then look
for structure in its inputs. Entries explain the identity an API expresses,
when that identity is useful, and what distinguishes nearby alternatives.
They cover Udon's public arithmetic and execution surface; closely related
constructors, accessors, and overloads share an entry. Buffer contracts and
complete signatures remain in the linked API definitions.

| Chapter | Questions it answers |
| --- | --- |
| [Fields](algebra/FIELDS.md) | Is this a product sum, rational expression, root, or running product? Which representation carries the right meaning? |
| [Polynomials](algebra/POLYNOMIALS.md) | Do I need a value, coefficients, a quotient, or a remainder? What can stay fixed between queries? |
| [Curves](algebra/CURVES.md) | Do I need one group operation, one scalar product, or a vector of scalar products? Which point representation fits? |
| [Multiscalar multiplication](algebra/MSM.md) | Can a weighted sum use sparse support, repeated bases, constant regions, successive differences, or shared scalars? |
| [FFTs](algebra/FFT.md) | Which polynomial do these evaluations represent? Can I change domain, work by residues, or combine interpolation with another operation? |
| [Execution](algebra/EXECUTION.md) | Is the computation one call, several independent results, or a pipeline with intermediate consumers? |

## Recognize a rearrangement

Here `a_i` are field elements, `G_i` are points on one curve, and polynomials
use ascending coefficients. Scalar coefficients of a group sum belong to that
curve's **scalar** field. Each linked entry states the additional hypotheses.

| Expression or input shape | Rearrangement and API |
| --- | --- |
| `sum a_i*b_i`, with optional squares or offsets | Keep the bilinear expression explicit with [product sums](algebra/FIELDS.md#dot-products-and-productsum). |
| Many unrelated divisions | Collect denominators with [batch inversion](algebra/FIELDS.md#batch-inversion), including denominators prepared by interpolation and Lagrange queries. |
| `z_(i+1) = z_i*a_i/b_i` | Use [fraction prefixes](algebra/FIELDS.md#fraction_prefixes-and-fraction_prefixes_in_place) when every partial product is needed. |
| Many coefficient vectors evaluated at one point | Retain powers in an [EvaluationPlan](algebra/POLYNOMIALS.md#evaluationplan). |
| `(f(X)-f(z))/(X-z)`, given coefficients | [Linear division](algebra/POLYNOMIALS.md#divide_linear_in_place) produces both the quotient and `f(z)`. |
| An opening quotient commitment, given domain evaluations | [Construct quotient samples](algebra/FFT.md#opening-quotients-from-evaluations) and commit in a Lagrange basis. |
| Values on a small arbitrary set of distinct nodes | Retain barycentric weights in a [polynomial InterpolationPlan](algebra/POLYNOMIALS.md#interpolationplan-for-arbitrary-nodes). |
| `aP+b(-P)` or repeated references to a base | [Coalesce](algebra/MSM.md#coalescingplan-and-coalescingkey) to `(a-b)P`; choose point or index equivalence deliberately. |
| `sum c*G_i + sum delta_i*G_i` | Retain the [basis sum](algebra/MSM.md#basissum) and compute only corrections. |
| `sum a_i*G_i`, with many equal adjacent coefficients | Use [suffix bases](algebra/MSM.md#suffixbasis) and differences `a_i-a_(i-1)`. |
| One scalar applied to many bases | [EisensteinTableBatch](algebra/CURVES.md#eisensteintablebatch) returns each product. |
| One scalar vector applied to many basis rows | [SharedScalarInput](algebra/MSM.md#sharedscalarinput) returns one weighted sum per row. |
| `f(w^k*X)` on a domain with root `w` | [Rotate natural node indices](algebra/FFT.md#domain-rotations) instead of transforming the polynomial. |
| A polynomial given on one domain, needed on a larger one | [Expansion](algebra/FFT.md#expansion-from-coefficients-or-evaluations) evaluates the same bounded-degree polynomial. |
| A polynomial supported at a few domain nodes | Factor out the complement's vanishing polynomial and use [short_product](algebra/FFT.md#expansionshort_product). |
| Almost all evaluations equal one constant | Write `f = c + sum (y_i-c)*L_i` and use [constant-prefix interpolation or extension](algebra/FFT.md#constant-prefix-interpolation). |
| A numerator divisible by `X^n-1`, sampled away from its roots | Use [VanishingDivision](algebra/FFT.md#vanishingdivision) for quotient pieces, or its factors for divided evaluations. |
| A sum of polynomials sampled on different domains | [Class interpolation](algebra/FFT.md#executioninterpolationplan-for-a-sum-of-classes) interpolates and adds coefficients with zero extension. |
| A coefficient commitment whose input is available as evaluations | Apply the inverse group FFT to the bases: [reference transforms](algebra/FFT.md#reference-transforms-and-group-bases). |

## Move linear work to the side that repeats

Evaluation, interpolation on fixed nodes, FFTs, and commitments in a fixed
basis are linear. For weights `w_i`, evaluate or commit
`sum w_i*f_i` either before or after taking the weighted sum. For example,
[`fold_weighted`](algebra/POLYNOMIALS.md#fold_weighted) combines coefficient
vectors, while an MSM combines their existing commitments. The useful choice
depends on which intermediate results the application still needs. Different
evaluation domains first need a common polynomial interpretation; adding their
raw arrays is generally wrong.

Changing the basis can reveal structure hidden in the coefficients. Constant
regions suggest `BasisSum`; piecewise constant rows suggest `SuffixBasis`;
evaluation-form commitments suggest a Lagrange basis. Each identity preserves
the result, but it preserves that result only together with its index mapping,
signs, and normalization. Preparation retains those facts for later calls.

## Keep three distinctions visible

A field residue, a canonical integer, and Montgomery storage are different
descriptions of data. `reduce()` chooses a unique stored representative;
`to_canonical_uint()` recovers the ordinary integer. Neither operation changes
an FFT's mathematical scale: an unscaled inverse still represents `n*c_i`
after field reduction.

An evaluation vector on `x_j = s*w^j`, with `n` nodes, determines a polynomial
modulo `X^n-s^n`. Its inverse FFT selects the representative of degree below
`n`. Pointwise products recover the full product only with a sufficient degree
bound; pointwise quotients additionally need a divisibility argument to be
ordinary polynomial quotients. A declared domain or layout does not prove
either fact.

Point addition and scalar multiplication produce group elements; MSMs sum such
products; shared-scalar and batch APIs can instead produce vectors of results.
Keep the required output shape explicit before choosing an API. Storage and
execution plans preserve these meanings. Their resource limits describe one
operation and do not establish algebraic validity or an application-wide
storage bound.

Udon's built-in arithmetic is variable-time, allocation-free, and available in
`no_std`. The reference concerns mathematical composition; the
[field](../crates/udon/src/field/mod.rs),
[curve](CURVES.md), [FFT](FFT.md), and [execution](EXECUTION.md) contracts govern
representation, validation, storage, and execution.
