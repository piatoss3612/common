//! Polynomial arithmetic over borrowed slices.
//!
//! Coefficients are in ascending degree order: `[a, b, c]` represents
//! `a + b*X + c*X^2`. Empty coefficient slices represent zero; trailing zero
//! coefficients remain part of a slice's extent.
//! [`InterpolationPlan`] recovers coefficients or evaluates directly from small
//! distinct point sets, reusing caller-owned barycentric weights. Its split
//! preparation can share denominator inversion with other arithmetic.
//! Operations do not allocate and accept both Pasta fields. Arithmetic is
//! variable-time.
//!
//! The unstable `traits` feature adds generic iterator evaluation, descending
//! quotient streaming, and geometric sums. Iterator recurrences dispatch
//! through `Field::mul_add`.
//! Native slice APIs remain available in every feature configuration.

mod division;
mod evaluation;
mod fold;
mod interpolation;
mod vanishing;

#[cfg(feature = "traits")]
pub use division::divide_linear_rev;
pub use division::{MonicDivisionError, divide_linear_in_place, divide_monic_in_place};
pub use evaluation::{EvaluationError, EvaluationPlan, evaluate};
#[cfg(feature = "traits")]
pub use evaluation::{evaluate_iter, geometric_sum};
pub use fold::{FoldError, fold_weighted};
pub use interpolation::{InterpolationError, InterpolationPlan, InterpolationPreparation};
pub use vanishing::{VanishingError, vanishing_polynomial};
