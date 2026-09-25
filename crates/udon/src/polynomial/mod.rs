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
//! The unstable `traits` feature adds generic iterator evaluation and descending
//! quotient streaming, geometric sums, and polynomial multiplication. Iterator
//! recurrences dispatch through `Field::mul_add`; products dispatch through
//! `FftField::multiply_polynomials`, including Pasta's coefficient expansion.
//! Native slice APIs remain available in every feature configuration.

mod division;
mod evaluation;
mod fold;
mod interpolation;
#[cfg(feature = "traits")]
mod multiplication;
mod vanishing;

#[cfg(feature = "traits")]
pub use division::divide_linear_rev;
pub use division::{MonicDivisionError, divide_linear_in_place, divide_monic_in_place};
pub use evaluation::{EvaluationError, EvaluationPlan, evaluate};
#[cfg(feature = "traits")]
pub use evaluation::{evaluate_iter, geometric_sum};
pub use fold::{FoldError, fold_weighted};
pub use interpolation::{InterpolationError, InterpolationPlan, InterpolationPreparation};
#[cfg(feature = "traits")]
pub use multiplication::multiply;
#[cfg(feature = "traits")]
pub(crate) use multiplication::{multiply_default, multiply_pasta};
pub use vanishing::{VanishingError, vanishing_polynomial};
