//! Curve contracts and Pasta arithmetic with caller-owned tables and scratch.
//!
//! Both curves have equation `y² = x³ + 5`, generator `(-1, 2)`, and prime
//! order. [`Pallas`] uses [`crate::field::Fp`] coordinates and
//! [`crate::field::Fq`] scalars; [`Vesta`] reverses that pairing.
//! [`AffinePoint`] stores a nonidentity point, [`Point`] also represents
//! identity, and [`ProjectivePoint`] avoids inversions during addition and doubling.
//! Use [`batch_normalize`] to share an inversion across projective results and
//! [`EisensteinTable`] or [`FixedBaseTable`] to prepare a base for repeated
//! scalar multiplication. [`glv_decompose`] and point endomorphisms are also
//! available to callers implementing their own scalar algorithms.
//! [`EisensteinScalar`] retains joint digits for compact tables, and
//! [`EisensteinTableBatch`] prepares or multiplies several bases together.
//! [`msm`] sums dense or indexed scalar/base terms with caller-owned scratch
//! and execution. Operators forward to these methods. The unstable `traits`
//! feature adds `Affine`, `Projective`, and their endomorphism capabilities
//! as consumer interfaces implemented through the same native arithmetic.
//!
//! Affine coordinates use [`crate::field::Reduced`] field elements;
//! projective coordinates and scalars may use loose residues. Constructors
//! establish the field and curve invariants, which trusted POD storage preserves
//! byte for byte.
//! Operations are variable-time and provide no constant-time guarantee for
//! secret inputs, including bases, scalars, and table contents. Setup and
//! execution require neither allocation nor a feature flag.
//!
//! ```
//! use zakura_udon::{curve::PallasPoint, field::Fq};
//! let generator = PallasPoint::GENERATOR;
//! assert_eq!(generator.mul_projective(&Fq::from_u64(2)),
//!            generator.double());
//! assert!(generator.add(&generator.neg()).is_identity());
//! assert_eq!(PallasPoint::from_bytes(generator.to_bytes()), Some(generator));
//! ```

pub(crate) mod pasta;
#[cfg(feature = "traits")]
mod traits;

pub use pasta::{
    AffinePoint, CurveError, CurveTableEntry, CurveTableRequirements, EisensteinScalar,
    EisensteinTable, EisensteinTableBatch, FixedBaseDescription, FixedBaseTable,
    IncompleteDoubleAndAdd, Pallas, PallasAffine, PallasPoint, PallasProjective, PastaCurve, Point,
    PreparedAffinePoint, ProjectivePoint, Vesta, VestaAffine, VestaPoint, VestaProjective,
    batch_normalize, glv_decompose,
};
#[cfg(feature = "traits")]
pub use traits::{Affine, EndomorphismAffine, EndomorphismProjective, Projective};

/// Multiscalar multiplication; also available at [`crate::msm`].
pub use crate::msm;
