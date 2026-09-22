//! Pallas and Vesta arithmetic with caller-owned tables and scratch.
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
//! and execution.
//!
//! Coordinates and scalars must satisfy [`PastaField`]'s reduced-residue
//! invariant. Stored affine points must also satisfy [`AffinePoint`]'s curve
//! invariant; POD layout checks alone do not establish either property.
//! Operations are variable-time and provide no constant-time guarantee for
//! secret inputs, including bases, scalars, and table contents. Setup and
//! execution require neither allocation nor a feature flag.
//!
//! ```
//! use zakura_udon::{curve::PallasPoint, field::Fq};
//! let generator = PallasPoint::GENERATOR;
//! assert_eq!(generator.mul_projective(&Fq::from_u64(2)).to_point(),
//!            generator.double());
//! assert!(generator.add(&generator.neg()).is_identity());
//! assert_eq!(PallasPoint::from_bytes(generator.to_bytes()), Some(generator));
//! ```

use core::{fmt, marker::PhantomData};

use crate::field::PastaField;

mod affine;
mod batch;
mod eisenstein;
mod eisenstein_batch;
mod encoding;
mod fixed_base;
mod glv;
pub mod msm;
mod parameters;
mod point;
mod projective;
mod scalar;
mod table_entry;

pub use batch::batch_normalize;
pub use eisenstein::{EisensteinScalar, EisensteinTable};
pub use eisenstein_batch::EisensteinTableBatch;
pub use fixed_base::{FixedBaseDescription, FixedBaseTable};
pub use glv::glv_decompose;
pub use parameters::{Pallas, PastaCurve, Vesta};
pub use table_entry::{CurveTableEntry, CurveTableRequirements, PreparedAffinePoint};

#[cfg(test)]
mod tests;

/// A nonidentity Pasta point in affine coordinates.
///
/// The stored layout is `x` followed by `y`, each in [`PastaField`]'s four-limb
/// Montgomery representation: 64 bytes with alignment 8. Bento storage requires
/// a little-endian target. [`bento::Pod`] checks memory layout, not the curve
/// equation or reduced residues. Arithmetic, equality, and encoding assume
/// reduced coordinates satisfying `y² = x³ + 5`; other stored bit patterns
/// remain memory-safe but can panic or give incorrect results.
/// Use [`from_xy`](Self::from_xy) or [`FixedBaseTable::bind`] to validate data
/// whose mathematical validity is not established by its producer.
///
/// Use [`Self::to_bytes`] for protocol encoding. [`crate::STORED_FORM`]
/// identifies the field representation only; artifact owners must separately
/// identify the curve and their record schema.
// SAFETY: The derive checks the coordinate fields and zero-sized marker for
// POD layout, including the absence of padding. Every coordinate bit pattern
// is valid to read and share. Curve operations use safe Rust; curve membership
// and reduced residues are mathematical requirements, not memory-safety ones.
// Future unsafe kernels must remain memory-safe for arbitrary coordinates.
#[derive(Clone, Copy, Eq, PartialEq, bento::Pod)]
#[repr(C)]
pub struct AffinePoint<C: PastaCurve> {
    x: PastaField<C::Base>,
    y: PastaField<C::Base>,
    marker: PhantomData<C>,
}

/// An affine Pasta point, including identity.
///
/// [`Default`] returns identity. Coordinates of a nonidentity point obey
/// [`AffinePoint`]'s invariants. This type does not implement [`bento::Pod`];
/// store nonidentity [`AffinePoint`] values or use [`Self::to_bytes`].
///
/// Operations that return affine results invert a field element when needed;
/// use [`ProjectivePoint`] to accumulate additions without these inversions.
#[derive(Clone, Copy, Eq, PartialEq)]
pub struct Point<C: PastaCurve>(Option<AffinePoint<C>>);

/// A Pasta point represented by Jacobian coordinates `(x, y, z)`.
///
/// For nonzero `z`, the affine coordinates are `(x / z², y / z³)` and satisfy
/// the curve equation. Every `z = 0` representation denotes identity, which is
/// also the [`Default`]. Equality compares group elements without inversion,
/// including differently scaled representations. Group operations do not
/// promise a particular scaling of their result.
///
/// This type does not implement [`bento::Pod`]; store [`AffinePoint`] values
/// instead.
#[derive(Clone, Copy)]
pub struct ProjectivePoint<C: PastaCurve> {
    x: PastaField<C::Base>,
    y: PastaField<C::Base>,
    z: PastaField<C::Base>,
    marker: PhantomData<C>,
}

/// A nonidentity affine Pallas point over [`crate::field::Fp`].
pub type PallasAffine = AffinePoint<Pallas>;
/// An affine Pallas point, including identity.
pub type PallasPoint = Point<Pallas>;
/// A Jacobian Pallas point.
pub type PallasProjective = ProjectivePoint<Pallas>;
/// A nonidentity affine Vesta point over [`crate::field::Fq`].
pub type VestaAffine = AffinePoint<Vesta>;
/// An affine Vesta point, including identity.
pub type VestaPoint = Point<Vesta>;
/// A Jacobian Vesta point.
pub type VestaProjective = ProjectivePoint<Vesta>;

/// Returns `x³ + 5`, the right-hand side of both Pasta curve equations.
///
/// `x` must satisfy [`PastaField`]'s reduced-residue invariant.
fn curve_rhs<C: PastaCurve>(x: &PastaField<C::Base>) -> PastaField<C::Base> {
    x.square().mul(x).add(&AffinePoint::<C>::B)
}

/// A rejected curve operation or multiplication table description.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CurveError {
    /// The fixed-base window width is outside `2..=8`.
    InvalidWindowBits {
        /// The supplied width.
        bits: u32,
    },
    /// The input does not meet the scalar or preparation facts of its resolved plan.
    IncompatibleMsmInput,
    /// An MSM Booth width is outside `4..=12`.
    #[cfg(test)]
    InvalidMsmWindow {
        /// Supplied window width.
        bits: u32,
    },
    /// A scalar is not below its modulus, exceeds its bit bound, or has an
    /// invalid bound.
    InvalidScalar {
        /// Position of the invalid scalar (zero for an invalid bound).
        position: usize,
    },
    /// The planner found no layout within the caller's temporary byte ceiling.
    ///
    /// The search follows [`crate::exec::ExecutionOptions`] and is not
    /// exhaustive; this does not establish a global minimum storage requirement.
    MemoryLimit {
        /// Supplied byte ceiling.
        limit: usize,
        /// Arithmetic workspace bytes required at the planner's stopping point.
        /// An unrepresentable total is reported as `usize::MAX`.
        required: usize,
    },
    /// The base has unreduced coordinates or fails the curve equation.
    InvalidBase,
    /// A table entry is invalid or differs from its specified multiple.
    InvalidTable,
    /// A requested buffer length cannot be represented by a Rust slice.
    SizeOverflow,
    /// A batch table does not contain a whole number of eight-entry tables.
    InvalidTableLayout,
    /// An indexed MSM refers past the end of its base slice.
    BaseIndexOutOfBounds {
        /// Position in the index slice.
        position: usize,
        /// Supplied base index.
        index: u32,
        /// Number of available bases.
        bases: usize,
    },
    /// An input or output buffer does not have the required exact length.
    LengthMismatch {
        /// The buffer's role.
        buffer: &'static str,
        /// Required length in elements.
        expected: usize,
        /// Supplied length in elements.
        actual: usize,
    },
    /// A scratch buffer is too short.
    ScratchTooSmall {
        /// The buffer's role.
        buffer: &'static str,
        /// Minimum length in elements.
        required: usize,
        /// Supplied length in elements.
        provided: usize,
    },
}

impl fmt::Display for CurveError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidWindowBits { bits } => write!(f, "window width {bits} is outside 2..=8"),
            Self::IncompatibleMsmInput => {
                f.write_str("input is incompatible with resolved MSM plan")
            }
            #[cfg(test)]
            Self::InvalidMsmWindow { bits } => write!(f, "MSM width {bits} is outside 4..=12"),
            Self::InvalidScalar { position } => {
                write!(f, "invalid MSM scalar at position {position}")
            }
            Self::MemoryLimit { limit, required } => {
                write!(f, "MSM needs {required} temporary bytes, limit is {limit}")
            }
            Self::InvalidBase => f.write_str("invalid curve base"),
            Self::InvalidTable => f.write_str("invalid curve table entry"),
            Self::SizeOverflow => f.write_str("curve buffer size overflows a slice length"),
            Self::InvalidTableLayout => {
                f.write_str("batch table length is not a multiple of eight")
            }
            Self::BaseIndexOutOfBounds {
                position,
                index,
                bases,
            } => {
                write!(
                    f,
                    "base index {index} at position {position} exceeds {bases} bases"
                )
            }
            Self::LengthMismatch {
                buffer,
                expected,
                actual,
            } => {
                write!(f, "{buffer} length is {actual}, expected {expected}")
            }
            Self::ScratchTooSmall {
                buffer,
                required,
                provided,
            } => {
                write!(
                    f,
                    "{buffer} scratch length is {provided}, requires at least {required}"
                )
            }
        }
    }
}

impl core::error::Error for CurveError {}

fn check_length(buffer: &'static str, expected: usize, actual: usize) -> Result<(), CurveError> {
    if actual != expected {
        return Err(CurveError::LengthMismatch {
            buffer,
            expected,
            actual,
        });
    }
    Ok(())
}

fn check_scratch(buffer: &'static str, required: usize, provided: usize) -> Result<(), CurveError> {
    if provided < required {
        return Err(CurveError::ScratchTooSmall {
            buffer,
            required,
            provided,
        });
    }
    Ok(())
}

const fn checked_count<T>(count: usize, per_item: usize) -> Result<usize, CurveError> {
    match count.checked_mul(per_item) {
        Some(length) if length <= isize::MAX as usize / core::mem::size_of::<T>() => Ok(length),
        _ => Err(CurveError::SizeOverflow),
    }
}
