//! Nonidentity points and compile-time coordinate validation.

use bento::const_arithmetic::{m255, u256};
use core::{fmt, marker::PhantomData};

use super::{AffinePoint, PastaCurve, Point, ProjectivePoint, curve_rhs};
use crate::field::{PastaField, PrimeModulus};

impl<C: PastaCurve> fmt::Debug for AffinePoint<C> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AffinePoint")
            .field("x", &self.x)
            .field("y", &self.y)
            .finish()
    }
}

impl<C: PastaCurve> AffinePoint<C> {
    /// The generator `(-1, 2)`.
    pub const GENERATOR: Self = Self {
        x: PastaField::from_montgomery_limbs(m255::from_u256!(
            &C::Base::MODULUS,
            &u256::sub_with_borrow!(&C::Base::MODULUS, &[1, 0, 0, 0]).0,
        )),
        y: PastaField::from_montgomery_limbs(m255::from_u64!(&C::Base::MODULUS, 2)),
        marker: PhantomData,
    };

    /// Checks reduced coordinates and the curve equation.
    ///
    /// Returns `None` for invalid coordinates, including `(0, 0)`. This method
    /// also rejects unreduced residues read through POD storage, before doing
    /// field arithmetic.
    pub fn from_xy(x: PastaField<C::Base>, y: PastaField<C::Base>) -> Option<Self> {
        if !x.is_reduced() || !y.is_reduced() || y.square() != curve_rhs(&x) {
            return None;
        }
        Some(Self {
            x,
            y,
            marker: PhantomData,
        })
    }

    /// Borrows the affine coordinates `(x, y)`.
    pub const fn coordinates(&self) -> (&PastaField<C::Base>, &PastaField<C::Base>) {
        (&self.x, &self.y)
    }

    /// Applies the order-three endomorphism `(x, y) -> (zeta * x, y)`.
    ///
    /// Here `zeta` is the coordinate field's [`PastaField::zeta`] value.
    /// This map equals multiplication by the scalar field's
    /// [`PastaField::zeta`] value.
    pub fn endomorphism(&self) -> Self {
        Self {
            x: self.x.mul(&PastaField::zeta()),
            ..*self
        }
    }

    /// Returns the additive inverse `(x, -y)`.
    pub fn neg(&self) -> Self {
        Self {
            x: self.x,
            y: self.y.neg(),
            marker: PhantomData,
        }
    }

    /// Includes this point in the identity-capable affine representation.
    pub const fn to_point(&self) -> Point<C> {
        Point(Some(*self))
    }

    /// Lifts this point to Jacobian coordinates with `z = 1`.
    pub const fn to_projective(&self) -> ProjectivePoint<C> {
        ProjectivePoint::from_affine(self)
    }

    /// Multiplies by a scalar using variable-time doubling and mixed addition.
    ///
    /// Processes the full canonical scalar; zero returns identity. The scalar
    /// must satisfy [`PastaField`]'s reduced-residue invariant. Uses bounded
    /// internal stack storage, with no caller table, scratch, or allocation.
    ///
    /// The current implementation uses an inversion-free binary ladder for
    /// scalars below `2^64`. Larger scalars prepare eight cached affine entries
    /// on the stack, using one field inversion per call, then run a GLV/Eisenstein
    /// ladder. The strategy is selected internally; stack frame sizes depend on
    /// the compiler and target.
    ///
    /// Use [`EisensteinTable`](super::EisensteinTable) or
    /// [`FixedBaseTable`](super::FixedBaseTable) to retain
    /// preparation for repeated multiplication of the same base.
    pub fn mul_projective(&self, scalar: &PastaField<C::Scalar>) -> ProjectivePoint<C> {
        // Short scalars use the binary ladder without paying for table setup.
        let scalar = scalar.to_canonical_uint();
        if scalar.highest_set_bit().is_none_or(|high| high < 64) {
            super::scalar::multiply_canonical(scalar, |point| point.add_mixed(self))
        } else {
            super::eisenstein::multiply_once(&self.to_projective(), scalar)
        }
    }

    // Const generic limbs let the compile-time arithmetic macros validate
    // coordinates even when the public wrapper occurs in a runtime expression.
    /// Expansion support for constructing a point from constant Montgomery limbs.
    ///
    /// Each coordinate consists of four little-endian limbs. Constant evaluation
    /// fails unless both residues are reduced and satisfy the curve equation.
    #[doc(hidden)]
    pub const fn __from_montgomery_coordinates<
        const X0: u64,
        const X1: u64,
        const X2: u64,
        const X3: u64,
        const Y0: u64,
        const Y1: u64,
        const Y2: u64,
        const Y3: u64,
    >() -> Self {
        const {
            let x = PastaField::from_montgomery_limbs([X0, X1, X2, X3]);
            let y = PastaField::from_montgomery_limbs([Y0, Y1, Y2, Y3]);
            let lhs = m255::mul!(&C::Base::MODULUS, &[Y0, Y1, Y2, Y3], &[Y0, Y1, Y2, Y3]);
            let rhs = m255::add!(
                &C::Base::MODULUS,
                &m255::mul!(
                    &C::Base::MODULUS,
                    &m255::mul!(&C::Base::MODULUS, &[X0, X1, X2, X3], &[X0, X1, X2, X3]),
                    &[X0, X1, X2, X3],
                ),
                &m255::from_u64!(&C::Base::MODULUS, 5),
            );
            let mut index = 0;
            while index < 4 {
                assert!(lhs[index] == rhs[index], "point must be on the curve");
                index += 1;
            }
            Self {
                x,
                y,
                marker: PhantomData,
            }
        }
    }
}

/// Constructs a nonidentity Pallas affine point from constant coordinates.
///
/// Both arguments must be [`Fp`](crate::field::Fp) constant expressions with
/// reduced residues satisfying `y² = x³ + 5`. Invalid values, including `(0, 0)`,
/// values from the wrong field, and runtime arguments fail compilation, even
/// when the macro invocation appears in a runtime expression.
///
/// ```
/// use zakura_udon::{curve::PallasAffine, fp_hex, pallas_affine};
/// const G: PallasAffine = pallas_affine!(
///     fp_hex!("0x40000000000000000000000000000000224698fc094cf91b992d30ed00000000"),
///     fp_hex!("0x0000000000000000000000000000000000000000000000000000000000000002"),
/// );
/// assert_eq!(G, PallasAffine::GENERATOR);
/// ```
#[macro_export]
macro_rules! pallas_affine {
    ($x:expr, $y:expr $(,)?) => {
        const {
            $crate::curve::PallasAffine::__from_montgomery_coordinates::<
                {
                    let value: $crate::field::Fp = $x;
                    value.montgomery_limbs()[0]
                },
                {
                    let value: $crate::field::Fp = $x;
                    value.montgomery_limbs()[1]
                },
                {
                    let value: $crate::field::Fp = $x;
                    value.montgomery_limbs()[2]
                },
                {
                    let value: $crate::field::Fp = $x;
                    value.montgomery_limbs()[3]
                },
                {
                    let value: $crate::field::Fp = $y;
                    value.montgomery_limbs()[0]
                },
                {
                    let value: $crate::field::Fp = $y;
                    value.montgomery_limbs()[1]
                },
                {
                    let value: $crate::field::Fp = $y;
                    value.montgomery_limbs()[2]
                },
                {
                    let value: $crate::field::Fp = $y;
                    value.montgomery_limbs()[3]
                },
            >()
        }
    };
}

/// Constructs a nonidentity Vesta affine point from constant coordinates.
///
/// Both arguments must be [`Fq`](crate::field::Fq) constant expressions. Enforces
/// the same coordinate checks as [`pallas_affine!`](crate::pallas_affine).
#[macro_export]
macro_rules! vesta_affine {
    ($x:expr, $y:expr $(,)?) => {
        const {
            $crate::curve::VestaAffine::__from_montgomery_coordinates::<
                {
                    let value: $crate::field::Fq = $x;
                    value.montgomery_limbs()[0]
                },
                {
                    let value: $crate::field::Fq = $x;
                    value.montgomery_limbs()[1]
                },
                {
                    let value: $crate::field::Fq = $x;
                    value.montgomery_limbs()[2]
                },
                {
                    let value: $crate::field::Fq = $x;
                    value.montgomery_limbs()[3]
                },
                {
                    let value: $crate::field::Fq = $y;
                    value.montgomery_limbs()[0]
                },
                {
                    let value: $crate::field::Fq = $y;
                    value.montgomery_limbs()[1]
                },
                {
                    let value: $crate::field::Fq = $y;
                    value.montgomery_limbs()[2]
                },
                {
                    let value: $crate::field::Fq = $y;
                    value.montgomery_limbs()[3]
                },
            >()
        }
    };
}
