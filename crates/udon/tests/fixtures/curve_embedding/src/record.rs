//! The artifact owner's schema, shared by its generator and consumer.

use udon::{
    curve::{
        AffinePoint, CurveError, FixedBaseDescription, FixedBaseRequirements, FixedBaseTable,
        PastaCurve,
    },
    field::PrimeModulus,
};

pub const DESCRIPTION: FixedBaseDescription = FixedBaseDescription { window_bits: 4 };
pub const REQUIREMENTS: FixedBaseRequirements = match DESCRIPTION.requirements() {
    Ok(required) => required,
    Err(_) => panic!("unsupported fixed-base description"),
};

// The owner assigns schema version 1 to this expanded-table layout. Numeric
// fields use u64, so host pointer width does not change the artifact schema.
#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
pub struct Record<C: PastaCurve> {
    pub schema: u64,
    pub base_modulus: [u64; 4],
    pub scalar_modulus: [u64; 4],
    pub montgomery_bits: u64,
    pub window_bits: u64,
    pub point_count: u64,
    pub base: AffinePoint<C>,
    pub entries: [AffinePoint<C>; REQUIREMENTS.affine_points],
}

impl<C: PastaCurve> Record<C> {
    pub fn empty() -> Self {
        Self {
            schema: 1,
            base_modulus: C::Base::MODULUS,
            scalar_modulus: C::Scalar::MODULUS,
            montgomery_bits: 256,
            window_bits: u64::from(DESCRIPTION.window_bits),
            point_count: REQUIREMENTS.affine_points as u64,
            base: AffinePoint::GENERATOR,
            entries: [AffinePoint::GENERATOR; REQUIREMENTS.affine_points],
        }
    }

    pub fn table(&self) -> Result<FixedBaseTable<'_, C>, CurveError> {
        assert!(
            self.schema == 1
                && self.base_modulus == C::Base::MODULUS
                && self.scalar_modulus == C::Scalar::MODULUS
                && self.montgomery_bits == 256
                && self.window_bits == u64::from(DESCRIPTION.window_bits)
                && self.point_count == REQUIREMENTS.affine_points as u64
                && self.base == AffinePoint::GENERATOR,
            "embedded metadata must match the curve and layout",
        );
        FixedBaseTable::bind(DESCRIPTION, &self.base, &self.entries)
    }
}
