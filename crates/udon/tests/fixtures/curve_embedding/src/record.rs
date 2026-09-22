//! The artifact owner's schema, shared by its generator and consumer.

use udon::{
    curve::{
        AffinePoint, CurveTableRequirements, EisensteinTable, FixedBaseDescription, FixedBaseTable,
        PastaCurve, PreparedAffinePoint,
    },
    fft::Domain,
};

pub const DESCRIPTION: FixedBaseDescription = FixedBaseDescription { window_bits: 4 };
pub const REQUIREMENTS: CurveTableRequirements = match DESCRIPTION.requirements() {
    Ok(required) => required,
    Err(_) => panic!("unsupported fixed-base description"),
};

// The record type and artifact filename identify the curve and table layout.
#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
pub struct Record<C: PastaCurve> {
    pub base: AffinePoint<C>,
    pub entries: [AffinePoint<C>; REQUIREMENTS.table_entries],
    pub cached: [PreparedAffinePoint<C>; REQUIREMENTS.table_entries],
    pub compact: [AffinePoint<C>; 8],
    pub compact_cached: [PreparedAffinePoint<C>; 8],
}

impl<C: PastaCurve> Record<C> {
    pub fn empty() -> Self {
        Self {
            base: AffinePoint::GENERATOR,
            entries: [AffinePoint::GENERATOR; REQUIREMENTS.table_entries],
            cached: [PreparedAffinePoint::from_affine(&AffinePoint::GENERATOR);
                REQUIREMENTS.table_entries],
            compact: [AffinePoint::GENERATOR; 8],
            compact_cached: [PreparedAffinePoint::from_affine(&AffinePoint::GENERATOR); 8],
        }
    }

    pub const fn tables(&self) -> Tables<'_, C> {
        let expanded = match FixedBaseTable::bind(DESCRIPTION, &self.base, &self.entries) {
            Ok(table) => table,
            Err(_) => panic!("unsupported fixed-base description"),
        };
        let cached = match FixedBaseTable::bind(DESCRIPTION, &self.base, &self.cached) {
            Ok(table) => table,
            Err(_) => panic!("unsupported fixed-base description"),
        };
        (
            expanded,
            cached,
            EisensteinTable::bind(&self.base, &self.compact),
            EisensteinTable::bind(&self.base, &self.compact_cached),
        )
    }
}

pub type Tables<'a, C> = (
    FixedBaseTable<'a, C>,
    FixedBaseTable<'a, C, PreparedAffinePoint<C>>,
    EisensteinTable<'a, C>,
    EisensteinTable<'a, C, PreparedAffinePoint<C>>,
);

pub const SRS_SIZE: usize = 8;

/// A structured reference string with coefficient and natural-order Lagrange bases.
#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
pub struct SrsRecord<C: PastaCurve> {
    pub coefficient: [PreparedAffinePoint<C>; SRS_SIZE],
    pub lagrange: [PreparedAffinePoint<C>; SRS_SIZE],
}

impl<C: PastaCurve> SrsRecord<C> {
    /// Creates placeholder bases for the generator.
    pub fn empty() -> Self {
        Self {
            coefficient: [PreparedAffinePoint::from_affine(&AffinePoint::GENERATOR); SRS_SIZE],
            lagrange: [PreparedAffinePoint::from_affine(&AffinePoint::GENERATOR); SRS_SIZE],
        }
    }

    /// The domain shared by the generator and consumer.
    pub fn domain(&self) -> Domain<C::Scalar> {
        Domain::for_size(SRS_SIZE).unwrap()
    }
}
