//! The artifact owner's schema, shared by its generator and consumer.

use udon::{
    curve::{
        AffinePoint, CurveError, CurveTableRequirements, EisensteinTable, FixedBaseDescription,
        FixedBaseTable, PastaCurve, PreparedAffinePoint,
    },
    fft::Domain,
    field::{PastaField, PrimeModulus},
};

pub const DESCRIPTION: FixedBaseDescription = FixedBaseDescription { window_bits: 4 };
pub const REQUIREMENTS: CurveTableRequirements = match DESCRIPTION.requirements() {
    Ok(required) => required,
    Err(_) => panic!("unsupported fixed-base description"),
};

// Schema 2 stores both GLV table kinds and both entry representations. Numeric
// metadata uses u64, independent of the host pointer width. Schema 1 used the
// obsolete 255-bit expanded layout and is rejected.
#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
pub struct Record<C: PastaCurve> {
    pub schema: u64,
    pub base_modulus: [u64; 4],
    pub scalar_modulus: [u64; 4],
    pub montgomery_bits: u64,
    pub window_bits: u64,
    pub point_count: u64,
    pub table_kinds: [u64; 4],
    pub entry_bytes: [u64; 4],
    pub base: AffinePoint<C>,
    pub entries: [AffinePoint<C>; REQUIREMENTS.table_entries],
    pub cached: [PreparedAffinePoint<C>; REQUIREMENTS.table_entries],
    pub compact: [AffinePoint<C>; 8],
    pub compact_cached: [PreparedAffinePoint<C>; 8],
}

impl<C: PastaCurve> Record<C> {
    pub fn empty() -> Self {
        Self {
            schema: 2,
            base_modulus: C::Base::MODULUS,
            scalar_modulus: C::Scalar::MODULUS,
            montgomery_bits: 256,
            window_bits: u64::from(DESCRIPTION.window_bits),
            point_count: REQUIREMENTS.table_entries as u64,
            table_kinds: [1, 1, 2, 2], // 1: expanded GLV, 2: compact Eisenstein
            entry_bytes: [64, 96, 64, 96],
            base: AffinePoint::GENERATOR,
            entries: [AffinePoint::GENERATOR; REQUIREMENTS.table_entries],
            cached: [PreparedAffinePoint::from_affine(&AffinePoint::GENERATOR);
                REQUIREMENTS.table_entries],
            compact: [AffinePoint::GENERATOR; 8],
            compact_cached: [PreparedAffinePoint::from_affine(&AffinePoint::GENERATOR); 8],
        }
    }

    pub fn tables(&self) -> Result<Tables<'_, C>, CurveError> {
        assert!(
            self.schema == 2
                && self.table_kinds == [1, 1, 2, 2]
                && self.entry_bytes == [64, 96, 64, 96]
                && self.base_modulus == C::Base::MODULUS
                && self.scalar_modulus == C::Scalar::MODULUS
                && self.montgomery_bits == 256
                && self.window_bits == u64::from(DESCRIPTION.window_bits)
                && self.point_count == REQUIREMENTS.table_entries as u64
                && self.base == AffinePoint::GENERATOR,
            "embedded metadata must match the curve and layout",
        );
        Ok((
            FixedBaseTable::bind(DESCRIPTION, &self.base, &self.entries)?,
            FixedBaseTable::bind(DESCRIPTION, &self.base, &self.cached)?,
            EisensteinTable::bind(&self.base, &self.compact)?,
            EisensteinTable::bind(&self.base, &self.compact_cached)?,
        ))
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
///
/// POD layout checks do not certify the root, basis order, or curve contents.
#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
pub struct SrsRecord<C: PastaCurve> {
    pub schema: u64,
    pub base_modulus: [u64; 4],
    pub scalar_modulus: [u64; 4],
    pub root: PastaField<C::Scalar>,
    pub coefficient: [PreparedAffinePoint<C>; SRS_SIZE],
    pub lagrange: [PreparedAffinePoint<C>; SRS_SIZE],
}

impl<C: PastaCurve> SrsRecord<C> {
    const SCHEMA: u64 = 1;

    fn expected_domain() -> Domain<C::Scalar> {
        Domain::for_size(SRS_SIZE).unwrap()
    }

    /// Creates matching metadata with placeholder bases for the generator.
    pub fn empty() -> Self {
        Self {
            schema: Self::SCHEMA,
            base_modulus: C::Base::MODULUS,
            scalar_modulus: C::Scalar::MODULUS,
            root: Self::expected_domain().root(),
            coefficient: [PreparedAffinePoint::from_affine(&AffinePoint::GENERATOR); SRS_SIZE],
            lagrange: [PreparedAffinePoint::from_affine(&AffinePoint::GENERATOR); SRS_SIZE],
        }
    }

    /// Returns the domain, panicking on incompatible curve or domain metadata.
    ///
    /// Point contents and basis order still require validation by the consumer.
    pub fn domain(&self) -> Domain<C::Scalar> {
        let domain = Self::expected_domain();
        assert!(
            self.schema == Self::SCHEMA
                && self.base_modulus == C::Base::MODULUS
                && self.scalar_modulus == C::Scalar::MODULUS
                && self.root == domain.root(),
            "embedded SRS metadata must match the curve and domain"
        );
        domain
    }
}
