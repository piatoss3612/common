use super::{
    CosetDomain, Domain, FftError, PastaField, PrimeModulus, check_domain_size, check_len,
};

/// Coefficient normalization expected by a residue power table.
///
/// Let `n` be the base size, `N` the extended size, and `g` and `w` the extended
/// coset's shift and canonical root. Entry `s*n + i` scales coefficient `i` in
/// residue `s`, where `0 <= s < N/n` and `0 <= i < n`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExpansionScaleNormalization {
    /// Entries are `(g * w^s)^i`, for ordinary coefficients.
    Coefficients,
    /// Entries are `n^-1 * (g * w^s)^i`, for coefficients multiplied by `n`.
    UnscaledInverse,
}

/// Residue-major powers bound to their extended coset and normalization.
///
/// [`Self::bind`] checks dimensions and every imported entry against the
/// declared domain and normalization.
/// Preparation returns the same domain-bound handle without revalidation.
/// [`ExpansionScaleNormalization`] defines each entry's formula and position.
/// [`Self::bind_trusted`] relies on the caller for correct contents.
#[derive(Clone, Copy)]
pub struct ExpansionScales<'a, M: PrimeModulus> {
    pub(super) base_size: usize,
    pub(super) extended: CosetDomain<M>,
    pub(super) normalization: ExpansionScaleNormalization,
    pub(super) values: &'a [PastaField<M>],
}

impl<M: PrimeModulus> core::fmt::Debug for ExpansionScales<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ExpansionScales")
            .field("base_size", &self.base_size)
            .field("extended", &self.extended)
            .field("normalization", &self.normalization)
            .field("values", &self.values)
            .finish()
    }
}

impl<'a, M: PrimeModulus> ExpansionScales<'a, M> {
    /// Checks entries against the declared domain, base size, and normalization.
    ///
    /// Shape errors follow [`Self::bind_trusted`]. Incorrect or unreduced entries
    /// return [`FftError::InvalidTables`]. Validation takes linear field work.
    pub fn bind(
        base_size: usize,
        extended: CosetDomain<M>,
        normalization: ExpansionScaleNormalization,
        values: &'a [PastaField<M>],
    ) -> Result<Self, FftError> {
        Self::bind_trusted(base_size, extended, normalization, values)?.validate()
    }

    /// Binds caller-trusted entries after checking the base size and slice length.
    ///
    /// Errors follow [`Self::requirements`], with [`FftError::LengthMismatch`]
    /// if `values` does not have the extended domain size.
    /// The caller must establish the reduced Montgomery entries described by
    /// [`ExpansionScaleNormalization`]. Incorrect contents can cause wrong
    /// results or panics, but not memory unsafety.
    pub fn bind_trusted(
        base_size: usize,
        extended: CosetDomain<M>,
        normalization: ExpansionScaleNormalization,
        values: &'a [PastaField<M>],
    ) -> Result<Self, FftError> {
        check_len(
            "scales",
            values.len(),
            Self::requirements(base_size, extended.size())?,
        )?;
        Ok(Self {
            base_size,
            extended,
            normalization,
            values,
        })
    }

    /// Number of retained fields, without constructing a domain.
    ///
    /// Both sizes follow [`Domain::for_size`]'s validity and overflow checks.
    /// Returns [`FftError::InvalidLayout`] if `base_size > extended_size`.
    pub const fn requirements(base_size: usize, extended_size: usize) -> Result<usize, FftError> {
        if let Err(e) = check_domain_size(base_size) {
            return Err(e);
        }
        if let Err(e) = check_domain_size(extended_size) {
            return Err(e);
        }
        if base_size > extended_size {
            return Err(FftError::InvalidLayout);
        }
        Ok(extended_size)
    }

    /// Generates residue powers into exactly sized caller storage.
    ///
    /// Entries follow [`ExpansionScaleNormalization`] and use reduced Montgomery
    /// representations. Lengths and errors follow [`Self::bind_trusted`]. All
    /// checks precede writes; initial destination values are overwritten.
    pub fn prepare(
        base_size: usize,
        extended: CosetDomain<M>,
        normalization: ExpansionScaleNormalization,
        values: &'a mut [PastaField<M>],
    ) -> Result<Self, FftError> {
        check_len(
            "scales",
            values.len(),
            Self::requirements(base_size, extended.size())?,
        )?;
        let first = match normalization {
            ExpansionScaleNormalization::Coefficients => PastaField::ONE,
            ExpansionScaleNormalization::UnscaledInverse => {
                Domain::<M>::for_size(base_size)?.size_inverse()
            }
        };
        let mut step = extended.shift();
        for residue in values.chunks_exact_mut(base_size) {
            let mut power = first;
            for value in residue {
                *value = power;
                power = power.mul(&step);
            }
            step = step.mul(&extended.domain().root());
        }
        Ok(Self {
            base_size,
            extended,
            normalization,
            values,
        })
    }

    /// Checks every mathematical entry and its reduced Montgomery representation.
    ///
    /// Returns [`FftError::InvalidTables`] for any incorrect or unreduced entry.
    pub fn validate(self) -> Result<Self, FftError> {
        let first = match self.normalization {
            ExpansionScaleNormalization::Coefficients => PastaField::ONE,
            ExpansionScaleNormalization::UnscaledInverse => {
                Domain::<M>::for_size(self.base_size)?.size_inverse()
            }
        };
        let mut step = self.extended.shift();
        for residue in self.values.chunks_exact(self.base_size) {
            let mut power = first;
            for value in residue {
                if value.montgomery_limbs() != power.montgomery_limbs() {
                    return Err(FftError::InvalidTables);
                }
                power = power.mul(&step);
            }
            step = step.mul(&self.extended.domain().root());
        }
        Ok(self)
    }

    /// Extended evaluation coset.
    pub const fn domain(self) -> CosetDomain<M> {
        self.extended
    }
    /// Size of the base transform.
    pub const fn base_size(self) -> usize {
        self.base_size
    }
    /// Meaning of the table's initial value in each residue.
    pub const fn normalization(self) -> ExpansionScaleNormalization {
        self.normalization
    }
    /// Borrowed entries in naturally numbered residue blocks.
    pub const fn as_slice(self) -> &'a [PastaField<M>] {
        self.values
    }
    /// Semantic metadata for downstream serialization or cache validation.
    pub const fn artifact(self) -> ExpansionScaleArtifact {
        ExpansionScaleArtifact {
            version: 1,
            modulus: M::MODULUS,
            montgomery_bits: 256,
            base_size: self.base_size,
            extended_size: self.extended.size(),
            shift: self.extended.shift().montgomery_limbs(),
            normalization: self.normalization,
        }
    }
}

/// Semantic metadata for downstream residue-table artifacts.
///
/// The owner chooses a byte format, framing, integrity checks, and persistence.
/// Version one uses the canonical roots from [`Domain`], residue-major powers,
/// and four Montgomery limbs per field, least significant first.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ExpansionScaleArtifact {
    /// Semantic format version, currently one.
    pub version: u32,
    /// Field modulus, in increasing limb significance.
    pub modulus: [u64; 4],
    /// Montgomery radix exponent, currently 256.
    pub montgomery_bits: u32,
    /// Coefficient count per residue.
    pub base_size: usize,
    /// Complete extended domain size.
    pub extended_size: usize,
    /// Canonical Montgomery representation of the extended coset shift.
    pub shift: [u64; 4],
    /// Normalization encoded in every residue's initial value.
    pub normalization: ExpansionScaleNormalization,
}

impl ExpansionScaleArtifact {
    /// Checks imported metadata against the intended operation.
    ///
    /// Returns [`FftError::InvalidTables`] for any metadata mismatch, or the size
    /// errors from [`ExpansionScales::requirements`]. Validate table entries
    /// separately with [`ExpansionScales::bind`] when binding decoded storage.
    pub fn validate<M: PrimeModulus>(
        self,
        base_size: usize,
        extended: CosetDomain<M>,
        normalization: ExpansionScaleNormalization,
    ) -> Result<(), FftError> {
        if self.version != 1
            || self.modulus != M::MODULUS
            || self.montgomery_bits != 256
            || self.base_size != base_size
            || self.extended_size != extended.size()
            || self.shift != extended.shift().montgomery_limbs()
            || self.normalization != normalization
        {
            return Err(FftError::InvalidTables);
        }
        ExpansionScales::<M>::requirements(base_size, extended.size())?;
        Ok(())
    }
}
