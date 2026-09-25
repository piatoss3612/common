use super::{
    CosetDomain, Domain, FftError, PastaField, PrimeModulus, assert_length, check_domain_size,
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
/// Preparation constructs the entries described by [`ExpansionScaleNormalization`].
/// [`Self::bind`] borrows trusted stored entries, checking dimensions and length.
#[derive(Clone, Copy)]
pub struct ExpansionScales<'a, M: PrimeModulus> {
    base_size: usize,
    extended: CosetDomain<M>,
    normalization: ExpansionScaleNormalization,
    values: &'a [PastaField<M>],
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
    /// Borrows trusted entries for the domain, base size, and normalization.
    ///
    /// Entries must follow [`ExpansionScaleNormalization`]'s formulas. Errors
    /// follow [`Self::requirements`]. Panics unless the storage has the extended
    /// domain size. Binding does not inspect entries.
    pub const fn bind(
        base_size: usize,
        extended: CosetDomain<M>,
        normalization: ExpansionScaleNormalization,
        values: &'a [PastaField<M>],
    ) -> Result<Self, FftError> {
        let required = match Self::requirements(base_size, extended.size()) {
            Ok(required) => required,
            Err(error) => return Err(error),
        };
        assert!(
            values.len() == required,
            "expansion scale table length mismatch"
        );
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
    /// Entries follow [`ExpansionScaleNormalization`]. Storage, error, and panic
    /// contracts follow [`Self::bind`]. All checks precede writes; initial destination values
    /// are overwritten.
    pub fn prepare(
        base_size: usize,
        extended: CosetDomain<M>,
        normalization: ExpansionScaleNormalization,
        values: &'a mut [PastaField<M>],
    ) -> Result<Self, FftError> {
        assert_length(
            "scales",
            Self::requirements(base_size, extended.size())?,
            values.len(),
        );
        let first = match normalization {
            ExpansionScaleNormalization::Coefficients => PastaField::ONE,
            ExpansionScaleNormalization::UnscaledInverse => {
                Domain::<M>::for_size(base_size)?.size_inverse()
            }
        };
        let mut step = extended.shift();
        let residues = values.len() / base_size;
        for (residue, row) in values.chunks_exact_mut(base_size).enumerate() {
            crate::field::fill_powers(first, step, row);
            if residue + 1 < residues {
                step = step.mul(&extended.domain().root());
            }
        }
        Ok(Self {
            base_size,
            extended,
            normalization,
            values,
        })
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
}
