use super::{
    CosetDomain, FftError, PastaField, PrimeModulus, check_domain_size, check_len, reverse,
};

/// Lengths of independently optional prepared tables for one domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TableRequirements {
    /// Number of `u32` entries in the bit-reversal permutation.
    pub permutation: usize,
    /// Number of field elements in each forward, inverse, or inverse-finish table.
    pub twiddles: usize,
    /// Number of field elements in the inverse scaling table.
    pub inverse_scales: usize,
}

impl TableRequirements {
    /// Returns table lengths from a domain size, without constructing a domain.
    ///
    /// This const query applies to both Pasta fields and any coset shift.
    /// Size limits and errors are those of [`super::Domain::for_size`].
    /// Callers may omit any table entirely.
    ///
    /// A size-`n` domain needs `n` permutation entries and `n/2` entries in
    /// each field table. For a singleton, the field tables are empty.
    pub const fn for_size(size: usize) -> Result<Self, FftError> {
        if let Err(error) = check_domain_size(size) {
            return Err(error);
        }
        Ok(Self::for_valid_size(size))
    }

    /// Returns the same lengths as [`Self::for_size`] for a validated domain.
    pub const fn for_domain<M: PrimeModulus>(domain: CosetDomain<M>) -> Self {
        Self::for_valid_size(domain.size())
    }

    const fn for_valid_size(size: usize) -> Self {
        Self {
            permutation: size,
            twiddles: size / 2,
            inverse_scales: size / 2,
        }
    }
}

/// Independently optional immutable tables borrowed by a [`super::Plan`].
///
/// Contents must belong to the plan's field and domain. Binding checks lengths
/// without regenerating the contents. Use [`Self::validate`] when checking a
/// generator or externally supplied artifact. Incorrect contents can produce
/// incorrect answers or panics, but do not compromise memory safety.
///
/// All field entries are reduced Montgomery residues, suitable for downstream
/// Bento POD storage. This descriptor contains references and is not an artifact
/// format. Omitted tables use arithmetic progressions or index bit reversal.
/// The default omits every table.
///
/// In the entry formulas below, `size`, `root`, and `shift` come from the bound
/// [`CosetDomain`]. Slice lengths are given by [`TableRequirements::for_domain`].
/// Only `inverse_finish` and `inverse_scales` depend on the shift; the other
/// tables can be reused for cosets of the same subgroup.
#[derive(Clone, Copy)]
pub struct Tables<'a, M: PrimeModulus> {
    /// Entry `i` is the reversal of the low `log2(size)` bits of `i`.
    pub bit_reversed: Option<&'a [u32]>,
    /// Entry `i` is `root^i`, for `i < size/2`.
    pub forward: Option<&'a [PastaField<M>]>,
    /// Entry `i` is `root^(-i)`, for `i < size/2`.
    pub inverse: Option<&'a [PastaField<M>]>,
    /// Entry `i` is `size^(-1) * shift^(-i) * root^(-i)`.
    pub inverse_finish: Option<&'a [PastaField<M>]>,
    /// Entry `i` is `size^(-1) * shift^(-i)`.
    pub inverse_scales: Option<&'a [PastaField<M>]>,
}

impl<M: PrimeModulus> core::fmt::Debug for Tables<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("Tables")
            .field("bit_reversed", &self.bit_reversed)
            .field("forward", &self.forward)
            .field("inverse", &self.inverse)
            .field("inverse_finish", &self.inverse_finish)
            .field("inverse_scales", &self.inverse_scales)
            .finish()
    }
}

impl<M: PrimeModulus> Default for Tables<'_, M> {
    fn default() -> Self {
        Self {
            bit_reversed: None,
            forward: None,
            inverse: None,
            inverse_finish: None,
            inverse_scales: None,
        }
    }
}

impl<M: PrimeModulus> Tables<'_, M> {
    pub(super) fn check_shape(self, domain: CosetDomain<M>) -> Result<(), FftError> {
        let requirements = TableRequirements::for_domain(domain);
        if let Some(table) = self.bit_reversed {
            check_len("bit_reversed", table.len(), requirements.permutation)?;
        }
        for (buffer, table) in [
            ("forward", self.forward),
            ("inverse", self.inverse),
            ("inverse_finish", self.inverse_finish),
        ] {
            if let Some(table) = table {
                check_len(buffer, table.len(), requirements.twiddles)?;
            }
        }
        if let Some(table) = self.inverse_scales {
            check_len("inverse_scales", table.len(), requirements.inverse_scales)?;
        }
        Ok(())
    }

    /// Checks lengths and every supplied entry against its mathematical value.
    ///
    /// This performs linear work without allocation. It also rejects unreduced
    /// Montgomery field entries, comparing stored limbs to generated canonical
    /// values without performing arithmetic on the supplied entries.
    /// Returns [`FftError::LengthMismatch`] for a wrong length or
    /// [`FftError::InvalidTables`] for a wrong entry. Omitted tables need no
    /// validation and are accepted.
    pub fn validate(self, domain: CosetDomain<M>) -> Result<(), FftError> {
        self.check_shape(domain)?;
        if let Some(indices) = self.bit_reversed {
            for (index, &entry) in indices.iter().enumerate() {
                if entry as usize != reverse(index, domain.domain().log_size()) {
                    return Err(FftError::InvalidTables);
                }
            }
        }
        let generators = generators(domain);
        for (table, (first, step)) in [
            (self.forward, generators.forward),
            (self.inverse, generators.inverse),
            (self.inverse_finish, generators.inverse_finish),
            (self.inverse_scales, generators.inverse_scales),
        ] {
            if let Some(table) = table {
                let mut expected = first;
                for entry in table {
                    if entry.montgomery_limbs() != expected.montgomery_limbs() {
                        return Err(FftError::InvalidTables);
                    }
                    expected = expected.mul(&step);
                }
            }
        }
        Ok(())
    }
}

/// Caller-provided destinations for preparing any subset of [`Tables`].
///
/// All supplied slices must have the exact lengths from [`TableRequirements`].
/// No table is written until every supplied length has been checked.
pub struct TablesMut<'a, M: PrimeModulus> {
    /// Destination for the bit-reversal permutation.
    pub bit_reversed: Option<&'a mut [u32]>,
    /// Destination for forward twiddles.
    pub forward: Option<&'a mut [PastaField<M>]>,
    /// Destination for inverse twiddles.
    pub inverse: Option<&'a mut [PastaField<M>]>,
    /// Destination for inverse twiddles with fused scaling.
    pub inverse_finish: Option<&'a mut [PastaField<M>]>,
    /// Destination for inverse coefficient scales.
    pub inverse_scales: Option<&'a mut [PastaField<M>]>,
}

impl<M: PrimeModulus> core::fmt::Debug for TablesMut<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TablesMut")
            .field("bit_reversed", &self.bit_reversed)
            .field("forward", &self.forward)
            .field("inverse", &self.inverse)
            .field("inverse_finish", &self.inverse_finish)
            .field("inverse_scales", &self.inverse_scales)
            .finish()
    }
}

impl<M: PrimeModulus> Default for TablesMut<'_, M> {
    fn default() -> Self {
        Self {
            bit_reversed: None,
            forward: None,
            inverse: None,
            inverse_finish: None,
            inverse_scales: None,
        }
    }
}

impl<'a, M: PrimeModulus> TablesMut<'a, M> {
    /// Fills the supplied tables and returns immutable borrows of their storage.
    ///
    /// Entries follow the formulas in [`Tables`]. Returns
    /// [`FftError::LengthMismatch`] without writing any table if a supplied
    /// slice has the wrong length. Omitted destinations remain omitted in the
    /// result; preparing the default descriptor succeeds without writing.
    pub fn prepare(self, domain: CosetDomain<M>) -> Result<Tables<'a, M>, FftError> {
        Tables {
            bit_reversed: self.bit_reversed.as_deref(),
            forward: self.forward.as_deref(),
            inverse: self.inverse.as_deref(),
            inverse_finish: self.inverse_finish.as_deref(),
            inverse_scales: self.inverse_scales.as_deref(),
        }
        .check_shape(domain)?;
        let indices = self.bit_reversed.map(|indices| {
            for (index, entry) in indices.iter_mut().enumerate() {
                *entry = reverse(index, domain.domain().log_size()) as u32;
            }
            &*indices
        });
        fn fill<M: PrimeModulus>(
            table: Option<&mut [PastaField<M>]>,
            (first, step): (PastaField<M>, PastaField<M>),
        ) -> Option<&[PastaField<M>]> {
            table.map(|table| {
                let mut value = first;
                for entry in table.iter_mut() {
                    *entry = value;
                    value = value.mul(&step);
                }
                &*table
            })
        }
        let generators = generators(domain);
        Ok(Tables {
            bit_reversed: indices,
            forward: fill(self.forward, generators.forward),
            inverse: fill(self.inverse, generators.inverse),
            inverse_finish: fill(self.inverse_finish, generators.inverse_finish),
            inverse_scales: fill(self.inverse_scales, generators.inverse_scales),
        })
    }
}

struct Generators<M: PrimeModulus> {
    forward: (PastaField<M>, PastaField<M>),
    inverse: (PastaField<M>, PastaField<M>),
    inverse_finish: (PastaField<M>, PastaField<M>),
    inverse_scales: (PastaField<M>, PastaField<M>),
}

fn generators<M: PrimeModulus>(domain: CosetDomain<M>) -> Generators<M> {
    let subgroup = domain.domain();
    Generators {
        forward: (PastaField::ONE, subgroup.root()),
        inverse: (PastaField::ONE, subgroup.inverse_root()),
        inverse_finish: (
            subgroup.size_inverse(),
            subgroup.inverse_root().mul(&domain.inverse_shift()),
        ),
        inverse_scales: (subgroup.size_inverse(), domain.inverse_shift()),
    }
}
