use super::{CosetDomain, FftError, PastaField, PrimeModulus, check_domain_size, check_length};

/// Lengths of independently optional prepared tables for one domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TableRequirements {
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
    /// A size-`n` domain needs `n/2` entries in each field table.
    /// For a singleton, the field tables are empty.
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
            twiddles: size / 2,
            inverse_scales: size / 2,
        }
    }
}

/// Independently optional immutable tables borrowed by a [`super::Plan`].
///
/// [`Self::bind`] checks lengths and mathematical contents once, returning a
/// handle that retains the domain and immutable borrows. [`Self::bind_trusted`]
/// skips content checks when the caller establishes correctness elsewhere.
///
/// All field entries must be reduced Montgomery residues for correct arithmetic.
/// Entries can use downstream Bento POD storage; this descriptor contains
/// references and is not an artifact format. Omitted tables use computed powers
/// and multiplicative recurrences. The default omits every table.
///
/// In the entry formulas below, `size`, `root`, and `shift` come from the bound
/// [`CosetDomain`]. Slice lengths are given by [`TableRequirements::for_domain`].
/// Only `inverse_finish` and `inverse_scales` depend on the shift; the other
/// tables can be reused for cosets of the same subgroup.
#[derive(Clone, Copy)]
pub struct Tables<'a, M: PrimeModulus> {
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
            forward: None,
            inverse: None,
            inverse_finish: None,
            inverse_scales: None,
        }
    }
}

impl<'a, M: PrimeModulus> Tables<'a, M> {
    /// Checks all supplied entries and binds the table set to its coset.
    ///
    /// Performs the linear content checks of [`Self::validate`]. Shift-dependent
    /// finish tables remain attached to this domain through plan construction.
    /// Returns [`FftError::LengthMismatch`] for a wrong length or
    /// [`FftError::InvalidTables`] for an incorrect or unreduced entry.
    pub fn bind(self, domain: CosetDomain<M>) -> Result<BoundTables<'a, M>, FftError> {
        self.validate(domain)?;
        Ok(BoundTables {
            domain,
            tables: self,
        })
    }

    /// Binds caller-trusted contents after checking all supplied lengths.
    ///
    /// Length errors follow [`Self::validate`]. The caller must establish that
    /// every entry matches [`Tables`]' formulas for this domain. Incorrect or
    /// unreduced entries can cause wrong results or panics, but not memory
    /// unsafety. Prefer [`Self::bind`] when importing unchecked artifacts.
    pub fn bind_trusted(self, domain: CosetDomain<M>) -> Result<BoundTables<'a, M>, FftError> {
        self.check_shape(domain)?;
        Ok(BoundTables {
            domain,
            tables: self,
        })
    }
    #[cfg(test)]
    pub(super) fn retained_bytes(self) -> Result<usize, FftError> {
        let mut bytes = 0usize;
        for table in [
            self.forward,
            self.inverse,
            self.inverse_finish,
            self.inverse_scales,
        ]
        .into_iter()
        .flatten()
        {
            bytes = bytes
                .checked_add(core::mem::size_of_val(table))
                .ok_or(FftError::SizeOverflow)?;
        }
        Ok(bytes)
    }
    pub(super) fn check_shape(self, domain: CosetDomain<M>) -> Result<(), FftError> {
        let requirements = TableRequirements::for_domain(domain);
        for (buffer, table) in [
            ("forward", self.forward),
            ("inverse", self.inverse),
            ("inverse_finish", self.inverse_finish),
        ] {
            if let Some(table) = table {
                check_length(buffer, requirements.twiddles, table.len())?;
            }
        }
        if let Some(table) = self.inverse_scales {
            check_length("inverse_scales", requirements.inverse_scales, table.len())?;
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
        let generators = generators(domain);
        for (table, (first, step)) in [
            (self.forward, generators.forward),
            (self.inverse, generators.inverse),
            (self.inverse_finish, generators.inverse_finish),
            (self.inverse_scales, generators.inverse_scales),
        ] {
            if let Some(table) = table {
                let mut expected = first;
                for (index, entry) in table.iter().enumerate() {
                    if entry.montgomery_limbs() != expected.montgomery_limbs() {
                        return Err(FftError::InvalidTables);
                    }
                    if index + 1 < table.len() {
                        expected = expected.mul(&step);
                    }
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
            forward: None,
            inverse: None,
            inverse_finish: None,
            inverse_scales: None,
        }
    }
}

impl<'a, M: PrimeModulus> TablesMut<'a, M> {
    /// Fills the supplied tables and returns a handle bound to their coset.
    ///
    /// Entries follow the formulas in [`Tables`]. Returns
    /// [`FftError::LengthMismatch`] without writing any table if a supplied
    /// slice has the wrong length. Omitted destinations remain omitted in the
    /// result; preparing the default descriptor succeeds without writing.
    pub fn prepare(self, domain: CosetDomain<M>) -> Result<BoundTables<'a, M>, FftError> {
        Tables {
            forward: self.forward.as_deref(),
            inverse: self.inverse.as_deref(),
            inverse_finish: self.inverse_finish.as_deref(),
            inverse_scales: self.inverse_scales.as_deref(),
        }
        .check_shape(domain)?;
        fn fill<M: PrimeModulus>(
            table: Option<&mut [PastaField<M>]>,
            (first, step): (PastaField<M>, PastaField<M>),
        ) -> Option<&[PastaField<M>]> {
            table.map(|table| {
                let mut value = first;
                let len = table.len();
                for (index, entry) in table.iter_mut().enumerate() {
                    *entry = value;
                    if index + 1 < len {
                        value = value.mul(&step);
                    }
                }
                &*table
            })
        }
        let generators = generators(domain);
        Ok(BoundTables {
            domain,
            tables: Tables {
                forward: fill(self.forward, generators.forward),
                inverse: fill(self.inverse, generators.inverse),
                inverse_finish: fill(self.inverse_finish, generators.inverse_finish),
                inverse_scales: fill(self.inverse_scales, generators.inverse_scales),
            },
        })
    }
}

/// Optional transform tables bound to their coset domain.
///
/// Native preparation and checked binding establish the formulas in [`Tables`].
/// [`Tables::bind_trusted`] instead relies on the caller for correct contents.
/// The immutable borrow retains these properties for every plan execution.
#[derive(Clone, Copy)]
pub struct BoundTables<'a, M: PrimeModulus> {
    domain: CosetDomain<M>,
    tables: Tables<'a, M>,
}

impl<M: PrimeModulus> core::fmt::Debug for BoundTables<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("BoundTables")
            .field("domain", &self.domain)
            .field("tables", &self.tables)
            .finish()
    }
}

impl<'a, M: PrimeModulus> BoundTables<'a, M> {
    /// Reuses these tables on another coset of the same subgroup.
    ///
    /// Ordinary forward and inverse twiddles retain their borrows without
    /// revalidation; any caller obligations from [`Tables::bind_trusted`] remain.
    /// If the shift changes, inverse-finish and inverse-scaling tables are
    /// omitted because their entries depend on that shift. An unchanged domain
    /// retains every table. This takes constant work without scanning entries.
    ///
    /// Returns [`FftError::InvalidTables`] if the subgroup sizes differ.
    pub fn for_coset(mut self, domain: CosetDomain<M>) -> Result<Self, FftError> {
        if self.domain.size() != domain.size() {
            return Err(FftError::InvalidTables);
        }
        if !self.domain.same_domain(domain) {
            self.tables.inverse_finish = None;
            self.tables.inverse_scales = None;
        }
        self.domain = domain;
        Ok(self)
    }

    /// Rechecks every supplied entry against the bound domain.
    ///
    /// Content checks and errors follow [`Tables::validate`].
    pub fn validate(self) -> Result<Self, FftError> {
        self.tables.validate(self.domain)?;
        Ok(self)
    }
    /// Constructs the plan for this table set's domain without rebinding it.
    pub const fn plan(self) -> super::Plan<'a, M> {
        super::Plan::new(self)
    }
    /// Domain shared by the borrowed tables and their transform plan.
    pub const fn domain(self) -> CosetDomain<M> {
        self.domain
    }
    pub(super) const fn tables(self) -> Tables<'a, M> {
        self.tables
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
