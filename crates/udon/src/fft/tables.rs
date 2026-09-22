use super::{CosetDomain, FftError, PastaField, PrimeModulus, Transform, check_domain_size};

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

/// Independently optional immutable tables borrowed by a [`super::Transform`].
///
/// [`Self::bind`] borrows trusted contents after checking lengths, retaining the
/// domain in the handle. Entries may use either representative of a loose field
/// value. Bento POD storage preserves those representations for direct use.
/// This descriptor contains references and is not an artifact format. Omitted
/// tables use computed powers and multiplicative recurrences. The default omits
/// every table.
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
    /// Borrows trusted tables for this coset without inspecting entries.
    ///
    /// Each supplied table must match the formulas in [`Tables`] for `domain`.
    /// Panics if its length differs from [`TableRequirements::for_domain`].
    pub const fn bind(self, domain: CosetDomain<M>) -> Transform<'a, M> {
        self.assert_shape(domain);
        Transform {
            domain,
            tables: self,
        }
    }

    const fn assert_shape(self, domain: CosetDomain<M>) {
        let requirements = TableRequirements::for_domain(domain);
        let tables = [self.forward, self.inverse, self.inverse_finish];
        let mut index = 0;
        while index < tables.len() {
            if let Some(table) = tables[index] {
                assert!(
                    table.len() == requirements.twiddles,
                    "twiddle table length mismatch"
                );
            }
            index += 1;
        }
        if let Some(table) = self.inverse_scales {
            assert!(
                table.len() == requirements.inverse_scales,
                "inverse scale table length mismatch"
            );
        }
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
    /// Entries follow the formulas in [`Tables`]. Panics before writes if the
    /// destinations violate [`TablesMut`]'s contract. Omitted destinations remain
    /// omitted in the result; preparing the default descriptor succeeds without
    /// writing.
    pub fn prepare(self, domain: CosetDomain<M>) -> Transform<'a, M> {
        Tables {
            forward: self.forward.as_deref(),
            inverse: self.inverse.as_deref(),
            inverse_finish: self.inverse_finish.as_deref(),
            inverse_scales: self.inverse_scales.as_deref(),
        }
        .assert_shape(domain);
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
        Transform {
            domain,
            tables: Tables {
                forward: fill(self.forward, generators.forward),
                inverse: fill(self.inverse, generators.inverse),
                inverse_finish: fill(self.inverse_finish, generators.inverse_finish),
                inverse_scales: fill(self.inverse_scales, generators.inverse_scales),
            },
        }
    }
}

impl<'a, M: PrimeModulus> super::Transform<'a, M> {
    /// Reuses these tables on another coset of the same subgroup.
    ///
    /// Ordinary forward and inverse twiddles retain their borrows without
    /// inspecting their entries.
    /// If the shift changes, inverse-finish and inverse-scaling tables are
    /// omitted because their entries depend on that shift. An unchanged domain
    /// retains every table. This takes constant work without scanning entries.
    ///
    /// Panics if the cosets have different subgroups.
    pub fn for_coset(mut self, domain: CosetDomain<M>) -> Self {
        assert_eq!(
            self.domain.size(),
            domain.size(),
            "cosets must share a subgroup"
        );
        if !self.domain.same_domain(domain) {
            self.tables.inverse_finish = None;
            self.tables.inverse_scales = None;
        }
        self.domain = domain;
        self
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
