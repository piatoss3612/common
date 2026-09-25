use super::{
    FftError, PastaField, PrimeModulus, assert_length, check_domain_size, check_field_count,
};

use crate::field::fill_powers;

/// Retained representation of subgroup powers, independent of value ordering.
///
/// Let `n` be [`TwiddleDescription::size`] and `w` its canonical root.
/// Size one needs no entries.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum TwiddleStorage {
    /// Powers `w^i` for `0 <= i < n/2`.
    Dense,
    /// Stage powers packed in ascending stage order.
    ///
    /// For each stage length `b = 2, 4, ..., n`, retain `w^(i*n/b)` for
    /// `0 <= i < b/2`. A smaller table retains only local stages of a transform.
    StagePacked,
}

/// Semantic description shared by const sizing, preparation, and imported data.
///
/// `size` selects the canonical root from [`Domain`](super::Domain).
/// [`TwiddleStorage`] defines the entry formulas and ordering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TwiddleDescription {
    /// Order of the canonical root used to generate the table.
    pub size: usize,
    /// Arrangement of retained powers.
    pub storage: TwiddleStorage,
}

impl TwiddleDescription {
    /// Exact number of retained field elements, without preparing a domain.
    ///
    /// Size validity and address-space limits follow [`super::Domain::for_size`].
    /// Storage overflow returns [`FftError::SizeOverflow`].
    pub const fn requirements(self) -> Result<usize, FftError> {
        if let Err(error) = check_domain_size(self.size) {
            return Err(error);
        }
        let half = self.size / 2;
        let count = match self.storage {
            TwiddleStorage::Dense => half,
            TwiddleStorage::StagePacked => self.size - 1,
        };
        check_field_count(count)
    }

    // Each sequence starts at one and advances by its canonical stage root.
    // Dense storage has only the final stage; packed storage concatenates all
    // stages. This lets preparation avoid per-entry powers.
    // Callers have already checked the description's size.
    fn stages<M: PrimeModulus>(self) -> impl Iterator<Item = (usize, PastaField<M>)> {
        let last = self.size.ilog2();
        let first = match self.storage {
            TwiddleStorage::Dense => last.max(1),
            TwiddleStorage::StagePacked => 1,
        };
        (first..=last).map(move |log| {
            let root = PastaField::root_of_unity(log).unwrap();
            ((1usize << log) / 2, root)
        })
    }
}

/// Borrowed, domain-described subgroup twiddles, independent of coset shifts.
///
/// A smaller table serves local stages of a larger transform; larger stages
/// generate their powers as needed. A larger table serves smaller transforms
/// using the nested canonical roots. Each table serves both forward and inverse
/// transforms. Preparation constructs the formulas in [`TwiddleStorage`];
/// [`Self::bind`] borrows trusted stored entries.
#[derive(Clone, Copy)]
pub struct TwiddleTable<'a, M: PrimeModulus> {
    description: TwiddleDescription,
    values: &'a [PastaField<M>],
    // Keep the stored orientation so kernels can borrow ordinary inverse tables
    // and conjugate on lookup without copying their entries.
    inverse: bool,
}

impl<M: PrimeModulus> core::fmt::Debug for TwiddleTable<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TwiddleTable")
            .field("description", &self.description)
            .field("values", &self.values)
            .finish()
    }
}

impl<'a, M: PrimeModulus> TwiddleTable<'a, M> {
    /// Borrows trusted canonical-root entries for their subgroup and storage order.
    ///
    /// Entries must follow [`TwiddleStorage`]'s formulas. Configuration errors
    /// follow [`TwiddleDescription::requirements`]. Panics unless `values` has
    /// the reported length. Binding does not inspect the entries.
    pub const fn bind(
        description: TwiddleDescription,
        values: &'a [PastaField<M>],
    ) -> Result<Self, FftError> {
        let required = match description.requirements() {
            Ok(required) => required,
            Err(error) => return Err(error),
        };
        assert!(values.len() == required, "twiddle table length mismatch");
        Ok(Self {
            description,
            values,
            inverse: false,
        })
    }

    /// Generates subgroup twiddles into exactly sized caller storage.
    ///
    /// Entries follow [`TwiddleStorage`]'s formulas. Storage, error, and panic
    /// contracts follow [`Self::bind`]. All checks precede writes; initial
    /// destination values are overwritten.
    pub fn prepare(
        description: TwiddleDescription,
        values: &'a mut [PastaField<M>],
    ) -> Result<Self, FftError> {
        assert_length("twiddles", description.requirements()?, values.len());
        let mut remaining = &mut *values;
        for (len, step) in description.stages::<M>() {
            let (stage, rest) = remaining.split_at_mut(len);
            fill_powers(PastaField::ONE, step, stage);
            remaining = rest;
        }
        Ok(Self {
            description,
            values,
            inverse: false,
        })
    }

    // Kernels borrow a transform's already checked table slices in either
    // root orientation without rechecking their lengths.
    pub(super) const fn trusted(
        description: TwiddleDescription,
        values: &'a [PastaField<M>],
        inverse: bool,
    ) -> Self {
        Self {
            description,
            values,
            inverse,
        }
    }

    pub(super) const fn is_inverse(self) -> bool {
        self.inverse
    }

    /// Generation semantics; the field is also fixed by the type parameter.
    pub const fn description(self) -> TwiddleDescription {
        self.description
    }
    /// Retained entries in the described order.
    pub const fn as_slice(self) -> &'a [PastaField<M>] {
        self.values
    }
}
