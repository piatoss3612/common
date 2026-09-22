use super::{
    FftError, PastaField, PrimeModulus, assert_length, check_domain_size, check_field_count,
    validate_length,
};

/// Retained representation of subgroup powers, independent of value ordering.
///
/// Let `n` be [`TwiddleDescription::size`] and `w` its canonical root, inverted
/// when [`TwiddleDescription::inverse`] is true. Size one needs no entries.
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
    /// Whether generation uses the inverse canonical root.
    pub inverse: bool,
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
    // stages. This lets preparation and validation avoid per-entry powers.
    // Callers have already checked the description's size.
    fn stages<M: PrimeModulus>(self) -> impl Iterator<Item = (usize, PastaField<M>)> {
        let last = self.size.ilog2();
        let first = match self.storage {
            TwiddleStorage::Dense => last,
            TwiddleStorage::StagePacked => 1,
        };
        (first..=last).map(move |log| {
            let root = if self.inverse {
                PastaField::root_of_unity_inverse(log)
            } else {
                PastaField::root_of_unity(log)
            }
            .unwrap();
            ((1usize << log) / 2, root)
        })
    }
}

/// Borrowed, domain-described subgroup twiddles, independent of coset shifts.
///
/// A smaller table serves local stages of a larger transform; larger stages
/// generate their powers as needed. A larger table serves smaller transforms
/// using the nested canonical roots. Either table direction can serve forward
/// and inverse transforms. Native preparation and [`Self::bind`] establish the
/// formulas in [`TwiddleStorage`]. [`Self::bind_trusted`] relies on the caller
/// for correct contents instead.
#[derive(Clone, Copy)]
pub struct TwiddleTable<'a, M: PrimeModulus> {
    description: TwiddleDescription,
    values: &'a [PastaField<M>],
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
    /// Checks every entry and binds its subgroup, direction, and storage order.
    ///
    /// Configuration errors follow [`TwiddleDescription::requirements`]. A wrong stored
    /// length returns [`FftError::LengthMismatch`]; incorrect or unreduced entries
    /// return [`FftError::InvalidTables`]. Validation takes linear field work.
    pub fn bind(
        description: TwiddleDescription,
        values: &'a [PastaField<M>],
    ) -> Result<Self, FftError> {
        validate_length("twiddles", description.requirements()?, values.len())?;
        Self::bind_trusted(description, values)?.validate()
    }

    /// Binds caller-trusted entries after checking dimensions and length.
    ///
    /// Configuration errors follow [`TwiddleDescription::requirements`]. Panics unless
    /// `values` has the reported length. The caller must establish the reduced
    /// Montgomery entries described by [`TwiddleStorage`]. Incorrect contents can cause
    /// wrong results or panics, but not memory unsafety.
    pub fn bind_trusted(
        description: TwiddleDescription,
        values: &'a [PastaField<M>],
    ) -> Result<Self, FftError> {
        assert_length("twiddles", description.requirements()?, values.len());
        Ok(Self {
            description,
            values,
        })
    }

    /// Generates subgroup twiddles into exactly sized caller storage.
    ///
    /// Entries follow [`TwiddleStorage`]'s formulas and use reduced Montgomery
    /// representations. Storage, error, and panic contracts follow
    /// [`Self::bind_trusted`]. All checks precede writes; initial destination values
    /// are overwritten.
    pub fn prepare(
        description: TwiddleDescription,
        values: &'a mut [PastaField<M>],
    ) -> Result<Self, FftError> {
        assert_length("twiddles", description.requirements()?, values.len());
        let mut remaining = &mut *values;
        for (len, step) in description.stages::<M>() {
            let (stage, rest) = remaining.split_at_mut(len);
            let mut power = PastaField::ONE;
            for value in stage {
                *value = power;
                power = power.mul(&step);
            }
            remaining = rest;
        }
        Ok(Self {
            description,
            values,
        })
    }

    /// Checks every entry against the canonical root, including reduced limbs.
    ///
    /// Returns [`FftError::InvalidTables`] for any incorrect or unreduced entry.
    /// Validation takes linear work without arithmetic on imported entries.
    pub fn validate(self) -> Result<Self, FftError> {
        // Regenerate expected powers from known reduced roots. Comparing limbs
        // rejects unreduced imports without feeding them into field arithmetic.
        let mut remaining = self.values;
        for (len, step) in self.description.stages::<M>() {
            let (stage, rest) = remaining.split_at(len);
            let mut power = PastaField::ONE;
            for value in stage {
                if value.montgomery_limbs() != power.montgomery_limbs() {
                    return Err(FftError::InvalidTables);
                }
                power = power.mul(&step);
            }
            remaining = rest;
        }
        Ok(self)
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

/// A declared power sequence with an explicit starting value and step.
///
/// Entry `i` represents `first * step^i`, with no implicit FFT normalization.
/// Empty sequences and zero starting values or steps are accepted. For example,
/// forward coefficient scales have `first = 1` and the coset shift as `step`.
/// Native preparation and [`Self::bind`] establish reduced fields and matching
/// entries. [`Self::bind_trusted`] relies on the caller for correct entries.
#[derive(Clone, Copy)]
pub struct PowerTable<'a, M: PrimeModulus> {
    first: PastaField<M>,
    step: PastaField<M>,
    values: &'a [PastaField<M>],
}

impl<M: PrimeModulus> core::fmt::Debug for PowerTable<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("PowerTable")
            .field("first", &self.first)
            .field("step", &self.step)
            .field("values", &self.values)
            .finish()
    }
}

impl<'a, M: PrimeModulus> PowerTable<'a, M> {
    /// Checks a sequence against its declared starting value and step.
    ///
    /// Returns [`FftError::InvalidTables`] for unreduced seeds or any incorrect
    /// or unreduced entry. Any length, including zero, is accepted. Validation
    /// performs linear work without arithmetic on imported entries.
    pub fn bind(
        first: PastaField<M>,
        step: PastaField<M>,
        values: &'a [PastaField<M>],
    ) -> Result<Self, FftError> {
        Self {
            first,
            step,
            values,
        }
        .validate()
    }

    /// Binds caller-trusted entries after checking that both seeds are reduced.
    ///
    /// Returns [`FftError::InvalidTables`] for an unreduced seed, even for an
    /// empty sequence. The caller must establish that entry `i` is the reduced
    /// Montgomery representation of `first * step^i`. Incorrect contents can
    /// cause wrong results or panics, but not memory unsafety.
    pub fn bind_trusted(
        first: PastaField<M>,
        step: PastaField<M>,
        values: &'a [PastaField<M>],
    ) -> Result<Self, FftError> {
        Self::check_seeds(first, step)?;
        Ok(Self {
            first,
            step,
            values,
        })
    }

    fn check_seeds(first: PastaField<M>, step: PastaField<M>) -> Result<(), FftError> {
        if !first.is_reduced() || !step.is_reduced() {
            return Err(FftError::InvalidTables);
        }
        Ok(())
    }

    /// Writes `first * step^i` into caller storage and returns its handle.
    ///
    /// Here `i` is the entry index. Any storage length, including zero, is accepted.
    /// Returns [`FftError::InvalidTables`] before writing for an unreduced seed.
    pub fn prepare(
        first: PastaField<M>,
        step: PastaField<M>,
        values: &'a mut [PastaField<M>],
    ) -> Result<Self, FftError> {
        Self::check_seeds(first, step)?;
        let mut power = first;
        for value in values.iter_mut() {
            *value = power;
            power = power.mul(&step);
        }
        Ok(Self {
            first,
            step,
            values,
        })
    }

    /// Checks all entries, with the content checks and errors of [`Self::bind`].
    pub fn validate(self) -> Result<Self, FftError> {
        Self::check_seeds(self.first, self.step)?;
        let mut power = self.first;
        for value in self.values {
            if value.montgomery_limbs() != power.montgomery_limbs() {
                return Err(FftError::InvalidTables);
            }
            power = power.mul(&self.step);
        }
        Ok(self)
    }
    /// Starting value, including any deliberate scale.
    pub const fn first(self) -> PastaField<M> {
        self.first
    }
    /// Multiplicative step between entries.
    pub const fn step(self) -> PastaField<M> {
        self.step
    }
    /// Retained sequence entries.
    pub const fn as_slice(self) -> &'a [PastaField<M>] {
        self.values
    }
}
