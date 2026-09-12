use super::{
    Domain, FftError, PastaField, PrimeModulus, check_domain_size, check_field_count, check_len,
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
    /// Powers `w^(i*chunk_len)` for `0 <= i < ceil((n/2)/chunk_len)`.
    ChunkSeeds {
        /// Positive power-of-two distance between retained seeds.
        chunk_len: usize,
    },
    /// Low powers followed by powers at multiples of `low_len`.
    ///
    /// Retain `w^i` for `0 <= i < min(low_len, n/2)`, then `w^(i*low_len)`
    /// for `0 <= i < ceil((n/2)/low_len)`. A lookup reconstructs a power with
    /// one multiplication.
    Factored {
        /// Positive power-of-two number of consecutive low powers.
        low_len: usize,
    },
}

/// Semantic description shared by const sizing, preparation, and imported data.
///
/// `size` selects the canonical root from [`Domain`]. [`TwiddleStorage`] defines
/// the entry formulas and ordering.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TwiddleDescription {
    /// Order of the canonical root used to generate the table.
    pub size: usize,
    /// Whether generation uses the inverse canonical root.
    pub inverse: bool,
    /// Arrangement and reconstruction policy.
    pub storage: TwiddleStorage,
}

impl TwiddleDescription {
    /// Exact number of retained field elements, without preparing a domain.
    ///
    /// Size validity and address-space limits follow [`Domain::for_size`].
    /// Returns [`FftError::InvalidExecution`] unless `chunk_len` or `low_len`,
    /// when present, is a positive power of two. These lengths may exceed the
    /// table size. Storage overflow returns [`FftError::SizeOverflow`].
    pub const fn requirements(self) -> Result<usize, FftError> {
        if let Err(error) = check_domain_size(self.size) {
            return Err(error);
        }
        let half = self.size / 2;
        let count = match self.storage {
            TwiddleStorage::Dense => half,
            TwiddleStorage::StagePacked => self.size - 1,
            TwiddleStorage::ChunkSeeds { chunk_len } => {
                if !chunk_len.is_power_of_two() {
                    return Err(FftError::InvalidExecution);
                }
                half.div_ceil(chunk_len)
            }
            TwiddleStorage::Factored { low_len } => {
                if !low_len.is_power_of_two() {
                    return Err(FftError::InvalidExecution);
                }
                if half == 0 {
                    0
                } else {
                    super::min(low_len, half) + half.div_ceil(low_len)
                }
            }
        };
        check_field_count(count)
    }

    fn exponent(self, index: usize) -> usize {
        match self.storage {
            TwiddleStorage::Dense => index,
            TwiddleStorage::StagePacked => {
                let half = 1 << (index + 1).ilog2();
                (index - (half - 1)) * (self.size / (2 * half))
            }
            TwiddleStorage::ChunkSeeds { chunk_len } => index * chunk_len,
            TwiddleStorage::Factored { low_len } => {
                let low = low_len.min(self.size / 2);
                if index < low {
                    index
                } else {
                    (index - low) * low_len
                }
            }
        }
    }
}

/// Borrowed, domain-described subgroup twiddles, independent of coset shifts.
///
/// A smaller table serves local stages of a larger transform; larger stages
/// generate their powers as needed. A larger table serves smaller transforms
/// using the nested canonical roots. Either table direction can serve forward
/// and inverse transforms. Bindings check shape; correct arithmetic requires
/// entries matching [`TwiddleStorage`], checked by [`Self::validate`]. Incorrect
/// contents can cause wrong results or panics, as described in the [module](super).
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
    /// Binds table dimensions without inspecting entries.
    ///
    /// Errors follow [`TwiddleDescription::requirements`], with
    /// [`FftError::LengthMismatch`] if `values` has a different length.
    /// Use [`Self::validate`] to check imported contents.
    pub fn bind(
        description: TwiddleDescription,
        values: &'a [PastaField<M>],
    ) -> Result<Self, FftError> {
        check_len("twiddles", values.len(), description.requirements()?)?;
        Ok(Self {
            description,
            values,
        })
    }

    /// Generates canonical Montgomery entries into exactly sized caller storage.
    ///
    /// Dimensions and errors follow [`Self::bind`]. All checks precede writes.
    pub fn prepare(
        description: TwiddleDescription,
        values: &'a mut [PastaField<M>],
    ) -> Result<Self, FftError> {
        check_len("twiddles", values.len(), description.requirements()?)?;
        let domain = Domain::<M>::for_size(description.size)?;
        let root = if description.inverse {
            domain.inverse_root()
        } else {
            domain.root()
        };
        // Each representation consists of short power progressions. Restart
        // only at progression boundaries, including packed stage boundaries.
        let mut previous = None;
        let mut power = PastaField::ONE;
        let mut previous_step = None;
        let mut step = PastaField::ONE;
        for (index, value) in values.iter_mut().enumerate() {
            let exponent = description.exponent(index);
            if let Some(prior) = previous
                && exponent >= prior
            {
                let difference = exponent - prior;
                if previous_step != Some(difference) {
                    step = root.pow_u64(difference as u64);
                    previous_step = Some(difference);
                }
                power = power.mul(&step);
            } else {
                power = root.pow_u64(exponent as u64);
            }
            *value = power;
            previous = Some(exponent);
        }
        Self::bind(description, values)
    }

    /// Checks every entry against the canonical root, including reduced limbs.
    ///
    /// Returns [`FftError::InvalidTables`] for any incorrect entry.
    pub fn validate(self) -> Result<Self, FftError> {
        let domain = Domain::<M>::for_size(self.description.size)?;
        let root = if self.description.inverse {
            domain.inverse_root()
        } else {
            domain.root()
        };
        for (index, value) in self.values.iter().enumerate() {
            let expected = root.pow_u64(self.description.exponent(index) as u64);
            if value.montgomery_limbs() != expected.montgomery_limbs() {
                return Err(FftError::InvalidTables);
            }
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

    pub(super) fn power(self, block: usize, index: usize, inverse: bool) -> PastaField<M> {
        debug_assert!(block <= self.description.size);
        let half = block / 2;
        if index == 0 {
            return PastaField::ONE;
        }
        // For the order-block root w, w^(-i) = -w^(block/2-i) when
        // 0 < i < block/2. Zero was handled above, so either table direction
        // can supply the other without storing a second table.
        let conjugate = inverse != self.description.inverse;
        let index = if conjugate { half - index } else { index };
        let exponent = index * (self.description.size / block);
        let power = match self.description.storage {
            TwiddleStorage::Dense => self.values[exponent],
            TwiddleStorage::StagePacked => self.values[half - 1 + index],
            TwiddleStorage::Factored { low_len } => self.values[exponent % low_len]
                .mul(&self.values[low_len.min(self.description.size / 2) + exponent / low_len]),
            TwiddleStorage::ChunkSeeds { chunk_len } => {
                let root = if self.description.inverse {
                    PastaField::root_of_unity_inverse(self.description.size.ilog2()).unwrap()
                } else {
                    PastaField::root_of_unity(self.description.size.ilog2()).unwrap()
                };
                self.values[exponent / chunk_len].mul(&root.pow_u64((exponent % chunk_len) as u64))
            }
        };
        if conjugate { power.neg() } else { power }
    }
}

/// A declared power sequence with an explicit starting value and step.
///
/// Entry `i` represents `first * step^i`, with no implicit FFT normalization.
/// Empty sequences and zero starting values or steps are accepted. For example,
/// forward coefficient scales have `first = 1` and the coset shift as `step`.
/// Correct arithmetic requires reduced fields and matching entries; binding
/// does not establish these properties. See [`Self::validate`] and the
/// module's [field representation contract](super).
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
    /// Binds declared sequence semantics without inspecting the entries.
    pub const fn bind(
        first: PastaField<M>,
        step: PastaField<M>,
        values: &'a [PastaField<M>],
    ) -> Self {
        Self {
            first,
            step,
            values,
        }
    }
    /// Writes `first * step^i` into caller storage and returns its descriptor.
    ///
    /// Here `i` is the entry index. Any storage length, including zero, is accepted.
    /// `first` and `step` must satisfy [`PastaField`]'s reduced representation
    /// contract for correct arithmetic.
    pub fn prepare(
        first: PastaField<M>,
        step: PastaField<M>,
        values: &'a mut [PastaField<M>],
    ) -> Self {
        let mut power = first;
        for value in values.iter_mut() {
            *value = power;
            power = power.mul(&step);
        }
        Self::bind(first, step, values)
    }
    /// Checks all entries, rejecting incorrect or unreduced representations.
    ///
    /// Returns [`FftError::InvalidTables`] if `first`, `step`, or any entry is
    /// unreduced, or if any entry differs from the declared sequence.
    pub fn validate(self) -> Result<Self, FftError> {
        let mut power = self.first;
        for value in [self.first, self.step] {
            if !value
                .montgomery_limbs()
                .iter()
                .rev()
                .cmp(M::MODULUS.iter().rev())
                .is_lt()
            {
                return Err(FftError::InvalidTables);
            }
        }
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

/// Semantic metadata for a downstream subgroup-table artifact.
///
/// This is a Rust descriptor, not a byte format or an integrity check. The owner
/// chooses serialization, filenames, and checksums. Compare this descriptor and
/// validate imported table contents separately from Bento's target layout checks.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TwiddleArtifact {
    /// Version of these semantics (currently one).
    pub version: u32,
    /// Field identity as its canonical modulus limbs, least significant first.
    pub modulus: [u64; 4],
    /// Montgomery radix exponent: a field value `x` is stored as
    /// `x * 2^montgomery_bits mod p`, where `p` is [`Self::modulus`].
    pub montgomery_bits: u32,
    /// Root orientation, dimensions, and table layout. Twiddles have no coset
    /// shift or inverse-size normalization.
    pub twiddles: TwiddleDescription,
}

impl TwiddleArtifact {
    /// Metadata to emit alongside a table of field `M`.
    ///
    /// This does not check dimensions; use [`Self::validate`] before importing it.
    pub const fn for_field<M: PrimeModulus>(twiddles: TwiddleDescription) -> Self {
        Self {
            version: 1,
            modulus: M::MODULUS,
            montgomery_bits: 256,
            twiddles,
        }
    }
    /// Checks the version, field representation, and declared table dimensions.
    ///
    /// Returns [`FftError::InvalidTables`] for a version, modulus, or Montgomery
    /// radix mismatch. Dimension errors follow [`TwiddleDescription::requirements`].
    /// This does not compare the description with an intended operation or inspect
    /// entries; compare metadata as needed and use [`TwiddleTable::validate`].
    pub fn validate<M: PrimeModulus>(self) -> Result<(), FftError> {
        if self != Self::for_field::<M>(self.twiddles) {
            return Err(FftError::InvalidTables);
        }
        self.twiddles.requirements().map(|_| ())
    }
}
