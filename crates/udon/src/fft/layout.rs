use super::{CosetDomain, FftError, PastaField, PrimeModulus, check_len, reverse};

/// Storage order of a complete evaluation vector.
///
/// For domain size `size`, natural row `j` evaluates the point `shift * root^j`
/// from the [`CosetDomain`], with `0 <= j < size`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum EvaluationLayout {
    /// Natural row `j` is stored at index `j`.
    Natural,
    /// Natural row `j` is stored at its `log2(size)`-bit reversal.
    BitReversed,
    /// Naturally numbered residue blocks, each with natural inner order.
    Residues(ResidueLayout),
}

impl EvaluationLayout {
    pub(super) fn check(self, size: usize) -> Result<(), FftError> {
        if let Self::Residues(layout) = self
            && layout.size() != size
        {
            return Err(FftError::InvalidLayout);
        }
        Ok(())
    }

    /// Maps a natural row to its storage index.
    ///
    /// Returns `None` unless `size` is a positive power of two, `row < size`,
    /// and any contained [`ResidueLayout`] has that same size.
    pub fn index(self, row: usize, size: usize) -> Option<usize> {
        if !size.is_power_of_two() || row >= size || self.check(size).is_err() {
            return None;
        }
        Some(match self {
            Self::Natural => row,
            Self::BitReversed => reverse(row, size.ilog2()),
            Self::Residues(layout) => layout.index(row)?,
        })
    }
}

/// Evaluations bound to their field, canonical root, shift, and storage order.
///
/// Binding is a structural check, not proof that the values evaluate a
/// particular polynomial. Mathematical consumers can compare domains before
/// modifying storage, without relying on equal slice lengths alone.
#[derive(Clone, Copy)]
pub struct EvaluationView<'a, M: PrimeModulus> {
    values: &'a [PastaField<M>],
    domain: CosetDomain<M>,
    layout: EvaluationLayout,
}

impl<M: PrimeModulus> core::fmt::Debug for EvaluationView<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("EvaluationView")
            .field("values", &self.values)
            .field("domain", &self.domain)
            .field("layout", &self.layout)
            .finish()
    }
}

impl<'a, M: PrimeModulus> EvaluationView<'a, M> {
    /// Checks the slice and layout dimensions, without checking field contents.
    ///
    /// Returns [`FftError::LengthMismatch`] unless `values` has the domain size,
    /// or [`FftError::InvalidLayout`] if a residue layout has a different size.
    pub fn bind(
        values: &'a [PastaField<M>],
        domain: CosetDomain<M>,
        layout: EvaluationLayout,
    ) -> Result<Self, FftError> {
        check_len("values", values.len(), domain.size())?;
        layout.check(domain.size())?;
        Ok(Self {
            values,
            domain,
            layout,
        })
    }

    /// The ordered mathematical domain, independent of storage layout.
    pub const fn domain(self) -> CosetDomain<M> {
        self.domain
    }
    /// The storage mapping.
    pub const fn layout(self) -> EvaluationLayout {
        self.layout
    }
    /// The borrowed values in storage order.
    pub const fn as_slice(self) -> &'a [PastaField<M>] {
        self.values
    }
    /// Looks up a natural domain row, returning `None` when it is out of range.
    pub fn get(self, row: usize) -> Option<&'a PastaField<M>> {
        self.values.get(self.layout.index(row, self.domain.size())?)
    }
    /// Looks up a natural row of a domain with the same shift and nested root.
    ///
    /// The supplied domain must be at least this view's size and have the same
    /// shift. Returns `None` if either condition fails, if `row` is out of range,
    /// or if it is not a multiple of the supplied size divided by the view's size.
    pub fn get_extended_row(self, row: usize, domain: CosetDomain<M>) -> Option<&'a PastaField<M>> {
        if domain.size() < self.domain.size()
            || row >= domain.size()
            || domain.shift() != self.domain.shift()
        {
            return None;
        }
        let stride = domain.size() / self.domain.size();
        if !row.is_multiple_of(stride) {
            return None;
        }
        self.get(row / stride)
    }
    /// Writes the pointwise product in the views' shared layout.
    ///
    /// Returns [`FftError::LengthMismatch`] unless `output` has the domain size,
    /// or [`FftError::InvalidLayout`] if the input domains or layouts differ.
    /// Validation precedes mutation. Field inputs follow the module's
    /// [representation contract](super).
    pub fn multiply_into(self, other: Self, output: &mut [PastaField<M>]) -> Result<(), FftError> {
        check_len("output", output.len(), self.domain.size())?;
        if !self.domain.same_domain(other.domain) || self.layout != other.layout {
            return Err(FftError::InvalidLayout);
        }
        for ((out, left), right) in output.iter_mut().zip(self.values).zip(other.values) {
            *out = left.mul(right);
        }
        Ok(())
    }
}

/// Groups evaluations by the remainder of their natural row index.
///
/// For `r` residues in `n` evaluations, natural row `s + r*k` is stored at
/// `s*(n/r) + k`, where `0 <= s < r` and `0 <= k < n/r`. Thus each residue
/// occupies one contiguous slice of `n/r` values. This differs from
/// [`CoefficientTiles`], which preserves natural coefficient order.
///
/// This descriptor checks dimensions only; it carries no field, root, or shift.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ResidueLayout {
    size: usize,
    residues: usize,
}

impl ResidueLayout {
    /// Constructs a layout with nonzero power-of-two dimensions.
    ///
    /// Returns [`FftError::InvalidLayout`] unless both dimensions are nonzero
    /// powers of two and `residues <= size`.
    pub fn new(size: usize, residues: usize) -> Result<Self, FftError> {
        if !size.is_power_of_two() || !residues.is_power_of_two() || residues > size {
            return Err(FftError::InvalidLayout);
        }
        Ok(Self { size, residues })
    }
    /// Total number of evaluations.
    pub const fn size(self) -> usize {
        self.size
    }
    /// Number of contiguous residue transforms.
    pub const fn residues(self) -> usize {
        self.residues
    }
    /// Number of evaluations in each residue.
    pub const fn rows(self) -> usize {
        self.size / self.residues
    }

    /// Maps a natural row to its stored offset, or returns `None` out of range.
    pub fn index(self, row: usize) -> Option<usize> {
        (row < self.size).then(|| (row % self.residues) * self.rows() + row / self.residues)
    }

    /// Maps a stored offset back to a natural row, or `None` out of range.
    pub fn natural_row(self, index: usize) -> Option<usize> {
        (index < self.size).then(|| index / self.rows() + self.residues * (index % self.rows()))
    }

    /// Maps a row of a larger domain into this subdomain's stored offset.
    ///
    /// For this mapping to identify the same evaluation point, the domains
    /// must use the same shift and roots from [`Domain`](super::Domain).
    /// Neither shift nor root is checked. Returns `None` for incompatible
    /// sizes, out-of-range rows, or rows not belonging to the smaller domain.
    pub fn index_at_extended_row(self, row: usize, extended_size: usize) -> Option<usize> {
        if !extended_size.is_power_of_two() || extended_size < self.size || row >= extended_size {
            return None;
        }
        let stride = extended_size / self.size;
        if !row.is_multiple_of(stride) {
            return None;
        }
        self.index(row / stride)
    }

    /// Copies natural-order evaluations into a distinct residue-major output.
    ///
    /// Both slices must have [`Self::size`] elements. Returns
    /// [`FftError::LengthMismatch`] before writing if either length differs.
    pub fn copy_from_natural<T: Copy>(self, input: &[T], output: &mut [T]) -> Result<(), FftError> {
        check_len("input", input.len(), self.size)?;
        check_len("output", output.len(), self.size)?;
        for (row, value) in input.iter().enumerate() {
            output[self.index(row).unwrap()] = *value;
        }
        Ok(())
    }

    /// Copies residue-major evaluations into a distinct natural-order output.
    ///
    /// Both slices must have [`Self::size`] elements. Returns
    /// [`FftError::LengthMismatch`] before writing if either length differs.
    pub fn copy_to_natural<T: Copy>(self, input: &[T], output: &mut [T]) -> Result<(), FftError> {
        check_len("input", input.len(), self.size)?;
        check_len("output", output.len(), self.size)?;
        for (row, value) in output.iter_mut().enumerate() {
            *value = input[self.index(row).unwrap()];
        }
        Ok(())
    }
}

/// An immutable view of field evaluations with a checked layout length.
///
/// Construction checks storage dimensions, not the evaluation domain or field
/// contents. Natural rows are mapped through [`ResidueLayout`].
#[derive(Clone, Copy)]
pub struct ResidueView<'a, M: PrimeModulus> {
    values: &'a [PastaField<M>],
    layout: ResidueLayout,
}

impl<M: PrimeModulus> core::fmt::Debug for ResidueView<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("ResidueView")
            .field("values", &self.values)
            .field("layout", &self.layout)
            .finish()
    }
}

impl<'a, M: PrimeModulus> ResidueView<'a, M> {
    /// Binds a layout after checking the slice length.
    ///
    /// Returns [`FftError::LengthMismatch`] unless `values.len() == layout.size()`.
    pub fn new(values: &'a [PastaField<M>], layout: ResidueLayout) -> Result<Self, FftError> {
        check_len("values", values.len(), layout.size())?;
        Ok(Self { values, layout })
    }
    /// The layout of the borrowed values.
    pub const fn layout(self) -> ResidueLayout {
        self.layout
    }
    /// Values in storage order.
    pub const fn as_slice(self) -> &'a [PastaField<M>] {
        self.values
    }
    /// An evaluation selected by its natural row, or `None` out of range.
    pub fn get(self, row: usize) -> Option<&'a PastaField<M>> {
        self.values.get(self.layout.index(row)?)
    }
    /// An evaluation selected by a compatible larger domain's row.
    ///
    /// Uses [`ResidueLayout::index_at_extended_row`], including its unchecked
    /// shift and root assumptions and its `None` cases.
    pub fn get_extended_row(self, row: usize, extended_size: usize) -> Option<&'a PastaField<M>> {
        self.values
            .get(self.layout.index_at_extended_row(row, extended_size)?)
    }
    /// Borrows a whole residue in increasing natural-row order.
    ///
    /// Returns `None` if `residue >= self.layout().residues()`.
    pub fn residue(self, residue: usize) -> Option<&'a [PastaField<M>]> {
        if residue >= self.layout.residues() {
            return None;
        }
        let start = residue * self.layout.rows();
        Some(&self.values[start..start + self.layout.rows()])
    }
}

/// Natural-order coefficients viewed as consecutive, equally sized tiles.
#[derive(Clone, Copy)]
pub struct CoefficientTiles<'a, M: PrimeModulus> {
    values: &'a [PastaField<M>],
    tile_len: usize,
}

impl<M: PrimeModulus> core::fmt::Debug for CoefficientTiles<'_, M> {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CoefficientTiles")
            .field("values", &self.values)
            .field("tile_len", &self.tile_len)
            .finish()
    }
}

impl<'a, M: PrimeModulus> CoefficientTiles<'a, M> {
    /// Checks that both lengths are nonzero powers of two and tiles fit exactly.
    ///
    /// Returns [`FftError::InvalidLayout`] if either length is zero or not a
    /// power of two, or if `tile_len > values.len()`.
    pub fn new(values: &'a [PastaField<M>], tile_len: usize) -> Result<Self, FftError> {
        if !values.len().is_power_of_two() || !tile_len.is_power_of_two() || tile_len > values.len()
        {
            return Err(FftError::InvalidLayout);
        }
        Ok(Self { values, tile_len })
    }
    /// Number of tiles.
    pub const fn tile_count(self) -> usize {
        self.values.len() / self.tile_len
    }
    /// All coefficients, in natural order.
    pub const fn as_slice(self) -> &'a [PastaField<M>] {
        self.values
    }
    /// A consecutive coefficient tile, or `None` out of range.
    pub fn tile(self, index: usize) -> Option<&'a [PastaField<M>]> {
        if index >= self.tile_count() {
            return None;
        }
        Some(&self.values[index * self.tile_len..(index + 1) * self.tile_len])
    }
}
