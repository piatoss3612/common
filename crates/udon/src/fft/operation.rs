use super::{ElementOrder, InverseScale};

/// Mathematical transform direction.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Direction {
    /// Evaluate coefficients on the coset.
    Forward,
    /// Interpolate evaluations to coefficients.
    Inverse,
}

/// Declared nonzero support in natural logical input order.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputSupport {
    /// Every domain position may be nonzero.
    Full,
    /// Only positions `0..length` may be nonzero.
    ///
    /// A separate input contains exactly this prefix; in-place execution ignores
    /// and overwrites its tail.
    /// For an inverse these positions are evaluations, not coefficients.
    Prefix(usize),
}

/// Location and lifetime of transform input.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum InputStorage {
    /// Read an immutable input view and write a separate output bank.
    Preserve,
    /// Read and overwrite the writable values bank.
    InPlace,
}

/// Mathematical and storage semantics fixed before executing an operation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct TransformRequest {
    /// Forward evaluation or inverse interpolation.
    pub direction: Direction,
    /// Declared input support; a prefix requires natural input order.
    pub support: InputSupport,
    /// Order of coefficients or evaluations in the input.
    pub input_order: ElementOrder,
    /// Desired order of coefficients or evaluations in the output.
    pub output_order: ElementOrder,
    /// Inverse-size factor; forward requests must use [`InverseScale::Normalized`].
    pub inverse_scale: InverseScale,
    /// In-place input or a preserved separate input view.
    pub input_storage: InputStorage,
}

impl TransformRequest {
    /// A full natural-order transform using in-place input.
    pub const fn new(direction: Direction) -> Self {
        Self {
            direction,
            support: InputSupport::Full,
            input_order: ElementOrder::Natural,
            output_order: ElementOrder::Natural,
            inverse_scale: InverseScale::Normalized,
            input_storage: InputStorage::InPlace,
        }
    }
    pub(super) const fn input_len(self, size: usize) -> usize {
        match self.support {
            InputSupport::Full => size,
            InputSupport::Prefix(len) => len,
        }
    }
}

/// Writable storage available to an incremental transform.
///
/// Fragment lengths describe the provider's physical storage, not an arithmetic
/// radix. A whole-bank lease permits order conversion without a retained copy.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StorageLayout {
    /// One bank whose complete mutable slice can be leased.
    Contiguous,
    /// Independently leased, equal power-of-two fragments.
    Fragments {
        /// Elements per physical fragment; clamped to the domain size.
        length: core::num::NonZeroUsize,
        /// Whether the provider can also lease the complete bank exclusively.
        whole_bank: bool,
    },
}
