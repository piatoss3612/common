//! Bounds carried by a field element's type, with no stored state flag.

/// A Montgomery representative in `[0, 2p)`.
///
/// Arithmetic returns this representation. Use [`super::PastaField::reduce`]
/// when a unique representative is needed for equality or a square root.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Loose {}

/// A unique Montgomery representative in `[0, p)`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Reduced {}

mod sealed {
    pub(in crate::field) trait Sealed {
        const REDUCED: bool;
    }
}

/// Selects the stored representative's bound.
///
/// Only [`Loose`] and [`Reduced`] implement this sealed trait. Both have the
/// same four-limb storage layout; the bound is known at compile time.
#[expect(
    private_bounds,
    reason = "the reduction flag is an implementation parameter"
)]
pub trait ReductionState: sealed::Sealed + Copy + Eq + Send + Sync + 'static {}

impl sealed::Sealed for Loose {
    const REDUCED: bool = false;
}
impl sealed::Sealed for Reduced {
    const REDUCED: bool = true;
}
impl ReductionState for Loose {}
impl ReductionState for Reduced {}
