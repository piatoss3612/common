//! Variable-time safegcd inversion of loose Pasta Montgomery residues.

use super::{PastaField, PrimeModulus, ReductionState};

#[cfg(test)]
mod tests;

#[cfg(test)]
std::thread_local! {
    static INVERSION_COUNT: core::cell::Cell<usize> = const { core::cell::Cell::new(0) };
}

#[cfg(test)]
pub(crate) fn count_inversions(f: impl FnOnce()) -> usize {
    INVERSION_COUNT.with(|count| {
        let before = count.get();
        f();
        count.get() - before
    })
}

impl<M: PrimeModulus, S: ReductionState> PastaField<M, S> {
    /// Returns the multiplicative inverse, or `None` for zero.
    ///
    /// The safegcd loop terminates according to the input.
    pub fn invert(&self) -> Option<PastaField<M>> {
        // Safegcd requires a representative below p. Reduce the stored
        // Montgomery value, preserving its scale; this also maps the loose
        // representation p of zero to the core's zero input.
        let inverse = super::safegcd::invert::<M>(&self.reduce().limbs)?;
        #[cfg(test)]
        INVERSION_COUNT.with(|count| count.set(count.get() + 1));
        Some(PastaField::from_montgomery(inverse))
    }
}
