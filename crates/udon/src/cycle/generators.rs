//! Borrowed commitment generators and their construction checks.

use crate::curve::{Affine, Pallas, PastaCurve, Point, Vesta};

/// Fixed generators of one curve with unknown discrete logarithm relationships
/// to each other.
pub trait FixedGenerators<C: Affine>: Send + Sync + 'static {
    /// The generators used to commit to vectors, such as polynomial
    /// coefficients.
    fn g(&self) -> &[C];

    /// The generator used for blinding.
    fn h(&self) -> &C;
}

/// Fixed generators of a Pasta curve, borrowed from static storage.
///
/// The parameter owner derives the points and keeps them alive for the
/// program's lifetime, for example as embedded constants or a leaked
/// allocation.
#[derive(Clone, Copy, Debug)]
pub struct Generators<C: PastaCurve> {
    g: &'static [Point<C>],
    h: Point<C>,
}

impl<C: PastaCurve> Generators<C> {
    /// Borrows the vector generators `g` and the blinding generator `h`.
    ///
    /// The parameter owner must derive the points with unknown discrete
    /// logarithm relationships. The checks below do not establish this
    /// property; the inputs must also satisfy [`Point`]'s invariants.
    ///
    /// # Panics
    ///
    /// Panics if `g` is empty, any generator is identity, or `h` appears in
    /// `g`. In a constant initializer this produces a compile error. The
    /// vector generators are not compared with each other here; see
    /// [`Self::are_distinct`].
    pub const fn new(g: &'static [Point<C>], h: Point<C>) -> Self {
        assert!(
            !g.is_empty(),
            "generators require at least one vector generator"
        );
        assert!(
            !h.is_identity(),
            "the blinding generator must not be identity"
        );
        let mut index = 0;
        while index < g.len() {
            assert!(!g[index].is_identity(), "a generator must not be identity");
            assert!(
                !same_point(&g[index], &h),
                "the blinding generator must not appear among the vector generators"
            );
            index += 1;
        }
        Self { g, h }
    }

    /// Returns whether all vector generators are distinct.
    ///
    /// A repeated generator gives a nontrivial vector whose commitment is
    /// identity. Honest derivation makes repeats negligible, so this check is
    /// for loaders decoding parameter files, where a slicing error can repeat
    /// entries. It compares every pair, so its cost is quadratic in the
    /// number of generators.
    ///
    /// This checks point equality only; it does not establish unknown
    /// discrete logarithm relationships.
    pub fn are_distinct(&self) -> bool {
        self.g
            .iter()
            .enumerate()
            .all(|(index, point)| !self.g[index + 1..].contains(point))
    }
}

// Point equality in constant evaluation: identity only equals identity, and
// nonidentity points compare their reduced coordinates limb by limb.
const fn same_point<C: PastaCurve>(a: &Point<C>, b: &Point<C>) -> bool {
    match (a.as_affine(), b.as_affine()) {
        (None, None) => true,
        (Some(a), Some(b)) => {
            let (ax, ay) = a.coordinates();
            let (bx, by) = b.coordinates();
            same_limbs(&ax.montgomery_limbs(), &bx.montgomery_limbs())
                && same_limbs(&ay.montgomery_limbs(), &by.montgomery_limbs())
        }
        _ => false,
    }
}

const fn same_limbs(a: &[u64; 4], b: &[u64; 4]) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
}

impl<C: PastaCurve> FixedGenerators<Point<C>> for Generators<C> {
    fn g(&self) -> &[Point<C>] {
        self.g
    }

    fn h(&self) -> &Point<C> {
        &self.h
    }
}

/// Fixed Pallas generators.
pub type PallasGenerators = Generators<Pallas>;
/// Fixed Vesta generators.
pub type VestaGenerators = Generators<Vesta>;
