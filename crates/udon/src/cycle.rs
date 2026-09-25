//! Curve cycles and the fixed parameters a proof system binds to them.
//!
//! A cycle pairs two curves whose scalar and base fields interchange. [`Cycle`]
//! names both fields and both curves through the field and curve traits,
//! together with the fixed generators and Poseidon instances a protocol uses
//! over them, so a proof system holds one type parameter and reaches every
//! parameter it needs. [`Pasta`] is the cycle of Pallas and Vesta.
//!
//! Generator derivation is not part of this crate. Generators are public
//! points with unknown discrete logarithm relationships, derived by the
//! parameter owner, typically by hash-to-curve, and supplied as borrowed
//! static slices. Construction rejects identity points and a blinding
//! generator that repeats a vector generator; [`Generators::are_distinct`]
//! additionally checks the vector generators against each other. Poseidon
//! instances come from [`crate::poseidon`].
//!
//! This example uses known multiples to demonstrate the parameter API and
//! arithmetic. These generator choices are insecure for cryptographic
//! commitments.
//!
//! ```
//! use zakura_udon::{
//!     curve::{Affine as _, PallasPoint, VestaPoint},
//!     cycle::{Cycle, FixedGenerators, Generators, Pasta, PastaParams},
//!     field::{Fp, Fq},
//! };
//!
//! // The parameter owner keeps derived generators alive for the program's
//! // lifetime; a leaked vector stands in for embedded or baked points here.
//! let multiple = |k| (PallasPoint::GENERATOR * Fq::from_u64(k)).to_point();
//! let pallas: &'static [PallasPoint] = Vec::leak(vec![multiple(1), multiple(2)]);
//! static VESTA: [VestaPoint; 1] = [VestaPoint::GENERATOR];
//! let params = PastaParams::new(
//!     Generators::new(pallas, multiple(3)),
//!     Generators::new(&VESTA, (VestaPoint::GENERATOR * Fp::from_u64(2)).to_point()),
//! );
//! let generators = Pasta::nested_generators(&params);
//! assert!(generators.are_distinct());
//! // g[0] * 2 + h * 3 = 2G + 9G
//! let commitment = generators.short_commit(Fq::from_u64(2), Fq::from_u64(3));
//! assert_eq!(commitment, multiple(11));
//! ```

use crate::{
    curve::{
        Affine, EndomorphismAffine, Pallas, PallasPoint, PastaCurve, Point, Projective as _, Vesta,
        VestaPoint,
    },
    field::{CubeRootField, DeferredField, FftField, Fp, Fq},
    poseidon::{PoseidonFp, PoseidonFq, PoseidonPermutation},
};

/// A cycle of two curves, each defined over the other's scalar field.
///
/// This proof-system interface requires radix-2 FFTs, deferred products, and
/// compatible order-three curve endomorphisms. The basic [`Affine`] and
/// [`crate::field::Field`] traits do not require those capabilities.
///
/// Implementations are zero-sized markers. Parameters that exist at runtime,
/// such as generators, live in [`Params`](Self::Params) and are reached
/// through the accessor functions.
pub trait Cycle: Copy + Default + Send + Sync + 'static {
    /// The field circuits are written over: the scalar field of the
    /// [`HostCurve`](Self::HostCurve) and the coordinate field of the
    /// [`NestedCurve`](Self::NestedCurve).
    type CircuitField: FftField + DeferredField + CubeRootField;

    /// The scalar field of the [`NestedCurve`](Self::NestedCurve) and the
    /// coordinate field of the [`HostCurve`](Self::HostCurve).
    type ScalarField: FftField + DeferredField + CubeRootField;

    /// The curve applications use for keys, signatures, and other primitives
    /// whose arithmetic circuits express over the
    /// [`CircuitField`](Self::CircuitField).
    type NestedCurve: EndomorphismAffine<Scalar = Self::ScalarField, Base = Self::CircuitField>;

    /// The curve the proof system commits with when proving circuits over the
    /// [`CircuitField`](Self::CircuitField).
    type HostCurve: EndomorphismAffine<Scalar = Self::CircuitField, Base = Self::ScalarField>;

    /// Fixed generators of the [`NestedCurve`](Self::NestedCurve).
    type NestedGenerators: FixedGenerators<Self::NestedCurve>;

    /// Fixed generators of the [`HostCurve`](Self::HostCurve).
    type HostGenerators: FixedGenerators<Self::HostCurve>;

    /// The Poseidon instance over the [`CircuitField`](Self::CircuitField).
    type CircuitPoseidon: PoseidonPermutation<Self::CircuitField>;

    /// The Poseidon instance over the [`ScalarField`](Self::ScalarField).
    type ScalarPoseidon: PoseidonPermutation<Self::ScalarField>;

    /// Runtime parameters: the generators, and whatever else the cycle needs.
    type Params: Send + Sync + 'static;

    /// Returns the fixed generators of the [`NestedCurve`](Self::NestedCurve).
    fn nested_generators(params: &Self::Params) -> &Self::NestedGenerators;

    /// Returns the fixed generators of the [`HostCurve`](Self::HostCurve).
    fn host_generators(params: &Self::Params) -> &Self::HostGenerators;

    /// Returns the Poseidon instance over the
    /// [`CircuitField`](Self::CircuitField).
    fn circuit_poseidon(params: &Self::Params) -> &Self::CircuitPoseidon;

    /// Returns the Poseidon instance over the
    /// [`ScalarField`](Self::ScalarField).
    fn scalar_poseidon(params: &Self::Params) -> &Self::ScalarPoseidon;
}

/// Fixed generators of one curve with unknown discrete logarithm relationships
/// to each other.
pub trait FixedGenerators<C: Affine>: Send + Sync + 'static {
    /// The generators used to commit to vectors, such as polynomial
    /// coefficients.
    fn g(&self) -> &[C];

    /// The generator used for blinding.
    fn h(&self) -> &C;

    /// Commits to one value: `g[0] * value + h * blind`.
    ///
    /// This operation is variable-time in both `value` and `blind`. It has
    /// no constant-time guarantee and must not be used when either scalar
    /// needs protection from timing or other execution side channels.
    /// The blinding scalar does not hide those execution traces.
    ///
    /// # Panics
    ///
    /// Panics if [`g`](Self::g) is empty.
    fn short_commit(&self, value: C::Scalar, blind: C::Scalar) -> C {
        (self.g()[0] * value + *self.h() * blind).to_affine()
    }
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

/// Runtime parameters of the [`Pasta`] cycle: the generators of both curves.
///
/// The Poseidon instances are compile-time constants and need no storage here.
#[derive(Clone, Copy, Debug)]
pub struct PastaParams {
    pallas: PallasGenerators,
    vesta: VestaGenerators,
}

impl PastaParams {
    /// Binds the generators of both curves.
    pub const fn new(pallas: PallasGenerators, vesta: VestaGenerators) -> Self {
        Self { pallas, vesta }
    }

    /// The Pallas generators.
    pub const fn pallas(&self) -> &PallasGenerators {
        &self.pallas
    }

    /// The Vesta generators.
    pub const fn vesta(&self) -> &VestaGenerators {
        &self.vesta
    }
}

/// The Pasta cycle: Pallas over [`Fp`] with scalars in [`Fq`] as the nested
/// curve, and Vesta over [`Fq`] with scalars in [`Fp`] as the host curve.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct Pasta;

impl Cycle for Pasta {
    type CircuitField = Fp;
    type ScalarField = Fq;
    type NestedCurve = PallasPoint;
    type HostCurve = VestaPoint;
    type NestedGenerators = PallasGenerators;
    type HostGenerators = VestaGenerators;
    type CircuitPoseidon = PoseidonFp;
    type ScalarPoseidon = PoseidonFq;
    type Params = PastaParams;

    fn nested_generators(params: &PastaParams) -> &PallasGenerators {
        &params.pallas
    }

    fn host_generators(params: &PastaParams) -> &VestaGenerators {
        &params.vesta
    }

    fn circuit_poseidon(_params: &PastaParams) -> &PoseidonFp {
        &PoseidonFp
    }

    fn scalar_poseidon(_params: &PastaParams) -> &PoseidonFq {
        &PoseidonFq
    }
}

#[cfg(test)]
mod tests {
    use std::vec::Vec;

    use super::*;
    use crate::{
        curve::{PallasProjective, ProjectivePoint},
        field::Field,
        poseidon::{PALLAS_BASE, PALLAS_SCALAR, PoseidonParameters},
    };

    /// `count` distinct nonidentity points, the generator multiples starting
    /// at `first`; disjoint ranges give disjoint points.
    fn leaked<C: PastaCurve>(count: usize, first: u64) -> &'static [Point<C>] {
        let points: Vec<Point<C>> = (first..first + count as u64)
            .map(|index| {
                Point::<C>::GENERATOR
                    .mul_projective(&crate::field::PastaField::from_u64(index * 7 + 3))
                    .to_point()
            })
            .collect();
        Vec::leak(points)
    }

    fn params() -> PastaParams {
        PastaParams::new(
            Generators::new(leaked::<Pallas>(3, 1), leaked::<Pallas>(1, 50)[0]),
            Generators::new(leaked::<Vesta>(2, 1), leaked::<Vesta>(1, 50)[0]),
        )
    }

    #[test]
    fn pasta_generators_are_reached_through_the_cycle() {
        let params = params();
        let pallas = Pasta::nested_generators(&params);
        let vesta = Pasta::host_generators(&params);
        assert_eq!(pallas.g().len(), 3);
        assert_eq!(vesta.g().len(), 2);
        assert_eq!(pallas.g(), params.pallas().g());
        assert_eq!(vesta.h(), params.vesta().h());
        assert!(pallas.g().iter().all(|point| !point.is_identity()));
        assert!(!pallas.h().is_identity());

        for (value, blind) in [(2u64, 3u64), (0, 5), (7, 0), (11, 13)] {
            let (value, blind) = (Fq::from_u64(value), Fq::from_u64(blind));
            let expected = pallas.g()[0]
                .mul_projective(&value)
                .add(&pallas.h().mul_projective(&blind))
                .to_point();
            assert_eq!(pallas.short_commit(value, blind), expected);
            let expected: ProjectivePoint<Vesta> = vesta.g()[0]
                .mul_projective(&Fp::from_u64(3))
                .add(&vesta.h().mul_projective(&Fp::from_u64(4)));
            assert_eq!(
                vesta.short_commit(Fp::from_u64(3), Fp::from_u64(4)),
                expected.to_point()
            );
        }
        let _: PallasProjective = pallas.g()[0] * Fq::from_u64(2);
    }

    fn check_poseidon<F: Field, P: PoseidonPermutation<F>, const T: usize>(
        instance: &P,
        parameters: &PoseidonParameters<F, T>,
    ) {
        assert_eq!(P::T, parameters.width());
        assert_eq!(P::RATE, parameters.rate());
        assert_eq!(P::FULL_ROUNDS, parameters.full_rounds);
        assert_eq!(P::PARTIAL_ROUNDS, parameters.partial_rounds);
        assert_eq!(P::ALPHA, parameters.alpha as isize);
        let rows: Vec<&[F]> = instance.round_constants().collect();
        assert_eq!(rows.len(), parameters.rounds());
        for (row, expected) in rows.iter().zip(parameters.round_constants) {
            assert_eq!(*row, &expected[..]);
        }
        let mds = instance.mds_matrix();
        assert_eq!(mds.len(), T);
        for (row, expected) in mds.zip(parameters.mds) {
            assert_eq!(row, &expected[..]);
        }
    }

    #[test]
    fn pasta_poseidon_instances_are_the_parameter_constants() {
        let params = params();
        check_poseidon(Pasta::circuit_poseidon(&params), &PALLAS_BASE);
        check_poseidon(Pasta::scalar_poseidon(&params), &PALLAS_SCALAR);
        assert_eq!(<Pasta as Cycle>::CircuitPoseidon::default(), PoseidonFp);
    }

    #[test]
    fn cycle_is_usable_through_generic_bounds() {
        fn commit<C: Cycle>(params: &C::Params, value: C::CircuitField) -> C::HostCurve {
            C::host_generators(params).short_commit(value, C::CircuitField::ONE)
        }
        fn width<C: Cycle>() -> usize {
            C::CircuitPoseidon::T + C::ScalarPoseidon::RATE
        }
        let params = params();
        let expected = params.vesta().g()[0]
            .mul_projective(&Fp::from_u64(9))
            .add(&params.vesta().h().to_projective())
            .to_point();
        assert_eq!(commit::<Pasta>(&params, Fp::from_u64(9)), expected);
        assert_eq!(width::<Pasta>(), 9);
    }

    #[test]
    #[should_panic(expected = "at least one vector generator")]
    fn generators_reject_an_empty_vector() {
        let _ = Generators::<Pallas>::new(&[], PallasPoint::GENERATOR);
    }

    #[test]
    #[should_panic(expected = "must not be identity")]
    fn generators_reject_identity() {
        static G: [PallasPoint; 2] = [PallasPoint::GENERATOR, PallasPoint::IDENTITY];
        let _ = Generators::<Pallas>::new(&G, PallasPoint::GENERATOR.neg());
    }

    #[test]
    #[should_panic(expected = "must not appear among the vector generators")]
    fn generators_reject_a_repeated_blinding_generator() {
        let g = leaked::<Pallas>(3, 1);
        let _ = Generators::<Pallas>::new(g, g[1]);
    }

    #[test]
    fn generator_distinctness_is_reported() {
        let distinct = Generators::new(leaked::<Vesta>(5, 1), leaked::<Vesta>(1, 50)[0]);
        assert!(distinct.are_distinct());
        assert!(Generators::new(leaked::<Vesta>(1, 1), leaked::<Vesta>(1, 50)[0]).are_distinct());

        let points = leaked::<Vesta>(4, 1);
        let repeated: &'static [VestaPoint] =
            Vec::leak(std::vec![points[0], points[1], points[2], points[1]]);
        assert!(!Generators::new(repeated, points[3]).are_distinct());
        // Reaching the same point by another route is still a repeat.
        let same: &'static [VestaPoint] = Vec::leak(std::vec![
            points[0],
            points[1].to_projective().double().to_point(),
            points[1].add(&points[1]).to_point(),
        ]);
        assert!(!Generators::new(same, points[3]).are_distinct());
    }
}
