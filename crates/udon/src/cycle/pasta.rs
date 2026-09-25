//! Pasta cycle binding and its runtime generator parameters.

use super::{Cycle, PallasGenerators, VestaGenerators};
use crate::{
    curve::{PallasPoint, VestaPoint},
    field::{Fp, Fq},
    poseidon::{PoseidonFp, PoseidonFq},
};

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
