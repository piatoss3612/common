#![no_std]
#![deny(warnings)]
#![forbid(unsafe_code)]

use bento::const_arithmetic::{U256, U320, m255, u256};
use udon::curve::{
    AffinePoint, CurveError, CurveTableRequirements, EisensteinScalar, EisensteinTable,
    EisensteinTableBatch, FixedBaseDescription, FixedBaseTable, Pallas, PallasAffine, PastaCurve,
    Point, PreparedAffinePoint, ProjectivePoint, Vesta, VestaAffine, batch_normalize,
    glv_decompose, msm,
};
use udon::exec::{Executor, SerialExecutor, TaskBudget, for_each_chunk_mut, for_each_mut};
use udon::fft::{
    Class, Domain, ElementOrder, ExecutionOptions, Expansion, ExpansionOptions, FftError, Plan,
    TableRequirements, TablesMut, interpolate_classes,
};
use udon::field::{Fp, Fq, PallasBase, PallasScalar, PastaField, PrimeModulus, ProductSum};

// Numeric word order is independent of the target's byte order. These
// assertions run during compilation, including on targets we cannot execute.
pub const MODULUS: U256 =
    u256::from_hex!("0x7fffffffffffffffffffffffffffffffffffffffffffffffffffffffffffffed");
pub const ENCODED: U256 = m255::from_u256!(&MODULUS, &[u64::MAX; 4]);
pub const ROOT: U256 = m255::two_adic_root_of_unity!(&[97, 0, 0, 0], 5, 5);
pub const RATIO: U320 = u256::round_shifted_ratio!(&[u64::MAX; 4], u128::MAX, 384);
pub const FP: Fp =
    udon::fp_hex!("0x0000000000000000000000000000000100000000000000000123456789abcdef");
pub const FQ: Fq =
    udon::fq_hex!("0x0000000000000000000000000000000100000000000000000123456789abcdef",);

const _: () = {
    let decoded = m255::to_u256!(&MODULUS, &ENCODED);
    assert!(decoded[0] == 37 && decoded[1] == 0 && decoded[2] == 0 && decoded[3] == 0);
    let root = m255::to_u256!(&[97, 0, 0, 0], &ROOT);
    assert!(root[0] == 28 && root[1] == 0 && root[2] == 0 && root[3] == 0);
    assert!(RATIO[0] == 1 && RATIO[1] == 0 && RATIO[2] == u64::MAX);
    assert!(RATIO[3] == u64::MAX && RATIO[4] == 0);
    let fp = m255::to_u256!(&PallasBase::MODULUS, &FP.montgomery_limbs());
    let fq = m255::to_u256!(&PallasScalar::MODULUS, &FQ.montgomery_limbs());
    assert!(fp[0] == 0x0123_4567_89ab_cdef && fp[1] == 0 && fp[2] == 1 && fp[3] == 0);
    assert!(fq[0] == fp[0] && fq[1] == fp[1] && fq[2] == fp[2] && fq[3] == fp[3]);
};

// Concrete wrappers force code generation of both fields' runtime kernels.
// Input bytes remain unknown at compile time, and no allocator is available.
fn field_operations<M: PrimeModulus>(
    wide: &[u8; 64],
    bytes: [u8; 32],
    log_size: u32,
) -> Option<[u8; 32]> {
    let value = PastaField::<M>::from_wide_bytes_reduced(wide);
    let other = PastaField::from_bytes(bytes)?;
    let inverse = value.invert()?;
    let root = other
        .sqrt()?
        .mul(&PastaField::root_of_unity(log_size)?)
        .mul(&PastaField::root_of_unity_inverse(log_size)?);
    let mut sum = ProductSum::new();
    sum.add_product(&value, &other);
    sum.add_term(&value.mul_sub_double_product(&root, &other, &inverse));
    let mut merged = ProductSum::new();
    merged.merge(&sum);
    Some(merged.finish().mul_add(&inverse, &root).to_bytes())
}

pub fn fp_operations(wide: &[u8; 64], bytes: [u8; 32], log_size: u32) -> Option<[u8; 32]> {
    field_operations::<PallasBase>(wide, bytes, log_size)
}

pub fn fq_operations(wide: &[u8; 64], bytes: [u8; 32], log_size: u32) -> Option<[u8; 32]> {
    field_operations::<PallasScalar>(wide, bytes, log_size)
}

pub const PALLAS: PallasAffine = udon::pallas_affine!(
    *PallasAffine::GENERATOR.coordinates().0,
    *PallasAffine::GENERATOR.coordinates().1,
);
pub const VESTA: VestaAffine = udon::vesta_affine!(
    *VestaAffine::GENERATOR.coordinates().0,
    *VestaAffine::GENERATOR.coordinates().1,
);

fn curve_operations<C: PastaCurve>(
    bytes: [u8; 32],
    scalar: &PastaField<C::Scalar>,
) -> Result<[u8; 32], CurveError> {
    let base = AffinePoint::<C>::from_bytes(bytes).ok_or(CurveError::InvalidBase)?;
    let point = base.mul_projective(scalar).double().add_mixed(&base);
    let points = [point, point.neg().add(&point)];
    let mut output = [Point::IDENTITY; 2];
    let mut normalization_scratch = [PastaField::ZERO; 2];
    batch_normalize(&points, &mut output, &mut normalization_scratch)?;
    const DESCRIPTION: FixedBaseDescription = FixedBaseDescription { window_bits: 4 };
    const REQUIREMENTS: CurveTableRequirements = match DESCRIPTION.requirements() {
        Ok(required) => required,
        Err(_) => panic!("unsupported fixed-base description"),
    };
    let mut entries = [AffinePoint::GENERATOR; REQUIREMENTS.table_entries];
    let mut projective = [ProjectivePoint::IDENTITY; REQUIREMENTS.projective_scratch];
    let mut field = [PastaField::ZERO; REQUIREMENTS.field_scratch];
    let table = FixedBaseTable::prepare(
        DESCRIPTION,
        &base,
        &mut entries,
        &mut projective,
        &mut field,
    )?;
    let bound = FixedBaseTable::bind(DESCRIPTION, &base, table.as_slice())?;
    let product = bound.mul(scalar);
    assert_eq!(product, base.to_projective().mul(scalar));
    let mut cached = [PreparedAffinePoint::from_affine(&base); REQUIREMENTS.table_entries];
    let cached =
        FixedBaseTable::prepare(DESCRIPTION, &base, &mut cached, &mut projective, &mut field)?;
    assert_eq!(cached.mul(scalar), product);
    let mut compact_entries = [PreparedAffinePoint::from_affine(&base); 8];
    let compact =
        EisensteinTable::prepare(&base, &mut compact_entries, &mut projective, &mut field)?;
    EisensteinTable::bind(&base, compact.as_slice())?.validate()?;
    assert_eq!(compact.mul(scalar), product);
    assert_eq!(
        compact.mul_prepared(&EisensteinScalar::new(scalar)),
        product
    );
    table_batch_operations(&base, scalar, product)?;
    msm_operations(&base, scalar)?;
    let (a, b) = glv_decompose::<C>(scalar);
    assert!(a != i128::MIN && b != i128::MIN);
    assert_eq!(
        base.endomorphism().to_projective(),
        base.to_point().endomorphism().to_projective()
    );
    assert_eq!(
        base.to_projective()
            .endomorphism()
            .endomorphism()
            .endomorphism(),
        base.to_projective()
    );
    assert_eq!(point, output[0].to_projective());
    assert!(output[1].is_identity());
    Ok(product.to_point().to_bytes())
}

pub fn pallas_operations(bytes: [u8; 32], scalar: &Fq) -> Result<[u8; 32], CurveError> {
    curve_operations::<Pallas>(bytes, scalar)
}

pub fn vesta_operations(bytes: [u8; 32], scalar: &Fp) -> Result<[u8; 32], CurveError> {
    curve_operations::<Vesta>(bytes, scalar)
}

fn table_batch_operations<C: PastaCurve>(
    base: &AffinePoint<C>,
    scalar: &PastaField<C::Scalar>,
    expected: ProjectivePoint<C>,
) -> Result<(), CurveError> {
    const N: usize = 64;
    const R: CurveTableRequirements = match EisensteinTableBatch::<Pallas>::requirements(N) {
        Ok(r) => r,
        Err(_) => panic!("unsupported table batch size"),
    };
    const MUL: usize = match EisensteinTableBatch::<Pallas>::multiplication_scratch(N) {
        Ok(n) => n,
        Err(_) => panic!("unsupported table batch size"),
    };
    const FIELD: usize = if MUL > R.field_scratch {
        MUL
    } else {
        R.field_scratch
    };
    let bases = [*base; N];
    let mut entries = [PreparedAffinePoint::from_affine(base); R.table_entries];
    let mut projective = [ProjectivePoint::IDENTITY; R.projective_scratch];
    let mut field = [PastaField::ZERO; FIELD];
    let batch = EisensteinTableBatch::prepare(
        &bases,
        &mut entries,
        &mut projective,
        &mut field,
        TaskBudget::SERIAL,
        &SerialExecutor,
    )?;
    let mut output = [ProjectivePoint::IDENTITY; N];
    batch.mul_prepared(
        &EisensteinScalar::new(scalar),
        &mut output,
        &mut field,
        TaskBudget::SERIAL,
        &SerialExecutor,
    )?;
    assert!(output.iter().all(|&p| p == expected));
    Ok(())
}

const MSM_OPTIONS: msm::ExecutionOptions = msm::ExecutionOptions::SERIAL
    .with_memory_limit(8192)
    .with_max_terms_per_pass(core::num::NonZeroUsize::new(17));
const MSM_SCRATCH: msm::Requirements =
    match msm::Input::<Pallas>::requirements_for_len(257, MSM_OPTIONS) {
        Ok(r) => r,
        Err(_) => panic!("unsupported MSM size"),
    };
const _: () = {
    assert!(matches!(
        msm::Input::<Pallas>::requirements_for_len(usize::MAX, MSM_OPTIONS),
        Err(CurveError::SizeOverflow)
    ));
    assert!(matches!(
        EisensteinTableBatch::<Pallas>::requirements(usize::MAX),
        Err(CurveError::SizeOverflow)
    ));
    if usize::BITS == 32 {
        assert!(matches!(
            msm::Input::<Pallas>::requirements_for_len(1 << 27, MSM_OPTIONS),
            Err(CurveError::SizeOverflow)
        ));
        assert!(matches!(
            EisensteinTableBatch::<Pallas>::multiplication_scratch(1 << 25),
            Err(CurveError::SizeOverflow)
        ));
    }
};

fn msm_operations<C: PastaCurve>(
    base: &AffinePoint<C>,
    scalar: &PastaField<C::Scalar>,
) -> Result<(), CurveError> {
    assert_eq!(
        msm::Input::<C>::requirements_for_len(257, MSM_OPTIONS)?,
        MSM_SCRATCH
    );
    let points = [base.to_point(), Point::IDENTITY];
    let indices: [u32; 257] = core::array::from_fn(|i| (i % 2) as u32);
    let scalars = [*scalar; 257];
    let selection = msm::Selection::indexed(msm::Bases::Points(&points), &indices)?;
    let input = selection.with_scalars(&scalars)?;
    let mut records = [msm::ScalarStorage::ZERO; MSM_SCRATCH.scalars()];
    let mut digits = [0; MSM_SCRATCH.digits()];
    let mut affine = [AffinePoint::GENERATOR; MSM_SCRATCH.affine()];
    let mut projective = [ProjectivePoint::IDENTITY; MSM_SCRATCH.projective()];
    let mut field = [PastaField::ZERO; MSM_SCRATCH.field()];
    let mut working_indices = [0; MSM_SCRATCH.indices()];
    let mut output = [ProjectivePoint::IDENTITY];
    msm::execute_batch(
        &[input],
        &mut output,
        MSM_OPTIONS,
        &SerialExecutor,
        msm::Scratch::new(
            &mut records,
            &mut digits,
            &mut affine,
            &mut projective,
            &mut field,
            &mut working_indices,
        ),
    )?;
    assert_eq!(
        output[0],
        base.mul_projective(&scalar.mul(&PastaField::from_u64(129)))
    );
    Ok(())
}

const _: () = {
    assert!(TaskBudget::new(0).is_none());
    let budget = TaskBudget::new(usize::MAX).unwrap();
    let (left, right) = budget.split_at(usize::MAX - 1).unwrap();
    assert!(left.get() == usize::MAX - 1 && right.get() == 1);
    let (jobs, inner) = budget.partition(2).unwrap();
    assert!(jobs == 2 && inner.get() == usize::MAX / 2);
};

// A concrete caller checks target code generation without std or an allocator;
// unused generic helpers would not exercise that boundary.
pub fn execution_operations(values: &mut [usize; 8], tasks: usize) -> Option<usize> {
    let budget = TaskBudget::new(tasks)?;
    let (left, right) = values.split_at_mut(3);
    let (left, right_len) = SerialExecutor.join(|| left, || right.len());
    let mut tiles = [left, right];
    for_each_mut(&mut tiles, budget, &SerialExecutor, |index, tile, inner| {
        for_each_chunk_mut(tile, 2, inner, &SerialExecutor, |chunk, values, _| {
            for value in values {
                *value = value.wrapping_add(index + chunk);
            }
        });
    });
    Some(
        values
            .iter()
            .fold(right_len, |sum, value| sum.wrapping_add(*value)),
    )
}

const FFT_SIZE: usize = 16;
const EXTENDED_FFT_SIZE: usize = 32;

// Sizing is evaluated for the consumer target, including its slice byte limit.
const _: () = {
    let options = ExecutionOptions::serial();
    let expansion = ExpansionOptions::serial();
    if usize::BITS == 32 {
        assert!(TableRequirements::for_size(1 << 25).is_ok());
        assert!(options.requirements(1 << 25).is_ok());
        assert!(matches!(
            TableRequirements::for_size(1 << 26),
            Err(FftError::SizeOverflow)
        ));
        assert!(matches!(
            options.requirements(1 << 26),
            Err(FftError::SizeOverflow)
        ));
        assert!(matches!(
            expansion.coefficient_requirements(1, 1 << 26),
            Err(FftError::SizeOverflow)
        ));
        assert!(matches!(
            expansion.evaluation_requirements(1, 1 << 26),
            Err(FftError::SizeOverflow)
        ));
        assert!(matches!(
            options.interpolation_requirements(1 << 26, &[]),
            Err(FftError::SizeOverflow)
        ));
    } else {
        assert!(TableRequirements::for_size((1u64 << 32) as usize).is_ok());
        assert!(options.requirements((1u64 << 32) as usize).is_ok());
        assert!(matches!(
            TableRequirements::for_size((1u64 << 33) as usize),
            Err(FftError::InvalidSize)
        ));
        assert!(matches!(
            options.requirements((1u64 << 33) as usize),
            Err(FftError::InvalidSize)
        ));
    }
};

fn fft_operations<M: PrimeModulus>(values: &mut [PastaField<M>; FFT_SIZE]) -> Result<(), FftError> {
    let domain = Domain::for_size(FFT_SIZE)?.subgroup();
    const TABLES: TableRequirements = match TableRequirements::for_size(FFT_SIZE) {
        Ok(required) => required,
        Err(_) => panic!("unsupported table size"),
    };
    let mut twiddles = [PastaField::ZERO; TABLES.twiddles];
    let tables = TablesMut {
        forward: Some(&mut twiddles),
        ..TablesMut::default()
    }
    .prepare(domain)?;
    let plan = Plan::new(tables);
    const OPTIONS: ExecutionOptions = ExecutionOptions {
        tile_len: 4,
        columns_per_task: 2,
        max_tasks: 1,
    };
    const SCRATCH: usize = match OPTIONS.requirements(FFT_SIZE) {
        Ok(required) => required.field_elements,
        Err(_) => panic!("unsupported transform configuration"),
    };
    let mut scratch = [PastaField::ZERO; SCRATCH];
    plan.forward(values, OPTIONS, &SerialExecutor, &mut scratch)?;
    let extended = Domain::for_size(EXTENDED_FFT_SIZE)?.coset(PastaField::zeta())?;
    let expansion = Expansion::new(plan, extended, None)?;
    let mut evaluations = [PastaField::ZERO; EXTENDED_FFT_SIZE];
    const EXPANSION_OPTIONS: ExpansionOptions = ExpansionOptions {
        max_residue_tasks: 1,
        transform: OPTIONS,
    };
    const EXPANSION_SCRATCH: usize =
        match EXPANSION_OPTIONS.evaluation_requirements(FFT_SIZE, EXTENDED_FFT_SIZE) {
            Ok(required) => required.field_elements,
            Err(_) => panic!("unsupported expansion configuration"),
        };
    let mut scratch = [PastaField::ZERO; EXPANSION_SCRATCH];
    expansion.evaluations(
        values,
        &mut evaluations,
        EXPANSION_OPTIONS,
        &SerialExecutor,
        &mut scratch,
    )?;
    let mut coefficients = [PastaField::ZERO; EXTENDED_FFT_SIZE];
    let mut output = Class::new(
        Plan::without_tables(extended),
        &mut coefficients,
        ElementOrder::BitReversed,
    )?;
    // Residue-major evaluations scatter directly into interpolation order.
    for (residue, values) in evaluations.chunks_exact(FFT_SIZE).enumerate() {
        output.scatter_strided(residue, 2, values)?;
    }
    let mut lifts = [Class::new(plan, values, ElementOrder::Natural)?];
    const INTERPOLATION_SCRATCH: usize =
        match OPTIONS.interpolation_requirements(EXTENDED_FFT_SIZE, &[FFT_SIZE]) {
            Ok(required) => required.field_elements,
            Err(_) => panic!("unsupported interpolation configuration"),
        };
    let mut scratch = [PastaField::ZERO; INTERPOLATION_SCRATCH];
    interpolate_classes(
        &mut output,
        &mut lifts,
        OPTIONS,
        &SerialExecutor,
        &mut scratch,
    )?;
    values.copy_from_slice(&coefficients[..FFT_SIZE]);
    Ok(())
}

pub fn fp_fft(values: &mut [Fp; FFT_SIZE]) -> Result<(), FftError> {
    fft_operations(values)
}

pub fn fq_fft(values: &mut [Fq; FFT_SIZE]) -> Result<(), FftError> {
    fft_operations(values)
}

// Deriving a concrete field record must not restrict ordinary arithmetic on
// big-endian targets. Requesting its storage below must still fail there.
#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
pub struct FieldRecord {
    pub fp: Fp,
    pub fq: Fq,
}

#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
pub struct CurveRecord {
    pub pallas: PallasAffine,
    pub vesta: VestaAffine,
    pub cached: PreparedAffinePoint<Pallas>,
}

#[repr(transparent)]
#[derive(Clone, Copy)]
#[cfg_attr(any(feature = "record", feature = "zero-array"), derive(bento::Pod))]
pub struct Value(pub u64);

impl bento::addchain::AdditionChain for Value {
    fn double(&self) -> Self {
        Self(self.0.wrapping_mul(2))
    }

    fn add(&self, rhs: &Self) -> Self {
        Self(self.0.wrapping_add(rhs.0))
    }
}

// Compiles on either endianness when no storage operation is requested.
pub fn scale(value: Value) -> Value {
    bento::addition_chain!(value, 181)
}

#[cfg(feature = "record")]
bento::embed_struct! {
    pub static RECORD: Value = "record.bin";
}

#[cfg(feature = "empty-record")]
#[repr(C)]
#[derive(Clone, Copy, bento::Pod)]
pub struct Empty;

#[cfg(feature = "empty-record")]
pub static EMPTY: &Empty = bento::AlignedBytes([]).as_value();

#[cfg(feature = "primitive")]
pub static PRIMITIVE: &u64 = bento::AlignedBytes([0; 8]).as_value();

#[cfg(feature = "field")]
pub static STORED_FP: &Fp = bento::AlignedBytes([0; 32]).as_value();

#[cfg(feature = "field")]
pub static STORED_FQ: &Fq = bento::AlignedBytes([0; 32]).as_value();

#[cfg(feature = "field-record")]
pub static STORED_FIELDS: &FieldRecord = bento::AlignedBytes([0; 64]).as_value();

#[cfg(feature = "curve")]
pub static STORED_PALLAS: &PallasAffine = bento::AlignedBytes([0; 64]).as_value();

#[cfg(feature = "curve")]
pub static STORED_VESTA: &VestaAffine = bento::AlignedBytes([0; 64]).as_value();

#[cfg(feature = "curve-record")]
pub static STORED_CURVES: &CurveRecord = bento::AlignedBytes([0; 224]).as_value();

#[cfg(feature = "zero-array")]
pub static ZERO_ARRAY: &[Value; 0] = bento::AlignedBytes([]).as_array();

#[cfg(feature = "curve")]
pub static STORED_CACHED_PALLAS: &PreparedAffinePoint<Pallas> =
    bento::AlignedBytes([0; 96]).as_value();
#[cfg(feature = "curve")]
pub static STORED_CACHED_VESTA: &PreparedAffinePoint<Vesta> =
    bento::AlignedBytes([0; 96]).as_value();
