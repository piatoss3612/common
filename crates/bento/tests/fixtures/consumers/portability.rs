#![no_std]
#![deny(warnings)]
#![forbid(unsafe_code)]

use bento::const_arithmetic::{U256, U320, m255, u256};
use udon::fft::{
    Class, Domain, ExecutionOptions, Expansion, ExpansionOptions, FftError, InputOrder, Plan,
    SerialExecutor, TableRequirements, TablesMut, interpolate_classes,
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
        InputOrder::BitReversed,
    )?;
    // Residue-major evaluations scatter directly into interpolation order.
    for (residue, values) in evaluations.chunks_exact(FFT_SIZE).enumerate() {
        output.scatter_strided(residue, 2, values)?;
    }
    let mut lifts = [Class::new(plan, values, InputOrder::Natural)?];
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

#[cfg(feature = "zero-array")]
pub static ZERO_ARRAY: &[Value; 0] = bento::AlignedBytes([]).as_array();
