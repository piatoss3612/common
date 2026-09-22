//! Field and affine point storage through the public APIs, also under Miri.
#![forbid(unsafe_code)]
#![cfg(target_endian = "little")]

use std::sync::OnceLock;

use bento::{AlignedBytes, bytes_of, bytes_of_slice};
use zakura_udon::{
    STORED_FORM,
    curve::{
        AffinePoint, Pallas, PallasAffine, PastaCurve, PreparedAffinePoint, Vesta, VestaAffine,
    },
    field::{Fp, Fq, PallasBase, PallasScalar, PastaField, PrimeModulus, Reduced, ReductionState},
    stored_form,
};

fn borrow_field<M: PrimeModulus, S: ReductionState>(
    value: PastaField<M, S>,
    bytes: &'static AlignedBytes<32>,
) {
    assert_eq!(size_of::<PastaField<M, S>>(), 32);
    assert_eq!(align_of::<PastaField<M, S>>(), 8);
    let expected: Vec<_> = value
        .montgomery_limbs()
        .into_iter()
        .flat_map(u64::to_le_bytes)
        .collect();
    assert_eq!(bytes_of(&value), expected);
    let stored: &PastaField<M, S> = bytes.as_value();
    assert_eq!(stored.montgomery_limbs(), value.montgomery_limbs());
    assert_eq!(bytes_of(stored).as_ptr(), bytes.0.as_ptr());
    assert_eq!(stored.mul(&PastaField::<M>::ONE).reduce(), value.reduce());
}

#[test]
fn fields_borrow_exact_bytes_in_both_reduction_states() {
    type FieldBytes = [[AlignedBytes<32>; 2]; 6];

    fn check<M: PrimeModulus>(storage: &'static OnceLock<FieldBytes>) {
        // Preserve the largest loose representative without normalizing it.
        let mut upper = M::MODULUS;
        let mut carry = 0;
        for limb in &mut upper {
            let sum = u128::from(*limb) * 2 + carry;
            *limb = sum as u64;
            carry = sum >> 64;
        }
        upper[0] -= 1;
        let values = [
            PastaField::<M>::ZERO,
            PastaField::ONE,
            PastaField::from_u64(7),
            PastaField::<M>::ONE.neg(),
            PastaField::from_montgomery_limbs(M::MODULUS),
            PastaField::from_montgomery_limbs(upper),
        ];
        let bytes = storage.get_or_init(|| {
            values.map(|value| {
                [
                    AlignedBytes(bytes_of(&value).try_into().unwrap()),
                    AlignedBytes(bytes_of(&value.reduce()).try_into().unwrap()),
                ]
            })
        });
        for (value, [loose, reduced]) in values.into_iter().zip(bytes) {
            borrow_field(value, loose);
            borrow_field(value.reduce(), reduced);
        }

        // A stored residue of one represents R^-1, not the field's one.
        static BYTES: AlignedBytes<32> = AlignedBytes({
            let mut bytes = [0; 32];
            bytes[0] = 1;
            bytes
        });
        let value: &PastaField<M, Reduced> = BYTES.as_value();
        assert_eq!(value.montgomery_limbs(), [1, 0, 0, 0]);
        assert_ne!(*value, PastaField::ONE);
        assert_eq!(bytes_of(value).as_ptr(), BYTES.0.as_ptr());
    }
    static FP_BYTES: OnceLock<FieldBytes> = OnceLock::new();
    static FQ_BYTES: OnceLock<FieldBytes> = OnceLock::new();
    check::<PallasBase>(&FP_BYTES);
    check::<PallasScalar>(&FQ_BYTES);
}

#[test]
fn affine_points_borrow_ready_to_use_coordinate_bytes() {
    fn check<C: PastaCurve>(storage: &'static OnceLock<AlignedBytes<64>>) {
        assert_eq!(size_of::<AffinePoint<C>>(), 64);
        assert_eq!(align_of::<AffinePoint<C>>(), 8);
        let generator = AffinePoint::<C>::GENERATOR;
        let (x, y) = generator.coordinates();
        let mut expected = Vec::from(bytes_of(x));
        expected.extend_from_slice(bytes_of(y));
        assert_eq!(bytes_of(&generator), expected);
        assert!(bytes_of_slice::<AffinePoint<C>>(&[]).is_empty());
        let bytes = storage.get_or_init(|| AlignedBytes(bytes_of(&generator).try_into().unwrap()));
        let stored: &AffinePoint<C> = bytes.as_value();
        assert_eq!(*stored, generator);
        assert_eq!(
            stored.to_projective().double(),
            generator.to_projective().double()
        );
        assert_eq!(bytes_of(stored).as_ptr(), bytes.0.as_ptr());
    }
    static PALLAS_BYTES: OnceLock<AlignedBytes<64>> = OnceLock::new();
    static VESTA_BYTES: OnceLock<AlignedBytes<64>> = OnceLock::new();
    check::<Pallas>(&PALLAS_BYTES);
    check::<Vesta>(&VESTA_BYTES);
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, bento::Pod)]
struct CurveRecord {
    pallas: [PallasAffine; 2],
    vesta: [VestaAffine; 2],
}

#[test]
fn affine_arrays_in_nested_records_round_trip() {
    let record = CurveRecord {
        pallas: [PallasAffine::GENERATOR, PallasAffine::GENERATOR.neg()],
        vesta: [VestaAffine::GENERATOR, VestaAffine::GENERATOR.neg()],
    };
    assert_eq!(size_of::<CurveRecord>(), 256);
    static RECORD: OnceLock<AlignedBytes<256>> = OnceLock::new();
    let bytes = RECORD.get_or_init(|| AlignedBytes(bytes_of(&record).try_into().unwrap()));
    let stored: &CurveRecord = bytes.as_value();
    assert_eq!(*stored, record);
    assert_eq!(bytes_of(stored).as_ptr(), bytes.0.as_ptr());
    assert!(
        stored.pallas[0]
            .to_point()
            .add(&stored.pallas[1].to_point())
            .is_identity()
    );
    assert!(
        stored.vesta[0]
            .to_point()
            .add(&stored.vesta[1].to_point())
            .is_identity()
    );
}

#[repr(C)]
#[derive(Clone, Copy, Debug, bento::Pod)]
struct Record {
    fp: Fp,
    fq: Fq<Reduced>,
    values: [Fp; 2],
}

#[test]
fn field_arrays_and_nested_records_round_trip() {
    let record = Record {
        fp: Fp::from_u64(7),
        fq: <Fq>::ONE.neg().reduce(),
        values: [Fp::ONE, Fp::from_u64(u64::MAX)],
    };
    assert_eq!(size_of::<Record>(), 128);
    static RECORD: OnceLock<AlignedBytes<128>> = OnceLock::new();
    let bytes = RECORD.get_or_init(|| AlignedBytes(bytes_of(&record).try_into().unwrap()));
    let stored: &Record = bytes.as_value();
    assert_eq!(bytes_of(stored), bytes_of(&record));
    assert_eq!(bytes_of(stored).as_ptr(), bytes.0.as_ptr());
    assert_eq!(
        (stored.fp.add(&stored.values[0])).reduce(),
        (<Fp>::from_u64(8)).reduce()
    );
    assert_eq!((stored.fq.add(&<Fq>::ONE)).reduce(), (<Fq>::ZERO).reduce());

    static ARRAY: OnceLock<AlignedBytes<64>> = OnceLock::new();
    let bytes =
        ARRAY.get_or_init(|| AlignedBytes(bytes_of_slice(&record.values).try_into().unwrap()));
    let stored: &[Fp; 2] = bytes.as_array();
    assert_eq!(
        (*stored)
            .iter()
            .map(|value| value.reduce())
            .collect::<Vec<_>>(),
        (record.values)
            .iter()
            .map(|value| value.reduce())
            .collect::<Vec<_>>()
    );
    assert_eq!(bytes_of_slice(stored).as_ptr(), bytes.0.as_ptr());
    assert!(bytes_of_slice::<Fp>(&[]).is_empty());
    assert!(bytes_of_slice::<Fq>(&[]).is_empty());
}

#[test]
fn producer_and_consumer_filenames_agree() {
    let generated = format!("values-{STORED_FORM}.bin");
    const EMBEDDED: &str = concat!("values-", stored_form!(), ".bin");
    assert_eq!(generated, EMBEDDED);
    assert_eq!(EMBEDDED, "values-mont-u64x4.bin");
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, bento::Pod)]
struct CachedCurveRecord {
    pallas: [PreparedAffinePoint<Pallas>; 2],
    vesta: [PreparedAffinePoint<Vesta>; 2],
}

#[test]
fn cached_point_arrays_borrow_ready_to_use_bytes() {
    fn check<C: PastaCurve>(storage: &'static OnceLock<AlignedBytes<96>>) {
        assert_eq!(size_of::<PreparedAffinePoint<C>>(), 96);
        assert_eq!(align_of::<PreparedAffinePoint<C>>(), 8);
        let base = AffinePoint::<C>::GENERATOR;
        let cached = PreparedAffinePoint::from_affine(&base);
        assert_eq!(cached.to_affine(), base);
        let (x, y) = base.coordinates();
        let mut expected = Vec::from(bytes_of(x));
        expected.extend_from_slice(bytes_of(&x.mul(&PastaField::<C::Base>::ZETA).reduce()));
        expected.extend_from_slice(bytes_of(y));
        assert_eq!(bytes_of(&cached), expected);
        assert!(bytes_of_slice::<PreparedAffinePoint<C>>(&[]).is_empty());
        let bytes = storage.get_or_init(|| AlignedBytes(bytes_of(&cached).try_into().unwrap()));
        let stored: &PreparedAffinePoint<C> = bytes.as_value();
        assert_eq!(*stored, cached);
        assert_eq!(stored.to_affine(), base);
        assert_eq!(bytes_of(stored).as_ptr(), bytes.0.as_ptr());
    }
    static PALLAS_BYTES: OnceLock<AlignedBytes<96>> = OnceLock::new();
    static VESTA_BYTES: OnceLock<AlignedBytes<96>> = OnceLock::new();
    check::<Pallas>(&PALLAS_BYTES);
    check::<Vesta>(&VESTA_BYTES);
    let pallas = PallasAffine::GENERATOR;
    let vesta = VestaAffine::GENERATOR;
    let record = CachedCurveRecord {
        pallas: [
            PreparedAffinePoint::from_affine(&pallas),
            PreparedAffinePoint::from_affine(&pallas.neg()),
        ],
        vesta: [
            PreparedAffinePoint::from_affine(&vesta),
            PreparedAffinePoint::from_affine(&vesta.neg()),
        ],
    };
    assert_eq!(size_of::<CachedCurveRecord>(), 384);
    static RECORD: OnceLock<AlignedBytes<384>> = OnceLock::new();
    let bytes = RECORD.get_or_init(|| AlignedBytes(bytes_of(&record).try_into().unwrap()));
    let stored: &CachedCurveRecord = bytes.as_value();
    assert_eq!(*stored, record);
    assert_eq!(bytes_of(stored).as_ptr(), bytes.0.as_ptr());
}
