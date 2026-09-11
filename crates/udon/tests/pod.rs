//! Field storage through the public APIs, also exercised under Miri.
#![forbid(unsafe_code)]
#![cfg(target_endian = "little")]

use std::sync::OnceLock;

use bento::{AlignedBytes, bytes_of, bytes_of_slice};
use zakura_udon::{
    STORED_FORM, StoredForm,
    field::{Fp, Fq, PallasBase, PallasScalar, PastaField, PrimeModulus},
    stored_form,
};

#[test]
fn fields_borrow_their_reduced_montgomery_bytes() {
    fn check<M: PrimeModulus>() {
        assert_eq!(size_of::<PastaField<M>>(), 32);
        assert_eq!(align_of::<PastaField<M>>(), 8);
        for value in [
            PastaField::<M>::ZERO,
            PastaField::ONE,
            PastaField::from_u64(7),
            PastaField::ONE.neg(),
        ] {
            let expected: Vec<_> = value
                .montgomery_limbs()
                .into_iter()
                .flat_map(u64::to_le_bytes)
                .collect();
            assert_eq!(bytes_of(&value), expected);
            assert_eq!(
                bytes_of(&value).as_ptr(),
                core::ptr::from_ref(&value).cast()
            );
        }

        // A stored residue of one represents R^-1, not the field's one.
        static BYTES: AlignedBytes<32> = AlignedBytes({
            let mut bytes = [0; 32];
            bytes[0] = 1;
            bytes
        });
        let value: &PastaField<M> = BYTES.as_value();
        assert_eq!(value.montgomery_limbs(), [1, 0, 0, 0]);
        assert_ne!(*value, PastaField::ONE);
        assert_eq!(value.mul(&PastaField::ONE), *value);
        assert_eq!(bytes_of(value).as_ptr(), BYTES.0.as_ptr());
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Eq, PartialEq, bento::Pod)]
struct Record {
    fp: Fp,
    fq: Fq,
    values: [Fp; 2],
}

#[test]
fn field_arrays_and_nested_records_round_trip() {
    let record = Record {
        fp: Fp::from_u64(7),
        fq: Fq::ONE.neg(),
        values: [Fp::ONE, Fp::from_u64(u64::MAX)],
    };
    assert_eq!(size_of::<Record>(), 128);
    static RECORD: OnceLock<AlignedBytes<128>> = OnceLock::new();
    let bytes = RECORD.get_or_init(|| AlignedBytes(bytes_of(&record).try_into().unwrap()));
    let stored: &Record = bytes.as_value();
    assert_eq!(*stored, record);
    assert_eq!(bytes_of(stored).as_ptr(), bytes.0.as_ptr());
    assert_eq!(stored.fp.add(&stored.values[0]), Fp::from_u64(8));
    assert_eq!(stored.fq.add(&Fq::ONE), Fq::ZERO);

    static ARRAY: OnceLock<AlignedBytes<64>> = OnceLock::new();
    let bytes =
        ARRAY.get_or_init(|| AlignedBytes(bytes_of_slice(&record.values).try_into().unwrap()));
    let stored: &[Fp; 2] = bytes.as_array();
    assert_eq!(*stored, record.values);
    assert_eq!(bytes_of_slice(stored).as_ptr(), bytes.0.as_ptr());
    assert!(bytes_of_slice::<Fp>(&[]).is_empty());
    assert!(bytes_of_slice::<Fq>(&[]).is_empty());
}

#[test]
fn descriptors_agree_across_supported_pointer_widths() {
    assert_eq!(StoredForm::ALL, &[StoredForm::MontU64x4]);
    assert_eq!(StoredForm::ACTIVE.descriptor(), "mont-u64x4");
    assert_eq!(STORED_FORM, stored_form!());
    assert_eq!(STORED_FORM, StoredForm::ACTIVE.descriptor());
    assert_eq!(
        concat!("values-", stored_form!(), ".bin"),
        "values-mont-u64x4.bin"
    );
    for width in ["32", "64"] {
        assert_eq!(StoredForm::for_target(width), StoredForm::ACTIVE);
    }
}
