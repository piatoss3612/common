//! Affine and prepared-point storage through the public POD APIs.
#![cfg(target_endian = "little")]
use bento::{AlignedBytes, bytes_of, bytes_of_slice};
use std::sync::OnceLock;
use zakura_udon::{
    curve::{
        AffinePoint, Pallas, PallasAffine, PastaCurve, PreparedAffinePoint, Vesta, VestaAffine,
    },
    field::PastaField,
};

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
