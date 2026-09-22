use super::{reference::Reference, *};
use crate::curve::eisenstein::REPRESENTATIVES;
use crate::test_support::modulus;
use num_bigint::BigUint;

fn compact<C: PastaCurve, E: CurveTableEntry<C> + Eq>() {
    let generator = AffinePoint::<C>::GENERATOR;
    for base in [
        generator,
        generator.neg(),
        generator.endomorphism(),
        *generator.to_point().double().as_affine().unwrap(),
    ] {
        let mut entries = [E::from_affine(&generator); 8];
        let mut projective = [ProjectivePoint::GENERATOR; 10];
        let mut field = [PastaField::ONE; 10];
        let table =
            EisensteinTable::prepare(&base, &mut entries, &mut projective, &mut field).unwrap();
        assert_eq!(table.base(), &base);
        assert_eq!(projective[8..], [ProjectivePoint::GENERATOR; 2]);
        assert_eq!(field[8..], [PastaField::ONE; 2]);
        table.validate().unwrap();
        let bound = EisensteinTable::bind(&base, table.as_slice()).unwrap();
        let trusted = EisensteinTable::bind_trusted(&base, table.as_slice()).unwrap();
        let p = modulus::<C::Base>();
        let reference = Reference::from_point(&base.to_point());
        let phi = Reference::from_point(&base.endomorphism().to_point());
        // Independent affine formulas establish representative ordering.
        for (&(a, b), entry) in REPRESENTATIVES.iter().zip(table.as_slice()) {
            let first = reference.mul(&BigUint::from(a as u8), &p);
            let mut second = phi.mul(&BigUint::from(b.unsigned_abs()), &p);
            if b < 0
                && let Some((_, y)) = second.coordinates.as_mut()
            {
                *y = &p - &*y;
            }
            first
                .add(&second, &p)
                .assert_point(&entry.affine().to_point());
        }
        for scalar in scalar_corpus::<C>() {
            let expected = crate::curve::scalar::multiply(&scalar, |sum| sum.add_mixed(&base));
            assert_eq!(table.mul(&scalar), expected);
            assert_eq!(bound.mul(&scalar), expected);
            assert_eq!(trusted.mul(&scalar), expected);
        }
        // The view does not retain either scratch borrow.
        projective.fill(ProjectivePoint::IDENTITY);
        field.fill(PastaField::ZERO);
        assert_eq!(table.mul(&PastaField::ONE), base.to_projective());
    }
}

fn errors<C: PastaCurve, E: CurveTableEntry<C> + Eq>() {
    let base = AffinePoint::<C>::GENERATOR;
    let invalid_base = AffinePoint {
        x: invalid_field(),
        ..base
    };
    let off_curve = AffinePoint {
        y: PastaField::ZERO,
        ..base
    };
    let mut entries = [E::from_affine(&base); 9];
    let mut projective = [ProjectivePoint::GENERATOR; 9];
    let mut field = [PastaField::ONE; 9];
    for (base, len, plen, flen) in [
        (base, 7, 8, 8),
        (base, 9, 8, 8),
        (base, 8, 7, 8),
        (base, 8, 8, 7),
        (invalid_base, 8, 8, 8),
        (off_curve, 8, 8, 8),
    ] {
        let old = (entries, projective, field);
        assert!(
            EisensteinTable::prepare(
                &base,
                &mut entries[..len],
                &mut projective[..plen],
                &mut field[..flen]
            )
            .is_err()
        );
        assert_eq!((entries, projective, field), old);
    }
    for len in [0, 7, 9] {
        assert!(EisensteinTable::bind(&base, &entries[..len]).is_err());
        assert!(EisensteinTable::bind_trusted(&base, &entries[..len]).is_err());
    }
    for base in [invalid_base, off_curve] {
        assert_eq!(
            EisensteinTable::bind(&base, &entries[..8]).unwrap_err(),
            CurveError::InvalidBase
        );
        assert_eq!(
            EisensteinTable::bind_trusted(&base, &entries[..8]).unwrap_err(),
            CurveError::InvalidBase
        );
    }
    EisensteinTable::prepare(&base, &mut entries[..8], &mut projective, &mut field).unwrap();
    let valid = entries;
    for index in 0..8 {
        entries[index] = E::from_affine(&base.neg());
        assert_eq!(
            EisensteinTable::bind(&base, &entries[..8]).unwrap_err(),
            CurveError::InvalidTable
        );
        assert_eq!(
            EisensteinTable::bind_trusted(&base, &entries[..8])
                .unwrap()
                .validate(),
            Err(CurveError::InvalidTable)
        );
        entries = valid;
    }
    entries.swap(0, 1);
    assert!(EisensteinTable::bind(&base, &entries[..8]).is_err());
    EisensteinTable::prepare(&base, &mut entries[..8], &mut projective, &mut field)
        .unwrap()
        .validate()
        .unwrap();
    assert_eq!(entries, valid);
}

fn caches<C: PastaCurve>() {
    let base = AffinePoint::<C>::GENERATOR;
    let description = FixedBaseDescription::default();
    let required = description.requirements().unwrap();
    let mut expanded = vec![PreparedAffinePoint::from_affine(&base); required.table_entries];
    let mut compact = [PreparedAffinePoint::from_affine(&base); 8];
    let mut projective = [ProjectivePoint::IDENTITY; 8];
    let mut field = [PastaField::ZERO; 8];
    FixedBaseTable::prepare_with(
        description,
        &base,
        &mut expanded,
        &mut projective,
        &mut field,
    )
    .unwrap();
    EisensteinTable::prepare(&base, &mut compact, &mut projective, &mut field).unwrap();
    // Preserve x/y while corrupting only the cache, then independently corrupt
    // either affine coordinate. Include reduced and unreduced damage.
    for coordinate in 0..3 {
        for bytes in [
            [0xff; 32],
            [0; 32],
            *bento::bytes_of(&PastaField::<C::Base>::ONE)
                .first_chunk::<32>()
                .unwrap(),
        ] {
            let mut raw = bento::AlignedBytes([0; 96]);
            raw.0.copy_from_slice(bento::bytes_of(&compact[0]));
            raw.0[coordinate * 32..(coordinate + 1) * 32].copy_from_slice(&bytes);
            let damaged: PreparedAffinePoint<C> =
                *std::boxed::Box::leak(std::boxed::Box::new(raw)).as_value();
            if coordinate == 1 {
                assert!(!damaged.valid_cache());
                assert_eq!(damaged.affine(), compact[0].affine());
            }
            let old_expanded = expanded[0];
            let old_compact = compact[0];
            expanded[0] = damaged;
            compact[0] = damaged;
            assert_eq!(
                FixedBaseTable::bind(description, &base, &expanded).unwrap_err(),
                CurveError::InvalidTable
            );
            assert_eq!(
                EisensteinTable::bind(&base, &compact).unwrap_err(),
                CurveError::InvalidTable
            );
            assert_eq!(
                FixedBaseTable::bind_trusted(description, &base, &expanded)
                    .unwrap()
                    .validate(),
                Err(CurveError::InvalidTable)
            );
            assert_eq!(
                EisensteinTable::bind_trusted(&base, &compact)
                    .unwrap()
                    .validate(),
                Err(CurveError::InvalidTable)
            );
            expanded[0] = old_expanded;
            compact[0] = old_compact;
        }
    }
}

#[test]
fn compact_tables_match_representatives_and_multiplication() {
    compact::<Pallas, PallasAffine>();
    compact::<Pallas, PreparedAffinePoint<Pallas>>();
    compact::<Vesta, VestaAffine>();
    compact::<Vesta, PreparedAffinePoint<Vesta>>();
}

#[test]
fn compact_errors_preserve_buffers_and_caches_are_checked() {
    errors::<Pallas, PallasAffine>();
    errors::<Pallas, PreparedAffinePoint<Pallas>>();
    errors::<Vesta, VestaAffine>();
    errors::<Vesta, PreparedAffinePoint<Vesta>>();
    caches::<Pallas>();
    caches::<Vesta>();
}
