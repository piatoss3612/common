use super::{reference::Reference, *};
use crate::curve::eisenstein::REPRESENTATIVES;
use crate::test_support::modulus;
use num_bigint::BigUint;

fn compact<C: PastaCurve, E: CurveTableEntry<C> + Eq>() {
    let generator = AffinePoint::<C>::GENERATOR;
    let mut bases = Vec::from([
        generator,
        generator.neg(),
        generator.endomorphism(),
        *generator
            .to_point()
            .double()
            .to_point()
            .as_affine()
            .unwrap(),
    ]);
    bases.extend(
        field_samples::<C::Scalar>()
            .filter(|scalar| !scalar.is_zero())
            .take(8)
            .map(|scalar| {
                *multiply(&scalar, |sum| sum.add_mixed(&generator))
                    .to_point()
                    .as_affine()
                    .unwrap()
            }),
    );
    for base in bases {
        let mut entries = [E::from_affine(&generator); 8];
        let mut projective = [ProjectivePoint::GENERATOR; 10];
        let mut field = [PastaField::ONE; 10];
        let table = EisensteinTable::prepare(&base, &mut entries, &mut projective, &mut field);
        assert_eq!(table.base(), &base);
        assert_eq!(projective[8..], [ProjectivePoint::GENERATOR; 2]);
        assert_eq!(
            (field[8..])
                .iter()
                .map(|value| value.reduce())
                .collect::<Vec<_>>(),
            ([PastaField::<_>::ONE; 2])
                .iter()
                .map(|value| value.reduce())
                .collect::<Vec<_>>()
        );
        let bound = EisensteinTable::bind(&base, table.as_array());
        let p = modulus::<C::Base>();
        let reference = Reference::from_point(&base.to_point());
        let phi = Reference::from_point(&base.endomorphism().to_point());
        let scaled_representatives = representatives(&scaled(&base.to_point(), 11));
        // Independent affine formulas establish representative ordering.
        for ((&(a, b), entry), projective) in REPRESENTATIVES
            .iter()
            .zip(table.as_array())
            .zip(scaled_representatives)
        {
            let first = reference.mul(&BigUint::from(a as u8), &p);
            let mut second = phi.mul(&BigUint::from(b.unsigned_abs()), &p);
            if b < 0
                && let Some((_, y)) = second.coordinates.as_mut()
            {
                *y = &p - &*y;
            }
            let expected = first.add(&second, &p);
            expected.assert_point(&entry.affine().to_point());
            expected.assert_point(&projective.to_point());
        }
        for scalar in scalar_corpus::<C>() {
            let expected = multiply(&scalar, |sum| sum.add_mixed(&base));
            assert_eq!(table.mul(&scalar), expected);
            assert_eq!(bound.mul(&scalar), expected);
        }
        // The view does not retain either scratch borrow.
        projective.fill(ProjectivePoint::IDENTITY);
        field.fill(PastaField::ZERO);
        assert_eq!(table.mul(&PastaField::<_>::ONE), base.to_projective());
    }
}

fn scratch_lengths<C: PastaCurve, E: CurveTableEntry<C> + Eq>() {
    let base = AffinePoint::<C>::GENERATOR;
    let mut entries = [E::from_affine(&base); 8];
    let mut projective = [ProjectivePoint::GENERATOR; 9];
    let mut field = [PastaField::ONE; 9];
    for (plen, flen) in [(7, 8), (8, 7)] {
        let old = (
            entries,
            projective,
            field.map(|value| value.montgomery_limbs()),
        );
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                EisensteinTable::prepare(
                    &base,
                    &mut entries,
                    &mut projective[..plen],
                    &mut field[..flen],
                );
            }))
            .is_err()
        );
        assert_eq!(
            (
                entries,
                projective,
                field.map(|value| value.montgomery_limbs())
            ),
            old
        );
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
fn compact_scratch_length_errors_preserve_buffers() {
    scratch_lengths::<Pallas, PallasAffine>();
    scratch_lengths::<Pallas, PreparedAffinePoint<Pallas>>();
    scratch_lengths::<Vesta, VestaAffine>();
    scratch_lengths::<Vesta, PreparedAffinePoint<Vesta>>();
}
