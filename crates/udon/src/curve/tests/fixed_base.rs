use super::{reference::Reference, *};
use crate::test_support::modulus;

fn tables<C: PastaCurve, E: CurveTableEntry<C>>() {
    let p = modulus::<C::Base>();
    // Use a nongenerator base as well as generator coverage in other tests.
    let base = *Point::<C>::GENERATOR.double().as_affine().unwrap();
    for window_bits in 2..=8 {
        let description = FixedBaseDescription { window_bits };
        let required = description.requirements().unwrap();
        let h = required.projective_scratch;
        let mut entries = vec![E::from_affine(&AffinePoint::GENERATOR); required.table_entries];
        let mut projective = vec![ProjectivePoint::GENERATOR; h + 2];
        let mut field = vec![PastaField::from_u64(77); h + 2];
        let table = FixedBaseTable::prepare_with(
            description,
            &base,
            &mut entries,
            &mut projective,
            &mut field,
        )
        .unwrap();
        assert_eq!(table.description(), description);
        assert_eq!(*table.base(), base);
        assert_eq!(table.as_slice().len(), required.table_entries);
        assert_eq!(&projective[h..], &[ProjectivePoint::GENERATOR; 2]);
        assert_eq!(&field[h..], &[PastaField::from_u64(77); 2]);
        table.validate().unwrap();
        // Every stored entry is checked using ordinary-integer affine formulas,
        // including the carry point and every partially filled top window.
        let mut reference_base = Reference::from_point(&base.to_point());
        for window in table.as_slice()[..required.table_entries - 1].chunks_exact(h) {
            let mut multiple = Reference::identity();
            for entry in window {
                multiple = multiple.add(&reference_base, &p);
                multiple.assert_point(&entry.affine().to_point());
            }
            for _ in 0..window_bits {
                reference_base = reference_base.add(&reference_base, &p);
            }
        }
        reference_base.assert_point(
            &table.as_slice()[required.table_entries - 1]
                .affine()
                .to_point(),
        );
        let mut scalars = scalar_corpus::<C>();
        // Sample full scalars around signed-window thresholds, including limb
        // boundaries. GLV changes their digits; direct recoder tests separately
        // exercise every half-width threshold and incoming-carry case.
        for bit in (window_bits as usize - 1..255).step_by(window_bits as usize) {
            let power =
                PastaField::from_canonical_uint(CanonicalUint::power_of_two(bit).unwrap()).unwrap();
            scalars.extend([
                power.sub(&PastaField::ONE),
                power,
                power.add(&PastaField::ONE),
            ]);
        }
        // These even-bit powers reach the final width-2 carry through the
        // second GLV half, with both signs. The window sweep above misses them.
        for bit in [176, 210, 214] {
            let power =
                PastaField::from_canonical_uint(CanonicalUint::power_of_two(bit).unwrap()).unwrap();
            scalars.extend([
                power.sub(&PastaField::ONE),
                power,
                power.add(&PastaField::ONE),
            ]);
        }
        let bound = FixedBaseTable::bind(description, &base, table.as_slice()).unwrap();
        let trusted = FixedBaseTable::bind_trusted(description, &base, table.as_slice()).unwrap();
        let mut carries = [[false; 2]; 2];
        for scalar in scalars {
            let (a, b) = glv_decompose::<C>(&scalar);
            for (rotation, half) in [a, b].into_iter().enumerate() {
                let (_, carry) = crate::curve::fixed_base::signed_window_digits(
                    half.unsigned_abs(),
                    window_bits as usize,
                );
                carries[rotation][usize::from(half < 0)] |= carry;
            }
            // Keep the oracle independent of GLV decomposition and recoding.
            let expected = crate::curve::scalar::multiply(&scalar, |sum| sum.add_mixed(&base));
            assert_eq!(
                table.mul(&scalar),
                expected,
                "width {window_bits}, scalar {scalar:?}"
            );
            assert_eq!(bound.mul(&scalar), expected);
            assert_eq!(trusted.mul(&scalar), expected);
        }
        assert_eq!(
            carries,
            [[false; 2], [window_bits == 2; 2]],
            "final-carry coverage at width {window_bits}"
        );
    }
}

#[test]
fn pallas_fixed_base_layouts_and_signed_digits() {
    tables::<Pallas, PallasAffine>();
    tables::<Pallas, PreparedAffinePoint<Pallas>>();
}
#[test]
fn vesta_fixed_base_layouts_and_signed_digits() {
    tables::<Vesta, VestaAffine>();
    tables::<Vesta, PreparedAffinePoint<Vesta>>();
}

fn rejections<C: PastaCurve>() {
    let description = FixedBaseDescription::default();
    let required = description.requirements().unwrap();
    assert_eq!(
        required,
        CurveTableRequirements {
            table_entries: 257,
            projective_scratch: 8,
            field_scratch: 8
        }
    );
    assert_eq!(
        FixedBaseDescription { window_bits: 8 }
            .requirements()
            .unwrap(),
        CurveTableRequirements {
            table_entries: 2049,
            projective_scratch: 128,
            field_scratch: 128
        }
    );
    let base = AffinePoint::<C>::GENERATOR;
    let invalid = AffinePoint {
        x: invalid_field(),
        ..base
    };
    let off_curve = AffinePoint {
        y: PastaField::ZERO,
        ..base
    };
    let mut entries = vec![base; required.table_entries];
    let mut projective = vec![ProjectivePoint::GENERATOR; required.projective_scratch];
    let mut field = vec![PastaField::from_u64(17); required.field_scratch];
    // Cover every fallible preparation check and compare all buffers, including
    // scratch that would have been modified had validation run too late.
    for (description, base, entry_len, projective_len, field_len, expected) in [
        (
            FixedBaseDescription { window_bits: 0 },
            base,
            257,
            8,
            8,
            CurveError::InvalidWindowBits { bits: 0 },
        ),
        (
            FixedBaseDescription { window_bits: 9 },
            base,
            257,
            8,
            8,
            CurveError::InvalidWindowBits { bits: 9 },
        ),
        (description, invalid, 257, 8, 8, CurveError::InvalidBase),
        (description, off_curve, 257, 8, 8, CurveError::InvalidBase),
        (
            description,
            base,
            256,
            8,
            8,
            CurveError::LengthMismatch {
                buffer: "entries",
                expected: 257,
                actual: 256,
            },
        ),
        (
            description,
            base,
            257,
            7,
            8,
            CurveError::ScratchTooSmall {
                buffer: "projective",
                required: 8,
                provided: 7,
            },
        ),
        (
            description,
            base,
            257,
            8,
            7,
            CurveError::ScratchTooSmall {
                buffer: "field",
                required: 8,
                provided: 7,
            },
        ),
    ] {
        let old = (entries.clone(), projective.clone(), field.clone());
        assert_eq!(
            FixedBaseTable::prepare_with(
                description,
                &base,
                &mut entries[..entry_len],
                &mut projective[..projective_len],
                &mut field[..field_len]
            )
            .unwrap_err(),
            expected
        );
        assert_eq!((&entries, &projective, &field), (&old.0, &old.1, &old.2));
    }
    for bits in [0, 1, 9, u32::MAX] {
        let invalid_description = FixedBaseDescription { window_bits: bits };
        assert_eq!(
            FixedBaseTable::bind_trusted(invalid_description, &base, &entries).unwrap_err(),
            CurveError::InvalidWindowBits { bits }
        );
        assert_eq!(
            FixedBaseTable::bind(invalid_description, &base, &entries).unwrap_err(),
            CurveError::InvalidWindowBits { bits }
        );
    }
    for invalid_base in [invalid, off_curve] {
        assert_eq!(
            FixedBaseTable::bind(description, &invalid_base, &entries).unwrap_err(),
            CurveError::InvalidBase
        );
        assert_eq!(
            FixedBaseTable::bind_trusted(description, &invalid_base, &entries).unwrap_err(),
            CurveError::InvalidBase
        );
    }
    FixedBaseTable::prepare_with(
        description,
        &base,
        &mut entries,
        &mut projective,
        &mut field,
    )
    .unwrap();
    let valid = entries.clone();
    for index in [0, 7, 8, required.table_entries - 1] {
        for replacement in [invalid, off_curve, base.neg()] {
            entries[index] = replacement;
            assert_eq!(
                FixedBaseTable::bind(description, &base, &entries).unwrap_err(),
                CurveError::InvalidTable
            );
            let trusted = FixedBaseTable::bind_trusted(description, &base, &entries).unwrap();
            assert_eq!(trusted.validate(), Err(CurveError::InvalidTable));
            entries.copy_from_slice(&valid);
        }
    }
    entries.swap(0, 1);
    assert_eq!(
        FixedBaseTable::bind(description, &base, &entries).unwrap_err(),
        CurveError::InvalidTable
    );
    entries.copy_from_slice(&valid);
    assert_eq!(
        FixedBaseTable::bind(description, &base.neg(), &entries).unwrap_err(),
        CurveError::InvalidTable
    );
    for len in [0, 256, 258] {
        let mut wrong = valid.clone();
        wrong.resize(len, base);
        for error in [
            FixedBaseTable::bind(description, &base, &wrong).unwrap_err(),
            FixedBaseTable::bind_trusted(description, &base, &wrong).unwrap_err(),
        ] {
            assert_eq!(
                error,
                CurveError::LengthMismatch {
                    buffer: "entries",
                    expected: 257,
                    actual: len
                }
            );
        }
    }
    // Reusing all buffers starts from their prior, arbitrary scratch contents.
    FixedBaseTable::prepare_with(
        description,
        &base,
        &mut entries,
        &mut projective,
        &mut field,
    )
    .unwrap()
    .validate()
    .unwrap();
    assert_eq!(entries, valid);
}

#[test]
fn fixed_base_errors_preserve_buffers_and_reject_damaged_storage() {
    rejections::<Pallas>();
    rejections::<Vesta>();
}
