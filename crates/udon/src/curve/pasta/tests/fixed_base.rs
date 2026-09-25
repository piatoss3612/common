use super::{reference::Reference, *};
use crate::field::pasta::test_support::{integer, modulus};

fn tables<C: PastaCurve, E: CurveTableEntry<C>>() {
    let p = modulus::<C::Base>();
    // Use a nongenerator base as well as generator coverage in other tests.
    let base = *Point::<C>::GENERATOR
        .double()
        .to_point()
        .as_affine()
        .unwrap();
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
            &mut projective[..h],
            &mut field[..h],
        )
        .unwrap();
        assert_eq!(table.description(), description);
        assert_eq!(*table.base(), base);
        assert_eq!(table.as_slice().len(), required.table_entries);
        assert_eq!(&projective[h..], &[ProjectivePoint::GENERATOR; 2]);
        assert_eq!(
            field[h..]
                .iter()
                .map(|value| value.reduce())
                .collect::<Vec<_>>(),
            [PastaField::<_>::from_u64(77); 2]
                .iter()
                .map(|value| value.reduce())
                .collect::<Vec<_>>()
        );
        // Every stored entry is checked using ordinary-integer affine formulas,
        // including the carry point and every partially filled top window.
        let mut reference_base = Reference::from_point(&base.to_point());
        let window_entries = 128_usize.div_ceil(window_bits as usize) * h;
        for window in table.as_slice()[..window_entries].chunks_exact(h) {
            let mut multiple = Reference::identity();
            for entry in window {
                multiple = multiple.add(&reference_base, &p);
                multiple.assert_point(&entry.affine().to_point());
            }
            for _ in 0..window_bits {
                reference_base = reference_base.add(&reference_base, &p);
            }
        }
        if window_bits == 2 {
            reference_base.assert_point(&table.as_slice()[window_entries].affine().to_point());
        }
        let mut scalars = scalar_corpus::<C>();
        // Scalar recoding must accept unreduced zero, one, and boundary limbs.
        let scalar_modulus = modulus::<C::Scalar>();
        let loose_one = (integer(&PastaField::<C::Scalar>::ONE.montgomery_limbs())
            + &scalar_modulus)
            .to_u64_digits();
        let loose_max = ((&scalar_modulus << 1_usize) - 1_u32).to_u64_digits();
        scalars.extend([
            PastaField::from_montgomery_limbs(C::Scalar::MODULUS),
            PastaField::from_montgomery_limbs(loose_one.try_into().unwrap()),
            PastaField::from_montgomery_limbs(loose_max.try_into().unwrap()),
            PastaField::from_montgomery_limbs([u64::MAX, u64::MAX, u64::MAX, (1 << 63) - 1]),
        ]);
        // Sample full scalars around signed-window thresholds, including limb
        // boundaries. GLV changes their digits; direct recoder tests separately
        // exercise every half-width threshold and incoming-carry case.
        for bit in (window_bits as usize - 1..255).step_by(window_bits as usize) {
            let power =
                PastaField::from_canonical_uint(CanonicalUint::power_of_two(bit).unwrap()).unwrap();
            scalars.extend([
                power.sub(&PastaField::<_>::ONE),
                power,
                power.add(&PastaField::<_>::ONE),
            ]);
        }
        // These even-bit powers reach the final width-2 carry through the
        // second GLV half, with both signs. The window sweep above misses them.
        for bit in [176, 210, 214] {
            let power =
                PastaField::from_canonical_uint(CanonicalUint::power_of_two(bit).unwrap()).unwrap();
            scalars.extend([
                power.sub(&PastaField::<_>::ONE),
                power,
                power.add(&PastaField::<_>::ONE),
            ]);
        }
        let bound = FixedBaseTable::bind(description, &base, table.as_slice()).unwrap();
        let mut carries = [[false; 2]; 2];
        for scalar in scalars {
            let (a, b) = glv_decompose::<C>(&scalar);
            for (rotation, half) in [a, b].into_iter().enumerate() {
                let (_, carry) = crate::curve::pasta::digits::signed_window_digits(
                    half.unsigned_abs(),
                    window_bits as usize,
                );
                carries[rotation][usize::from(half < 0)] |= carry;
            }
            // Keep the oracle independent of GLV decomposition and recoding.
            let expected = multiply(&scalar, |sum| sum.add_mixed(&base));
            assert_eq!(
                table.mul(&scalar),
                expected,
                "width {window_bits}, scalar {scalar:?}"
            );
            assert_eq!(bound.mul(&scalar), expected);
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
            table_entries: 256,
            projective_scratch: 8,
            field_scratch: 8
        }
    );
    assert_eq!(
        FixedBaseDescription { window_bits: 8 }
            .requirements()
            .unwrap(),
        CurveTableRequirements {
            table_entries: 2048,
            projective_scratch: 128,
            field_scratch: 128
        }
    );
    let base = AffinePoint::<C>::GENERATOR;
    let mut entries = vec![base; required.table_entries];
    let mut projective = vec![ProjectivePoint::GENERATOR; required.projective_scratch];
    let mut field = vec![PastaField::from_u64(17); required.field_scratch];
    // Cover every fallible preparation check and compare all buffers, including
    // scratch that would have been modified had validation run too late.
    for (description, base, entry_len, projective_len, field_len, expected) in [
        (
            FixedBaseDescription { window_bits: 0 },
            base,
            256,
            8,
            8,
            CurveError::InvalidWindowBits { bits: 0 },
        ),
        (
            FixedBaseDescription { window_bits: 9 },
            base,
            256,
            8,
            8,
            CurveError::InvalidWindowBits { bits: 9 },
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
        assert_eq!((&entries, &projective), (&old.0, &old.1));
        assert_eq!(bento::bytes_of_slice(&field), bento::bytes_of_slice(&old.2));
    }
    for (entry_len, projective_len, field_len) in [(255, 8, 8), (256, 7, 8), (256, 8, 7)] {
        let old = (entries.clone(), projective.clone(), field.clone());
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = FixedBaseTable::prepare_with(
                    description,
                    &base,
                    &mut entries[..entry_len],
                    &mut projective[..projective_len],
                    &mut field[..field_len],
                );
            }))
            .is_err()
        );
        assert_eq!((&entries, &projective), (&old.0, &old.1));
        assert_eq!(bento::bytes_of_slice(&field), bento::bytes_of_slice(&old.2));
    }
    for bits in [0, 1, 9, u32::MAX] {
        let invalid_description = FixedBaseDescription { window_bits: bits };
        assert_eq!(
            FixedBaseTable::bind(invalid_description, &base, &entries).unwrap_err(),
            CurveError::InvalidWindowBits { bits }
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
    for len in [0, 255, 257] {
        let mut wrong = valid.clone();
        wrong.resize(len, base);
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                let _ = FixedBaseTable::bind(description, &base, &wrong);
            }))
            .is_err()
        );
    }
    // Reusing all buffers starts from their prior, arbitrary scratch contents.
    FixedBaseTable::prepare_with(
        description,
        &base,
        &mut entries,
        &mut projective,
        &mut field,
    )
    .unwrap();
    assert_eq!(entries, valid);
}

#[test]
fn fixed_base_configuration_and_length_errors_preserve_buffers() {
    rejections::<Pallas>();
    rejections::<Vesta>();
}

fn preparation_batches<C: PastaCurve, E: CurveTableEntry<C> + bento::Pod>() {
    let base = *Point::<C>::GENERATOR
        .double()
        .to_point()
        .as_affine()
        .unwrap();
    for window_bits in 2..=8 {
        let description = FixedBaseDescription { window_bits };
        let required = description.requirements().unwrap();
        let n = 128_usize.div_ceil(window_bits as usize);
        let h = required.projective_scratch;
        let carry = usize::from(window_bits == 2);
        let total = required.table_entries;
        let sentinel = E::from_affine(&AffinePoint::GENERATOR);
        let mut reference = vec![sentinel; total];
        FixedBaseTable::prepare_with(
            description,
            &base,
            &mut reference,
            &mut vec![ProjectivePoint::IDENTITY; h],
            &mut vec![PastaField::ZERO; h],
        )
        .unwrap();
        // Cross window boundaries, exercise a short final batch, and include
        // width 2's carry both alone and alongside the last windows. Unequal
        // buffers must use the smaller allowance without changing the layout.
        for (projective_len, field_len, expected_inversions) in [
            (h, h, n + carry),
            (h + 1, h + 1, n),
            (2 * h, 2 * h, n.div_ceil(2) + carry),
            (2 * h + 1, 2 * h + 1, n.div_ceil(2)),
            (3 * h, 3 * h, n.div_ceil(3)),
            (n * h - 1, n * h - 1, 2),
            (n * h, n * h, 1 + carry),
            (total, total, 1),
            (total + 3, total + 5, 1),
            (total, h, n + carry),
            (h, total, n + carry),
        ] {
            let capacity = projective_len.min(field_len).min(total);
            // Enough retained capacity for width 8 must not override the
            // explicitly selected layout when extra scratch also fits it.
            let mut entries = vec![sentinel; 2050];
            let mut projective = vec![ProjectivePoint::GENERATOR; projective_len + 2];
            let mut field = vec![PastaField::from_u64(77); field_len + 2];
            let inversions = crate::field::count_inversions(|| {
                let table = FixedBaseTable::prepare_with(
                    description,
                    &base,
                    &mut entries,
                    &mut projective[..projective_len],
                    &mut field[..field_len],
                )
                .unwrap();
                assert_eq!(table.description(), description);
                assert_eq!(
                    bento::bytes_of_slice(table.as_slice()),
                    bento::bytes_of_slice(&reference)
                );
            });
            assert_eq!(
                inversions, expected_inversions,
                "width {window_bits}, projective {projective_len}, field {field_len}"
            );
            assert!(
                entries[total..]
                    .iter()
                    .all(|entry| bento::bytes_of(entry) == bento::bytes_of(&sentinel))
            );
            assert!(
                projective[capacity..]
                    .iter()
                    .all(|p| *p == ProjectivePoint::GENERATOR)
            );
            assert!(
                field[capacity..]
                    .iter()
                    .all(|f| f.reduce() == PastaField::from_u64(77))
            );
        }
    }
}

#[test]
fn fixed_base_scratch_allowance_controls_inversions_without_changing_entries() {
    preparation_batches::<Pallas, PallasAffine>();
    preparation_batches::<Pallas, PreparedAffinePoint<Pallas>>();
    preparation_batches::<Vesta, VestaAffine>();
    preparation_batches::<Vesta, PreparedAffinePoint<Vesta>>();
}

fn fixed_base<C: PastaCurve>() {
    let base = AffinePoint::<C>::GENERATOR;
    let scalar = PastaField::<C::Scalar>::TWO_INVERSE;
    for (capacity, scratch, window_bits) in [
        (129, 2, 2),
        (2048, 2, 2),
        (2047, 128, 7),
        (2048, 128, 8),
        (2048, 2048, 8),
    ] {
        let mut entries = vec![base.neg(); capacity + 1];
        let mut projective = vec![base.to_projective(); scratch + 1];
        let mut field = vec![PastaField::from_u64(42); scratch + 1];
        let table = FixedBaseTable::prepare(
            &base,
            &mut entries[..capacity],
            &mut projective[..scratch],
            &mut field[..scratch],
        )
        .unwrap();
        assert_eq!(table.description().window_bits, window_bits);
        assert_eq!(table.mul(&scalar), base.mul_projective(&scalar));
        let required = table.description().requirements().unwrap();
        assert!(
            entries[required.table_entries..]
                .iter()
                .all(|v| *v == base.neg())
        );
        assert!(
            projective[scratch..]
                .iter()
                .all(|v| *v == base.to_projective())
        );
        assert!(
            field[scratch..]
                .iter()
                .all(|v| v.reduce() == PastaField::from_u64(42))
        );
    }
}

#[test]
fn fixed_base_preparation_respects_retained_and_temporary_capacity() {
    fixed_base::<Pallas>();
    fixed_base::<Vesta>();
}
