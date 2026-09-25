use super::*;

fn reference<C: PastaCurve, E: CurveTableEntry<C>>(
    tables: &[FixedBaseTable<'_, C, E>],
    scalars: &[PastaField<C::Scalar>],
) -> ProjectivePoint<C> {
    tables
        .iter()
        .zip(scalars)
        .fold(ProjectivePoint::IDENTITY, |sum, (table, scalar)| {
            sum.add(&multiply(scalar, |sum| sum.add_mixed(table.base())))
        })
}

fn check<C: PastaCurve, E: CurveTableEntry<C>>() {
    let g = AffinePoint::<C>::GENERATOR;
    let bases = [g, g, g.neg(), g.endomorphism(), g, g.neg(), g];
    let mut storage = Vec::new();
    for (i, base) in bases.iter().enumerate() {
        let description = FixedBaseDescription {
            window_bits: i as u32 + 2,
        };
        let n = description.requirements().unwrap().table_entries;
        let mut entries = vec![E::from_affine(&g); n];
        FixedBaseTable::prepare_with(
            description,
            base,
            &mut entries,
            &mut vec![ProjectivePoint::IDENTITY; n],
            &mut vec![PastaField::ZERO; n],
        )
        .unwrap();
        storage.push(entries);
    }
    let tables: Vec<_> = storage
        .iter()
        .enumerate()
        .map(|(i, entries)| {
            FixedBaseTable::bind(
                FixedBaseDescription {
                    window_bits: i as u32 + 2,
                },
                &bases[i],
                entries,
            )
            .unwrap()
        })
        .collect();
    let mut corpus = vec![
        vec![PastaField::ZERO; bases.len()],
        vec![PastaField::ONE; bases.len()],
        vec![PastaField::from_montgomery_limbs(C::Scalar::MODULUS); bases.len()],
        field_samples::<C::Scalar>().take(bases.len()).collect(),
        vec![
            PastaField::ONE,
            PastaField::<_>::ONE.neg(),
            PastaField::from_u64(2),
            PastaField::from_u64(17),
            PastaField::from_u64(u64::MAX),
            PastaField::from_i64(-7),
            PastaField::ZERO,
        ],
    ];
    for bit in [176, 210, 214] {
        let power =
            PastaField::from_canonical_uint(CanonicalUint::power_of_two(bit).unwrap()).unwrap();
        corpus.push(vec![power; bases.len()]);
        corpus.push(vec![power.neg(); bases.len()]);
    }
    let loose = PastaField::from_montgomery_limbs([u64::MAX, u64::MAX, u64::MAX, (1 << 63) - 1]);
    corpus.push(vec![loose; bases.len()]);
    for scalars in &corpus {
        for n in 0..=tables.len() {
            let expected = reference(&tables[..n], &scalars[..n]);
            for capacity in [1, 2, 3, 7, 16, 65, 1024] {
                let mut points = vec![g; capacity + 1];
                let mut fields = vec![PastaField::ONE; capacity + 2];
                assert_eq!(
                    FixedBaseTable::sum(
                        &tables[..n],
                        &scalars[..n],
                        &mut points[..capacity],
                        &mut fields[..capacity],
                    ),
                    expected,
                    "n={n}, capacity={capacity}"
                );
                assert_eq!(points[capacity], g);
                assert!(fields[capacity..].iter().all(PastaField::is_one));
            }
        }
    }
    let count = FixedBaseTable::sum_scratch_len(&tables).unwrap();
    assert_eq!(
        count,
        [129_usize, 86, 64, 52, 44, 38, 32].iter().sum::<usize>()
    );
    let mut points = vec![g; count + 3];
    let mut fields = vec![PastaField::ONE; count + 5];
    assert_eq!(
        FixedBaseTable::sum(&tables, &corpus[3], &mut points, &mut fields),
        reference(&tables, &corpus[3]),
    );
    assert_eq!(&points[count..], &[g; 3]);
    assert!(fields[count..].iter().all(PastaField::is_one));
    for scalars in &corpus {
        assert_eq!(
            FixedBaseTable::sum(&tables, scalars, &mut points[..65], &mut fields[..65]),
            reference(&tables, scalars),
        );
    }
    for len in 0..70 {
        for shape in 0..3 {
            let mut points: Vec<_> = (0..len)
                .map(|i| {
                    if (shape == 1 && i % 2 == 1) || (shape == 2 && i >= len / 2) {
                        g.neg()
                    } else {
                        g
                    }
                })
                .collect();
            let expected = points
                .iter()
                .fold(ProjectivePoint::IDENTITY, |sum, p| sum.add_mixed(p));
            let result = crate::curve::reduce::sum(&mut points, &mut vec![PastaField::ZERO; len]);
            assert_eq!(result, expected);
        }
    }
}

#[test]
fn expanded_sum_selection_and_complete_trees() {
    check::<Pallas, AffinePoint<Pallas>>();
    check::<Vesta, AffinePoint<Vesta>>();
    check::<Pallas, crate::curve::PreparedAffinePoint<Pallas>>();
    check::<Vesta, crate::curve::PreparedAffinePoint<Vesta>>();
}

fn contracts<C: PastaCurve>() {
    let g = AffinePoint::<C>::GENERATOR;
    let mut entries = [g; 256];
    let table = FixedBaseTable::prepare_with(
        FixedBaseDescription::default(),
        &g,
        &mut entries,
        &mut [ProjectivePoint::IDENTITY; 8],
        &mut [PastaField::ZERO; 8],
    )
    .unwrap();
    assert_eq!(FixedBaseTable::<C>::sum_scratch_len(&[]), Ok(0));
    assert!(FixedBaseTable::<C>::sum(&[], &[], &mut [], &mut []).is_identity());
    // Empty support and zero scalars leave all supplied storage untouched.
    let mut points = [g; 130];
    let mut fields = [PastaField::from_u64(71); 129];
    for (tables, scalars) in [(&[][..], &[][..]), (&[table][..], &[PastaField::ZERO][..])] {
        assert!(FixedBaseTable::sum(tables, scalars, &mut points, &mut fields).is_identity());
        assert_eq!(points, [g; 130]);
        assert_eq!(
            bento::bytes_of_slice(&fields),
            bento::bytes_of_slice(&[PastaField::<C::Base>::from_u64(71); 129])
        );
    }
    for (tables, scalars, affine_len, field_len) in [
        (0, 1, 130, 129),
        (1, 0, 130, 129),
        (1, 2, 130, 129),
        (1, 1, 0, 129),
        (1, 1, 130, 0),
        (1, 1, 0, 0),
    ] {
        let old_fields = fields;
        assert!(
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                FixedBaseTable::sum(
                    &[table][..tables],
                    &[PastaField::ONE; 2][..scalars],
                    &mut points[..affine_len],
                    &mut fields[..field_len],
                );
            }))
            .is_err()
        );
        assert_eq!(points, [g; 130]);
        assert_eq!(
            bento::bytes_of_slice(&fields),
            bento::bytes_of_slice(&old_fields)
        );
    }
    let scalars = [PastaField::from_u64(87); 2];
    for (affine_len, field_len) in [(1, 129), (130, 1), (3, 129), (130, 3)] {
        points.fill(g);
        fields.fill(PastaField::from_u64(71));
        let count = affine_len.min(field_len);
        let result = FixedBaseTable::sum(
            &[table, table],
            &scalars,
            &mut points[..affine_len],
            &mut fields[..field_len],
        );
        assert_eq!(result, reference(&[table, table], &scalars));
        assert!(points[count..].iter().all(|p| *p == g));
        assert!(
            fields[count..]
                .iter()
                .all(|f| f.reduce() == PastaField::<C::Base>::from_u64(71).reduce())
        );
    }
}

#[test]
fn expanded_sum_counts_and_scratch_contracts() {
    contracts::<Pallas>();
    contracts::<Vesta>();
}
