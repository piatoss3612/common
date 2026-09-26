use super::*;

fn small_transforms<M: PrimeModulus>() {
    // Debug must be available with only the public field-modulus bound.
    fn assert_debug<T: core::fmt::Debug>() {}
    assert_debug::<Domain<PastaField<M>>>();
    assert_debug::<CosetDomain<M>>();
    assert_debug::<Transform<'_, M>>();
    assert_debug::<Tables<'_, M>>();
    assert_debug::<TablesMut<'_, M>>();
    assert_debug::<Expansion<'_, M>>();
    assert_debug::<Class<'_, M>>();

    const OPTIONS: [Strategy; 3] = [
        Strategy::SERIAL,
        Strategy {
            tile_len: 1,
            columns_per_task: 3,
            max_tasks: 3,
        },
        Strategy {
            tile_len: 4,
            columns_per_task: 3,
            max_tasks: 5,
        },
    ];
    const SCRATCH: [[usize; OPTIONS.len()]; 7] = {
        let mut counts = [[0; OPTIONS.len()]; 7];
        let mut log = 0;
        while log < counts.len() {
            let mut index = 0;
            while index < OPTIONS.len() {
                counts[log][index] = match OPTIONS[index].requirements(1 << log) {
                    Ok(required) => required,
                    Err(_) => panic!("unsupported test configuration"),
                };
                index += 1;
            }
            log += 1;
        }
        counts
    };
    for log in 0..=6 {
        let subgroup = Domain::<PastaField<M>>::new(log).unwrap();
        let by_size = Domain::<PastaField<M>>::for_size(1 << log).unwrap();
        assert_eq!(by_size.size(), subgroup.size());
        assert_eq!(by_size.log_size(), log);
        assert_eq!((by_size.root()).reduce(), (subgroup.root()).reduce());
        assert_eq!(
            (by_size.inverse_root()).reduce(),
            (subgroup.inverse_root()).reduce()
        );
        assert_eq!(
            (by_size.size_inverse()).reduce(),
            (subgroup.size_inverse()).reduce()
        );
        let coefficients = inputs(subgroup.size());
        for coset in [false, true] {
            let domain = if coset {
                subgroup.coset()
            } else {
                subgroup.subgroup()
            };
            let shift: PastaField<M> = if coset {
                PastaField::ZETA
            } else {
                PastaField::ONE
            };
            assert_eq!(domain.shift().reduce(), shift.reduce());
            assert_eq!(
                domain.inverse_shift().reduce(),
                shift.invert().unwrap().reduce()
            );
            let prepared = Prepared::new(domain);
            let expected = direct(&coefficients, domain);
            // Every table is independently optional, including combinations
            // where final scales are prepared but ordinary twiddles are not.
            for mask in 0..8 {
                let tables = Tables {
                    forward: (mask & 1 != 0).then_some(prepared.forward.as_slice()),
                    inverse: (mask & 2 != 0).then_some(prepared.inverse.as_slice()),
                    inverse_finish: (mask & 4 != 0).then_some(prepared.finish.as_slice()),
                };
                let plan = tables.bind(domain);
                for (index, options) in OPTIONS.into_iter().enumerate() {
                    let count = SCRATCH[log as usize][index];
                    assert_eq!(plan.scratch_requirements_with(options).unwrap(), count);
                    let sentinel = PastaField::from_u64(99);
                    let mut scratch = vec![sentinel; count + 3];
                    let mut output = vec![PastaField::ZERO; coefficients.len()];
                    plan.forward_into_with(
                        &coefficients,
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                    assert_eq!(reduced(&output), reduced(&expected), "forward log={log}");
                    assert_loose_bound(&output);
                    plan.inverse_with(&mut output, options, &SerialExecutor, &mut scratch)
                        .unwrap();
                    assert_eq!(
                        reduced(&output),
                        reduced(&coefficients),
                        "inverse log={log}"
                    );
                    let evaluations = expected.clone();
                    plan.inverse_into_with(
                        &evaluations,
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                    assert_eq!(
                        reduced(&output),
                        reduced(&coefficients),
                        "inverse_into log={log}"
                    );
                    assert_eq!(reduced(&evaluations), reduced(&expected));
                    // Construct the input order independently of Transform::permute.
                    for (row, value) in expected.iter().enumerate() {
                        let bits = (0..log).fold(0, |bits, bit| (bits << 1) | ((row >> bit) & 1));
                        output[bits] = *value;
                    }
                    plan.inverse_bit_reversed_with(
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                    assert_eq!(
                        reduced(&output),
                        reduced(&coefficients),
                        "inverse_bit_reversed log={log}"
                    );
                    assert_eq!(
                        bytes_of_slice(&scratch[count..]),
                        bytes_of_slice(&[sentinel; 3])
                    );
                    assert_loose_bound(&scratch);
                }
            }
        }
    }
}

#[test]
fn small_dfts_both_fields_all_table_modes() {
    small_transforms::<PallasBase>();
    small_transforms::<PallasScalar>();
}

fn prefixes<M: PrimeModulus>() {
    for log in 0..=8 {
        let domain = Domain::<PastaField<M>>::new(log).unwrap().coset();
        let prepared = Prepared::new(domain);
        let input = inputs(domain.size());
        for table in [Tables::default(), prepared.tables()] {
            let plan = table.bind(domain);
            for len in [
                0,
                1,
                2,
                3,
                5,
                domain.size() / 2,
                domain.size() - 1,
                domain.size(),
            ]
            .into_iter()
            .filter(|&len| len <= domain.size())
            {
                let expected = reference_coset(&input[..len], domain);
                for tile_len in [1, 4, 16, 256] {
                    let options = Strategy {
                        tile_len,
                        columns_per_task: 3,
                        max_tasks: 3,
                    };
                    let mut scratch =
                        vec![PastaField::ZERO; plan.scratch_requirements_with(options).unwrap()];
                    let mut output = vec![PastaField::ONE; domain.size()];
                    plan.forward_prefix_with(
                        &input[..len],
                        &mut output,
                        options,
                        &SerialExecutor,
                        &mut scratch,
                    )
                    .unwrap();
                    assert_eq!(
                        reduced(&output),
                        reduced(&expected),
                        "prefix log={log} len={len} tile={tile_len}"
                    );
                }
            }
        }
    }
}

#[test]
fn zero_suffix_skips_rounds_across_tile_boundaries() {
    prefixes::<PallasBase>();
    prefixes::<PallasScalar>();
}

fn source_sizes<M: PrimeModulus>() {
    for log in 11..=14 {
        let domain = Domain::<PastaField<M>>::new(log).unwrap().coset();
        let prepared = Prepared::new(domain);
        let plan = prepared.tables().bind(domain);
        let coefficients = inputs(domain.size());
        let expected = reference_coset(&coefficients, domain);
        let options = Strategy {
            tile_len: 2048,
            columns_per_task: 257,
            max_tasks: 3,
        };
        let mut scratch = vec![PastaField::ZERO; plan.scratch_requirements_with(options).unwrap()];
        let mut output = coefficients.clone();
        plan.forward_with(&mut output, options, &Threads, &mut scratch)
            .unwrap();
        assert_eq!(reduced(&output), reduced(&expected));
        plan.inverse_with(&mut output, options, &Threads, &mut scratch)
            .unwrap();
        assert_eq!(reduced(&output), reduced(&coefficients));
    }
}

#[test]
fn original_domain_sizes_with_scoped_parallel_execution() {
    source_sizes::<PallasBase>();
    source_sizes::<PallasScalar>();
}

fn larger_transform<M: PrimeModulus>() {
    // Verify the complete root ladder without allocating enormous transforms.
    for log in 1..=32 {
        let root = PastaField::<M>::root_of_unity(log).unwrap();
        let inverse = PastaField::<M>::root_of_unity_inverse(log).unwrap();
        assert_eq!(
            (root.square()).reduce(),
            (PastaField::<_>::root_of_unity(log - 1).unwrap()).reduce()
        );
        assert_eq!(
            (inverse.square()).reduce(),
            (PastaField::<_>::root_of_unity_inverse(log - 1).unwrap()).reduce()
        );
        assert_eq!(
            (root.mul(&inverse)).reduce(),
            (PastaField::<_>::ONE).reduce()
        );
        assert_eq!(
            (root.pow_u64(1u64 << (log - 1))).reduce(),
            (PastaField::<_>::ONE.neg()).reduce()
        );
    }
    let subgroup = Domain::<PastaField<M>>::for_size(1 << 16).unwrap();
    assert_eq!(subgroup.log_size(), 16);
    let domain = subgroup.coset();
    let prepared = Prepared::new(domain);
    let plan = prepared.tables().bind(domain);
    let coefficients = inputs(domain.size());
    let expected = reference_coset(&coefficients, domain);
    let options = Strategy {
        max_tasks: 8,
        ..Strategy::default()
    };
    let mut scratch = vec![PastaField::ZERO; plan.scratch_requirements_with(options).unwrap()];
    let executor = CountJoins::default();
    let mut output = coefficients.clone();
    plan.forward_with(&mut output, options, &executor, &mut scratch)
        .unwrap();
    assert_eq!(reduced(&output), reduced(&expected));
    assert!(executor.0.load(Ordering::Relaxed) > 0);
    plan.inverse_with(&mut output, options, &executor, &mut scratch)
        .unwrap();
    assert_eq!(reduced(&output), reduced(&coefficients));

    let mut reference = coefficients.clone();
    reference::transform(&mut reference, &subgroup.root());
    reference::inverse_transform(
        &mut reference,
        &subgroup.inverse_root(),
        &subgroup.size_inverse(),
    );
    assert_eq!(reduced(&reference), reduced(&coefficients));
}

#[test]
fn larger_transforms_and_all_root_orders_match_reference() {
    larger_transform::<PallasBase>();
    larger_transform::<PallasScalar>();
}
