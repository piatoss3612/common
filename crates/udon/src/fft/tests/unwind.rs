use super::*;

#[test]
fn loose_values_remain_valid_on_unwind_and_serial_join_completes_both_jobs() {
    let a = <Fp>::ONE.neg();
    let b = <Fp>::from_u64(2).neg();
    let mut values = [a, b];
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            let (left, right) = values.split_at_mut(1);
            crate::field::pasta::butterfly::butterfly(&mut left[0], &mut right[0], None);
            panic!("interrupt loose region");
        }))
        .is_err()
    );
    assert_eq!(reduced(&values), reduced(&[a.add(&b), a.sub(&b)]));
    assert_loose_bound(&values);
    let count = AtomicUsize::new(0);
    assert!(
        catch_unwind(|| SerialExecutor.join(
            || panic!("first job"),
            || {
                count.fetch_add(1, Ordering::SeqCst);
            }
        ))
        .is_err()
    );
    assert_eq!(count.load(Ordering::SeqCst), 1);
}

#[test]
fn executor_panics_leave_public_buffers_within_loose_bounds() {
    struct Panics;
    impl Executor for Panics {
        fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
        where
            L: FnOnce() -> A + Send,
            R: FnOnce() -> B + Send,
            A: Send,
            B: Send,
        {
            SerialExecutor.join(left, right);
            panic!("executor failure");
        }
    }
    let domain = Domain::<Fp>::new(6).unwrap().subgroup();
    let plan = Transform::new(domain);
    let options = Strategy {
        tile_len: 8,
        columns_per_task: 4,
        max_tasks: 2,
    };
    let mut values = inputs(domain.size());
    let mut scratch = vec![Fp::ZERO; plan.scratch_requirements_with(options).unwrap()];
    assert!(
        catch_unwind(AssertUnwindSafe(|| plan.forward_with(
            &mut values,
            options,
            &Panics,
            &mut scratch
        )))
        .is_err()
    );
    assert_loose_bound(&values);
    assert_loose_bound(&scratch);
    // The private fused path also rejects a class interrupted by an executor.
    let mut class = Class::new(plan, &mut values, ElementOrder::Natural);
    assert!(
        catch_unwind(AssertUnwindSafe(|| interpolate_classes(
            &mut class,
            &mut [],
            options,
            &Panics,
            &mut scratch
        )))
        .is_err()
    );
    assert_loose_bound(class.values);
    assert_loose_bound(&scratch);
    let partial = class.values.to_vec();
    let scratch_before = scratch.clone();
    assert_eq!(
        interpolate_classes(&mut class, &mut [], options, &SerialExecutor, &mut scratch),
        Err(FftError::InvalidClassState)
    );
    assert_eq!(bytes_of_slice(class.values), bytes_of_slice(&partial));
    assert_eq!(bytes_of_slice(&scratch), bytes_of_slice(&scratch_before));
}

#[test]
fn inverse_panics_leave_outputs_and_scratch_within_loose_bounds() {
    fn check<M: PrimeModulus>() {
        let options = Strategy {
            tile_len: 16,
            columns_per_task: 3,
            max_tasks: 3,
        };
        let subgroup = Domain::<PastaField<M>>::new(7).unwrap();
        let input = inputs(subgroup.size());
        for coset in [false, true] {
            let domain = if coset {
                subgroup.coset()
            } else {
                subgroup.subgroup()
            };
            let prepared = Prepared::new(domain);
            for tables in [Tables::default(), prepared.tables()] {
                let plan = tables.bind(domain);
                let count = plan.scratch_requirements_with(options).unwrap();
                let mut scratch = vec![PastaField::ONE; count + 2];
                let joins = CountJoins::default();
                plan.inverse_with(&mut input.clone(), options, &joins, &mut scratch)
                    .unwrap();
                // Interrupt each scheduling boundary, including after terminal
                // kernels have stored their results and after only some
                // columns have been copied back to the caller's output.
                for index in 0..joins.take() {
                    let mut values = input.clone();
                    scratch.fill(PastaField::ONE);
                    let executor = FailAt {
                        calls: AtomicUsize::new(0),
                        index,
                    };
                    assert!(
                        catch_unwind(AssertUnwindSafe(|| plan.inverse_with(
                            &mut values,
                            options,
                            &executor,
                            &mut scratch,
                        )))
                        .is_err()
                    );
                    assert_loose_bound(&values);
                    assert_loose_bound(&scratch);
                    assert_eq!(
                        bytes_of_slice(&scratch[count..]),
                        bytes_of_slice(&[PastaField::<M>::ONE; 2])
                    );
                }
            }
        }
    }
    check::<PallasBase>();
    check::<PallasScalar>();
}

#[test]
fn nested_expansion_panics_leave_all_scratch_partitions_canonical() {
    let base = Transform::new(Domain::<Fp>::new(6).unwrap().subgroup());
    let expansion = Expansion::new(base, Domain::new(9).unwrap().subgroup(), None).unwrap();
    let options = ExpansionStrategy {
        max_residue_tasks: 3,
        transform: Strategy {
            tile_len: 8,
            columns_per_task: 3,
            max_tasks: 3,
        },
    };
    let input = inputs(base.domain().size());
    let mut scratch = vec![Fp::ONE; expansion.coefficient_scratch_with(options).unwrap()];
    let count = CountJoins::default();
    base.inverse_with(&mut input.clone(), options.transform, &count, &mut scratch)
        .unwrap();
    let inverse_joins = count.take();
    let factors = vec![Fp::ONE; expansion.layout().size()];
    let factor = expansion.view(&factors);
    for operation in 0..3 {
        let mut output = factors.clone();
        let executor = FailAt {
            calls: AtomicUsize::new(0),
            // For evaluation input, reach the residue jobs after the inverse.
            index: if operation == 1 { inverse_joins + 1 } else { 1 },
        };
        assert!(
            catch_unwind(AssertUnwindSafe(|| match operation {
                0 => expansion.coefficients_with(
                    &input,
                    &mut output,
                    options,
                    &executor,
                    &mut scratch,
                ),
                1 => expansion.evaluations_with(
                    &input,
                    &mut output,
                    options,
                    &executor,
                    &mut scratch,
                ),
                _ => expansion.short_product_with(
                    &input[..5],
                    factor,
                    &mut output,
                    options,
                    &executor,
                    &mut scratch,
                ),
            }))
            .is_err()
        );
        assert_loose_bound(&output);
        assert_loose_bound(&scratch);
    }
}
