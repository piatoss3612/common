use crate::bridge::{
    executor::RayonExecutor,
    fft::{ClassBuilder, FftWorkspace, OwnedTables},
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use zakura_udon::{
    exec::{Executor, SerialExecutor, TaskBudget, for_each_mut},
    fft::{
        self, Class, ClassState, Domain, ElementOrder, ExecutionOptions, Expansion, ExpansionOrder,
        ExpansionStorage, ExpansionStrategy, InterpolationOptions, InverseScale, ResourceBudget,
        interpolate_classes_parallel, reference,
    },
    field::{PallasBase, PallasScalar, PastaField, PrimeModulus},
};

const N: usize = 2048;

fn coefficients<M: PrimeModulus>(size: usize, seed: usize) -> Vec<PastaField<M>> {
    (0..size)
        .map(|i| PastaField::from_u64((i * i + seed + 1) as u64).neg())
        .collect()
}

// The reference uses a different transform schedule from the blocked and
// bit-reversed paths under test, comparing their results in natural order.
fn evaluate<M: PrimeModulus>(
    coefficients: &[PastaField<M>],
    domain: fft::CosetDomain<M>,
) -> Vec<PastaField<M>> {
    let mut values = vec![PastaField::ZERO; domain.size()];
    let mut power = PastaField::ONE;
    for (value, coefficient) in values.iter_mut().zip(coefficients) {
        *value = coefficient.mul(&power);
        power = power.mul(&domain.shift());
    }
    reference::transform(&mut values, &domain.domain().root());
    values
}

fn geometry(budget: TaskBudget) -> ExecutionOptions {
    ExecutionOptions {
        tile_len: 64,
        columns_per_task: 2,
        max_tasks: budget.get(),
    }
}

fn pipeline<M: PrimeModulus>() {
    let base = OwnedTables::<M>::new(Domain::for_size(N).unwrap().subgroup());
    let classes: [_; 4] = core::array::from_fn(|i| {
        OwnedTables::<M>::new(
            Domain::for_size(N << i)
                .unwrap()
                .coset(PastaField::zeta())
                .unwrap(),
        )
    });
    let plans = classes.each_ref().map(OwnedTables::plan);
    let base_plan = base.plan();
    let expansion = Expansion::new(base_plan, plans[3].domain(), None).unwrap();
    let class_sizes = plans.map(|plan| plan.domain().size());
    let output_size = class_sizes[3];
    // Reference transforms depend on the inputs and domains, so all pool
    // configurations and workspace replays can share their results.
    let expansion_cases: [[_; 2]; 2] = core::array::from_fn(|repetition| {
        core::array::from_fn(|index| {
            let original = coefficients::<M>(N, index + repetition);
            let input = evaluate(&original, base_plan.domain());
            let expected = evaluate(&original, plans[3].domain());
            (original, input, expected)
        })
    });
    let products: [_; 2] = core::array::from_fn(|index| {
        let prefix = coefficients::<M>(16, index + 11);
        let expected = evaluate(&prefix, plans[3].domain());
        (prefix, expected)
    });
    let class_cases: [[_; 4]; 2] = core::array::from_fn(|repetition| {
        core::array::from_fn(|i| {
            let polynomial = coefficients::<M>(class_sizes[i], i + repetition + 7);
            let evaluations = evaluate(&polynomial, plans[i].domain());
            (polynomial, evaluations)
        })
    });
    let table_bytes = base.capacity_bytes()
        + classes
            .iter()
            .map(OwnedTables::capacity_bytes)
            .sum::<usize>();
    for threads in [1, 3, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let executor = RayonExecutor(&pool);
        let budget = TaskBudget::new(threads).unwrap();
        let (_, inner) = budget.partition(2).unwrap();
        let strategy = ExpansionStrategy {
            transform: geometry(inner),
            budget: ResourceBudget::for_tasks(inner.get()),
        };
        let retained = expansion
            .configure(
                ExpansionOrder::Residues,
                ExpansionStorage::CoefficientWorkspace {
                    scale: InverseScale::Unscaled,
                },
                strategy,
            )
            .unwrap();
        let forward = expansion
            .configure(
                ExpansionOrder::Residues,
                ExpansionStorage::Coefficients,
                strategy,
            )
            .unwrap();
        let interpolation = InterpolationOptions {
            transform: geometry(inner),
            max_class_tasks: 2,
            max_tasks: inner.get(),
        };
        let scratch = retained
            .requirements()
            .scratch_fields
            .max(forward.requirements().scratch_fields)
            .max(
                interpolation
                    .requirements(output_size, &class_sizes[..3])
                    .unwrap()
                    .scratch_fields,
            )
            .max(
                geometry(inner)
                    .requirements(output_size)
                    .unwrap()
                    .field_elements,
            );
        assert_eq!(retained.requirements().coefficient_fields, N);
        let mut workspaces: [_; 2] = core::array::from_fn(|_| {
            FftWorkspace::<M>::new(
                retained.requirements().coefficient_fields,
                output_size,
                class_sizes.iter().sum(),
                scratch,
            )
        });
        let capacities = workspaces.each_ref().map(FftWorkspace::capacities);
        for (expansion_cases, class_cases) in expansion_cases.iter().zip(&class_cases) {
            for_each_mut(
                &mut workspaces,
                budget,
                &executor,
                |index, work, allowance| {
                    assert_eq!(allowance, inner);
                    let (original, input, expected) = &expansion_cases[index];
                    let view = retained
                        .execute_with_workspace(
                            input,
                            &mut work.output,
                            &mut work.coefficients,
                            &executor,
                            &mut work.scratch,
                        )
                        .unwrap();
                    assert_eq!(
                        view.normalization_factor(),
                        base_plan.domain().domain().size_inverse()
                    );
                    for (stored, coefficient) in view.as_slice().iter().zip(original) {
                        assert_eq!(*stored, coefficient.mul(&PastaField::from_u64(N as u64)));
                    }
                    let output = retained.view(&work.output).unwrap();
                    for (row, value) in expected.iter().enumerate() {
                        assert_eq!(output.get(row), Some(value));
                    }

                    // Carry the scale with the coefficients into another expansion
                    // and into a normal forward-prefix transform.
                    forward
                        .execute_coefficients(view, &mut work.product, &executor, &mut work.scratch)
                        .unwrap();
                    assert_eq!(work.output, work.product);
                    plans[3]
                        .forward_prefix(
                            view,
                            &mut work.product,
                            geometry(inner),
                            &executor,
                            &mut work.scratch,
                        )
                        .unwrap();
                    assert_eq!(&work.product, expected);
                    let (prefix, short) = &products[index];
                    forward
                        .execute_product_into(
                            prefix.as_slice(),
                            output,
                            &mut work.product,
                            &executor,
                            &mut work.scratch,
                        )
                        .unwrap();
                    let product = forward.view(&work.product).unwrap();
                    for row in 0..output_size {
                        assert_eq!(product.get(row), Some(&short[row].mul(&expected[row])));
                    }

                    // Disjoint residue blocks can fill bit-reversed class storage
                    // directly, without an intermediate permutation buffer.
                    let (a, rest) = work.classes.split_at_mut(class_sizes[0]);
                    let (b, rest) = rest.split_at_mut(class_sizes[1]);
                    let (c, d) = rest.split_at_mut(class_sizes[2]);
                    let mut builders = [
                        ClassBuilder::new(plans[0], a, 1),
                        ClassBuilder::new(plans[1], b, 2),
                        ClassBuilder::new(plans[2], c, 4),
                        ClassBuilder::new(plans[3], d, 8),
                    ];
                    for (i, (builder, (_, evaluations))) in
                        builders.iter_mut().zip(class_cases).enumerate()
                    {
                        let residues = 1 << i;
                        // Reverse completion order to catch completion-count and
                        // storage-order assumptions in the consumer adapter.
                        for residue in (0..residues).rev() {
                            let values: Vec<_> = evaluations
                                .iter()
                                .skip(residue)
                                .step_by(residues)
                                .copied()
                                .collect();
                            builder.submit(residue, &values).unwrap();
                        }
                    }
                    let [a, b, c, mut output] = builders.map(|builder| builder.finish().unwrap());
                    let mut lifts = [a, b, c];
                    interpolate_classes_parallel(
                        &mut output,
                        &mut lifts,
                        interpolation,
                        &executor,
                        &mut work.scratch,
                    )
                    .unwrap();
                    assert_eq!(output.state(), ClassState::Coefficients);
                    for (lift, (polynomial, _)) in lifts.iter().zip(class_cases) {
                        assert_eq!(lift.values(), polynomial);
                    }
                    for (i, value) in output.values().iter().enumerate() {
                        let expected = class_cases
                            .iter()
                            .filter_map(|(polynomial, _)| polynomial.get(i))
                            .fold(PastaField::ZERO, |sum, value| sum.add(value));
                        assert_eq!(*value, expected);
                    }
                },
            );
            assert_eq!(
                workspaces.each_ref().map(FftWorkspace::capacities),
                capacities
            );
        }
        eprintln!(
            "FFT {threads} tasks: scratch required {} bytes/owner; retained workspace {} bytes/owner; tables {table_bytes} bytes",
            scratch * 32,
            workspaces[0].capacity_bytes()
        );
    }
}

#[test]
fn expansion_products_and_fused_classes() {
    pipeline::<PallasBase>();
    pipeline::<PallasScalar>();
}

#[test]
fn incomplete_producers_and_panics_require_refill() {
    let domain = Domain::<PallasBase>::for_size(16).unwrap().subgroup();
    let plan = fft::Plan::without_tables(domain);
    let mut storage = [PastaField::ZERO; 16];
    {
        let mut builder = ClassBuilder::new(plan, &mut storage, 2);
        builder.submit(1, &[PastaField::ONE; 8]).unwrap();
        assert_eq!(
            builder.submit(1, &[PastaField::ONE; 8]),
            Err("duplicate producer range")
        );
        assert_eq!(
            builder.submit(0, &[PastaField::ONE; 7]),
            Err("invalid producer range")
        );
        assert!(builder.finish().is_err());
    }
    struct PanicExecutor;
    impl Executor for PanicExecutor {
        fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
        where
            L: FnOnce() -> A + Send,
            R: FnOnce() -> B + Send,
            A: Send,
            B: Send,
        {
            // Both provided jobs complete before failure is propagated.
            let result = SerialExecutor.join(left, right);
            drop(result);
            panic!("executor failed after jobs completed")
        }
    }
    let mut lift_storage = [PastaField::ONE; 16];
    let options = InterpolationOptions {
        transform: ExecutionOptions {
            tile_len: 2,
            columns_per_task: 1,
            max_tasks: 2,
        },
        max_class_tasks: 1,
        max_tasks: 2,
    };
    let mut scratch =
        vec![PastaField::ZERO; options.requirements(16, &[16]).unwrap().scratch_fields];
    {
        storage.fill(PastaField::ONE);
        let mut output = Class::new(plan, &mut storage, ElementOrder::Natural).unwrap();
        let mut lifts = [Class::new(plan, &mut lift_storage, ElementOrder::Natural).unwrap()];
        assert!(
            catch_unwind(AssertUnwindSafe(|| {
                interpolate_classes_parallel(
                    &mut output,
                    &mut lifts,
                    options,
                    &PanicExecutor,
                    &mut scratch,
                )
                .unwrap();
            }))
            .is_err()
        );
        assert_eq!(output.state(), ClassState::Consumed);
        assert!(output.scatter(0, &[PastaField::ONE]).is_err());
        assert!(
            interpolate_classes_parallel(
                &mut output,
                &mut lifts,
                options,
                &SerialExecutor,
                &mut scratch
            )
            .is_err()
        );
    }
    storage.fill(PastaField::ONE);
    lift_storage.fill(PastaField::ONE);
    let mut output = Class::new(plan, &mut storage, ElementOrder::Natural).unwrap();
    let mut lifts = [Class::new(plan, &mut lift_storage, ElementOrder::Natural).unwrap()];
    interpolate_classes_parallel(
        &mut output,
        &mut lifts,
        options,
        &SerialExecutor,
        &mut scratch,
    )
    .unwrap();
    assert_eq!(output.values()[0], PastaField::from_u64(2));
    assert!(output.values()[1..].iter().all(PastaField::is_zero));
}
