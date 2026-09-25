use crate::{
    buffers::{ClassBuilder, FftWorkspace, OwnedTables},
    executor_adapter::RayonExecutor,
};
use std::panic::{AssertUnwindSafe, catch_unwind};
use zakura_udon::{
    exec::{ExecutionOptions, Executor, SerialExecutor, TaskBudget, for_each_mut},
    fft::{
        self, Direction, Domain, ElementOrder, EvaluationLayout, EvaluationView, Expansion,
        ExpansionOrder, ExpansionStorage, InputStorage, InputSupport, InverseScale, StorageLayout,
        TransformRequest,
        execution::{ExpansionPlan, InterpolationPlan},
        reference,
    },
    field::{PallasBase, PallasScalar, PastaField, PrimeModulus},
};

const N: usize = 2048;

fn coefficients<M: PrimeModulus>(size: usize, seed: usize) -> Vec<PastaField<M>> {
    (0..size)
        .map(|i| PastaField::<_>::from_u64((i * i + seed + 1) as u64).neg())
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
    ExecutionOptions::default().with_task_budget(budget)
}

fn pipeline<M: PrimeModulus>() {
    let base = OwnedTables::<M>::new(Domain::for_size(N).unwrap().subgroup());
    let classes: [_; 4] =
        core::array::from_fn(|i| OwnedTables::<M>::new(Domain::for_size(N << i).unwrap().coset()));
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
        let retained = ExpansionPlan::new(
            expansion,
            ExpansionStorage::CoefficientWorkspace {
                scale: InverseScale::Unscaled,
            },
            ExpansionOrder::Residues,
            InputSupport::Full,
            ElementOrder::Natural,
            StorageLayout::Contiguous,
            geometry(inner),
        )
        .unwrap();
        let forward = ExpansionPlan::new(
            expansion,
            ExpansionStorage::Coefficients,
            ExpansionOrder::Residues,
            InputSupport::Full,
            ElementOrder::Natural,
            StorageLayout::Contiguous,
            geometry(inner),
        )
        .unwrap();
        let interpolation = InterpolationPlan::new(
            [plans[3], plans[0], plans[1], plans[2]].map(|plan| (plan, ElementOrder::BitReversed)),
            false,
            StorageLayout::Contiguous,
            geometry(inner).with_memory_limit(0),
        )
        .unwrap();
        let scratch = retained
            .scratch_fields()
            .max(forward.scratch_fields())
            .max(plans[3].scratch_requirements(geometry(inner)).unwrap());
        assert_eq!(retained.coefficient_fields(), N);
        let mut workspaces: [_; 2] = core::array::from_fn(|_| {
            FftWorkspace::<M>::new(N, output_size, class_sizes.iter().sum(), scratch)
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
                        .execute(
                            input,
                            &mut work.output,
                            &mut work.coefficients,
                            None,
                            &mut work.scratch,
                            &executor,
                        )
                        .unwrap();
                    assert_eq!(
                        (view.normalization_factor()).reduce(),
                        (base_plan.domain().domain().size_inverse()).reduce()
                    );
                    for (stored, coefficient) in view.as_slice().iter().zip(original) {
                        assert_eq!(
                            (*stored).reduce(),
                            (coefficient.mul(&PastaField::<_>::from_u64(N as u64))).reduce()
                        );
                    }
                    let output = EvaluationView::bind(
                        &work.output,
                        plans[3].domain(),
                        EvaluationLayout::Residues(expansion.layout()),
                    );
                    for (row, value) in expected.iter().enumerate() {
                        assert_eq!(
                            (output.get(row)).map(|value| value.reduce()),
                            (Some(value)).map(|value| value.reduce())
                        );
                    }

                    // Carry the scale with the coefficients into another expansion
                    // and into a normal forward-prefix transform.
                    forward
                        .with_coefficient_scale(view.normalization_factor())
                        .execute(
                            view.as_slice(),
                            &mut work.product,
                            &mut [],
                            None,
                            &mut work.scratch,
                            &executor,
                        );
                    assert_eq!(
                        (work.output)
                            .iter()
                            .map(|value| value.reduce())
                            .collect::<Vec<_>>(),
                        (work.product)
                            .iter()
                            .map(|value| value.reduce())
                            .collect::<Vec<_>>()
                    );
                    plans[3]
                        .execute(
                            TransformRequest {
                                input_storage: InputStorage::Preserve,
                                support: InputSupport::Prefix(view.as_slice().len()),
                                ..TransformRequest::new(Direction::Forward)
                            },
                            Some(view),
                            &mut work.product,
                            geometry(inner),
                            &executor,
                            &mut work.scratch,
                        )
                        .unwrap();
                    assert_eq!(
                        work.product
                            .iter()
                            .map(|value| value.reduce())
                            .collect::<Vec<_>>(),
                        (expected)
                            .iter()
                            .map(|value| value.reduce())
                            .collect::<Vec<_>>()
                    );
                    let (prefix, short) = &products[index];
                    ExpansionPlan::new(
                        expansion,
                        ExpansionStorage::Coefficients,
                        ExpansionOrder::Residues,
                        InputSupport::Prefix(prefix.len()),
                        ElementOrder::Natural,
                        StorageLayout::Contiguous,
                        geometry(inner),
                    )
                    .unwrap()
                    .execute(
                        prefix,
                        &mut work.product,
                        &mut [],
                        Some(&work.output),
                        &mut work.scratch,
                        &executor,
                    );
                    let product = EvaluationView::bind(
                        &work.product,
                        plans[3].domain(),
                        EvaluationLayout::Residues(expansion.layout()),
                    );
                    for row in 0..output_size {
                        assert_eq!(
                            (product.get(row)).map(|value| value.reduce()),
                            (Some(&short[row].mul(&expected[row]))).map(|value| value.reduce())
                        );
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
                    let [a, b, c, output] = builders.map(|builder| builder.finish().unwrap());
                    interpolation.execute(
                        [output, a, b, c],
                        [&mut [], &mut [], &mut [], &mut []],
                        &executor,
                    );
                    for (lift, (polynomial, _)) in [a, b, c].iter().zip(class_cases) {
                        assert_eq!(
                            (*lift)
                                .iter()
                                .map(|value| value.reduce())
                                .collect::<Vec<_>>(),
                            (polynomial)
                                .iter()
                                .map(|value| value.reduce())
                                .collect::<Vec<_>>()
                        );
                    }
                    for (i, value) in output.iter().enumerate() {
                        let expected = class_cases
                            .iter()
                            .filter_map(|(polynomial, _)| polynomial.get(i))
                            .fold(PastaField::ZERO, |sum, value| sum.add(value));
                        assert_eq!((*value).reduce(), (expected).reduce());
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
    let domain = Domain::<PastaField<PallasBase>>::for_size(16)
        .unwrap()
        .subgroup();
    let plan = fft::Transform::new(domain);
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
    let interpolation = InterpolationPlan::new(
        [(plan, ElementOrder::Natural); 2],
        false,
        StorageLayout::Contiguous,
        ExecutionOptions::default().with_task_budget(TaskBudget::new(2).unwrap()),
    )
    .unwrap();
    storage.fill(PastaField::ONE);
    assert!(
        catch_unwind(AssertUnwindSafe(|| {
            interpolation.execute(
                [&mut storage, &mut lift_storage],
                [&mut [], &mut []],
                &PanicExecutor,
            );
        }))
        .is_err()
    );
    // After an unwind, the owner refills evaluations before reusing either bank.
    storage.fill(PastaField::ONE);
    lift_storage.fill(PastaField::ONE);
    interpolation.execute(
        [&mut storage, &mut lift_storage],
        [&mut [], &mut []],
        &SerialExecutor,
    );
    assert_eq!(
        (storage[0]).reduce(),
        (PastaField::<_>::from_u64(2)).reduce()
    );
    assert!(storage[1..].iter().all(PastaField::is_zero));
}
