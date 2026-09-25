//! MSM preparation and execution respect caller resource limits.
use crate::{
    curve::{AffinePoint, CurveError, Pallas, PastaCurve, ProjectivePoint, Vesta},
    exec::{
        ExecutionOptions, SerialExecutor, TaskBudget,
        execution::{Identity, TaskStorage},
    },
    field::PastaField,
    msm::{
        Bases, Input, PreparedScalars, Requirements, ScalarStorage, Scratch,
        execution::{MsmPlan, MsmRun, ParallelMsmRun, ProducedInput},
    },
};
use core::num::NonZeroUsize;
use std::{vec, vec::Vec};

struct Workspace<C: PastaCurve> {
    records: Vec<ScalarStorage<C>>,
    digits: Vec<u8>,
    affine: Vec<AffinePoint<C>>,
    projective: Vec<ProjectivePoint<C>>,
    field: Vec<PastaField<C::Base>>,
    indices: Vec<usize>,
}
impl<C: PastaCurve> Workspace<C> {
    fn new(r: Requirements) -> Self {
        Self {
            records: vec![ScalarStorage::ZERO; r.scalars()],
            digits: vec![0; r.digits()],
            affine: vec![AffinePoint::GENERATOR; r.affine()],
            projective: vec![ProjectivePoint::IDENTITY; r.projective()],
            field: vec![PastaField::ZERO; r.field()],
            indices: vec![0; r.indices()],
        }
    }
    fn scratch(&mut self) -> Scratch<'_, C> {
        Scratch::new(
            &mut self.records,
            &mut self.digits,
            &mut self.affine,
            &mut self.projective,
            &mut self.field,
            &mut self.indices,
        )
    }
}

fn msm<C: PastaCurve>() {
    for count in [4095, 4096, 4097, 32767, 32768, 32769] {
        let bases = vec![AffinePoint::<C>::GENERATOR; count];
        let scalars: Vec<_> = (0..count)
            .map(|i| {
                PastaField::<C::Scalar>::from_u64(i as u64 + 1).mul(&PastaField::<_>::TWO_INVERSE)
            })
            .collect();
        let sum = scalars
            .iter()
            .fold(PastaField::ZERO, |sum, scalar| sum.add(scalar));
        let expected = AffinePoint::<C>::GENERATOR.mul_projective(&sum);
        let mut records = vec![ScalarStorage::<C>::ZERO; count];
        let prepared =
            PreparedScalars::prepare(&scalars, &mut records, TaskBudget::SERIAL, &SerialExecutor);
        for tasks in [1, 4] {
            let options =
                ExecutionOptions::default().with_task_budget(TaskBudget::new(tasks).unwrap());
            let plan = MsmPlan::new(count, options).unwrap();
            let length = prepared.cache_len(&plan);
            if count == 4096 {
                assert!(length > 0);
            }
            if count == 32768 {
                assert_eq!(length, 0);
            }
            let mut digits = vec![0xa5; length + 1];
            let cached = prepared.cache(&plan, &mut digits, options.task_budget(), &SerialExecutor);
            let input = Input::new_prepared(Bases::Affine(&bases), cached);
            let cached_plan = MsmPlan::for_input(&input, options).unwrap();
            let required = cached_plan.requirements();
            if length != 0 {
                assert_eq!(required.digits(), 0, "requested cache must be consumed");
            }
            assert_eq!(required.scalars(), 0);
            let mut workspace = Workspace::new(required);
            assert_eq!(
                cached_plan.execute(input, &SerialExecutor, workspace.scratch()),
                expected
            );
            assert_eq!(digits[length], 0xa5);
        }
    }
    let bases = vec![AffinePoint::<C>::GENERATOR; 4096];
    let scalars = vec![PastaField::<C::Scalar>::TWO_INVERSE; bases.len()];
    let input = Input::new(Bases::Affine(&bases), &scalars);
    let options = ExecutionOptions::default().with_task_budget(TaskBudget::new(4).unwrap());
    let bounded = options.with_memory_limit(64 * 1024);
    let required = input.requirements(bounded).unwrap();
    assert!(required.bytes::<C>().unwrap() <= 64 * 1024);
    let mut workspace = Workspace::new(required);
    let expected = bases[0].mul_projective(&PastaField::<_>::from_u64(2048));
    // Actual typed capacities alone must induce a fitting implementation.
    assert_eq!(
        input
            .execute(options, &SerialExecutor, workspace.scratch())
            .unwrap(),
        expected
    );
    assert!(matches!(
        input.requirements(options.with_memory_limit(0)),
        Err(CurveError::MemoryLimit { .. })
    ));
    let plan = MsmPlan::for_input(&input, bounded).unwrap();
    assert!(matches!(
        plan.retained_for_slots(NonZeroUsize::new(1024).unwrap()),
        Err(CurveError::MemoryLimit { .. })
    ));

    let mut records = vec![ScalarStorage::<C>::ZERO; scalars.len()];
    let prepared =
        PreparedScalars::prepare(&scalars, &mut records, TaskBudget::SERIAL, &SerialExecutor);
    for fragment in [256, 2048] {
        let plan = MsmPlan::for_produced(
            ProducedInput::dense(Bases::Affine(&bases)),
            NonZeroUsize::new(fragment).unwrap(),
            options,
        )
        .unwrap();
        assert!(plan.grain() <= fragment);
        assert_eq!(prepared.cache_len(&plan), 0);
        let mut workspace = Workspace::new(plan.requirements());
        assert_eq!(
            plan.execute(input, &SerialExecutor, workspace.scratch()),
            expected
        );
    }
    let specialized = MsmPlan::for_input(
        &Input::new_prepared(Bases::Affine(&bases), prepared),
        options,
    )
    .unwrap();
    let mut identities = [Identity::new()];
    let mut tasks = [[const { TaskStorage::EMPTY }; 1]];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = ParallelMsmRun::new(specialized, input, &mut identities, &mut tasks);
        }))
        .is_err()
    );
}

#[test]
fn scalar_caches_follow_resolved_plans_and_capacities() {
    msm::<Pallas>();
    msm::<Vesta>();
}

#[test]
fn zero_length_cache_requests_preserve_existing_preparation() {
    fn check<C: PastaCurve>() {
        let bases = [AffinePoint::<C>::GENERATOR; 16];
        let mut records = [ScalarStorage::<C>::ZERO; 16];
        let mut digits = Vec::new();
        let options = ExecutionOptions::DEFAULT;
        let cached = {
            let scalars = [PastaField::<C::Scalar>::ONE; 16];
            let prepared = PreparedScalars::prepare(
                &scalars,
                &mut records,
                TaskBudget::SERIAL,
                &SerialExecutor,
            );
            let plan = MsmPlan::new(scalars.len(), options).unwrap();
            let len = prepared.cache_len(&plan);
            assert!(len > 0);
            digits.resize(len, 0);
            let executor = SerialExecutor;
            let cached = prepared.cache(&plan, &mut digits, TaskBudget::new(3).unwrap(), &executor);
            assert_eq!(cached.retained_bytes(), prepared.retained_bytes() + len);
            cached
        };
        // The handle outlives the original scalars, preparation handle, plan,
        // and executor. Only the records and cache bytes remain borrowed.
        let input = Input::new_prepared(Bases::Affine(&bases), cached);
        let short = MsmPlan::for_input(&input, options).unwrap();
        let mismatched = MsmPlan::new(bases.len() + 1, options).unwrap();
        let fragmented = MsmPlan::for_produced(
            ProducedInput::dense(Bases::Affine(&bases)),
            NonZeroUsize::new(8).unwrap(),
            options,
        )
        .unwrap();
        for plan in [short, mismatched, fragmented] {
            assert_eq!(cached.cache_len(&plan), 0);
            let mut unused = [0xa5];
            let reused = cached.cache(
                &plan,
                &mut unused,
                TaskBudget::new(3).unwrap(),
                &SerialExecutor,
            );
            assert_eq!(reused.retained_bytes(), cached.retained_bytes());
            let plan = MsmPlan::new(bases.len(), options).unwrap();
            let mut workspace = Workspace::new(plan.requirements());
            assert_eq!(
                plan.execute(
                    Input::new_prepared(Bases::Affine(&bases), reused),
                    &SerialExecutor,
                    workspace.scratch(),
                ),
                bases[0].mul_projective(&PastaField::<C::Scalar>::from_u64(16)),
            );
            assert_eq!(unused, [0xa5]);
        }
    }
    check::<Pallas>();
    check::<Vesta>();
}

fn cached_plan_slots<C: PastaCurve>() {
    let bases = [AffinePoint::<C>::GENERATOR; 16];
    let scalars = [PastaField::<C::Scalar>::TWO_INVERSE; 16];
    let mut records = [ScalarStorage::<C>::ZERO; 16];
    let prepared =
        PreparedScalars::prepare(&scalars, &mut records, TaskBudget::SERIAL, &SerialExecutor);
    let options = ExecutionOptions::default();
    let plan = MsmPlan::<C>::new(scalars.len(), options).unwrap();
    let mut digits = vec![0; prepared.cache_len(&plan)];
    assert!(!digits.is_empty());
    let cached = prepared.cache(&plan, &mut digits, TaskBudget::SERIAL, &SerialExecutor);
    let input = Input::new_prepared(Bases::Affine(&bases), cached);
    let plan = MsmPlan::for_input(&input, options).unwrap();
    assert_eq!(plan.requirements().digits(), 0);

    let mut identity = Identity::new();
    let mut storage = [const { TaskStorage::EMPTY }; 1];
    assert!(
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            let _ = MsmRun::new_partition(plan, input, 0..8, &mut identity, &mut storage);
        }))
        .is_err()
    );
    for rebind in [false, true] {
        let mut identities = core::array::from_fn(|_| Identity::new());
        let mut storage = [const { [const { TaskStorage::EMPTY }; 1] }; 2];
        let run = if rebind {
            let empty = Input::new(Bases::Affine(&[]), &[]);
            let empty_plan = MsmPlan::new(0, options).unwrap();
            let mut run =
                ParallelMsmRun::new(empty_plan, empty, &mut identities, &mut storage).unwrap();
            run.rebind(plan, input).unwrap();
            run
        } else {
            ParallelMsmRun::new(plan, input, &mut identities, &mut storage).unwrap()
        };
        let mut requests = [None];
        assert_eq!(run.ready_slot_from(0, 0, &mut requests), 1);
        assert_eq!(run.ready_slot_from(1, 0, &mut requests), 0);
        assert!(!run.is_failed());
    }
}

#[test]
fn cached_plans_allow_unused_parallel_slots() {
    cached_plan_slots::<Pallas>();
    cached_plan_slots::<Vesta>();
}
