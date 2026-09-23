//! Public resource constraints, resolved preparation, and adaptive execution.

use core::num::NonZeroUsize;
use zakura_udon::{
    curve::{
        AffinePoint, CurveError, FixedBaseTable, Pallas, PastaCurve, ProjectivePoint, Vesta,
        msm::{
            Bases, Input, PreparedScalars, Requirements, ScalarStorage, Scratch,
            run::{MsmPlan, MsmRun, ParallelMsmRun, ProducedInput},
        },
    },
    exec::{
        ExecutionOptions, SerialExecutor, TaskBudget,
        run::{Identity, TaskError, TaskStorage},
    },
    fft::{
        Direction, Domain, ElementOrder, Expansion, ExpansionOrder, ExpansionStorage, InputStorage,
        InputSupport, StorageLayout, Transform, TransformRequest,
        run::{ExpansionPlan, ExpansionRun, FftPlan},
    },
    field::{PallasBase, PallasScalar, PastaField, PrimeModulus},
};

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
            let cached = prepared.cache(&plan, &mut digits);
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
    let cached = prepared.cache(&plan, &mut digits);
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

fn fft<M: PrimeModulus>() {
    for size in [64, 2048, 4096] {
        let domain = Domain::<M>::for_size(size).unwrap().coset();
        let transform = Transform::new(domain);
        for tasks in [1, 4] {
            for limit in [0, 64 * 32, size * 32] {
                let options = ExecutionOptions::default()
                    .with_task_budget(TaskBudget::new(tasks).unwrap())
                    .with_memory_limit(limit);
                for length in [0, 1, 17, size] {
                    let input: Vec<_> = (0..length)
                        .map(|i| PastaField::from_u64(i as u64 + 2))
                        .collect();
                    for order in [ElementOrder::Natural, ElementOrder::BitReversed] {
                        let request = TransformRequest {
                            input_storage: InputStorage::Preserve,
                            support: InputSupport::Prefix(length),
                            output_order: order,
                            ..TransformRequest::new(Direction::Forward)
                        };
                        let plan =
                            FftPlan::new(transform, request, StorageLayout::Contiguous, options)
                                .unwrap();
                        assert!(plan.retained_fields() * 32 <= limit);
                        let mut scratch =
                            vec![PastaField::from_u64(99); plan.retained_fields() + 3];
                        let mut output = vec![PastaField::ZERO; size];
                        plan.execute(
                            Some(&input),
                            &mut output,
                            None,
                            &mut scratch,
                            &SerialExecutor,
                        );
                        assert!(
                            scratch[plan.retained_fields()..]
                                .iter()
                                .all(|v| v.reduce() == PastaField::from_u64(99))
                        );
                        for row in [0, 1, size / 3, size - 1] {
                            let point = domain
                                .shift()
                                .mul(&domain.domain().root().pow_u64(row as u64));
                            let expected = input
                                .iter()
                                .rev()
                                .fold(PastaField::ZERO, |acc, coefficient| {
                                    acc.mul(&point).add(coefficient)
                                });
                            let index = if order == ElementOrder::Natural {
                                row
                            } else {
                                row.reverse_bits() >> (usize::BITS - size.ilog2())
                            };
                            assert_eq!((output[index]).reduce(), (expected).reduce());
                        }
                        let mut direct = vec![PastaField::ZERO; size];
                        transform
                            .execute(
                                request,
                                Some(input.as_slice().into()),
                                &mut direct,
                                options,
                                &SerialExecutor,
                                &mut scratch,
                            )
                            .unwrap();
                        assert_eq!(
                            (direct)
                                .iter()
                                .map(|value| value.reduce())
                                .collect::<Vec<_>>(),
                            (output)
                                .iter()
                                .map(|value| value.reduce())
                                .collect::<Vec<_>>()
                        );
                    }
                }
            }
        }
    }
    let base = Transform::new(Domain::<M>::for_size(4096).unwrap().subgroup());
    let expansion = Expansion::new(base, Domain::for_size(8192).unwrap().subgroup(), None).unwrap();
    let plan = ExpansionPlan::new(
        expansion,
        ExpansionStorage::CoefficientWorkspace {
            scale: zakura_udon::fft::InverseScale::Normalized,
        },
        ExpansionOrder::Residues,
        InputSupport::Full,
        ElementOrder::Natural,
        StorageLayout::Fragments {
            length: NonZeroUsize::new(64).unwrap(),
            whole_bank: false,
        },
        ExecutionOptions::default().with_memory_limit(2 * 4096 * 32),
    )
    .unwrap();
    assert!(plan.snapshot_fields() > 0);
    let mut identities = core::array::from_fn::<_, 3, _>(|_| Identity::new());
    let mut tasks = [const { [const { TaskStorage::EMPTY }; 1] }; 3];
    assert!(matches!(
        ExpansionRun::new(plan, false, &mut identities, &mut tasks),
        Err(TaskError::Storage)
    ));
}

#[test]
fn transform_selection_obeys_constraints_and_mathematical_layouts() {
    fft::<PallasBase>();
    fft::<PallasScalar>();
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
