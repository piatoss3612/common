use super::{
    fft_pipeline::{Banks, Lease},
    run_pool::{self, Work},
};
use std::num::NonZeroUsize;
use zakura_udon::{
    exec::{
        SerialExecutor,
        run::{Completion, Identity, Task, TaskStorage},
    },
    fft::{
        ClassState, Codelet, Direction, Domain, ElementOrder, FftError, Plan, TransformRequest,
        run::{
            AdditionKernel, Bank, FftKernel, FftPlan, InterpolationPlan, InterpolationRun, Request,
            WorkKind,
        },
    },
    field::{PallasBase, PallasScalar, PastaField, PrimeModulus},
};

#[expect(
    clippy::large_enum_variant,
    reason = "fixed envelopes avoid task allocation"
)]
enum Job<'a, M: PrimeModulus> {
    Transform(usize, Task<'a, FftKernel<'a, M>, Lease<'a, M>>),
    Add(usize, Task<'a, AdditionKernel<M>, Lease<'a, M>>),
}
impl<'a, M: PrimeModulus> Work for Job<'a, M> {
    type Completion = (
        usize,
        bool,
        Completion<'a, Lease<'a, M>, Result<(), FftError>>,
    );
    fn execute(&mut self) {
        match self {
            Self::Transform(_, task) => task.execute().unwrap(),
            Self::Add(_, task) => task.execute().unwrap(),
        }
    }
    fn complete(self) -> Self::Completion {
        match self {
            Self::Transform(i, task) => (i, false, task.finish()),
            Self::Add(i, task) => (i, true, task.finish()),
        }
    }
}

fn check<M: PrimeModulus>() {
    const CLASSES: usize = 5;
    let sizes = [64, 8, 64, 32, 64];
    let shifts = [7, 11, 7, 13, 7];
    for consume in [false, true] {
        for tile in [8, 64] {
            for flip in [false, true] {
                let plans = core::array::from_fn::<_, CLASSES, _>(|i| {
                    Plan::without_tables(
                        Domain::<M>::for_size(sizes[i])
                            .unwrap()
                            .coset(PastaField::from_u64(shifts[i]))
                            .unwrap(),
                    )
                });
                let orders = core::array::from_fn::<_, CLASSES, _>(|i| {
                    if (i % 2 == 0) == flip {
                        ElementOrder::Natural
                    } else {
                        ElementOrder::BitReversed
                    }
                });
                let coefficients: Vec<Vec<_>> = sizes
                    .iter()
                    .enumerate()
                    .map(|(c, &n)| {
                        (0..n)
                            .map(|i| PastaField::from_u64((i * i + c + 1) as u64))
                            .collect()
                    })
                    .collect();
                let mut expected = coefficients[0].clone();
                for lift in &coefficients[1..] {
                    for (out, value) in expected.iter_mut().zip(lift) {
                        *out = out.add(value);
                    }
                }
                let transforms = core::array::from_fn(|i| {
                    FftPlan::new(
                        plans[i],
                        TransformRequest {
                            input_order: orders[i],
                            ..TransformRequest::new(Direction::Inverse)
                        },
                        NonZeroUsize::new(tile).unwrap(),
                        Codelet::Radix4,
                        false,
                    )
                    .unwrap()
                });
                let plan = InterpolationPlan::new(transforms, consume).unwrap();
                let banks = Banks::new(&[64; CLASSES * 2], tile);
                for i in 0..CLASSES {
                    let mut evaluations = coefficients[i].clone();
                    plans[i]
                        .forward(
                            &mut evaluations,
                            Default::default(),
                            &SerialExecutor,
                            &mut [],
                        )
                        .unwrap();
                    if orders[i] == ElementOrder::BitReversed {
                        let original = evaluations.clone();
                        for (j, value) in evaluations.iter_mut().enumerate() {
                            *value = original[j.reverse_bits() >> (usize::BITS - sizes[i].ilog2())];
                        }
                    }
                    banks.write(i, &evaluations);
                }
                let mut ids = core::array::from_fn(|_| core::array::from_fn(|_| Identity::new()));
                let mut slots =
                    [const { [const { [const { TaskStorage::EMPTY }; 3] }; 2] }; CLASSES];
                let mut run = InterpolationRun::new(plan, &mut ids, &mut slots).unwrap();
                let mut released = [false; CLASSES];
                run_pool::scoped(4, 3, |pool| {
                    while !run.is_complete() {
                        for class in 0..CLASSES {
                            let mut cursor = 0;
                            loop {
                                let mut ready = [None];
                                if !pool.available()
                                    || run.ready_transform_from(class, cursor, &mut ready) == 0
                                {
                                    break;
                                }
                                let request = ready[0].take().unwrap();
                                cursor = request.key.index() + 1;
                                if let Some(task) = run
                                    .try_claim_transform(class, request.clone(), || {
                                        banks.acquire(&request, class, class, CLASSES + class, 0)
                                    })
                                    .unwrap()
                                {
                                    assert!(pool.submit(Job::Transform(class, task)).is_ok());
                                }
                            }
                            let mut cursor = 0;
                            loop {
                                let mut ready = [None];
                                if !pool.available()
                                    || run.ready_addition_from(class, cursor, &mut ready) == 0
                                {
                                    break;
                                }
                                let add = ready[0].take().unwrap();
                                cursor = add.key.index() + 1;
                                let request = Request {
                                    key: add.key,
                                    kind: WorkKind::Initialize,
                                    write: (Bank::Values, add.write.clone()),
                                    pair: None,
                                    read: Some((Bank::Input, add.read.clone())),
                                    factor: None,
                                    estimate: add.estimate,
                                };
                                if let Some(task) = run
                                    .try_claim_addition(class, add, || {
                                        banks.acquire(&request, class, 0, CLASSES, 0)
                                    })
                                    .unwrap()
                                {
                                    assert!(pool.submit(Job::Add(class, task)).is_ok());
                                }
                            }
                        }
                        let (class, addition, receipt) =
                            pool.receive().expect("admitted interpolation progress");
                        let published = if addition {
                            run.complete_addition(class, receipt).unwrap()
                        } else {
                            run.complete_transform(class, receipt).unwrap()
                        };
                        assert_eq!(published.error, None);
                        if let Some(class) = published.released {
                            assert!(!released[class]);
                            released[class] = true;
                        }
                        drop(published);
                    }
                });
                assert_eq!(banks.read(0), expected);
                assert_eq!(run.state(0), Some(ClassState::Coefficients));
                for class in 1..CLASSES {
                    assert!(released[class]);
                    assert_eq!(
                        run.state(class),
                        Some(if consume {
                            ClassState::Consumed
                        } else {
                            ClassState::Coefficients
                        })
                    );
                    if !consume {
                        assert_eq!(&banks.read(class)[..sizes[class]], &coefficients[class]);
                    }
                }
            }
        }
    }
}

#[test]
fn independent_class_inverses_and_fused_merges_match_polynomial_sums() {
    check::<PallasBase>();
    check::<PallasScalar>();
}
