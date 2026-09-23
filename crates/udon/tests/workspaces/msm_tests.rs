use crate::bridge::{
    executor::{RayonExecutor, with_side_work},
    msm::MsmWorkspace,
};
use zakura_udon::{
    curve::{
        AffinePoint, EisensteinTable, EisensteinTableBatch, Pallas, PastaCurve, Point,
        PreparedAffinePoint, ProjectivePoint, Vesta, batch_normalize,
        msm::{
            Bases, BasisSum, CoalescingKey, CoalescingPlan, IndexedCoalescingPlan, Input,
            Selection, SuffixBasis,
        },
    },
    exec::{ExecutionOptions, SerialExecutor, TaskBudget},
    field::PastaField,
};

fn bases<C: PastaCurve>(n: usize) -> Vec<AffinePoint<C>> {
    let mut next = ProjectivePoint::GENERATOR;
    let projective: Vec<_> = (0..n)
        .map(|_| {
            let value = next;
            next = next.add(&ProjectivePoint::GENERATOR);
            value
        })
        .collect();
    let mut points = vec![Point::IDENTITY; n];
    batch_normalize(&projective, &mut points, &mut vec![PastaField::ZERO; n]);
    points.iter().map(|p| *p.as_affine().unwrap()).collect()
}

// A binary ladder, independent of Udon's GLV, fixed-base and MSM schedules.
fn expected<C: PastaCurve>(scalar: PastaField<C::Scalar>) -> ProjectivePoint<C> {
    let mut result = ProjectivePoint::<C>::IDENTITY;
    for byte in scalar.to_bytes().iter().rev() {
        for bit in (0..8).rev() {
            result = result.double();
            if byte & (1 << bit) != 0 {
                result = result.add(&ProjectivePoint::GENERATOR);
            }
        }
    }
    result
}

fn representations<C: PastaCurve>() {
    let bases = bases::<C>(17);
    let cached: Vec<_> = bases.iter().map(PreparedAffinePoint::from_affine).collect();
    let r = EisensteinTableBatch::<C>::requirements(bases.len()).unwrap();
    let mut entries = vec![AffinePoint::GENERATOR; r.table_entries];
    let compact = EisensteinTableBatch::prepare(
        &bases,
        &mut entries,
        &mut vec![ProjectivePoint::IDENTITY; r.projective_scratch],
        &mut vec![PastaField::ZERO; r.field_scratch],
        TaskBudget::SERIAL,
        &SerialExecutor,
    );
    let indices: Vec<_> = (0..37).map(|i| (i * 7 % 17) as u32).collect();
    let mut workspace = MsmWorkspace::new();
    for source in [
        Bases::Affine(&bases),
        Bases::Prepared(&cached),
        Bases::Compact(compact),
    ] {
        for selection in [
            Selection::new(source),
            Selection::indexed(source, &indices).unwrap(),
        ] {
            for round in 0..2 {
                let scalars: Vec<_> = (0..selection.len())
                    .map(|i| {
                        PastaField::<_>::from_u64((i + round + 2) as u64)
                            .invert()
                            .unwrap()
                    })
                    .collect();
                let inputs = [selection.with_scalars(&scalars)];
                let sum = scalars
                    .iter()
                    .enumerate()
                    .fold(PastaField::ZERO, |sum, (i, scalar)| {
                        let index = if selection.len() == bases.len() {
                            i
                        } else {
                            indices[i] as usize
                        };
                        sum.add(&scalar.mul(&PastaField::<_>::from_u64(index as u64 + 1)))
                    });
                let mut result = [ProjectivePoint::IDENTITY];
                let mut run = workspace
                    .prepare(
                        &inputs,
                        ExecutionOptions::default().with_memory_limit(32768),
                    )
                    .unwrap();
                assert!(run.temporary_bytes() <= 32768);
                assert!(run.requirements().bytes::<C>().unwrap() <= run.temporary_bytes());
                run.execute(&mut result, &SerialExecutor);
                let expected = expected(sum);
                assert_eq!(result[0], expected);
                // The same borrowed plan can execute again using dirty scratch.
                run.execute(&mut result, &SerialExecutor);
                assert_eq!(result[0], expected);
            }
        }
    }
}

#[test]
fn owned_msm_dense_indexed_cached_and_compact() {
    representations::<Pallas>();
    representations::<Vesta>();
}

fn shrinking_batches<C: PastaCurve>() {
    const N: usize = 2048;
    let bases = bases::<C>(N);
    let cached: Vec<_> = bases.iter().map(PreparedAffinePoint::from_affine).collect();
    let indices: [Vec<u32>; 2] = core::array::from_fn(|side| {
        (0..N / 2)
            .map(|i| ((i * 17 + side * 3) % N) as u32)
            .collect()
    });
    let mut entries = [[AffinePoint::GENERATOR; 8]; 2];
    for (side, entries) in entries.iter_mut().enumerate() {
        EisensteinTable::prepare(
            &bases[3 + side * 4],
            entries,
            &mut [ProjectivePoint::IDENTITY; 8],
            &mut [PastaField::ZERO; 8],
        );
    }
    let fixed = [
        EisensteinTable::bind(&bases[3], &entries[0]),
        EisensteinTable::bind(&bases[7], &entries[1]),
    ];
    for threads in [1, 3, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let executor = RayonExecutor(&pool);
        let budget = TaskBudget::new(threads).unwrap();
        let mut workspace = MsmWorkspace::new();
        let mut warmed = None;
        let mut peak_required = 0;
        for repetition in 0..2 {
            for round in 0..11 {
                let len = N >> (round + 1);
                let selections = indices.each_ref().map(|indices| {
                    Selection::indexed(Bases::Prepared(&cached), &indices[..len]).unwrap()
                });
                let rows: [Vec<_>; 2] = core::array::from_fn(|side| {
                    let step = PastaField::<_>::from_u64((round + side + repetition + 2) as u64)
                        .invert()
                        .unwrap();
                    (0..len)
                        .map(|i| step.mul(&PastaField::<_>::from_u64(i as u64)))
                        .collect()
                });
                let inputs = [
                    selections[0].with_scalars(&rows[0]),
                    selections[1].with_scalars(&rows[1]),
                ];
                let side_scalars: [_; 4] =
                    core::array::from_fn(|i| PastaField::from_u64((i + round + 2) as u64));
                let (mut result, side) = with_side_work(
                    &executor,
                    budget,
                    |main| {
                        let mut result = [ProjectivePoint::IDENTITY; 2];
                        let options = ExecutionOptions::default()
                            .with_task_budget(main)
                            .with_memory_limit(2 * 1024 * 1024);
                        let mut run = workspace.prepare(&inputs, options).unwrap();
                        peak_required = peak_required.max(run.temporary_bytes());
                        run.execute(&mut result, &executor);
                        result
                    },
                    |side| {
                        assert_eq!(side, budget);
                        core::array::from_fn::<_, 2, _>(|i| {
                            fixed[0]
                                .mul(&side_scalars[2 * i])
                                .add(&fixed[1].mul(&side_scalars[2 * i + 1]))
                        })
                    },
                );
                for i in 0..2 {
                    result[i] = result[i].add(&side[i]);
                }
                let mut points = [Point::IDENTITY; 2];
                batch_normalize(&result, &mut points, &mut [PastaField::ZERO; 2]);
                for i in 0..2 {
                    let sum = rows[i]
                        .iter()
                        .zip(&indices[i])
                        .fold(PastaField::ZERO, |sum, (scalar, index)| {
                            sum.add(&scalar.mul(&PastaField::<_>::from_u64(u64::from(*index) + 1)))
                        })
                        .add(&side_scalars[2 * i].mul(&PastaField::<_>::from_u64(4)))
                        .add(&side_scalars[2 * i + 1].mul(&PastaField::<_>::from_u64(8)));
                    assert_eq!(points[i], expected::<C>(sum).to_point());
                }
            }
            if let Some(capacities) = warmed {
                assert_eq!(workspace.capacities(), capacities);
            } else {
                warmed = Some(workspace.capacities());
            }
        }
        assert!(workspace.capacity_bytes() >= peak_required);
        eprintln!(
            "MSM {threads} tasks: peak required {peak_required} bytes; retained capacity {} bytes",
            workspace.capacity_bytes()
        );
    }
}

#[test]
fn shrinking_batches_reuse_owned_storage() {
    shrinking_batches::<Pallas>();
    shrinking_batches::<Vesta>();
}

fn suffix_rows<C: PastaCurve>() {
    const N: usize = 513;
    let bases: Vec<_> = bases::<C>(N)
        .iter()
        .enumerate()
        .map(|(i, point)| {
            if i % 7 == 0 {
                Point::IDENTITY
            } else {
                point.to_point()
            }
        })
        .collect();
    let mut sums = vec![Point::IDENTITY; N];
    let basis = SuffixBasis::prepare(
        &bases,
        &mut sums,
        &mut [ProjectivePoint::IDENTITY; 17],
        &mut [PastaField::ZERO; 3],
    );
    let unsigned: Vec<_> = (0..N).map(|i| (1_u128 << 96) + (i / 11) as u128).collect();
    let fields: Vec<_> = (0..N)
        .map(|i| {
            PastaField::<C::Scalar>::from_u64((i / 13 + 2) as u64)
                .invert()
                .unwrap()
        })
        .collect();
    let unsigned_fields: Vec<_> = unsigned
        .iter()
        .map(|n| {
            PastaField::from_canonical_uint(zakura_udon::field::CanonicalUint::from_limbs([
                *n as u64,
                (n >> 64) as u64,
                0,
                0,
            ]))
            .unwrap()
        })
        .collect();
    let answers = [&unsigned_fields, &fields].map(|row| {
        expected::<C>(
            row.iter()
                .enumerate()
                .filter(|(i, _)| i % 7 != 0)
                .fold(PastaField::ZERO, |sum, (i, scalar)| {
                    sum.add(&scalar.mul(&PastaField::<_>::from_u64(i as u64 + 1)))
                }),
        )
    });
    let mut integer_differences = vec![0; N];
    let mut field_differences = vec![PastaField::ZERO; N];
    let inputs = [
        basis
            .with_monotone_unsigned(&unsigned, &mut integer_differences)
            .unwrap(),
        basis.with_scalars(&fields, &mut field_differences),
    ];
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let executor = RayonExecutor(&pool);
        let mut workspace = MsmWorkspace::new();
        for limit in [32768, 65536] {
            let options = ExecutionOptions::default()
                .with_task_budget(TaskBudget::new(4).unwrap())
                .with_memory_limit(limit);
            let mut output = [ProjectivePoint::GENERATOR; 2];
            let mut run = workspace.prepare(&inputs, options).unwrap();
            assert!(run.temporary_bytes() <= limit);
            assert!(run.requirements().bytes::<C>().unwrap() <= run.temporary_bytes());
            run.execute(&mut output, &SerialExecutor);
            assert_eq!(output, answers);
            // Enter even a one-worker pool from one of its workers.
            pool.install(|| run.execute(&mut output, &executor));
            assert_eq!(output, answers);
            pool.install(|| run.execute(&mut output, &executor));
            assert_eq!(output, answers);
        }
        assert!(
            workspace
                .prepare(&inputs, ExecutionOptions::default().with_memory_limit(1))
                .is_err()
        );
        let mut output = [ProjectivePoint::IDENTITY; 2];
        workspace
            .prepare(
                &inputs,
                ExecutionOptions::default().with_memory_limit(32768),
            )
            .unwrap()
            .execute(&mut output, &executor);
        assert_eq!(output, answers);
    }
}

#[test]
fn suffix_differences_reuse_bounded_workspaces_and_selected_workers() {
    suffix_rows::<Pallas>();
    suffix_rows::<Vesta>();
}

fn constant_regions<C: PastaCurve>() {
    const N: usize = 513;
    let bases: Vec<_> = bases::<C>(N).iter().map(AffinePoint::to_point).collect();
    let region = BasisSum::prepare(&bases[3..N - 2]);
    let indices: Vec<_> = (0..N).map(|i| ((i * 17) % (N - 5)) as u32).collect();
    let differences: Vec<_> = (0..N)
        .map(|i| {
            PastaField::<C::Scalar>::from_u64(i as u64 + 2)
                .invert()
                .unwrap()
        })
        .collect();
    let constant = PastaField::<C::Scalar>::from_u64(7);
    let extra_bases = [bases[0], bases[N - 1]];
    let extra_scalars = [
        PastaField::from_u64(3),
        PastaField::<C::Scalar>::from_u64(11).neg(),
    ];
    let inputs = [
        region.corrections(&indices, &differences).unwrap(),
        Input::new(Bases::Points(&extra_bases), &extra_scalars),
    ];
    let dense = (3..N - 2).fold(PastaField::ZERO, |sum, i| {
        sum.add(&constant.mul(&PastaField::<_>::from_u64(i as u64 + 1)))
    });
    let corrected = indices.iter().zip(&differences).fold(dense, |sum, (i, d)| {
        sum.add(&d.mul(&PastaField::<_>::from_u64(u64::from(*i) + 4)))
    });
    let answer = expected::<C>(
        corrected
            .add(&extra_scalars[0])
            .add(&extra_scalars[1].mul(&PastaField::<_>::from_u64(N as u64))),
    );
    for threads in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let executor = RayonExecutor(&pool);
        let mut workspace = MsmWorkspace::new();
        for limit in [32768, 65536] {
            let options = ExecutionOptions::default()
                .with_task_budget(TaskBudget::new(4).unwrap())
                .with_memory_limit(limit);
            let mut run = workspace.prepare(&inputs, options).unwrap();
            assert!(run.temporary_bytes() <= limit);
            let mut output = [ProjectivePoint::IDENTITY; 2];
            for _ in 0..2 {
                pool.install(|| run.execute(&mut output, &executor));
                assert_eq!(
                    region
                        .sum()
                        .mul_projective(&constant)
                        .add(&output[0])
                        .add(&output[1]),
                    answer
                );
            }
            run.execute(&mut output, &SerialExecutor);
            assert_eq!(
                region
                    .sum()
                    .mul_projective(&constant)
                    .add(&output[0])
                    .add(&output[1]),
                answer
            );
        }
        assert!(
            workspace
                .prepare(&inputs, ExecutionOptions::default().with_memory_limit(1))
                .is_err()
        );
        let mut output = [ProjectivePoint::IDENTITY; 2];
        workspace
            .prepare(&inputs, ExecutionOptions::default())
            .unwrap()
            .execute(&mut output, &SerialExecutor);
        assert_eq!(
            region
                .sum()
                .mul_projective(&constant)
                .add(&output[0])
                .add(&output[1]),
            answer
        );
    }
}

#[test]
fn constant_region_corrections_include_extras_with_bounded_workers() {
    constant_regions::<Pallas>();
    constant_regions::<Vesta>();
}

fn coalescing<C: PastaCurve>() {
    let original = bases::<C>(129);
    let indices: Vec<_> = (0..513)
        .map(|i| ((i * 17) % original.len()) as u32)
        .collect();
    let points: Vec<_> = indices
        .iter()
        .enumerate()
        .map(|(i, &index)| {
            let point = original[index as usize].to_point();
            if i % 2 == 0 { point } else { point.neg() }
        })
        .collect();
    let mut keys = vec![CoalescingKey::EMPTY; indices.len()];
    let plan = CoalescingPlan::prepare(&points, &mut keys);
    let mut order = vec![0; indices.len()];
    let indexed =
        IndexedCoalescingPlan::prepare(Bases::Affine(&original), &indices, &mut order).unwrap();
    let mut output_points = vec![Point::IDENTITY; plan.groups()];
    let mut output_indices = vec![0; indexed.groups()];
    let mut point_sums = vec![PastaField::ZERO; plan.groups()];
    let mut index_sums = vec![PastaField::ZERO; indexed.groups()];
    let mut workspace = MsmWorkspace::new();
    for round in 0..3 {
        let row: Vec<_> = (0..indices.len())
            .map(|i| {
                if round == 1 {
                    PastaField::ZERO
                } else {
                    PastaField::<C::Scalar>::from_u64((i + round + 2) as u64)
                        .invert()
                        .unwrap()
                }
            })
            .collect();
        let scalar =
            indices
                .iter()
                .zip(&row)
                .enumerate()
                .fold(PastaField::ZERO, |sum, (i, (&index, s))| {
                    let term = s.mul(&PastaField::<C::Scalar>::from_u64(u64::from(index) + 1));
                    if i % 2 == 0 {
                        sum.add(&term)
                    } else {
                        sum.sub(&term)
                    }
                });
        let unsigned = indices
            .iter()
            .zip(&row)
            .fold(PastaField::ZERO, |sum, (&index, s)| {
                sum.add(&s.mul(&PastaField::<C::Scalar>::from_u64(u64::from(index) + 1)))
            });
        let inputs = [
            plan.with_scalars(&row, &mut output_points, &mut point_sums),
            indexed.with_scalars(&row, &mut output_indices, &mut index_sums),
        ];
        let answers = [expected::<C>(scalar), expected::<C>(unsigned)];
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let executor = RayonExecutor(&pool);
            for limit in [8192, 65536] {
                let options = ExecutionOptions::default()
                    .with_task_budget(TaskBudget::new(4).unwrap())
                    .with_memory_limit(limit);
                let mut run = workspace.prepare(&inputs, options).unwrap();
                assert!(run.temporary_bytes() <= limit);
                let mut output = [ProjectivePoint::IDENTITY; 2];
                for _ in 0..2 {
                    pool.install(|| run.execute(&mut output, &executor));
                    assert_eq!(output, answers);
                }
                run.execute(&mut output, &SerialExecutor);
                assert_eq!(output, answers);
            }
        }
    }
}

#[test]
fn coalesced_inputs_reuse_bounded_scratch_on_one_and_four_workers() {
    coalescing::<Pallas>();
    coalescing::<Vesta>();
}

fn nonzero_support<C: PastaCurve>() {
    let bases = bases::<C>(513);
    let selection = Selection::new(Bases::Affine(&bases));
    let extras = [bases[0], bases[512]];
    let extra_scalars = [
        PastaField::<C::Scalar>::from_u64(7),
        PastaField::<C::Scalar>::ONE.neg(),
    ];
    let mut indices = vec![0; bases.len()];
    let mut scalars = vec![PastaField::ZERO; bases.len()];
    let mut workspace = MsmWorkspace::new();
    for stride in [0, 1, 2, 17, 513] {
        let row: Vec<_> = (0..bases.len())
            .map(|i| {
                if stride != 0 && i % stride == 0 {
                    PastaField::<C::Scalar>::from_u64(i as u64 + 2)
                        .invert()
                        .unwrap()
                } else {
                    PastaField::ZERO
                }
            })
            .collect();
        let total = row.iter().enumerate().fold(
            PastaField::<C::Scalar>::from_u64(7).sub(&PastaField::<C::Scalar>::from_u64(513)),
            |sum, (i, s)| sum.add(&s.mul(&PastaField::<C::Scalar>::from_u64(i as u64 + 1))),
        );
        let answer = expected::<C>(total);
        let compact = selection
            .with_nonzero_scalars(&row, &mut indices, &mut scalars)
            .unwrap();
        let inputs = [compact, Input::new(Bases::Affine(&extras), &extra_scalars)];
        for threads in [1, 4] {
            let pool = rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap();
            let executor = RayonExecutor(&pool);
            for limit in [8192, 65536] {
                let options = ExecutionOptions::default()
                    .with_task_budget(TaskBudget::new(4).unwrap())
                    .with_memory_limit(limit);
                let mut run = workspace.prepare(&inputs, options).unwrap();
                assert!(run.temporary_bytes() <= limit);
                let mut output = [ProjectivePoint::IDENTITY; 2];
                for _ in 0..2 {
                    pool.install(|| run.execute(&mut output, &executor));
                    assert_eq!(output[0].add(&output[1]), answer);
                }
                run.execute(&mut output, &SerialExecutor);
                assert_eq!(output[0].add(&output[1]), answer);
            }
        }
    }
}

#[test]
fn nonzero_support_keeps_extras_with_bounded_workers() {
    nonzero_support::<Pallas>();
    nonzero_support::<Vesta>();
}
