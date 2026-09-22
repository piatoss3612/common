use crate::bridge::{
    executor::{RayonExecutor, with_side_work},
    msm::MsmWorkspace,
};
use zakura_udon::{
    curve::{
        AffinePoint, EisensteinTable, EisensteinTableBatch, Pallas, PastaCurve, Point,
        PreparedAffinePoint, ProjectivePoint, Vesta, batch_normalize,
        msm::{Bases, Selection},
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
    )
    .unwrap();
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
                        PastaField::from_u64((i + round + 2) as u64)
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
                        sum.add(&scalar.mul(&PastaField::from_u64(index as u64 + 1)))
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
        )
        .unwrap();
    }
    let fixed = [
        EisensteinTable::bind(&bases[3], &entries[0]).unwrap(),
        EisensteinTable::bind(&bases[7], &entries[1]).unwrap(),
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
                    let step = PastaField::from_u64((round + side + repetition + 2) as u64)
                        .invert()
                        .unwrap();
                    (0..len)
                        .map(|i| step.mul(&PastaField::from_u64(i as u64)))
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
                            sum.add(&scalar.mul(&PastaField::from_u64(u64::from(*index) + 1)))
                        })
                        .add(&side_scalars[2 * i].mul(&PastaField::from_u64(4)))
                        .add(&side_scalars[2 * i + 1].mul(&PastaField::from_u64(8)));
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
