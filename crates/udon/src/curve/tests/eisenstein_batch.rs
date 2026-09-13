use super::*;
use crate::exec::{Executor, SerialExecutor, TaskBudget};

struct Pool;
impl Executor for Pool {
    fn join<L, R, A, B>(&self, left: L, right: R) -> (A, B)
    where
        L: FnOnce() -> A + Send,
        R: FnOnce() -> B + Send,
        A: Send,
        B: Send,
    {
        rayon::join(left, right)
    }
}

fn batches<C: PastaCurve, E: CurveTableEntry<C> + Eq>() {
    let g = AffinePoint::<C>::GENERATOR;
    let bases: Vec<_> = (1..=129)
        .map(|i| {
            *g.mul_projective(&PastaField::from_u64(i))
                .to_point()
                .as_affine()
                .unwrap()
        })
        .collect();
    for n in [0, 1, 7, 8, 15, 31, 32, 33, 63, 64, 65, 99, 128, 129] {
        let r = EisensteinTableBatch::<C, E>::requirements(n).unwrap();
        let mut entries = vec![E::from_affine(&g); r.table_entries];
        let mut projective = vec![ProjectivePoint::GENERATOR; r.projective_scratch + 1];
        let mut field = vec![PastaField::ONE; r.field_scratch + 1];
        for tasks in [1, 3, 8] {
            let budget = TaskBudget::new(tasks).unwrap();
            let tables = EisensteinTableBatch::prepare(
                &bases[..n],
                &mut entries,
                &mut projective,
                &mut field,
                budget,
                &Pool,
            )
            .unwrap();
            assert_eq!(projective[r.projective_scratch], ProjectivePoint::GENERATOR);
            assert_eq!(field[r.field_scratch], PastaField::ONE);
            EisensteinTableBatch::<C, E>::bind(tables.as_slice()).unwrap();
            assert!(tables.get(n).is_none());
            let required = EisensteinTableBatch::<C, E>::multiplication_scratch(n).unwrap();
            let mut scratch = vec![PastaField::ONE; required + 1];
            let mut output = vec![ProjectivePoint::GENERATOR; n];
            for scalar in scalar_corpus::<C>().into_iter().step_by(7) {
                let prepared = EisensteinScalar::new(&scalar);
                tables
                    .mul_prepared(&prepared, &mut output, &mut scratch, budget, &Pool)
                    .unwrap();
                assert_eq!(scratch[required], PastaField::ONE);
                for (i, base) in bases[..n].iter().enumerate() {
                    let expected =
                        crate::curve::scalar::multiply(&scalar, |sum| sum.add_mixed(base));
                    assert_eq!(output[i], expected, "n={n}, tasks={tasks}, term={i}");
                    assert_eq!(tables.get(i).unwrap().mul_prepared(&prepared), expected);
                }
            }
        }
    }
}

#[test]
fn pallas() {
    rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap()
        .install(|| {
            batches::<Pallas, AffinePoint<Pallas>>();
            batches::<Pallas, PreparedAffinePoint<Pallas>>();
        });
}
#[test]
fn vesta() {
    // Nested joins must also finish when no other worker is available.
    rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .unwrap()
        .install(|| {
            batches::<Vesta, AffinePoint<Vesta>>();
            batches::<Vesta, PreparedAffinePoint<Vesta>>();
        });
}

#[test]
fn errors_preserve_preparation_and_multiplication_buffers() {
    type C = Pallas;
    let g = AffinePoint::<C>::GENERATOR;
    for n in [7, 8, 63, 64, 65] {
        let r = EisensteinTableBatch::<C>::requirements(n).unwrap();
        for failure in 0..4 {
            let mut bases = vec![g; n];
            let mut entries = vec![g; r.table_entries];
            let mut projective = vec![ProjectivePoint::GENERATOR; r.projective_scratch];
            let mut field = vec![PastaField::ONE; r.field_scratch];
            if failure == 0 {
                entries.pop();
            }
            if failure == 1 {
                field.pop();
            }
            if failure == 2 && projective.pop().is_none() {
                continue;
            }
            if failure == 3 {
                bases[n - 1].x = invalid_field();
            }
            assert!(
                EisensteinTableBatch::prepare(
                    &bases,
                    &mut entries,
                    &mut projective,
                    &mut field,
                    TaskBudget::SERIAL,
                    &SerialExecutor
                )
                .is_err()
            );
            assert!(entries.iter().all(|&p| p == g));
            assert!(projective.iter().all(|&p| p == ProjectivePoint::GENERATOR));
            assert!(field.iter().all(|&f| f == PastaField::ONE));
        }
        let mut entries = vec![g; r.table_entries];
        let mut projective = vec![ProjectivePoint::IDENTITY; r.projective_scratch];
        let mut field = vec![PastaField::ZERO; r.field_scratch];
        let tables = EisensteinTableBatch::prepare(
            &vec![g; n],
            &mut entries,
            &mut projective,
            &mut field,
            TaskBudget::SERIAL,
            &SerialExecutor,
        )
        .unwrap();
        let required = EisensteinTableBatch::<C>::multiplication_scratch(n).unwrap();
        let mut field = vec![PastaField::ONE; required];
        let mut output = vec![ProjectivePoint::GENERATOR; n + 1];
        assert!(
            tables
                .mul(
                    &PastaField::ONE,
                    &mut output,
                    &mut field,
                    TaskBudget::SERIAL,
                    &SerialExecutor
                )
                .is_err()
        );
        if required != 0 {
            assert!(
                tables
                    .mul(
                        &PastaField::ONE,
                        &mut output[..n],
                        &mut field[..required - 1],
                        TaskBudget::SERIAL,
                        &SerialExecutor
                    )
                    .is_err()
            );
        }
        assert!(output.iter().all(|&p| p == ProjectivePoint::GENERATOR));
        assert!(field.iter().all(|&f| f == PastaField::ONE));
    }
    assert!(matches!(
        EisensteinTableBatch::<C>::bind(&[g; 7]),
        Err(CurveError::InvalidTableLayout)
    ));
    assert!(matches!(
        EisensteinTableBatch::<C>::bind(&[g; 8]),
        Err(CurveError::InvalidTable)
    ));
    assert!(matches!(
        EisensteinTableBatch::<C>::requirements(usize::MAX),
        Err(CurveError::SizeOverflow)
    ));
    assert!(matches!(
        EisensteinTableBatch::<C>::multiplication_scratch(usize::MAX),
        Err(CurveError::SizeOverflow)
    ));
}
