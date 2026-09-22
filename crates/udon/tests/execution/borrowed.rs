//! Minimal movable owners: typed borrows cross threads, erased views stay local.
use std::thread;
use zakura_udon::{
    curve::{
        AffinePoint, Pallas, ProjectivePoint,
        msm::{self, run as msm_run},
    },
    exec::{
        ExecutionOptions, SerialExecutor, TaskBudget,
        run::{Identity, TaskStorage},
    },
    fft::{self, run as fft_run},
    field::{Fp, Fq, PallasBase},
};

fn require_send<T: Send>(_: &T) {}

struct FftSlices<'a> {
    values: &'a mut [Fp],
    source: &'a [Fp],
}
impl fft_run::Resources<PallasBase> for FftSlices<'_> {
    fn buffers(&mut self) -> fft_run::Buffers<'_, PallasBase> {
        fft_run::Buffers {
            values: self.values,
            pair: &mut [],
            source: &self.source,
            factor: &[],
        }
    }
}

#[test]
fn fft_task_owns_nonstatic_slices_on_a_scoped_worker() {
    let source = [Fp::ONE, Fp::from_u64(2), Fp::ZERO, Fp::ZERO];
    let mut values = [Fp::ZERO; 4];
    let domain = fft::Domain::for_size(4).unwrap().subgroup();
    let plan = fft_run::FftPlan::new(
        fft::Transform::new(domain),
        fft::TransformRequest {
            input_storage: fft::InputStorage::Preserve,
            ..fft::TransformRequest::new(fft::Direction::Forward)
        },
        fft::StorageLayout::Contiguous,
        ExecutionOptions::default(),
    )
    .unwrap();
    let mut identity = Identity::new();
    let mut slots = [TaskStorage::EMPTY];
    let mut run = fft_run::FftRun::new(plan, false, &mut identity, &mut slots);
    let mut ready = [None];
    assert_eq!(run.ready(&mut ready), 1);
    let mut task = run
        .try_claim(ready[0].take().unwrap(), || {
            Some(FftSlices {
                values: &mut values,
                source: &source,
            })
        })
        .unwrap()
        .unwrap();
    require_send(&task);
    let receipt = thread::scope(|scope| {
        scope
            .spawn(move || {
                task.execute().unwrap();
                task.finish()
            })
            .join()
            .unwrap()
    });
    assert_eq!(run.complete(receipt).unwrap().error, None);
    assert!(run.is_complete());
    for (row, value) in values.iter().enumerate() {
        let point = domain.domain().root().pow_u64(row as u64);
        assert_eq!(
            (*value).reduce(),
            (<Fp>::ONE.add(&<Fp>::from_u64(2).mul(&point))).reduce()
        );
    }
    assert_eq!((source[1]).reduce(), (<Fp>::from_u64(2)).reduce());
}

struct MsmSlices<'a> {
    scratch: msm::Scratch<'a, Pallas>,
    output: &'a mut [ProjectivePoint<Pallas>],
    partials: &'a [ProjectivePoint<Pallas>],
}
impl msm_run::Resources<Pallas> for MsmSlices<'_> {
    fn buffers(&mut self) -> msm_run::Buffers<'_, Pallas> {
        msm_run::Buffers {
            records: &[],
            digits: &[],
            scratch: self.scratch.reborrow(),
            buckets: &mut [],
            output: self.output,
            partials: &self.partials,
        }
    }
}

#[test]
fn msm_tasks_own_nonstatic_slices_on_scoped_workers() {
    let bases = [AffinePoint::<Pallas>::GENERATOR; 2];
    let scalars = [Fq::from_u64(2), Fq::from_u64(3)];
    let mut records = [msm::ScalarStorage::ZERO; 2];
    let prepared =
        msm::PreparedScalars::prepare(&scalars, &mut records, TaskBudget::SERIAL, &SerialExecutor);
    let input = msm::Input::new_prepared(msm::Bases::Affine(&bases), prepared);
    let plan = msm_run::MsmPlan::new(2, ExecutionOptions::default()).unwrap();
    let required = plan.temporary();
    let mut affine = vec![AffinePoint::GENERATOR; required.affine()];
    let mut projective = vec![ProjectivePoint::IDENTITY; required.projective()];
    let mut field = vec![Fp::ZERO; required.field()];
    let mut indices = vec![0; required.indices()];
    let mut partial = [ProjectivePoint::IDENTITY];
    let mut identity = Identity::new();
    let mut slots = [TaskStorage::EMPTY];
    let mut run = msm_run::MsmRun::new(plan, input, &mut identity, &mut slots);
    while run.result().is_none() {
        let mut ready = [None];
        assert_eq!(run.ready(&mut ready), 1);
        let request = ready[0].take().unwrap();
        // Prepared small scalars need no shared recoding or preparation writes.
        assert_eq!(request.scratch.scalars() + request.scratch.digits(), 0);
        assert_eq!(request.read_scalars + request.read_digits, 0);
        let (output, partials) = if request.output_slot.is_some() {
            (&mut partial[..], &[][..])
        } else {
            (&mut [][..], &partial[..])
        };
        let mut task = run
            .try_claim(request, || {
                Some(MsmSlices {
                    scratch: msm::Scratch::new(
                        &mut [],
                        &mut [],
                        &mut affine,
                        &mut projective,
                        &mut field,
                        &mut indices,
                    ),
                    output,
                    partials,
                })
            })
            .unwrap()
            .unwrap();
        require_send(&task);
        let receipt = thread::scope(|scope| {
            scope
                .spawn(move || {
                    task.execute().unwrap();
                    task.finish()
                })
                .join()
                .unwrap()
        });
        assert_eq!(run.complete(receipt).unwrap().error, None);
    }
    let generator = bases[0].to_projective();
    assert_eq!(
        run.result(),
        Some(generator.double().double().add(&generator))
    );
}
