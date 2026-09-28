//! FFT tasks can own nonstatic typed borrows across scoped workers.
use std::thread;
use zakura_udon::{
    exec::{
        ExecutionOptions,
        execution::{Identity, TaskStorage},
    },
    fft::{self, execution as fft_run},
    field::{Fp, PallasBase},
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
    // Existing callers can still name the plan through `run`.
    let plan: zakura_udon::fft::execution::FftPlan<'_, PallasBase> = fft_run::FftPlan::new(
        fft::Transform::new(domain),
        fft::TransformRequest {
            input_storage: fft::InputStorage::Preserve,
            ..fft::TransformRequest::new(fft::Direction::Forward)
        },
        fft::StorageLayout::Contiguous,
        ExecutionOptions::default(),
    )
    .unwrap();
    let mut identity: zakura_udon::exec::execution::Identity = Identity::new();
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
