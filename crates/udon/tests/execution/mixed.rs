#[path = "../support/admission.rs"]
pub(crate) mod admission;
#[path = "../support/fft_run.rs"]
mod fft_run;
#[path = "../support/mixed_run.rs"]
mod mixed_run;
#[path = "../support/msm_run.rs"]
#[allow(dead_code)]
mod msm_run;
#[path = "../support/run_pool.rs"]
pub(crate) mod run_pool;
#[test]
fn heterogeneous_shrinking_rounds_share_one_arena_and_release_consumers() {
    let fixture = mixed_run::Fixture::new(1024, 1);
    for workers in [1, 3, 4, 16] {
        for queue in [1, 3] {
            mixed_run::scoped_checked(&fixture, workers, queue, |run| {
                let first = run();
                let second = run();
                assert_eq!(first.bytes, second.bytes);
                assert!(first.bytes < mixed_run::CEILING);
                assert_eq!(first.results, second.results);
                assert_eq!(first.rounds, 22);
                assert!(first.tasks > 1000);
                assert!(first.early_consumers > 0);
                assert_eq!(first.peak.0[4], 1);
                assert!(first.peak.0[6] <= queue);
            });
        }
    }
}
