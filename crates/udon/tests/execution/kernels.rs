// Kernel-forcing protocol tests compile inside Udon so tuning stays private.
#[path = "../support/admission.rs"]
mod admission;
#[path = "fft.rs"]
mod fft;
#[path = "../support/fft_run.rs"]
mod fft_run;
#[path = "msm.rs"]
mod msm;
#[path = "../support/msm_run.rs"]
mod msm_run;
#[path = "../support/run_pool.rs"]
#[allow(dead_code)]
mod run_pool;

#[path = "../support/fft_pipeline.rs"]
mod fft_pipeline;

mod frontier;
mod mixed;
