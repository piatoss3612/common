//! Public FFT execution, workspaces, and stored tables.
#![forbid(unsafe_code)]

#[path = "../harness/mod.rs"]
mod harness;

#[path = "../harness/executor.rs"]
#[allow(dead_code)] // This consumer only uses the pool adapter.
mod executor_adapter;

#[path = "../../src/exec/execution/test_pool.rs"]
#[allow(dead_code)]
mod run_pool;

#[path = "../../src/fft/execution/test_pipeline.rs"]
mod fft_pipeline;

mod borrowed;
mod buffers;
mod embedding;
mod expansion;
#[cfg(feature = "traits")]
#[path = "../harness/field_model.rs"]
mod field_model;
mod interpolation;
#[cfg(feature = "traits")]
mod traits;
mod workspace;
