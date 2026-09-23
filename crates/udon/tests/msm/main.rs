//! Public MSM execution and caller-owned workspaces.
#![forbid(unsafe_code)]

#[path = "../harness/executor.rs"]
mod executor_adapter;

mod borrowed;
mod buffers;
mod workspace;
