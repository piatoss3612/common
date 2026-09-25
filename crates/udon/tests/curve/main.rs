//! Public curve arithmetic, generic contracts, and storage.
#![forbid(unsafe_code)]

#[path = "../harness/mod.rs"]
mod harness;

#[path = "../harness/field_model.rs"]
mod field_model;

mod constants;
mod edwards;
mod embedding;
mod pod;
mod traits;
