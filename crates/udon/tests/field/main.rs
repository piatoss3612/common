//! Public field arithmetic, generic contracts, and storage.
#![forbid(unsafe_code)]

#[path = "../harness/mod.rs"]
mod harness;

#[path = "../harness/field_model.rs"]
#[cfg(feature = "traits")]
mod field_model;

mod constants;
mod embedding;
mod pod;
#[cfg(feature = "traits")]
mod traits;
