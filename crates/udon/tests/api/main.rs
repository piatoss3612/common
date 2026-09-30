//! Cross-domain compiler checks of the public API.
#![forbid(unsafe_code)]

#[path = "../harness/mod.rs"]
mod harness;

mod boundaries;
mod traits;
