//! Dependency arrangements and target portability across the public facade.
#![forbid(unsafe_code)]

#[path = "../harness/mod.rs"]
mod harness;

mod dependencies;
mod portability;
