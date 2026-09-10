//! Shared POD storage utilities and compile-time support for field implementations.

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

#[expect(
    unused_imports,
    reason = "the facade wiring precedes core's public items; this \
              expectation goes unfulfilled (and gets removed) once they land"
)]
pub use bento_core::*;
