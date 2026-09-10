//! Shared support traits and storage behind the `bento` facade.
//!
//! The [`addchain`] module defines the target operations used by addition chains.
//! [`Pod`] defines the storage contract used by the byte views and file embedding
//! macros.

#![no_std]
#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod addchain;

#[allow(unsafe_code)]
mod pod;
pub use pod::{AlignedBytes, MAX_ALIGN, Pod, bytes_of, bytes_of_slice};
