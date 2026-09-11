//! Shared support traits, reference arithmetic, and storage behind `bento`.
//!
//! The [`addchain`] module defines the target operations used by addition chains.
//! [`Pod`] defines the storage contract used by the byte views and file embedding
//! macros. [`const_arithmetic`] derives field constants from primitive parameters.

#![no_std]
#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

pub mod addchain;
pub mod const_arithmetic;

#[allow(unsafe_code)]
mod pod;
pub use pod::{AlignedBytes, MAX_ALIGN, Pod, bytes_of, bytes_of_slice};

#[doc(hidden)]
pub use pod::Layout as __PodLayout;
