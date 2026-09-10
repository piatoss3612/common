//! Procedural macros over `bento-core`.
//!
//! Macros are exposed and documented through the `bento` facade. This crate
//! contains their implementation and is not intended as a direct dependency.
//!
//! Entry points here only parse input, resolve dependency paths, and invoke
//! `helpers::macro_body`. Expansion lives in `derive` or `proc` and uses
//! `proc_macro2::TokenStream` and `syn::Result` so it can be tested without the
//! compiler's procedural macro context. Shared arithmetic belongs in
//! `bento-core`; parsing and token generation belong here.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

mod derive;
mod helpers;
mod path_resolution;
mod proc;
