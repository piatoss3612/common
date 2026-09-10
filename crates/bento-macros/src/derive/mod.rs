//! Derive macro implementations, one module per derive.
//!
//! Each module exposes `derive(input: syn::DeriveInput, ...)`, taking resolved
//! paths explicitly and returning `syn::Result<proc_macro2::TokenStream>`.
//! Keep input validation, expansion, and unit tests together in that module.
//! Mark generated implementations with `#[automatically_derived]`.
