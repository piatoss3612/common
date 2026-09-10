//! Conventions for derive macro implementations.
//!
//! Add one module per derive, with a `derive` function that accepts
//! [`syn::DeriveInput`] and resolved dependency paths and returns expansion
//! tokens through [`syn::Result`].
//! Keep input validation, expansion, and unit tests together in that module.
//! Mark generated implementations with `#[automatically_derived]`.
