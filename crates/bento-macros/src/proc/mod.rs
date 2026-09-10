//! Function-like procedural macro implementations, one module per macro.
//!
//! Define an `Input` implementing `syn::parse::Parse` when a built-in syntax
//! node does not suffice. Expose `evaluate(input: Input, ...)`, taking any
//! resolved paths explicitly and returning `syn::Result<proc_macro2::TokenStream>`.
//! Keep parsing, expansion, and unit tests together in that module.
