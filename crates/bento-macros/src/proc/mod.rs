//! Function-like procedural macro implementations, one module per macro.
//!
//! [`addition_chain`] parses fixed scalars and emits scaling operations.
//!
//! Define an input type implementing [`syn::parse::Parse`] when a built-in
//! syntax node does not suffice. Expose an `evaluate` function that accepts the
//! parsed input and resolved dependency paths and returns expansion tokens
//! through [`syn::Result`].
//! Keep parsing, expansion, and unit tests together in that module.

pub(crate) mod addition_chain;
