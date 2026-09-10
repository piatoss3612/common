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
//!
//! Report invalid input with `syn::Error`; reserve panics for internal
//! invariants. In generated code, interpolate the supplied `BentoCorePath` for
//! library items and use absolute `::core` paths for standard types to support
//! `no_std` callers.
//!
//! Document and explicitly re-export each macro from `bento`. Test parsing and
//! expansion in the implementation module, and use the facade dev-dependency
//! for tests of generated behavior. Expansions that reference library items
//! also need separate Cargo consumer tests to exercise dependency resolution.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

mod derive;
mod helpers;
mod path_resolution;
mod proc;

// Documentation lives on the facade's re-export.
#[expect(missing_docs)]
#[proc_macro]
pub fn addition_chain(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as proc::addition_chain::Input);
    helpers::macro_body(|| {
        let core = path_resolution::BentoCorePath::resolve()?;
        proc::addition_chain::evaluate(input, core)
    })
}
