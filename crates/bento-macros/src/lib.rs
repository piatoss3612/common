//! Procedural macro implementations for the `bento` facade.
//!
//! Macros are exposed and documented through the `bento` facade. This crate
//! contains their implementation and is not intended as a direct dependency.
//!
//! Macros run on the build host, while their output must compile for the
//! caller's target. Shared support interfaces and reference arithmetic belong
//! in [`bento_core`]; parsing and token generation belong here.

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]

mod derive;
mod helpers;
mod path_resolution;
mod proc;

// Documentation lives on the facade's re-export.
#[expect(missing_docs)]
#[proc_macro_derive(Pod, attributes(pod))]
pub fn pod(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as syn::DeriveInput);
    helpers::macro_body(|| {
        let core = derive::pod::core_path(&input)?;
        derive::pod::derive(input, core)
    })
}

// Documentation lives on the facade's wrapper.
#[expect(missing_docs)]
#[proc_macro]
pub fn addition_chain(input: proc_macro::TokenStream) -> proc_macro::TokenStream {
    let input = syn::parse_macro_input!(input as proc::addition_chain::Invocation);
    helpers::macro_body(|| proc::addition_chain::evaluate(input.input, input.core))
}
