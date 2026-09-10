//! Shared entry-point plumbing.

use proc_macro2::TokenStream;
use syn::Result;

/// Turn an expansion result into compiler tokens, preserving error spans.
#[expect(
    dead_code,
    reason = "entry-point plumbing for the first procedural macro"
)]
pub fn macro_body<F>(f: F) -> proc_macro::TokenStream
where
    F: FnOnce() -> Result<TokenStream>,
{
    f().unwrap_or_else(|e| e.into_compile_error()).into()
}
