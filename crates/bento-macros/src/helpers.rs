//! Shared entry-point plumbing.

use proc_macro2::TokenStream;
use syn::Result;

/// Turn an expansion result into compiler tokens, preserving error spans.
pub fn macro_body<F>(f: F) -> proc_macro::TokenStream
where
    F: FnOnce() -> Result<TokenStream>,
{
    f().unwrap_or_else(|e| e.into_compile_error()).into()
}
