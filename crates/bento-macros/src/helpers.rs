//! Error reporting shared by macro entry points.
//!
//! [`macro_body`] turns expansion errors into compiler diagnostics at their
//! original source spans.

use proc_macro2::TokenStream;
use syn::Result;

/// Converts expansion errors into compiler diagnostics, preserving error spans.
pub fn macro_body<F>(f: F) -> proc_macro::TokenStream
where
    F: FnOnce() -> Result<TokenStream>,
{
    f().unwrap_or_else(|e| e.into_compile_error()).into()
}
