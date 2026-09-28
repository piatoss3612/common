//! Support paths for generated code.
//!
//! Derives discover ordinary facade aliases as a convenience. Manifest lookup
//! cannot establish whether a dependency is active in the current compilation;
//! exceptional consumers provide an explicit path. Function-like macros receive
//! the defining facade's hygienic path through its declarative wrapper.

use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Span, TokenStream};
use quote::ToTokens;
use syn::{Error, Ident, Path, Result, parse_quote};

/// A path through which generated code can reach the support interface.
#[derive(Clone)]
pub struct BentoCorePath(Path);

impl From<Path> for BentoCorePath {
    fn from(path: Path) -> Self {
        Self(path)
    }
}

impl ToTokens for BentoCorePath {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.0.to_tokens(tokens);
    }
}

impl Default for BentoCorePath {
    fn default() -> Self {
        Self(parse_quote!(::bento_core))
    }
}

impl BentoCorePath {
    /// Discovers a facade dependency, without selecting unrelated core entries.
    pub fn resolve() -> Result<Self> {
        let found = crate_name("zakura-bento").map_err(|_| {
            Error::new(
                Span::call_site(),
                "cannot discover zakura-bento; use #[pod(crate = path)] to name the support path",
            )
        })?;
        facade_path(found).map(Self)
    }
}

fn facade_path(found: FoundCrate) -> Result<Path> {
    // The library has a self alias; integration tests and doctests see its
    // external crate name. `crate` would instead name the doctest harness.
    let name = match found {
        FoundCrate::Itself => "zakura_bento".into(),
        FoundCrate::Name(name) => name,
    };
    let name = syn::parse_str::<Ident>(&name)
        .or_else(|_| syn::parse_str::<Ident>(&format!("r#{name}")))?;
    Ok(parse_quote!(::#name))
}

#[cfg(test)]
mod tests {
    use quote::quote;

    use super::*;

    #[test]
    fn facade_aliases_and_self_paths() {
        for (found, expected) in [
            (FoundCrate::Name("bento".into()), quote!(::bento)),
            (FoundCrate::Name("renamed".into()), quote!(::renamed)),
            (FoundCrate::Name("type".into()), quote!(::r#type)),
            (FoundCrate::Itself, quote!(::zakura_bento)),
        ] {
            let path = facade_path(found).unwrap();
            assert_eq!(quote!(#path).to_string(), expected.to_string());
        }
    }
}
