//! Resolve the caller's path to items defined in `bento-core`.
//!
//! Prefer a direct dependency on `zakura-bento-core`, then fall back to the
//! `zakura-bento` facade's root re-exports. Look up package names rather than
//! assuming dependency names: both crates can be renamed in `Cargo.toml`.
//! Self references use `crate`.
//!
//! In a doctest, `crate` refers to the generated test crate. The facade's
//! doctests resolve through its direct core dependency, which takes precedence
//! over its self lookup. Verify changes with those doctests and the separate
//! Cargo consumers in `tests/consumers.rs`, as well as the unit tests here.

use proc_macro_crate::{FoundCrate, crate_name};
use proc_macro2::{Span, TokenStream};
use quote::ToTokens;
use syn::{Error, Ident, Path, Result, parse_quote};

/// The path through which generated code can reach `bento-core` items.
#[derive(Clone)]
pub struct BentoCorePath(Path);

impl ToTokens for BentoCorePath {
    fn to_tokens(&self, tokens: &mut TokenStream) {
        self.0.to_tokens(tokens)
    }
}

impl Default for BentoCorePath {
    /// A deterministic path for expansion tests; entry points use `resolve`.
    fn default() -> Self {
        Self(parse_quote!(::bento_core))
    }
}

impl BentoCorePath {
    pub fn resolve() -> Result<Self> {
        bento_core_path(
            crate_name("zakura-bento-core").ok(),
            crate_name("zakura-bento").ok(),
        )
        .map(Self)
    }
}

fn bento_core_path(core: Option<FoundCrate>, facade: Option<FoundCrate>) -> Result<Path> {
    Ok(match (core, facade) {
        (Some(FoundCrate::Itself), _) => parse_quote!(crate),
        (Some(FoundCrate::Name(name)), _) | (None, Some(FoundCrate::Name(name))) => {
            let name = syn::parse_str::<Ident>(&name)
                .or_else(|_| syn::parse_str::<Ident>(&format!("r#{name}")))?;
            parse_quote!(::#name)
        }
        (None, Some(FoundCrate::Itself)) => parse_quote!(crate),
        (None, None) => {
            return Err(Error::new(
                Span::call_site(),
                "failed to find zakura-bento or zakura-bento-core; add zakura-bento to your Cargo.toml dependencies",
            ));
        }
    })
}

#[cfg(test)]
mod tests {
    use quote::quote;

    use super::*;

    #[test]
    fn test_dependency_paths() {
        let cases = [
            (
                Some(FoundCrate::Name("bento_core".into())),
                None,
                quote!(::bento_core),
            ),
            (
                None,
                Some(FoundCrate::Name("bento".into())),
                quote!(::bento),
            ),
            (
                None,
                Some(FoundCrate::Name("renamed".into())),
                quote!(::renamed),
            ),
            (
                None,
                Some(FoundCrate::Name("type".into())),
                quote!(::r#type),
            ),
            (
                Some(FoundCrate::Name("renamed_core".into())),
                Some(FoundCrate::Name("renamed_facade".into())),
                quote!(::renamed_core),
            ),
            (Some(FoundCrate::Itself), None, quote!(crate)),
            (None, Some(FoundCrate::Itself), quote!(crate)),
            (
                Some(FoundCrate::Name("bento_core".into())),
                Some(FoundCrate::Itself),
                quote!(::bento_core),
            ),
        ];

        for (core, facade, expected) in cases {
            let resolved = BentoCorePath(bento_core_path(core, facade).unwrap());
            assert_eq!(quote!(#resolved).to_string(), expected.to_string());
        }
    }

    #[test]
    fn test_missing_dependency() {
        let error = bento_core_path(None, None).err().unwrap();
        assert_eq!(
            error.to_string(),
            "failed to find zakura-bento or zakura-bento-core; add zakura-bento to your Cargo.toml dependencies",
        );
    }

    #[test]
    fn test_resolve_manifest_dependency() {
        // The inherited workspace dependency renames zakura-bento-core to bento-core.
        let resolved = BentoCorePath::resolve().unwrap();
        let expected = BentoCorePath::default();
        assert_eq!(quote!(#resolved).to_string(), quote!(#expected).to_string());
    }
}
