//! Derivation of the conditional [`Pod`] contract from a struct's field types.
//!
//! Bounds apply to complete field types so marker parameters do not acquire
//! unnecessary [`Pod`] bounds. Recursive validation through [`Pod::ASSERT_LAYOUT`]
//! establishes field validity and excludes padding and unsupported target
//! layouts. Assertions stay in the associated constant so generic structs are
//! checked for the instantiation being stored.
//!
//! [`Pod`]: bento_core::Pod
//! [`Pod::ASSERT_LAYOUT`]: bento_core::Pod::ASSERT_LAYOUT

use proc_macro2::TokenStream;
use quote::{quote, quote_spanned};
use syn::{Data, DeriveInput, LitInt, Path, parse_quote, parse_quote_spanned, spanned::Spanned};

use crate::path_resolution::BentoCorePath;

/// Resolves support items, honoring an explicit path for facade re-exports.
pub fn core_path(input: &DeriveInput) -> syn::Result<BentoCorePath> {
    crate_override(input)?.map_or_else(BentoCorePath::resolve, |path| Ok(path.into()))
}

fn crate_override(input: &DeriveInput) -> syn::Result<Option<Path>> {
    let mut path = None;
    for attribute in input
        .attrs
        .iter()
        .filter(|attr| attr.path().is_ident("pod"))
    {
        attribute.parse_nested_meta(|meta| {
            if !meta.path.is_ident("crate") {
                return Err(meta.error("expected `crate = path`"));
            }
            if path.is_some() {
                return Err(meta.error("duplicate Pod crate path"));
            }
            path = Some(meta.value()?.parse()?);
            Ok(())
        })?;
    }
    Ok(path)
}

/// Validates a struct's representation and emits its conditional implementation.
pub fn derive(input: DeriveInput, core: BentoCorePath) -> syn::Result<TokenStream> {
    let Data::Struct(data) = &input.data else {
        return Err(syn::Error::new_spanned(&input, "Pod requires a struct"));
    };
    for field in &data.fields {
        for attribute in &field.attrs {
            if attribute.path().is_ident("pod") {
                return Err(syn::Error::new_spanned(
                    attribute,
                    "Pod attributes are supported only on the struct",
                ));
            }
        }
    }
    let mut representation = false;
    for attribute in &input.attrs {
        if attribute.path().is_ident("repr") {
            attribute.parse_nested_meta(|meta| {
                if meta.path.is_ident("C") || meta.path.is_ident("transparent") {
                    representation = true;
                    Ok(())
                } else if meta.path.is_ident("align") {
                    let content;
                    syn::parenthesized!(content in meta.input);
                    content.parse::<LitInt>()?;
                    if !content.is_empty() {
                        return Err(content.error("expected one alignment"));
                    }
                    Ok(())
                } else {
                    Err(meta.error(
                        "Pod supports only repr(C), repr(transparent), and repr(align(...))",
                    ))
                }
            })?;
        }
    }
    if !representation {
        return Err(syn::Error::new_spanned(
            &input.ident,
            "Pod requires repr(C) or repr(transparent)",
        ));
    }

    let name = &input.ident;
    let fields: Vec<_> = data.fields.iter().map(|field| &field.ty).collect();
    let mut generics = input.generics.clone();
    let predicates = &mut generics.make_where_clause().predicates;
    predicates.push(parse_quote!(Self: ::core::marker::Copy + ::core::marker::Sync + 'static));
    for ty in &fields {
        predicates.push(parse_quote_spanned!(ty.span()=> #ty: #core::Pod));
    }
    let (impl_generics, ty_generics, where_clause) = generics.split_for_impl();
    let field_assertions = fields
        .iter()
        .map(|ty| quote_spanned!(ty.span()=> let () = <#ty as #core::Pod>::ASSERT_LAYOUT;));

    // The representation fixes field order, and each field implements `Pod`.
    // Recursive validation establishes field validity and target layout. Equality
    // with the sum of field sizes excludes both interior and trailing padding.
    // Metadata and its inherent check belong to the actual trait, so callers
    // cannot replace safety-critical helpers through the chosen support path.
    Ok(quote! {
        #[automatically_derived]
        unsafe impl #impl_generics #core::Pod for #name #ty_generics #where_clause {
            const ASSERT_LAYOUT: () = {
                #(#field_assertions)*
                <Self as #core::Pod>::__LAYOUT.assert_record(
                    &[#(<#fields as #core::Pod>::__LAYOUT),*]
                );
            };
        }
    })
}

#[cfg(test)]
mod tests;
