use quote::{ToTokens, quote};
use syn::parse_quote;

use super::*;

#[rustfmt::skip]
#[test]
fn rejects_unsupported_items_and_representations() {
    for (input, expected) in [
        (
            quote!(struct Record(u32);),
            "Pod requires repr(C) or repr(transparent)",
        ),
        (
            quote!(#[repr(align(8))] struct Record(u64);),
            "Pod requires repr(C) or repr(transparent)",
        ),
        (quote!(#[repr(C)] enum Record { A, B }), "Pod requires a struct"),
        (quote!(#[repr(C)] union Record { a: u32, b: u32 }), "Pod requires a struct"),
        (quote!(#[repr(C, packed)] struct Record(u32);), "Pod supports only repr(C)"),
        (quote!(#[repr(C, packed(2))] struct Record(u32);), "Pod supports only repr(C)"),
        (quote!(#[repr(u32)] struct Record(u32);), "Pod supports only repr(C)"),
        (quote!(#[repr(C, align(8, 16))] struct Record(u64);), "expected one alignment"),
        (quote!(#[repr(C, align())] struct Record(u64);), "expected integer literal"),
    ] {
        let input = syn::parse2(input).unwrap();
        let error = derive(input, BentoCorePath::default()).unwrap_err();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[rustfmt::skip]
#[test]
fn parses_explicit_paths_and_rejects_invalid_options() {
    let input = parse_quote! {
        #[pod(crate = crate::support)]
        #[repr(C)]
        struct Record(u32);
    };
    assert_eq!(
        core_path(&input).unwrap().to_token_stream().to_string(),
        quote!(crate::support).to_string(),
    );
    for (attributes, expected) in [
        (quote!(#[pod(unknown)]), "expected `crate = path`"),
        (
            quote!(#[pod(crate = a, crate = b)]),
            "duplicate Pod crate path",
        ),
        (
            quote!(#[pod(crate = a)] #[pod(crate = b)]),
            "duplicate Pod crate path",
        ),
        (quote!(#[pod(crate = "bento")]), "expected identifier"),
        (quote!(#[pod(crate)]), "expected `=`"),
    ] {
        let input = syn::parse2(quote!(#attributes #[repr(C)] struct Record(u32);)).unwrap();
        let error = crate_override(&input).err().unwrap();
        assert!(error.to_string().contains(expected), "{error}");
    }
}

#[rustfmt::skip]
#[test]
fn expansion_preserves_generics_and_bounds_complete_field_types() {
    let core: Path = parse_quote!(::renamed::support);
    let output = derive(
        parse_quote! {
            #[repr(C)]
            struct Block<T: Copy, M: ?Sized, const N: usize = 4>
            where
                T: Sync,
            {
                values: [T; N],
                marker: ::core::marker::PhantomData<M>,
            }
        },
        core.into(),
    )
    .unwrap();

    // Interpolated field types leave adjacent closing `>` tokens separate.
    let expected = quote! {
        #[automatically_derived]
        unsafe impl<T: Copy, M: ?Sized, const N: usize> ::renamed::support::Pod for Block<T, M, N>
        where
            T: Sync,
            Self: ::core::marker::Copy + ::core::marker::Sync + 'static,
            [T; N]: ::renamed::support::Pod,
            ::core::marker::PhantomData<M>: ::renamed::support::Pod
        {
            const ASSERT_LAYOUT: () = {
                let () = <::core::primitive::u8 as ::renamed::support::Pod>::ASSERT_LAYOUT;
                let () = <[T; N] as ::renamed::support::Pod>::ASSERT_LAYOUT;
                let () = <::core::marker::PhantomData<M> as ::renamed::support::Pod>::ASSERT_LAYOUT;
                ::core::assert!(
                    ::core::mem::size_of::<Self>() == 0
                        + ::core::mem::size_of::<[T; N]>()
                        + ::core::mem::size_of::<::core::marker::PhantomData<M> >(),
                    "Pod struct must have no padding"
                );
                ::core::assert!(
                    ::core::mem::align_of::<Self>() <= ::renamed::support::MAX_ALIGN,
                    "over-aligned Pod type"
                );
            };
        }
    };
    assert_eq!(output.to_string(), expected.to_string());
    syn::parse2::<syn::ItemImpl>(output).unwrap();
}

#[rustfmt::skip]
#[test]
fn accepts_named_tuple_unit_and_transparent_structs() {
    for input in [
        quote!(#[repr(C)] struct Named { value: u64 }),
        quote!(#[repr(C)] #[repr(align(64))] struct Tuple([u64; 8]);),
        quote!(#[repr(C)] struct Unit;),
        quote!(#[repr(C)] struct Empty {}),
        quote!(#[repr(transparent)] struct Wrapper(u32);),
    ] {
        let output = derive(syn::parse2(input).unwrap(), BentoCorePath::default()).unwrap();
        syn::parse2::<syn::ItemImpl>(output).unwrap();
    }
}
