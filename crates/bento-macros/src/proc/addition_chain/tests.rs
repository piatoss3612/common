use super::*;
use syn::parse_quote;

#[rustfmt::skip]
#[test]
fn expression_parser_handles_commas_and_all_integer_radices() {
    for tokens in [
        quote!(f::<A, B>(a, b), 181),
        quote!((|a, b| a + b)(1, 2), 0xb5,),
        quote!({ let x = 1; x }, 0o265),
        quote!(x, 0b1011_0101),
    ] {
        let input: Input = syn::parse2(tokens).unwrap();
        assert_eq!(limbs_from_decimal(input.scalar.base10_digits()), [181]);
        syn::parse2::<Expr>(evaluate(input, BentoCorePath::default()).unwrap()).unwrap();
    }
}

#[test]
fn rejects_unsupported_scalar_syntax() {
    for tokens in [
        quote!(x),
        quote!(, 2),
        quote!(x, -1),
        quote!(x, SCALAR),
        quote!(x, 1 + 2),
        quote!(x, "12"),
        quote!(x, 2, extra),
    ] {
        assert!(syn::parse2::<Input>(tokens.clone()).is_err(), "{tokens}");
    }
    for tokens in [quote!(x, 0), quote!(x, 0x00)] {
        let error = evaluate(syn::parse2(tokens).unwrap(), BentoCorePath::default()).unwrap_err();
        assert_eq!(
            error.to_string(),
            "addition_chain! scalar must be nonzero; the trait has no identity operation"
        );
    }
    let error = evaluate(parse_quote!(x, 1u64), BentoCorePath::default()).unwrap_err();
    assert_eq!(
        error.to_string(),
        "addition_chain! scalar must be an unsuffixed integer literal"
    );
}

#[test]
fn decoding_is_not_limited_to_target_integer_sizes() {
    for (digits, limbs) in [
        ("0", vec![]),
        ("18446744073709551615", vec![u64::MAX]),
        ("18446744073709551616", vec![0, 1]),
        ("18446744073709551621", vec![5, 1]),
        ("340282366920938463463374607431768211456", vec![0, 0, 1]),
    ] {
        assert_eq!(limbs_from_decimal(digits), limbs);
    }
}

#[rustfmt::skip]
#[test]
fn expansion_unrolls_a_windowed_chain_with_qualified_calls() {
    // This snapshot intentionally fixes the tie-breaking and code-generation policy.
    // 181 uses ten operations, including the odd table, versus eleven for binary.
    let expansion = evaluate(parse_quote!(f(y), 0xb5), BentoCorePath::default()).unwrap();
    let expected = quote! {{
        let __bento_odd_0 = (f(y));
        let __bento_doubled = ::bento_core::addchain::AdditionChain::double(&__bento_odd_0);
        let __bento_odd_1 = ::bento_core::addchain::AdditionChain::add(&__bento_odd_0, &__bento_doubled);
        let __bento_odd_2 = ::bento_core::addchain::AdditionChain::add(&__bento_odd_1, &__bento_doubled);
        let mut __bento_accumulator = {
            fn __bento_clone<T: ::bento_core::addchain::AdditionChain>(value: &T) -> T {
                ::core::clone::Clone::clone(value)
            }
            __bento_clone(&__bento_odd_2)
        };
        __bento_accumulator = ::bento_core::addchain::AdditionChain::double(&__bento_accumulator);
        __bento_accumulator = ::bento_core::addchain::AdditionChain::double(&__bento_accumulator);
        __bento_accumulator = ::bento_core::addchain::AdditionChain::double(&__bento_accumulator);
        __bento_accumulator = ::bento_core::addchain::AdditionChain::add(&__bento_accumulator, &__bento_odd_2);
        __bento_accumulator = ::bento_core::addchain::AdditionChain::double(&__bento_accumulator);
        __bento_accumulator = ::bento_core::addchain::AdditionChain::double(&__bento_accumulator);
        __bento_accumulator = ::bento_core::addchain::AdditionChain::add(&__bento_accumulator, &__bento_odd_0);
        __bento_accumulator
    }};
    assert_eq!(expansion.to_string(), expected.to_string());
    syn::parse2::<Expr>(expansion).unwrap();
}
