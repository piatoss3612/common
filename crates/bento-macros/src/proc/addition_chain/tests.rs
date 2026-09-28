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

#[test]
fn emission_modes_preserve_qualified_calls_and_compact_long_runs() {
    for mode in [quote!(compact), quote!(unrolled), quote!(batched)] {
        let input = syn::parse2(quote!(f(y), 0x1_0000000000000001, emission = #mode)).unwrap();
        let expansion = evaluate(input, BentoCorePath::default()).unwrap();
        let text = expansion.to_string();
        assert!(text.contains(":: bento_core :: addchain :: AdditionChain"));
        if mode.to_string() == "compact" {
            assert!(text.contains("for _ in"));
        }
        if mode.to_string() == "batched" {
            assert!(text.contains("double_n_add"));
        }
        syn::parse2::<Expr>(expansion).unwrap();
    }
}

#[test]
fn supplied_chains_are_replayed_exactly_before_emission() {
    for mode in [quote!(compact), quote!(unrolled), quote!(batched)] {
        let input = syn::parse2(quote!(x, 9, chain = |a| {
            let b = double(a, 3);
            let c = add(b, a);
            c
        }, emission = #mode))
        .unwrap();
        syn::parse2::<Expr>(evaluate(input, BentoCorePath::default()).unwrap()).unwrap();
    }
    for body in [
        quote!(|a| {
            let b = double(a, 2);
            b
        }), // Wrong exponent.
        quote!(|a| {
            let b = add(c, a);
            let c = double(a, 3);
            b
        }),
        quote!(|a| {
            let a = double(a, 3);
            a
        }),
        quote!(|a| {
            let b = double(a, 0);
            b
        }),
        quote!(|a| {
            let b = double(a, 18446744073709551615);
            b
        }),
        quote!(|a| {
            let b = double(a, 4);
            b
        }), // Overshoots.
        quote!(|a| {
            let b = other(a, 3);
            b
        }),
        quote!(|a| {
            let mut b = double(a, 3);
            b
        }),
        quote!(|a| {
            let b = add(a, a);
            b;
        }),
    ] {
        let input = syn::parse2(quote!(x, 9, chain = #body)).unwrap();
        assert!(evaluate(input, BentoCorePath::default()).is_err());
    }
}

#[test]
fn supplied_chain_coefficients_cross_integer_word_boundaries() {
    use num_bigint::BigUint;
    for bits in [63usize, 64, 127, 128, 255, 256, 511] {
        let count = LitInt::new(&bits.to_string(), Span::call_site());
        let power = BigUint::from(1u8) << bits;
        for (scalar, valid) in [(&power + 1u8, true), (power, false)] {
            let scalar = LitInt::new(&scalar.to_str_radix(10), Span::call_site());
            let input = syn::parse2(quote!(x, #scalar, chain = |a| {
                let b = double(a, #count);
                let c = add(b, a);
                c
            }))
            .unwrap();
            assert_eq!(evaluate(input, BentoCorePath::default()).is_ok(), valid);
        }
    }
}

#[test]
fn derived_exponents_require_the_exact_factorization() {
    let input: Input = syn::parse2(quote!(
        x,
        tonelli_shanks(
            "0x0000000000000000000000000000000000000000000000000000000000000061",
            5
        )
    ))
    .unwrap();
    assert_eq!(input.scalar.base10_digits(), "1");
    for (modulus, adicity) in [
        ("61", 5),
        (
            "0000000000000000000000000000000000000000000000000000000000000061",
            5,
        ),
        (
            "0x0000000000000000000000000000000000000000000000000000000000000061",
            4,
        ),
        (
            "0x0000000000000000000000000000000000000000000000000000000000000062",
            1,
        ),
        (
            "0x0000000000000000000000000000000000000000000000000000000000000001",
            0,
        ),
    ] {
        let adicity = LitInt::new(&adicity.to_string(), Span::call_site());
        assert!(syn::parse2::<Input>(quote!(x, tonelli_shanks(#modulus, #adicity))).is_err());
    }
}

#[test]
fn internal_invocations_require_an_explicit_support_path() {
    let invocation: Invocation = syn::parse2(quote!(crate = crate::support; value, 3)).unwrap();
    let core = invocation.core;
    assert_eq!(
        quote!(#core).to_string(),
        quote!(crate::support).to_string()
    );
    assert!(syn::parse2::<Invocation>(quote!(value, 3)).is_err());
}

#[test]
fn wide_decimal_decoding_matches_an_independent_integer() {
    use num_bigint::BigUint;
    for bits in [1usize, 63, 64, 65, 127, 128, 129, 255, 256, 257, 511, 1024] {
        let power = BigUint::from(1u8) << bits;
        for number in [&power - 1u8, power.clone(), &power + 1u8] {
            assert_eq!(
                limbs_from_decimal(&number.to_str_radix(10)),
                number.to_u64_digits()
            );
        }
    }
}
