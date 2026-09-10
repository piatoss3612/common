//! Parse a fixed scalar and emit calls to the addition-chain support trait.
//!
//! Scalar decoding and scheduling are host-only implementation details.
//! Keeping them here leaves `bento-core` with just the target support trait,
//! requiring neither dependencies nor an allocator for addition chains.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{
    Error, Expr, LitInt, Result, Token,
    parse::{Parse, ParseStream},
};

use crate::path_resolution::BentoCorePath;

mod schedule;

/// A value expression and a nonzero, unsuffixed scalar literal.
///
/// Use syn's full expression parser so commas inside closures and generic
/// arguments do not need extra parentheses to distinguish them from the
/// macro's argument separator.
pub(crate) struct Input {
    value: Expr,
    scalar: LitInt,
}

impl Parse for Input {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let value = input.parse()?;
        input.parse::<Token![,]>()?;
        if input.peek(Token![-]) {
            return Err(input.error("addition_chain! scalar must be positive"));
        }
        let scalar = input.parse()?;
        input.parse::<Option<Token![,]>>()?;
        if !input.is_empty() {
            return Err(input.error("unexpected tokens after the scalar"));
        }
        Ok(Self { value, scalar })
    }
}

/// Validate the scalar, plan on the host, and emit a single target expression.
pub(crate) fn evaluate(input: Input, core: BentoCorePath) -> Result<TokenStream> {
    let Input { value, scalar } = input;
    if !scalar.suffix().is_empty() {
        return Err(Error::new(
            scalar.span(),
            "addition_chain! scalar must be an unsuffixed integer literal",
        ));
    }
    let limbs = limbs_from_decimal(scalar.base10_digits());
    let Some(schedule) = schedule::plan(&limbs) else {
        return Err(Error::new(
            scalar.span(),
            "addition_chain! scalar must be nonzero; the trait has no identity operation",
        ));
    };

    // Mixed-site spans keep generated bindings separate from caller bindings.
    // UFCS also avoids requiring trait imports or selecting inherent methods.
    let odd = |index| format_ident!("__bento_odd_{index}", span = Span::mixed_site());
    let base = odd(0);
    let doubled = format_ident!("__bento_doubled", span = Span::mixed_site());
    let accumulator = format_ident!("__bento_accumulator", span = Span::mixed_site());
    let clone = format_ident!("__bento_clone", span = Span::mixed_site());
    let support = quote!(#core::addchain::AdditionChain);
    let table = if schedule.max_odd_index == 0 {
        quote!()
    } else {
        let entries = (1..=schedule.max_odd_index).map(|index| {
            let current = odd(index);
            let previous = odd(index - 1);
            quote!(let #current = #support::add(&#previous, &#doubled);)
        });
        quote! {
            let #doubled = #support::double(&#base);
            #(#entries)*
        }
    };
    let first = odd(schedule.first);
    // Reassignment drops superseded accumulators after each operation. Chains
    // completed by table preparation alone need no mutable binding.
    let mutability = (!schedule.steps.is_empty()).then(|| quote!(mut));
    let steps = schedule.steps.iter().map(|step| match step {
        schedule::Step::Double => {
            quote!(#accumulator = #support::double(&#accumulator);)
        }
        schedule::Step::AddOdd(index) => {
            let entry = odd(*index);
            quote!(#accumulator = #support::add(&#accumulator, &#entry);)
        }
    });
    Ok(quote! {{
        let #base = (#value);
        #table
        let #mutability #accumulator = {
            // Enforce the trait even for scalar one. Unlike local bindings,
            // item names are not hygienic at mixed site; keep this helper out
            // of the caller expression's scope.
            fn #clone<T: #support>(value: &T) -> T {
                ::core::clone::Clone::clone(value)
            }
            #clone(&#first)
        };
        #(#steps)*
        #accumulator
    }})
}

/// Decode syn's normalized decimal digits without imposing a target word size.
/// Zero is the empty vector; all other values have a nonzero final limb.
fn limbs_from_decimal(digits: &str) -> Vec<u64> {
    let mut limbs = Vec::new();
    for digit in digits.bytes() {
        debug_assert!(digit.is_ascii_digit());
        let mut carry = u128::from(digit - b'0');
        for limb in &mut limbs {
            let value = u128::from(*limb) * 10 + carry;
            *limb = value as u64;
            carry = value >> 64;
        }
        if carry != 0 {
            limbs.push(carry as u64);
        }
    }
    limbs
}

#[cfg(test)]
mod tests;
