//! Parsing and expansion of fixed-scalar addition chains.
//!
//! [`Input`] parses a macro invocation, and [`evaluate`] validates its scalar
//! and emits calls to [`bento_core::addchain::AdditionChain`]. The [`schedule`]
//! module chooses the sequence of operations.
//!
//! Scalar decoding and scheduling run on the host, so [`bento_core`] needs
//! neither dependencies nor an allocator to supply the target support trait.

use proc_macro2::{Span, TokenStream};
use quote::{format_ident, quote};
use syn::{
    Error, Expr, LitInt, Result, Token,
    parse::{Parse, ParseStream},
};

use crate::path_resolution::BentoCorePath;

mod schedule;

/// The internal protocol used by the facade wrapper and direct core consumers.
pub(crate) struct Invocation {
    pub(crate) core: BentoCorePath,
    pub(crate) input: Input,
}

impl Parse for Invocation {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        input.parse::<Token![crate]>()?;
        input.parse::<Token![=]>()?;
        let core = input.parse::<syn::Path>()?.into();
        input.parse::<Token![;]>()?;
        Ok(Self {
            core,
            input: input.parse()?,
        })
    }
}

/// A parsed invocation whose scalar still needs semantic validation.
///
/// [`evaluate`] checks that the scalar is nonzero and unsuffixed before
/// generating code.
pub(crate) struct Input {
    value: Expr,
    scalar: LitInt,
}

impl Parse for Input {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        // The full expression parser distinguishes commas inside closures and
        // generic arguments from the macro's argument separator.
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

/// Validates the scalar and expands the invocation into a scaling expression.
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

    // Mixed-site spans isolate caller locals, but constants still participate
    // in pattern resolution. Functions in the helper's enclosing scope shield
    // every generated binding name. The caller expression stays outside that
    // scope as the argument to the generated function.
    let odd = |index| format_ident!("__bento_odd_{index}", span = Span::mixed_site());
    let base = odd(0);
    let doubled = format_ident!("__bento_doubled", span = Span::mixed_site());
    let accumulator = format_ident!("__bento_accumulator", span = Span::mixed_site());
    let chain = format_ident!("__bento_chain", span = Span::mixed_site());
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
    let shields = (0..=schedule.max_odd_index)
        .map(odd)
        .chain([doubled.clone(), accumulator.clone()]);
    Ok(quote! {
        ({
            #(#[allow(dead_code)] fn #shields() {})*
            fn #chain<__BentoValue: #support>(#base: __BentoValue) -> __BentoValue {
                #table
                let #mutability #accumulator = __BentoValue::clone(&#first);
                #(#steps)*
                #accumulator
            }
            #chain
        })(#value)
    })
}

/// Decodes decimal digits into little-endian limbs without a fixed integer size.
///
/// Accepts the normalized digits from [`LitInt::base10_digits`]. Zero is the
/// empty vector; all other values have a nonzero final limb.
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
