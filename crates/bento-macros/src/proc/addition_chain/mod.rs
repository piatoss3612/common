//! Parsing and expansion of fixed-scalar addition chains.
//!
//! [`Invocation`] parses the support path and arguments, and [`evaluate`]
//! validates the scalar and emits calls to
//! [`bento_core::addchain::AdditionChain`]. The [`schedule`] module chooses the
//! sequence of operations.
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

mod chain;
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
    chain: Option<syn::ExprClosure>,
    emission: Emission,
}

#[derive(Clone, Copy)]
enum Emission {
    Compact,
    Unrolled,
    Batched,
}

impl Parse for Input {
    fn parse(input: ParseStream<'_>) -> Result<Self> {
        let value = input.parse()?;
        input.parse::<Token![,]>()?;
        if input.peek(Token![-]) {
            return Err(input.error("addition_chain! scalar must be positive"));
        }
        let scalar = if input.peek(syn::Ident) {
            derived_scalar(input)?
        } else {
            input.parse()?
        };
        let mut chain = None;
        let mut emission = None;
        while !input.is_empty() {
            input.parse::<Token![,]>()?;
            if input.is_empty() {
                break;
            }
            let option: syn::Ident = input.parse()?;
            input.parse::<Token![=]>()?;
            if option == "chain" && chain.is_none() {
                chain = Some(input.parse()?);
            } else if option == "emission" && emission.is_none() {
                let mode: syn::Ident = input.parse()?;
                emission = Some(match mode.to_string().as_str() {
                    "compact" => Emission::Compact,
                    "unrolled" => Emission::Unrolled,
                    "batched" => Emission::Batched,
                    _ => {
                        return Err(Error::new_spanned(
                            mode,
                            "expected compact, unrolled, or batched",
                        ));
                    }
                });
            } else {
                return Err(Error::new_spanned(
                    option,
                    "unknown or duplicate addition-chain option",
                ));
            }
        }
        Ok(Self {
            value,
            scalar,
            chain,
            emission: emission.unwrap_or(Emission::Compact),
        })
    }
}

// This deliberately narrow derived form does not evaluate arbitrary Rust.
fn derived_scalar(input: ParseStream<'_>) -> Result<LitInt> {
    use bento_core::const_arithmetic::u256;
    let name: syn::Ident = input.parse()?;
    if name != "tonelli_shanks" {
        return Err(Error::new_spanned(
            name,
            "expected an integer literal or tonelli_shanks(modulus, two_adicity)",
        ));
    }
    let args;
    syn::parenthesized!(args in input);
    let modulus: syn::LitStr = args.parse()?;
    args.parse::<Token![,]>()?;
    let two_adicity: LitInt = args.parse()?;
    if !args.is_empty() || !two_adicity.suffix().is_empty() {
        return Err(args.error("expected modulus string and unsuffixed two-adicity"));
    }
    let literal = modulus.value();
    let digits = literal.strip_prefix("0x").unwrap_or(&literal);
    if !literal.starts_with("0x")
        || digits.len() != 64
        || !digits.bytes().all(|c| c.is_ascii_hexdigit())
    {
        return Err(Error::new_spanned(
            modulus,
            "modulus must contain 0x followed by exactly 64 hexadecimal digits",
        ));
    }
    let p = u256::from_hex(&literal);
    let s: u32 = two_adicity.base10_parse()?;
    if p[0] & 1 == 0 || !u256::ge(&p, &[3, 0, 0, 0]) || !(1..=255).contains(&s) {
        return Err(Error::new_spanned(
            name,
            "requires an odd modulus above two and two-adicity in 1..=255",
        ));
    }
    let pm1 = u256::sub_u64(&p, 1);
    let trailing = pm1.iter().take_while(|limb| **limb == 0).count() as u32 * 64
        + pm1
            .iter()
            .find(|limb| **limb != 0)
            .unwrap()
            .trailing_zeros();
    if trailing != s {
        return Err(Error::new_spanned(
            two_adicity,
            "two-adicity must equal the valuation of modulus minus one",
        ));
    }
    let e = u256::tonelli_shanks_exponent(&p, s);
    Ok(LitInt::new(
        &format!("0x{:016x}{:016x}{:016x}{:016x}", e[3], e[2], e[1], e[0]),
        modulus.span(),
    ))
}

/// Validates the scalar and expands the invocation into a scaling expression.
pub(crate) fn evaluate(input: Input, core: BentoCorePath) -> Result<TokenStream> {
    let Input {
        value,
        scalar,
        chain,
        emission,
    } = input;
    if !scalar.suffix().is_empty() {
        return Err(Error::new(
            scalar.span(),
            "addition_chain! scalar must be an unsuffixed integer literal",
        ));
    }
    let limbs = limbs_from_decimal(scalar.base10_digits());
    if limbs.is_empty() {
        return Err(Error::new(
            scalar.span(),
            "addition_chain! scalar must be nonzero; the trait has no identity operation",
        ));
    };
    let chain = match chain {
        Some(chain) => chain::Chain::supplied(chain, &limbs)?,
        None => chain::Chain::planned(schedule::plan(&limbs).unwrap()),
    };
    Ok(chain.emit(core, value, emission))
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
