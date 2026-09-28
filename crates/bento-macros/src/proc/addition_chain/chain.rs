//! A shared, validated operation graph for planned and supplied chains.

use super::{schedule, *};
use syn::{ExprCall, ExprClosure, Ident, Pat, Stmt, spanned::Spanned};

#[derive(Clone)]
pub(super) enum Operation {
    Double(usize, usize),
    Add(usize, usize),
}

pub(super) struct Chain {
    // Value zero is the input; operation i produces value i + 1.
    operations: Vec<Operation>,
    result: usize,
}

impl Chain {
    pub(super) fn planned(schedule: schedule::Schedule) -> Self {
        let mut chain = Self {
            operations: Vec::new(),
            result: 0,
        };
        let mut odd = vec![0];
        if schedule.max_odd_index != 0 {
            let doubled = chain.push(Operation::Double(0, 1));
            for index in 1..=schedule.max_odd_index {
                let next = chain.push(Operation::Add(odd[index - 1], doubled));
                odd.push(next);
            }
        }
        let mut current = odd[schedule.first];
        let mut doubles = 0;
        for step in schedule.steps {
            match step {
                schedule::Step::Double => doubles += 1,
                schedule::Step::AddOdd(index) => {
                    if doubles != 0 {
                        current = chain.push(Operation::Double(current, doubles));
                        doubles = 0;
                    }
                    current = chain.push(Operation::Add(current, odd[index]));
                }
            }
        }
        if doubles != 0 {
            current = chain.push(Operation::Double(current, doubles));
        }
        chain.result = current;
        chain
    }

    fn push(&mut self, operation: Operation) -> usize {
        self.operations.push(operation);
        self.operations.len()
    }

    pub(super) fn supplied(closure: ExprClosure, scalar: &[u64]) -> Result<Self> {
        if !closure.attrs.is_empty()
            || closure.constness.is_some()
            || closure.lifetimes.is_some()
            || closure.inputs.len() != 1
            || closure.asyncness.is_some()
            || closure.movability.is_some()
            || closure.capture.is_some()
            || !matches!(closure.output, syn::ReturnType::Default)
        {
            return Err(Error::new_spanned(
                &closure,
                "chain requires |input| { ...; result }",
            ));
        }
        let mut names = vec![binding(&closure.inputs[0])?];
        let mut coefficients = vec![vec![1u64]];
        let Expr::Block(block) = *closure.body else {
            return Err(Error::new_spanned(closure.body, "chain requires a block"));
        };
        if !block.attrs.is_empty() || block.label.is_some() {
            return Err(Error::new_spanned(block, "unsupported chain block"));
        }
        let mut chain = Self {
            operations: Vec::new(),
            result: 0,
        };
        let mut statements = block.block.stmts.iter();
        let Some(last) = statements.next_back() else {
            return Err(Error::new_spanned(block, "chain requires a result"));
        };
        for statement in statements {
            let Stmt::Local(local) = statement else {
                return Err(Error::new_spanned(
                    statement,
                    "chain accepts only let bindings",
                ));
            };
            let name = binding(&local.pat)?;
            if names.contains(&name) {
                return Err(Error::new_spanned(name, "chain bindings must be unique"));
            }
            let Some(init) = &local.init else {
                return Err(Error::new_spanned(
                    local,
                    "chain binding requires an operation",
                ));
            };
            if init.diverge.is_some() || !local.attrs.is_empty() {
                return Err(Error::new_spanned(local, "unsupported chain binding"));
            }
            let Expr::Call(ExprCall {
                attrs, func, args, ..
            }) = &*init.expr
            else {
                return Err(Error::new_spanned(
                    init.expr.clone(),
                    "expected double(value, count) or add(lhs, rhs)",
                ));
            };
            if !attrs.is_empty() {
                return Err(Error::new_spanned(
                    &init.expr,
                    "unsupported chain operation attribute",
                ));
            }
            let operation = name_of(func)?;
            if args.len() != 2 {
                return Err(Error::new_spanned(
                    args,
                    "chain operations require two arguments",
                ));
            }
            let lhs = lookup(&args[0], &names)?;
            let (op, coefficient) = if operation == "double" {
                let Expr::Lit(literal) = &args[1] else {
                    return Err(Error::new_spanned(
                        &args[1],
                        "doubling count must be a positive literal",
                    ));
                };
                if !literal.attrs.is_empty() {
                    return Err(Error::new_spanned(
                        literal,
                        "unsupported doubling count attribute",
                    ));
                }
                let syn::Lit::Int(count) = &literal.lit else {
                    return Err(Error::new_spanned(
                        literal,
                        "doubling count must be a positive literal",
                    ));
                };
                let count_value = count.base10_parse::<usize>()?;
                // A positive chain cannot shrink again. Bounding shifts by
                // the target prevents unbounded host allocations from counts.
                if !count.suffix().is_empty() || count_value == 0 || count_value > scalar.len() * 64
                {
                    return Err(Error::new_spanned(
                        count,
                        "doubling count must be positive and no larger than the scalar width",
                    ));
                }
                let mut coefficient = coefficients[lhs].clone();
                for _ in 0..count_value {
                    coefficient = add(&coefficient, &coefficient);
                }
                (Operation::Double(lhs, count_value), coefficient)
            } else if operation == "add" {
                let rhs = lookup(&args[1], &names)?;
                (
                    Operation::Add(lhs, rhs),
                    add(&coefficients[lhs], &coefficients[rhs]),
                )
            } else {
                return Err(Error::new_spanned(operation, "expected double or add"));
            };
            if coefficient.len() > scalar.len()
                || (coefficient.len() == scalar.len()
                    && coefficient.iter().rev().cmp(scalar.iter().rev()).is_gt())
            {
                return Err(Error::new_spanned(
                    local,
                    "chain coefficient exceeds the scalar",
                ));
            }
            names.push(name);
            coefficients.push(coefficient);
            chain.push(op);
        }
        let Stmt::Expr(result, None) = last else {
            return Err(Error::new_spanned(
                last,
                "chain must end with a bound value",
            ));
        };
        chain.result = lookup(result, &names)?;
        if coefficients[chain.result] != scalar {
            return Err(Error::new_spanned(
                result,
                "chain does not compute the requested scalar",
            ));
        }
        Ok(chain)
    }

    pub(super) fn emit(self, core: BentoCorePath, value: Expr, emission: Emission) -> TokenStream {
        let support = quote!(#core::addchain::AdditionChain);
        let name = |i| format_ident!("__bento_value_{i}", span = Span::mixed_site());
        let helper = format_ident!("__bento_chain", span = Span::mixed_site());
        let input = name(0);
        let mut last_use = vec![0; self.operations.len() + 1];
        let mut uses = vec![0; last_use.len()];
        for (i, op) in self.operations.iter().enumerate() {
            let mut record = |j: usize| {
                last_use[j] = i + 1;
                uses[j] += 1;
            };
            match *op {
                Operation::Double(a, _) => record(a),
                Operation::Add(a, b) => {
                    record(a);
                    record(b);
                }
            }
        }
        last_use[self.result] = last_use.len();
        uses[self.result] += 1;
        let mut statements = Vec::new();
        let mut i = 0;
        while i < self.operations.len() {
            let start = i + 1;
            let mut output = start;
            let expression = match self.operations[i] {
                Operation::Add(a, b) => {
                    let (a, b) = (name(a), name(b));
                    quote!(#support::add(&#a, &#b))
                }
                Operation::Double(a, count) => {
                    let a = name(a);
                    // A sole following consumer can fuse a run with its add.
                    if matches!(emission, Emission::Batched)
                        && uses[output] == 1
                        && let Some(Operation::Add(lhs, rhs)) = self.operations.get(i + 1)
                        && *lhs == output
                        && *rhs != output
                    {
                        let b = name(*rhs);
                        i += 1;
                        output += 1;
                        quote!(#support::double_n_add(&#a, #count, &#b))
                    } else if matches!(emission, Emission::Batched) {
                        quote!(#support::double_n(&#a, #count))
                    } else {
                        let acc = format_ident!("__bento_accumulator", span = Span::mixed_site());
                        if matches!(emission, Emission::Unrolled) || count <= 8 {
                            let steps = (1..count).map(|_| quote!(#acc = #support::double(&#acc);));
                            let mutable = (count > 1).then(|| quote!(mut));
                            quote!({ let #mutable #acc = #support::double(&#a); #(#steps)* #acc })
                        } else {
                            quote!({ let mut #acc = #support::double(&#a);
                                for _ in 1..#count { #acc = #support::double(&#acc); } #acc })
                        }
                    }
                }
            };
            let out = name(output);
            statements.push(quote!(let #out = #expression;));
            // Drop values at their last consumer, including values consumed
            // by the first half of a fused operation. The fused temporary was
            // never materialized and must not be named here.
            for (j, last) in last_use.iter().enumerate().take(output + 1) {
                if output != start && j == start {
                    continue;
                }
                if (start..=output).contains(last) || (j == output && uses[j] == 0) {
                    let binding = name(j);
                    statements.push(quote!(::core::mem::drop(#binding);));
                }
            }
            i += 1;
        }
        let result = name(self.result);
        // Mixed-site spans isolate caller locals, but constants still participate
        // in pattern resolution. Functions in the helper's enclosing scope shield
        // every generated binding name. The caller expression stays outside that
        // scope as the argument to the generated function.
        let shields = (0..=self.operations.len()).map(name).chain([format_ident!(
            "__bento_accumulator",
            span = Span::mixed_site()
        )]);
        quote! {
            ({
                #(#[allow(dead_code)] fn #shields() {})*
                fn #helper<__BentoValue: #support>(#input: __BentoValue) -> __BentoValue {
                    #(#statements)*
                    #result
                }
                #helper
            })(#value)
        }
    }
}

fn binding(pattern: &Pat) -> Result<Ident> {
    if let Pat::Ident(name) = pattern
        && name.attrs.is_empty()
        && name.by_ref.is_none()
        && name.mutability.is_none()
        && name.subpat.is_none()
    {
        return Ok(name.ident.clone());
    }
    Err(Error::new_spanned(
        pattern,
        "chain requires a plain binding name",
    ))
}

fn name_of(expr: &Expr) -> Result<Ident> {
    if let Expr::Path(path) = expr
        && path.attrs.is_empty()
        && path.qself.is_none()
        && let Some(name) = path.path.get_ident()
    {
        return Ok(name.clone());
    }
    Err(Error::new(expr.span(), "expected a chain name"))
}

fn lookup(expr: &Expr, names: &[Ident]) -> Result<usize> {
    let name = name_of(expr)?;
    names
        .iter()
        .position(|candidate| *candidate == name)
        .ok_or_else(|| Error::new_spanned(name, "unknown or forward chain reference"))
}

fn add(a: &[u64], b: &[u64]) -> Vec<u64> {
    let mut out = Vec::new();
    let mut carry = 0u128;
    for i in 0..a.len().max(b.len()) {
        carry +=
            u128::from(a.get(i).copied().unwrap_or(0)) + u128::from(b.get(i).copied().unwrap_or(0));
        out.push(carry as u64);
        carry >>= 64;
    }
    if carry != 0 {
        out.push(carry as u64);
    }
    out
}
