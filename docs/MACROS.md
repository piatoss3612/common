# Macros

[`bento-macros`](../crates/bento-macros/src/lib.rs) implements macros exposed
through the [`bento`](../crates/bento/src/lib.rs) facade. Parsing and expansion
run on the build host; emitted code compiles for the caller's target. See the
[crate guide](CRATES.md) for dependency boundaries.

The facade also exports declarative wrappers for compile-time reference
arithmetic. These are defined in `bento-core` and evaluated by the Rust compiler.

## Authoring conventions

For procedural macros, keep compiler entry points thin. Parse and expand with
`syn::Result` and `proc_macro2::TokenStream` so most behavior can be tested outside
the procedural macro context. Report invalid input with `syn::Error` at the
relevant span; reserve panics for internal invariants. Reject unsupported helper
attributes, including misplaced ones, instead of silently ignoring them.

Document and deliberately export public macros from the facade. Generated
code must support the target's `no_std` context. Use qualified paths, but do not
assume that `::core` or caller-visible helper names authenticate safety checks.
For generated unsafe implementations, anchor the proof in the actual support
trait and code whose meaning the consumer cannot substitute. Layout and other
target properties must be checked in target-compiled code.

Generated names must survive caller locals, constants, imports, and repeated
invocations. Preserve the documented evaluation count and the caller's control
flow when wrapping expressions. Exercise these properties in real consumers;
see the [testing guide](TESTING.md).

## Support paths

Prefer a facade wrapper that forwards `$crate` to the implementation. The
public `addition_chain!(value, scalar)` macro uses this approach, so aliases,
build dependencies, and indirect re-exports retain the correct support path.
Direct users of the implementation crate must supply its internal protocol:
`addition_chain!(crate = support_path; value, scalar)`.

Derives cannot receive `$crate` from a declarative wrapper. The `Pod` derive
looks up the facade dependency's Cargo name for ordinary callers. Use
`#[pod(crate = path)]` on the struct for direct-core consumers, build-only
dependencies, indirect re-exports, or ambiguous dependency arrangements. That
path must expose the intended `Pod` trait. Manifest discovery identifies names;
it does not determine whether an optional, target, or development dependency is
active. Explicit paths must name support available in that compilation context.

## Storage derivation

The `Pod` derive validates representation and field bounds, then emits recursive
layout assertions. Measurements and record checks belong to
[core-owned metadata](../crates/bento-core/src/pod/layout.rs) obtained through
the actual `Pod` trait, so replacing a caller's `core` path or
re-exporting the trait with counterfeit helpers cannot bypass validation.
Metadata is not a validation witness: storage consumers must still evaluate
`Pod::ASSERT_LAYOUT`, including for empty values. See the [POD guide](POD.md)
and the [trait contract](../crates/bento-core/src/pod/mod.rs).

## Reference arithmetic

The [`const_arithmetic`](../crates/bento/src/const_arithmetic/mod.rs) facade
exports arithmetic macros and integer storage aliases explicitly. Calls such
as `m255::mul!(&MODULUS, &A, &B)` expand to the corresponding core function
inside `const { ... }`. The compiler rejects runtime arguments and runs the
operation's input checks even when the macro appears in a runtime expression.

The [API documentation](../crates/bento/src/const_arithmetic/mod.rs) explains
which constant expressions can be passed through the wrappers and how to
compose derivations. The [Montgomery module](../crates/bento/src/const_arithmetic/m255.rs)
also demonstrates inferred and explicit table lengths.

The wrappers are defined in core so `$crate` identifies the implementation
through dependency aliases, build dependencies, and indirect macro re-exports.
Each wrapper accepts only its operation's arguments; it provides no way to
select another core item or return an arithmetic context. Runtime arithmetic
belongs in Udon or another consumer.
