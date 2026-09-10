# bento-macros

Implementation crate for procedural macros exposed by `bento`. Compiler entry
points delegate to ordinary Rust expansion functions, with shared error handling
and dependency-path resolution.

- `src/lib.rs` declares entry points and dispatches to implementations.
- `src/helpers.rs` converts `syn::Error` into spanned compiler diagnostics.
- `src/path_resolution.rs` resolves `bento-core` items through a direct dependency
  or the `bento` facade, including renamed dependencies.
- `src/derive/<name>.rs` implements each derive through a `derive` function.
- `src/proc/<name>.rs` implements each function-like macro through an `evaluate`
  function and, when needed, an `Input` parser.

The crate currently provides infrastructure; it does not export a macro yet.

See [the crate development guide](../README.md) for workspace dependency aliases,
the crate layers, macro path resolution, and publication conventions.

## Adding a macro

Keep `proc_macro::TokenStream` at the compiler boundary. Parse with
`syn::parse_macro_input!`, resolve only the paths the expansion needs, and pass
them into the implementation. For example, once `proc::example` exists:

```rust,ignore
use helpers::macro_body;
use proc_macro::TokenStream;
use syn::parse_macro_input;

// Documentation lives on the facade's re-export.
#[allow(missing_docs)]
#[proc_macro]
pub fn example(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as proc::example::Input);
    macro_body(|| {
        let bento_core_path = path_resolution::BentoCorePath::resolve()?;
        proc::example::evaluate(input, bento_core_path)
    })
}
```

The implementation returns `syn::Result<proc_macro2::TokenStream>` and uses
`quote!` for output. Interpolate the supplied `BentoCorePath` for library items
and use absolute `::core` paths for standard types so generated code works in
`no_std` callers. Validate unsupported input with `syn::Error::new` or
`syn::Error::new_spanned`; reserve panics for internal invariants. Put reusable
arithmetic in `bento-core`, separate from parsing and code generation.

Derives follow the same pattern with `syn::DeriveInput`, an implementation in
`derive`, and `#[automatically_derived]` on generated implementations. Add
attribute parsing or syntax substitution helpers when a macro needs them.
Remove the relevant temporary `dead_code` expectations when the entry point
starts using the scaffold.

Document and explicitly re-export each macro from `bento`. Use the existing
facade dev-dependency for compilation tests or doctests that invoke that public
path. Expansion unit tests can use `syn::parse_quote!` and
`BentoCorePath::default()` without invoking Cargo or the compiler's macro API.
Test supported syntax, rejected input, and generated behavior; include renamed
dependencies when an expansion references library items.
