# Crate development

The workspace uses short dependency names in Rust code and `zakura-` package
names for Cargo and the registry. Shared package metadata and internal dependency
declarations live in the root [`Cargo.toml`](../Cargo.toml); each member explicitly
inherits what it uses.

## Shared package metadata

The root `[workspace.package]` table defines `version`, `authors`, `edition`,
`rust-version` (MSRV), `license`, and `publish` for all four crates. Each member
inherits these fields in its `[package]` table:

```toml
[package]
name = "zakura-udon"
description = """
Optimized Pasta field and curve arithmetic
"""
version.workspace = true
authors.workspace = true
edition.workspace = true
rust-version.workspace = true
license.workspace = true
publish.workspace = true
```

Edit shared values in the root manifest; names, descriptions, and library target
settings stay in the individual crates. Every member currently shares one
version, so changing `workspace.package.version` updates all four packages.
The package version and dependency version requirements are separate fields;
when registry requirements are added for publication, maintain them in
`[workspace.dependencies]` alongside the shared package version.

Cargo materializes inherited package metadata when packaging, so published
manifests carry these values without requiring the original workspace. See
Cargo's [package inheritance reference](https://doc.rust-lang.org/cargo/reference/workspaces.html#the-package-table).

## Names and workspace dependencies

| Directory | Cargo package | Dependency key | Rust dependency path |
| --- | --- | --- | --- |
| `bento/` | `zakura-bento` | `bento` | `bento::` |
| `bento-core/` | `zakura-bento-core` | `bento-core` | `bento_core::` |
| `bento-macros/` | `zakura-bento-macros` | `bento-macros` | `bento_macros::` |
| `udon/` | `zakura-udon` | `udon` | `udon::` |

The root manifest defines each alias, package identity, and path once:

```toml
[workspace.dependencies]
bento = { package = "zakura-bento", path = "crates/bento" }
bento-core = { package = "zakura-bento-core", path = "crates/bento-core" }
bento-macros = { package = "zakura-bento-macros", path = "crates/bento-macros" }
udon = { package = "zakura-udon", path = "crates/udon" }
```

A member such as `udon` opts in with:

```toml
[dependencies]
bento.workspace = true
```

Cargo's `package` field selects the package; the dependency key supplies the
name available to that member's Rust code. Hyphens become underscores in Rust.
Workspace dependency paths are relative to the workspace root. Inheritance also
works in `[dev-dependencies]` and `[build-dependencies]`.

The table does not add dependencies to every member or make dependencies
transitively available. In particular, `bento-core` inherits no dependencies. Add new
internal aliases to the root table, then inherit them only where needed.
Third-party dependencies currently remain in the members that use them.

Package declarations, `Cargo.lock`, registry URLs, and macro package lookups
retain the `zakura-` names. For package-oriented commands, use those names, for
example `cargo test -p zakura-bento-macros`.

This is dependency renaming, not a `[lib] name` override. Library target names
still default to the package name with underscores, such as `zakura_bento`.
Inside a library, refer to its own items through `crate::`. Integration tests
and doctests are separate crates; when importing the library being tested under
a short name, they can write `use zakura_bento as bento;`. A public doctest can
hide that setup with `# use zakura_bento as bento;` so the visible example uses
`bento::`. Such an import does not change the dependency names in Cargo's
manifest or the macro resolver's lookup.

See Cargo's references for [dependency renaming](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#renaming-dependencies-in-cargotoml),
[workspace inheritance](https://doc.rust-lang.org/cargo/reference/workspaces.html#the-dependencies-table),
and [library target names](https://doc.rust-lang.org/cargo/reference/cargo-targets.html#the-name-field).

## Crate layers

| Crate | Role | Normal workspace dependencies |
| --- | --- | --- |
| `bento-core` | Shared traits, storage support, and reference arithmetic; `no_std` | None |
| `bento-macros` | Parsing, validation, and code generation on the build host; uses `std` | `bento-core` |
| `bento` | Public facade; `no_std` | `bento-core`, `bento-macros` |
| `udon` | Optimized field and curve arithmetic; `no_std` | `bento` |

These roles describe the intended implementation; the crates are currently
scaffolds. Consumers use the `bento` facade. It re-exports core items at its root,
and each procedural macro will be explicitly re-exported and documented there.
No procedural macros are exported yet. `udon` reaches shared support through
`bento` without declaring a direct dependency on `bento-core`.

`bento-core` is the bottom layer and cannot invoke the macros through the
facade under this dependency structure. Shared arithmetic belongs there so
both ordinary library code and macros can use it. Macro execution happens on
the build host: macro implementation code calls that arithmetic through its own
`bento_core` dependency. The generated Rust code is then compiled in the
caller's crate for the target. These are separate dependency contexts.

`bento-macros` also has a development dependency on `bento` for tests using the
public facade. Cargo permits this development dependency cycle; it does not
introduce a normal dependency from the macro implementation back to the facade.

## Procedural macro structure and paths

Compiler entry points in `bento-macros/src/lib.rs` parse input, resolve the
dependency paths needed by the expansion, and delegate through
`helpers::macro_body`. Expansion functions in `derive/` and `proc/` use
`proc_macro2::TokenStream` and `syn::Result`, keeping them testable outside the
compiler's procedural macro context. Parsing and token generation belong in
the macro crate; reusable arithmetic belongs in core. The
[`bento-macros` guide](bento-macros/README.md) describes adding an entry point,
reporting errors, testing expansions, and exposing the macro through `bento`.

Generated paths must match the caller's dependencies. The
[`BentoCorePath` resolver](bento-macros/src/path_resolution.rs) uses
[`proc-macro-crate`](https://docs.rs/proc-macro-crate/3.5.0/proc_macro_crate/)
to look up stable package identities in the caller's manifest, including
inherited workspace dependencies:

1. Look for a direct dependency on package `zakura-bento-core`.
2. Otherwise, look for package `zakura-bento` and use its root re-exports.
3. Emit the discovered dependency name as an absolute Rust path. A lookup that
   identifies the current package (`FoundCrate::Itself`) maps to `crate`.
4. Report a compiler error if neither package is available.

| Caller's dependencies | Path to core items in the expansion |
| --- | --- |
| Facade aliased to `bento` | `::bento` |
| Facade aliased to `support` | `::support` |
| Facade without a rename | `::zakura_bento` |
| Core aliased to `bento-core` | `::bento_core` |
| Core aliased to `support-core`, plus a facade dependency | `::support_core` |

The direct core dependency takes precedence when both packages are present.
It is an implementation option, not a requirement for consumers: a dependency
on the facade is sufficient. The macro crate's own core dependency is not
automatically visible in the caller. Resolving by package identity and then
using the caller's alias avoids hardcoding either `::bento` or
`::zakura_bento` into generated code.

Expansion functions interpolate the supplied path with `quote!` and use
absolute `::core` paths for standard types, preserving support for `no_std`
callers. `BentoCorePath::default()` is a deterministic `::bento_core` path for
unit tests; real entry points use `resolve()`.

### Self aliases

`extern crate self as zakura_bento_core;` would give the current crate an
additional name within its own code, allowing `::zakura_bento_core::Item` to
refer to its own `Item`. It does not add a dependency or change the name seen
by downstream crates. We do not need that alias: core does not invoke the
macros, and it has no effect on either the macro implementation's dependency
name or the paths available to a downstream expansion.

A self alias can help a macro expansion that references a fixed external crate
name when invoked inside that same library, if its dependency structure allows
such invocation. Our resolver instead handles `FoundCrate::Itself` with
`crate`. This case is not a blanket solution for doctests: there, `crate`
refers to the generated test crate. When adding actual macros, test their
intended invocation sites, including doctests and renamed dependencies.

## Local development and publication

All packages currently inherit `publish = false` from `[workspace.package]`, and
internal dependencies use local paths without registry version requirements.
Code already uses the short aliases. An external checkout can use the same
convention before publication:

```toml
[dependencies]
bento = { package = "zakura-bento", path = "../udon/crates/bento" }
udon = { package = "zakura-udon", path = "../udon/crates/udon" }
```

Adjust those paths relative to the consuming manifest. Once the packages are
published, the equivalent registry dependencies will be:

```toml
[dependencies]
bento = { package = "zakura-bento", version = "0.1" }
udon = { package = "zakura-udon", version = "0.1" }
```

Before enabling publication, add version requirements alongside the internal
paths in the root workspace table. Keep the facade, core, and macros in
lockstep: pin core and macros to the exact matching release so generated code
and the facade's re-exports cannot resolve to incompatible implementations.
For an illustrative `0.1.0` release, the implementation entries would be:

```toml
[workspace.dependencies]
bento-core = { package = "zakura-bento-core", path = "crates/bento-core", version = "=0.1.0" }
bento-macros = { package = "zakura-bento-macros", path = "crates/bento-macros", version = "=0.1.0" }
```

Other internal entries used by published packages also need appropriate version
requirements. Cargo uses the local path during workspace development and
checks that its package version satisfies the requirement. Packaging resolves
inherited workspace dependencies into concrete entries and removes local paths
from the registry manifest while retaining package identities, aliases, and
version requirements. Published crates therefore do not need this repository's
workspace to build. See Cargo's [path and registry dependency rules](https://doc.rust-lang.org/cargo/reference/specifying-dependencies.html#multiple-locations).

Macro path resolution uses the same package identities and caller aliases
before and after publication; switching from a path to a registry dependency
does not require different expansion code. Downstream users may choose other
aliases, or inherit them from their own workspace, and the resolver discovers
those names. Keep the prefixed package strings in its lookups even though our
Rust dependency paths use short names.
