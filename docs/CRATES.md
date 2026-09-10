# Crate development

The workspace uses short dependency names in Rust code and `zakura-` package
names for Cargo and the registry. Shared package metadata and internal dependency
declarations live in the root [`Cargo.toml`](../Cargo.toml); each member explicitly
inherits what it uses.

## Related reorganization

When a change introduces a new structure or convention, review the surrounding
code, tests, and documentation for related reorganization. Consider whether
existing material should follow the same structure, and include that
reorganization when it makes the result more consistent and easier to navigate.
Keep it tied to the change and update affected paths, links, and test harnesses.

For example, adding a `pod/` directory to the
[macro fixtures](../crates/bento-macros/tests/fixtures/) calls for grouping the
existing addition-chain fixtures under `addition_chain/` too. Fixtures shared by
both features belong in a common directory such as `consumers/`.

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
| `crates/bento/` | `zakura-bento` | `bento` | `bento::` |
| `crates/bento-core/` | `zakura-bento-core` | `bento-core` | `bento_core::` |
| `crates/bento-macros/` | `zakura-bento-macros` | `bento-macros` | `bento_macros::` |
| `crates/udon/` | `zakura-udon` | `udon` | `udon::` |

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
transitively available. In particular, `bento-core` inherits no dependencies.
Add new internal aliases to the root table, then inherit them only where needed.
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

| Crate | Intended role | Normal workspace dependencies |
| --- | --- | --- |
| `bento-core` | Shared traits, storage support, and reference arithmetic; `no_std` | None |
| `bento-macros` | Parsing, validation, and code generation on the build host; uses `std` | `bento-core` |
| `bento` | Public facade; `no_std` | `bento-core`, `bento-macros` |
| `udon` | Optimized field and curve arithmetic; `no_std` | `bento` |

The current implementation provides
[`addition_chain!`](../crates/bento/src/lib.rs), its
[`AdditionChain`](../crates/bento-core/src/addchain.rs) support trait, and
[POD storage and embedding](POD.md). Reference arithmetic in core, and field and
curve arithmetic in `udon`, remain scaffolded. Consumers use the `bento` facade,
which re-exports core items at its root and explicitly re-exports each macro.
Procedural macros are documented on the facade; declarative embedding macros
carry their documentation through the re-export. `udon` declares its support
dependency through `bento`.

`bento-core` is the bottom layer and cannot invoke the macros through the
facade under this dependency structure. Shared arithmetic belongs there so
both ordinary library code and macros can use it. Macros execute on the build
host; generated Rust code is compiled in the caller's crate for the target.
These are separate dependency contexts.

`bento-macros` also has a development dependency on `bento` for tests using the
public facade. Cargo permits this development dependency cycle; it does not
introduce a normal dependency from the macro implementation back to the facade.

## Procedural macros

See the [macro authoring conventions](MACROS.md#authoring-conventions)
for error handling and testing requirements. The
[macro guide](MACROS.md) also links to the addition-chain implementation
and tests.

Generated paths must match the caller's dependencies. The
[path resolver](../crates/bento-macros/src/path_resolution.rs) discovers the caller's
aliases; a dependency on the facade is sufficient. Support interfaces are
documented in their owning modules, such as
[`addchain`](../crates/bento-core/src/addchain.rs).

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
