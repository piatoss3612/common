# Crate development

This repository is one virtual Cargo workspace, with members under `crates/`.
The root [manifest](../Cargo.toml) owns shared package metadata and internal
dependency declarations. Members inherit the fields and dependencies they use;
crate names, descriptions, and target settings stay in their own manifests.

## Dependencies and names

Use short dependency keys in Rust code and `zakura-` package names in Cargo
commands and registry metadata. Declare internal aliases once in
`[workspace.dependencies]`, then opt in with `name.workspace = true` in the
appropriate dependency table. Inheritance does not make a dependency available
to every member or make transitive dependencies directly accessible.

For example, `bento-core` selects package `zakura-bento-core` and supplies the
Rust path `bento_core::`. Hyphens become underscores. Library target names still
default to their package names; integration tests and doctests can import
`zakura_bento as bento`. A Rust import does not change Cargo's dependency names.
Third-party dependencies belong in the members that use them unless sharing
their declaration serves an actual workspace need.

## Crate boundaries

| Crate | Responsibility |
| --- | --- |
| `bento-core` | Shared traits, storage support, and planned reference arithmetic; `no_std` |
| `bento-macros` | Parsing, validation, and code generation on the build host |
| `bento` | Public `no_std` facade over core and macros |
| `udon` | Planned optimized field and curve arithmetic; `no_std` |

Keep dependencies directed from arithmetic consumers through the facade to
support code. Core must not depend on the facade or invoke its procedural
macros. Expose facade items deliberately; adding a public implementation helper
must not automatically extend the facade's API.

Procedural macros and build scripts execute on the host. Generated code and
embedded representations must satisfy the target's layout and platform
requirements. Keep host parsing and generation dependencies out of target
libraries. Reference arithmetic used to derive constants belongs in support
code; artifact formats and their generators belong with the data's owner.

As arithmetic grows, distinguish memory validity, mathematical invariants, and
side-channel guarantees. Safe constructors must establish any invariants needed
by safe operations, including their unsafe internals. A fixed operation schedule
alone does not establish constant-time behavior of the operations it calls.
State which inputs may be secret and justify any constant-time claim at the
layer that implements it.

Keep unsafe code small and document its proof at the operation or implementation
that relies on it. Generated unsafe implementations are part of this surface:
review how their safety-critical names resolve in consumers. Do not rely on
host layout, trusted generators, or intended callers to satisfy requirements
that a public safe API permits arbitrary callers to bypass.

## Procedural macros

The [macro guide](MACROS.md) defines expansion and path conventions. Declarative
wrappers should carry library paths through `$crate` when possible. Manifest
lookup discovers names, not which dependencies Cargo activated. Provide an
explicit path for contexts where discovery is insufficient instead of trying
to reproduce Cargo's resolver inside a macro.

## Changes and publication

When introducing a structure or convention, update directly affected code,
tests, and links together. Organize around responsibilities that will remain
useful as the repository grows; avoid reorganizing unrelated code for symmetry.
Choose tests by the boundary they exercise; see the [testing guide](TESTING.md).

Packages currently inherit `publish = false` and use local dependency paths.
Before publication, add registry version requirements alongside those paths,
review package metadata and contents, and test consumers of the packaged crates.
Pin implementation dependencies when generated code and support must evolve in
lockstep. Such pins coordinate a facade's own dependencies; they do not prevent
a consumer from declaring additional, incompatible versions.
