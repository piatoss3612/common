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
| `bento-core` | Shared traits, storage support, and reference arithmetic; `no_std` |
| `bento-macros` | Parsing, validation, and code generation on the build host |
| `bento` | Public `no_std` facade over core and macros |
| `udon` | Runtime Pasta fields, curves, and field FFTs; `no_std` |

Keep dependencies directed from arithmetic consumers through the facade to
support code. Core must not depend on the facade or invoke its procedural
macros. Expose facade items deliberately; adding a public implementation helper
must not automatically extend the facade's API.

The facade lists each exported item explicitly, including items within public
modules. Reference arithmetic is exposed through macros that place the entire
implementation call inside `const { ... }`. Do not export arithmetic functions
or contexts through the facade: Udon must use this support only at compile time.
POD storage APIs retain their existing const methods.

Within core's `pod` module, the trait and primitive implementations define the
storage contract, `layout` owns target measurements and validation, and
`storage` owns byte views and aligned buffers. The facade preserves the public
paths while the macro crate emits checks through the core-owned trait metadata.

Use associated constants for fixed values tied to a type, such as field
parameters and execution presets. Perform construction and representation checks
in constant initializers so they do not depend on optimizer constant folding.
A `const fn` is also callable at runtime; declaring it `const` does not force
compile-time evaluation of those calls. Keep functions for input-dependent
operations and constructors for mutable working state.

Use named statics for large shared tables and expose borrowed references through
constants or lookup methods. An array-valued associated constant does not
guarantee shared storage.

Procedural macros and build scripts execute on the host. Generated code and
embedded representations must satisfy the target's layout and platform
requirements. Keep host parsing and generation dependencies out of target
libraries. Reference arithmetic used to derive constants belongs in support
code; artifact formats and their generators belong with the data's owner.

Keep curve and FFT artifact schemas and execution runtimes downstream: Udon
borrows caller tables, buffers, scratch, and, for parallel FFTs, an executor.
Udon APIs do not allocate. See the [curve guide](CURVES.md) and
[FFT guide](FFT.md) for preparation workflows and
[crate docs](../crates/udon/src/lib.rs) for feature definitions.

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

## Udon module boundaries

Udon owns runtime arithmetic and the contracts needed to execute it. Bento
supplies constant derivation and POD tools; it does not own Pasta's runtime
representations or arithmetic kernels. Artifact schemas, generator derivation,
allocating workspaces, and worker runtimes belong to their downstream owners.

| Source module | Responsibility |
| --- | --- |
| `field/` | Public exports of native field APIs and optional consumer interfaces |
| `field/pasta/` | Pasta representations, parameters, and optimized field implementations |
| `field/consumer/` | Optional field trait and its operator adapter |
| `curve/` | Public exports of native curve APIs and optional consumer interfaces |
| `curve/pasta/` | Pasta point representations, coordinate kernels, and fixed-base tables |
| `curve/consumer/` | Optional curve traits and their affine/projective operator adapters |
| `fft/` | Domains, full transforms, layouts, tables, and transform plans |
| `msm/` | Multiscalar multiplication, scalar preparation, scheduling, and task plans |
| `exec/` | Shared executor contracts, operation budgets, and scoped work helpers |
| `exec/execution/` | Common incremental task, completion, and frontier protocol |
| `cycle/` | Optional cycle contracts, borrowed generator containers, and Pasta bindings |
| `poseidon/` | Fixed Pasta parameter sets and optional consumer trait views |
| `polynomial/` | Native Pasta polynomial arithmetic and optional generic evaluation, linear division, and geometric sums |

The `field` and `curve` modules explicitly re-export their concrete types;
callers use paths such as `field::Fp` and `curve::Pallas`.
Their private Pasta modules keep representation-specific code separate from
generic contracts. Each domain's `consumer/traits.rs` defines its optional
contracts; `consumer/adapter.rs` owns the wrappers, their operators, and their
trait implementations. The unstable `traits` feature gates each `consumer`
module, along with generic polynomial iterators and `cycle`.
Native arithmetic must not depend on the consumer contracts, even when the
feature is enabled. Native types expose explicit arithmetic methods. The
optional `field::FieldAdapter`, `curve::AffineAdapter`, and
`curve::ProjectiveAdapter` wrappers own the operator implementations and borrow
native buffers through transparent views without allocation or copying.
The `random`, `low_u64`, `dot`, and `dot_iter` helpers use concrete Pasta values
and need no feature. They live beside native encoding and product arithmetic
under `field/pasta/` and are re-exported from `field`.
Field butterfly kernels live in `field/pasta/butterfly/`;
they are the small arithmetic steps used by `fft/`, not a second transform API.
Likewise, `curve/pasta/reduce.rs` owns the coordinate formulas used by MSM
bucket reduction, keeping raw affine coordinates private to the curve implementation.

FFT and MSM are sibling modules for bulk arithmetic over fields and curves.
Both depend on `exec`, which supplies shared contracts
without owning threads or allocating a pool. Within FFT, `planning` resolves
geometry and scratch, `request` describes the requested operation, and
`normalization` applies inverse scaling. The `fft/execution` and `msm/execution`
modules own their plans and task kernels; `exec/execution` owns the common protocol.

These are source modules within one library crate. Runtime arithmetic kernels
stay in Udon; Bento supplies compile-time derivation and code generation.
The Pasta field implementation also owns its stored-representation descriptor
and shared field test helpers. `STORED_FORM` and `stored_form!` remain exported
at the crate root for artifact producers and consumers.

Keep arithmetic, encoding, cycle, and resource-limit assertions beside their
implementation, including checks written entirely through public methods.
Keep Pasta field and curve suites under `field/pasta/tests` and
`curve/pasta/tests`, including their consumer adapter and helper tests.
Enable tests of the optional consumer interfaces with `traits`.
Use separate consumers for macro resolution, independent trait implementations,
caller-owned adapters, and generated artifacts. Group them by field, curve, FFT,
MSM, or shared execution. Put each domain's storage tests and consumer fixtures in
that domain, alongside its other tests. Shared test machinery belongs in the
test harness rather than another suite. See the
[testing guide](TESTING.md#udon-test-layout) for the test entry points.

## Procedural macros

The [macro guide](MACROS.md) defines expansion and path conventions. Declarative
wrappers should carry library paths through `$crate` when possible. Manifest
lookup discovers names, not which dependencies Cargo activated. Provide an
explicit path for contexts where discovery is insufficient instead of trying
to reproduce Cargo's resolver inside a macro.

## Maintaining the workspace

When introducing a structure or convention, update directly affected code,
tests, and links together. Organize around responsibilities that will remain
useful as the repository grows; avoid reorganizing unrelated code for symmetry.
Choose tests by the boundary they exercise; see the [testing guide](TESTING.md).

Packages currently inherit `publish = false` and use local dependency paths.
