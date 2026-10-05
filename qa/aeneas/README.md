# Aeneas proofs of Udon arithmetic

This package extracts the production Rust arithmetic with Charon and Aeneas,
then checks handwritten Lean proofs. The package covers four layers of field
arithmetic, canonical integer and byte encodings, Pasta primality, roots of
unity, the fixed square-root exponentiation schedules, generic Tonelli–Shanks
correction, and all six default Fp/Fq square-root entry points.
The inversion parameter checks and signed-limb input conversion are also
proved; the full inversion routine remains in progress.

The checked word theorems in [Proofs.lean](proofs/Proofs.lean) cover `adc`,
`mac`, and `sbb` for all valid inputs. With `B = 2^64`, `adc` and `mac` return
the low remainder and high quotient of their exact integer sum or product.
`sbb` accepts an incoming borrow at most one and proves an outgoing borrow
at most one and `low + rhs + borrow = lhs + B * outgoing_borrow`.
Each theorem also proves successful execution with overflow checks enabled.

The five multi-limb theorems additionally prove comparison of the represented
256-bit integers, exact addition and subtraction including the outgoing carry
or borrow, and exact 512-bit multiplication and squaring. These proofs use the
unchanged production `word.rs` imported by the parameter harness. `val4` and
`val8` interpret little-endian limb arrays as natural numbers; their definitions
are in [Common.lean](proofs/Common.lean).

| Routine | Postcondition |
| --- | --- |
| `compare_limbs` | Returned ordering agrees with `val4` |
| `add_limbs` | `val4(out) + 2^256 * carry = val4(lhs) + val4(rhs)`, carry at most one |
| `subtract_limbs` | `val4(out) + val4(rhs) = val4(lhs) + 2^256 * borrow`, borrow at most one |
| `multiply_wide` | `val8(out) = val4(lhs) * val4(rhs)` |
| `square_wide` | `val8(out) = val4(value)^2` |

The seven Montgomery routines are also proved. Write `p` for the selected
Pasta modulus and `R = 2^256`:

| Routine | Input bound and postcondition |
| --- | --- |
| `reduce_once` | Input below `2p`; canonical result `input mod p` below `p` |
| `reduce_twice_modulus` | Five-limb input below `4p`; result `input mod 2p` below `2p` |
| `montgomery_reduce_unreduced` | Input `T < pR + p²`; result `u < 3p` with `Ru = T + mp`, `m < R` |
| `montgomery_reduce` | Input `T < pR`; result below `p` with `Ru ≡ T (mod p)` |
| `montgomery_multiply` | Both operands below `2p`, or product below `pR`; result `u < 2p` with `Ru = ab + mp`, `m < R` |
| `montgomery_square` | Input below `2p`; result `u < 2p` with `Ru = a² + mp`, `m < R` |
| `square_run` | Every `usize` count and either factor branch; loose bounds and the weighted power congruence below |

For `n` squarings without a final factor, `square_run` proves
`R^(2^n - 1) * u ≡ a^(2^n) (mod p)`. With a loose final factor `f`, it proves
`R^(2^n) * u ≡ a^(2^n) * f (mod p)`. Thus it covers arbitrary run lengths
rather than a fixed collection of exponentiation schedules. All these theorems
include successful execution of the extracted loops, checked operations,
debug assertions, slice conversion, and unwrap.

[Parameters.lean](proofs/Parameters.lean) checks the required modulus shape,
Montgomery inverse, doubled modulus, and integer bounds for both actual Pasta
modulus literals. [Algebra.lean](proofs/Algebra.lean) proves the stronger closure
argument for arbitrary loose products documented in the production kernel.
No primality assumption is needed for these integer and congruence theorems.
[NativeBridge.lean](proofs/NativeBridge.lean) connects the harness to the native
parameter dictionaries and proves definitional equality of the shared kernels.
Both native instances discharge the concrete modulus and Montgomery constant
requirements. [NativeConstants.lean](proofs/NativeConstants.lean) also checks
the compiled `R` and `R2` values against their integer specifications.

The native theorems use `decode(a) = (val4(a) * R_inverse) mod p`, with the
inverse checked for each Pasta modulus. Multiplication and squaring prove the
ordinary modular product and square of the decoded inputs. Addition,
subtraction, negation, doubling, triple, multiplication by four and eight,
and the composed multiply-add and multiply-subtract methods also have modular
correctness and successful-execution proofs. These preserve the loose bound
below `2p`; the specialized small multiples return below `p`.

Halving preserves either representation bound and computes multiplication by
`(p + 1) / 2` modulo `p`. Exponentiation handles every `u64` exponent, including
zero. Its proof checks the leading-zero calculation, reverse iterator, bit
tests, decreasing loop measure, and accumulated exponent.

The representation proofs cover reduction, widening, Montgomery limb import
and access, zero, one, and the zero and one predicates. Integer constructors
cover every `u64`, `u128`, and `i64`, including `i64::MIN`, for both loose and
reduced results. Input preconditions describe valid stored field values:
below `2p` for loose values and below `p` for reduced values. The generic
arithmetic proofs also accept tighter operand bounds.

The encoding proofs cover both fields and representation states.
`from_canonical_uint` and `from_bytes` reject exactly the integers at or above
the modulus. Accepted inputs decode to the supplied ordinary integer.
`from_uint_reduced` reduces every 256-bit input modulo the selected prime.
`to_canonical_uint` and `to_bytes` produce the decoded integer below the modulus.
The byte proofs use an independent little-endian `Nat.ofDigits` interpretation;
they check the indexed loops, slice bounds, conversion and unwrap. Canonical
bytes round-trip exactly. Field values round-trip with the same decoded value
and the requested representation bound. `is_odd` tests the decoded integer's
parity. These are universal input theorems.

The root-of-unity proofs cover both fields and representation states, including
every `u32` index. Indices above 32 return `None`. Valid indices return canonical
Montgomery limbs whose decoded values are primitive roots of order `2^log_size`.
Forward roots equal `5^((p - 1) / 2^log_size)`; inverse roots multiply those
values to one modulo `p`. [NativeRootTables.lean](proofs/NativeRootTables.lean)
checks the actual compiled tables with kernel `decide`, and
[NativeRootOrder.lean](proofs/NativeRootOrder.lean) derives exact orders from
the full-power and half-power checks. The lookup proofs check the index cast,
array access and representation constructor.

[NativeSqrtChains.lean](proofs/NativeSqrtChains.lean) proves both compiled
addition chains compute `value^((t - 1) / 2)`, where `p - 1 = t * 2^32`, for
every valid loose input. These proofs follow the extracted schedules and use
the checked native multiplication and square-run routines.

[NativeTonelli.lean](proofs/NativeTonelli.lean) and
[NativeTonelliSqrt.lean](proofs/NativeTonelliSqrt.lean) prove the extracted generic
Tonelli–Shanks routines under explicit field-operation and root-callback
contracts. The order-search loop respects its bounds and terminates. The
correction loops preserve the root equation, terminate, and discharge the
checked increments and assertions. The alternate routine's flag identifies
squares and its returned value squares to the input or its fixed nonsquare
multiple. The ordinary routine returns a root or rejects a nonsquare, including
the zero case. [NativeSqrtBridge.lean](proofs/NativeSqrtBridge.lean) connects the
concrete model to the checked native arithmetic and proves equality with the
separately extracted generic routines. The compiled root tables, fixed powers,
zero and one predicates, multiplication, squaring and representation conversions
discharge every contract for both Pasta fields. Checked Euler powers establish
that each compiled top root is a nonsquare; Pasta primality supplies the starting
power identity.

[NativeSqrtCaps.lean](proofs/NativeSqrtCaps.lean) proves successful execution of
`sqrt`, `sqrt_alt` and `sqrt_ratio` for every valid reduced Fp/Fq input. `sqrt`
returns a reduced root, or `None` exactly for nonsquares. `sqrt_alt` returns a
square flag and a reduced root of either the input or its fixed nonsquare
multiple. For nonzero numerator and denominator, `sqrt_ratio` returns a reduced
root whose square times the denominator equals the numerator or its fixed
nonsquare multiple; the flag identifies whether the ratio is square. A zero
numerator returns `(true, 0)`, including when both inputs are zero. A nonzero
numerator with zero denominator returns `(false, 0)`. These proofs use the
default feature configuration; `sqrt-table-large` remains unproved.

[NativeSafegcdParameters.lean](proofs/NativeSafegcdParameters.lean) checks both
compiled signed-62 modulus representations, the five-limb offset `p * 2^63`,
and every entry in the twelve-element Montgomery correction tables. Entry `k`
decodes to `4^(k + 1)` modulo the selected prime.
[NativeSigned62.lean](proofs/NativeSigned62.lean) proves the extracted
`to_signed62` loop succeeds for every four-limb input below `2^255` and
preserves its integer value in radix `2^62`. Its first four digits lie in
`[0, 2^62)` and its top digit lies in `[0, 128)`. The proof checks all five
packing steps, including the shifts, masks, casts, array accesses and updates.
The divstep loop and coefficient updates still need correctness proofs before
these results establish inversion.

[PastaPrimality.lean](proofs/PastaPrimality.lean) proves primality of the two
compiled Pasta moduli using Lucas certificates and recursively checked prime
factors. [Pratt.lean](proofs/Pratt.lean) supplies bounded modular exponentiation
and connects its computations to Mathlib's Lucas theorem. The numeric checks
use Lean's kernel `decide`; certificate generation supplies no trusted premise.

| Layer | Checked claim |
| --- | --- |
| 1. Words | Exact `adc`, `mac`, and `sbb`, including carries and borrows |
| 2. Limbs | Comparison, addition, subtraction, wide multiplication and squaring |
| 3. Montgomery | Reduction, multiplication, squaring, and square runs |
| 4. Native fields | Concrete Fp/Fq parameters, arithmetic, representation, predicates and integer constructors |
| 5. Remaining field routines | Encodings, parity, primality, roots and all six default square-root methods checked; inversion and the additional scopes listed below remain in progress |

[catalog.json](catalog.json) records the proof modules and theorem census.
[provenance.json](provenance.json) pins Udon source hashes, Charon, Aeneas,
Lean, Mathlib, and the extraction compiler. The source hashes are checked on
every run, so subsequent proof commits can reproduce the same arithmetic.

## Reproduce

Use the official Aeneas release
[`nightly-2026.10.04-557eff8`](https://github.com/AeneasVerif/aeneas/releases/tag/nightly-2026.10.04-557eff8),
which includes Charon and the Lean backend. Its published macOS archive
SHA256 is `813251bac79ee357e14c6f5141ddfc1d65ff58f624c1644e1f163ed10c5c7966`.
The host used here is macOS arm64. Install the pinned Rust nightly including
`rustc-dev` and `rust-src`, and Lean 4.31.0. Prepare the Lean backend's pinned
Lake dependencies and Mathlib cache before an offline run.

The concrete square-root extraction uses a source build of the same pinned
Charon revision with [charon-retain-cyclic-bounds.patch](charon-retain-cyclic-bounds.patch).
Its source archive is available at the URL recorded in `charon_sqrt` in the
provenance file; its SHA256 is
`8fb8d08240affd9db31d92dc2eac80048ea6be45979d62cbaa975be2e1c5db92`.
The build script checks the archive, patch, complete regular-file source tree
and source links, builds both binaries with the pinned compiler, and records
their hashes and exact build commands. Cargo dependencies must be cached for
an offline build.

```console
python3 qa/aeneas/build_sqrt_charon.py --archive /path/to/charon-source.tar.gz --output target/aeneas/sqrt-charon --offline
python3 qa/aeneas/reproduce.py --tools /path/to/aeneas-release --sqrt-tools target/aeneas/sqrt-charon --offline
```

`--tools` names the extracted release directory containing `aeneas`, `charon`,
and `backends/lean`. Python 3.11 or later is required. The script creates a
fresh directory under `target/aeneas/`. Use `--output /path/to/new-directory`
to choose its location; it must not already exist. Results, exact commands, timings,
generated Lean files, and logs are retained there.
`--sqrt-tools` names the source-build directory containing its checked manifest.
Every other extraction stage uses the official release binaries.

The script selects the native `word.rs` from the Udon crate. It also translates
all limb and Montgomery kernels through a small parameter harness that imports
production `word.rs` and `montgomery.rs` by path. All fifteen word, limb, and
Montgomery routines have correctness theorems for the specified input bounds.
The harness reads both Pasta modulus literals and derives the doubled modulus
and Montgomery coefficient.
The native extraction additionally imports the actual Udon crate through
[native_wrapper.rs](native_wrapper.rs). It checks that the wrapper's Cargo
dependencies remain pinned to the workspace lockfile. Native parameter values
come from the production compile-time derivations and are checked by Lean.
The extraction uses the default Udon feature configuration.
The generic square-root routines are separately selected from the production
crate. Their model includes the extracted standard-library tuple comparison
and Boolean inequality; it needs no external definitions or translation adapter.

The production refactors reuse the already-proved limb kernels in addition,
subtraction and halving, use an indexed loop for small multiples, and name the
compile-time `R + p` constant used by `is_one`. This avoids unsupported mutable
iterator models and missing generic promoted constants in the pinned frontend.
Encoding uses indexed byte loops and explicit branches. Inversion names the
compile-time Bézout offset, expands the four signed coefficient conversions, and
uses widening casts and explicit bounds. The default native extraction includes
inversion bodies; extraction alone supplies no correctness claim for inversion.
The square-root assertions use explicit range bounds and tuple equality so
both conditions remain checked during extraction. The concrete square-root
model imports the production crate through [sqrt_wrapper.rs](sqrt_wrapper.rs).

## Trust boundary

The proof census accepts Lean's standard `propext`, `Classical.choice`, and
`Quot.sound`. Multiplication and `square_run` also mention Aeneas's
`core.fmt.Formatter`: an abstract **type** used by the standard-library model
of the debug trait passed to `unwrap`. It supplies no arithmetic proposition.
The proof establishes the successful conversion and unwrap; it does not model
formatting on the error path. The exact per-theorem allowance is recorded in
the catalog and checked on each run. The generated model and proof sources must
contain no proof holes, extra declared axioms, or external-model placeholders.
Extraction enables
both overflow checks and debug assertions. Specifications require successful
execution and the stated mathematical result.

The native translation uses four checked adapters in [adapters/](adapters/):

- Closed, typed literal globals become pure initializer functions; existing
  functions, types and dictionary implementations are retained.
- The unused parameter-dictionary back edge and square-root callback fields
  are removed. The two transparent callback functions remain as independent
  proof roots, with unchanged bodies. Their selection metadata and declaration
  order reflect the projected dictionary. The adapter rejects references to
  either erased component from retained bodies. It is unsuitable for extracting
  square-root calls.
- The generic `ONE` initializer receives an assertion of literal `true` to
  correct its inferred effect type. All other definitions are checked unchanged.
- Charon's explicitly selected private halving root is retained by marking
  its extraction metadata as local. Its body and Rust visibility are retained.

[native_support/](native_support/) supplies two fully defined standard-library
extensions: the Rust `Ordering` discriminants, checked against Charon's export,
and identity cloning for the zero-sized `PhantomData` marker. The reproducer
allows only the exact supplied external-definition file, never a generated
template with placeholders.

The concrete square-root model has three additional translation details:

- The pinned Charon patch changes only the cycle case in `has_assoc_types`,
  retaining ADT trait clauses instead of recursively expanding cyclic removed
  clauses. The patch changes no Rust arithmetic body. The source-build manifest
  and binary hashes are checked on every reproduction.
- [project_unused_parent.py](adapters/project_unused_parent.py) removes the
  unused parameter-dictionary back edge while retaining its callback fields.
  It rejects references to the removed parent and checks that all functions,
  types, globals and other dictionary contents are unchanged. Constants use
  the same checked normalization and effect adapters as the native model.
- [sqrt_support/FunsExternal.lean](sqrt_support/FunsExternal.lean) fully defines
  marker cloning, short-circuit tuple equality and `Option::map`. It also links
  the two parameter callbacks to their already-verified extracted Rust bodies
  through the same four-limb representation. No callback arithmetic is assumed.
  The generic-routine equality proofs also check the tuple-comparison definition
  against the separately extracted standard-library implementation. The
  reproducer checks the exact five external declarations and permits only the
  supplied definitions, which contain no placeholders.

The theorems concern the extracted Lean definitions. Rust compilation, Charon,
Aeneas, its standard-library models, and the documented adapters and extensions
form the translation trust boundary. The source hashes pin the workspace Rust
sources and dependency manifests; the tool hashes pin the extraction binaries.
Lean checks the arithmetic proofs and their axiom census. The first four layers
use integer bounds and modular congruences. Primality has its own checked
certificates.

Inversion, the `sqrt-table-large` feature, reducing byte strings of other widths,
product accumulation, curves and FFTs have no correctness theorem in this package yet.
The proofs do not establish constant-time behavior.
