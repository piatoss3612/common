# Aeneas proofs of Udon arithmetic

This package extracts the production Rust arithmetic with Charon and Aeneas,
then checks handwritten Lean proofs. The package covers four layers of field
arithmetic; inversion, square roots, and canonical encodings remain in progress.

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

| Layer | Checked claim |
| --- | --- |
| 1. Words | Exact `adc`, `mac`, and `sbb`, including carries and borrows |
| 2. Limbs | Comparison, addition, subtraction, wide multiplication and squaring |
| 3. Montgomery | Reduction, multiplication, squaring, and square runs |
| 4. Native fields | Concrete Fp/Fq parameters, arithmetic, representation, predicates and integer constructors |
| 5. Remaining field routines | Inversion, square roots and canonical encodings: in progress |

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

```console
python3 qa/aeneas/reproduce.py --tools /path/to/aeneas-release --offline
```

`--tools` names the extracted release directory containing `aeneas`, `charon`,
and `backends/lean`. Python 3.11 or later is required. The script creates a
fresh directory under `target/aeneas/`. Use `--output /path/to/new-directory`
to choose its location; it must not already exist. Results, exact commands, timings,
generated Lean files, and logs are retained there.

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

The production refactors reuse the already-proved limb kernels in addition,
subtraction and halving, use an indexed loop for small multiples, and name the
compile-time `R + p` constant used by `is_one`. This avoids unsupported mutable
iterator models and missing generic promoted constants in the pinned frontend.

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
- The unused parameter-dictionary back edge and two unused square-root
  callbacks are removed. The adapter rejects references to either from retained
  arithmetic bodies. It is unsuitable for extracting square-root calls.
- The generic `ONE` initializer receives an assertion of literal `true` to
  correct its inferred effect type. All other definitions are checked unchanged.
- Charon's explicitly selected private halving root is retained by marking
  its extraction metadata as local. Its body and Rust visibility are retained.

[native_support/](native_support/) supplies two fully defined standard-library
extensions: the Rust `Ordering` discriminants, checked against Charon's export,
and identity cloning for the zero-sized `PhantomData` marker. The reproducer
allows only the exact supplied external-definition file, never a generated
template with placeholders.

The theorems concern the extracted Lean definitions. Rust compilation, Charon,
Aeneas, its standard-library models, and the documented adapters and extensions
form the translation trust boundary. The source hashes pin the workspace Rust
sources and dependency manifests; the tool hashes pin the extraction binaries.
Lean checks the arithmetic proofs and their axiom census. No primality theorem
is required for the current integer and modular arithmetic claims.

Inversion, square roots, canonical encodings, product accumulation, curves and
FFTs have no correctness theorem in this package yet. The proofs do not
establish constant-time behavior.
