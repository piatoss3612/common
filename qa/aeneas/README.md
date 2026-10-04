# Aeneas proofs of Udon arithmetic

This package extracts the production Rust arithmetic with Charon and Aeneas,
then checks handwritten Lean proofs. The Rust implementation is unchanged.

The checked word theorems in [Proofs.lean](proofs/Proofs.lean) cover `adc`,
`mac`, and `sbb` for all valid inputs. With `B = 2^64`, `adc` and `mac` return
the low remainder and high quotient of their exact integer sum or product.
`sbb` accepts an incoming borrow at most one and proves an outgoing borrow
at most one and `low + rhs + borrow = lhs + B * outgoing_borrow`.
Each theorem also proves successful execution with overflow checks enabled.

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
and `backends/lean`. The script creates a fresh directory under the system
temporary directory. Use `--output /path/to/new-directory` to choose its
location; it must not already exist. Results, exact commands, timings,
generated Lean files, and logs are retained there.

The script selects the native `word.rs` from the Udon crate. It also translates
all limb and Montgomery kernels through a small parameter harness that imports
production `word.rs` and `montgomery.rs` by path. Translation of those additional
routines is not a correctness theorem for them. The harness reads both Pasta
modulus literals and derives the doubled modulus and Montgomery coefficient;
it bypasses the complete parameter traits and constant table generation.

## Trust boundary

The proof census accepts only Lean's standard `propext`, `Classical.choice`,
and `Quot.sound`. The generated model and proof sources must contain no proof
holes, extra declared axioms, or external-model placeholders. Extraction enables
both overflow checks and debug assertions. Specifications require successful
execution and the stated mathematical result.

The theorems concern the extracted Lean definitions. Rust compilation, Charon,
Aeneas, and Aeneas's standard-library models form the translation trust boundary.
The complete native Fp/Fq API, its parameter refinement, modular multiplication,
inversion, square roots, encodings, curves, and FFTs have no correctness theorem
in this package yet. Udon's variable-time arithmetic is unchanged; these proofs
do not establish constant-time behavior.
