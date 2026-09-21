# Documentation

Explain what readers need to use an API or maintain its implementation: purpose,
semantics, constraints, and the reasons behind non-obvious choices. Scale the
explanation to the subject. A simple operation may need one sentence; a complex
abstraction may need background, examples, and a discussion of tradeoffs.

The [README](../README.md) is the repository entry point. Keep short guides to
cross-API workflows and relationships in `docs/`, API contracts in rustdoc with
their definitions, and implementation reasoning beside the code it explains.
Link to the authoritative explanation instead of maintaining parallel accounts
or inventories of current files.

## Audience and contracts

An item exposed through a public facade has a public contract, wherever it is
implemented. Document its current purpose and consequential semantics that are
not apparent from the signature: accepted inputs, validation, errors, panics,
mutation on failure, representation, ownership, invariants, and non-obvious
costs. Review re-exported docs as readers see them. Internal documentation can
explain algorithms, representations, and maintenance constraints without making
them public guarantees.

Describe current capabilities rather than chronology, migration stories,
roadmaps, or requirements of named external consumers. Preserve necessary
algorithm attribution and license notices.

Use `///` and `//!` for the abstraction and its contract; use `//` for local
reasoning. Comments should explain why code is necessary or correct rather than
narrate its syntax. Avoid promising incidental implementation behavior.

Distinguish memory safety, mathematical validity, storage and layout
requirements, side-channel properties, performance guarantees, and caller
policy. State what is checked and what is assumed, including trusted
constructors and generated data. Name the scope of each guarantee: one item,
one task, one invocation, or a whole provision. Avoid promising incidental
kernel choices or heuristics. For cryptographic code, make the assumptions of
side-channel claims explicit.

Unsafe items need a `# Safety` section stating the caller's or implementer's
obligations precisely. Explain how unsafe operations discharge those obligations
where they occur.

## Presentation

Use direct sentences, concrete terminology, and examples that explain meaningful
relationships. Avoid filler and restating straightforward signatures. Introduce
unfamiliar concepts before relying on them; use headings when they help readers
navigate, without imposing a template on every module.

Start Rust doc blocks with a brief summary and separate further details with a
blank line. Wrap prose at roughly 80 characters, use backticks for code names,
and use conventional Markdown heading levels. Prefer executable examples with
assertions; hide setup only when it distracts from the documented use.

Use Rustdoc intra-doc links for API items and relative links in repository
guides. Link to named items, files, or section anchors, not line numbers. Define
mathematical notation locally and ensure it renders in the supported renderer;
keep plain code comments readable without mathematical rendering.

## Review

Read documentation alongside the implementation and referenced definitions.
Check that the claims are accurate, useful to the intended audience, and no
stronger than the code and validation support. Test consequential claims about
validation, mutation, ownership, and reachability through the public interface,
including re-exports, trait bounds, macro expansion, and supported features.
Remove obsolete material and unnecessary duplication. Run the relevant examples
and the documentation checks in
[CI](../.github/workflows/ci.yml); warnings must fail the documentation build.
