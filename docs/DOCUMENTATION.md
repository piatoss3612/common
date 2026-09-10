# Documentation

Documentation explains an API's purpose and use. Readers need context that a
signature alone cannot provide. Supply that context by introducing the problem,
its solution, and the API's role in that solution, with links to the definitions
it relies on.

Use this guide when writing or reviewing documentation. It covers Rust doc
comments (`///`, `//!`), module documentation, and code comments (`//`). General
writing rules also apply to READMEs and other prose guides. Formatting rules
specific to Rustdoc are identified below.

## Audience and contract

Write for the users of the API being documented. Public-facing crate and item
documentation describes the supported abstraction, how to use it, and the
behavior callers can rely on. Treat statements in public documentation as part
of the API contract. Omit internal implementation details from public
documentation.

Internal APIs can document implementation details for their users. Crates such
as `bento-core` are internal even though their items use `pub`. Internal module
and item documentation can explain algorithms, representations, invariants, and
the decisions their maintainers need to understand. Keep local implementation
reasoning beside the relevant code.

Public module documentation should name only public API items and link to
them. If an item cannot be linked as part of that API, omit it. Internal module
documentation may discuss and link the internal API it serves.

Check where documentation reaches readers. Documentation on an internal item
may also appear through a public facade's re-export. Review that presentation
against the public API's intended contract. Linking from public documentation
to an internal implementation is not a substitute for documenting the public
abstraction.

For example, the public `addition_chain!` documentation can explain scaling and
its support interface. Planner window widths, tie-breaking, and temporary
bindings belong with the internal planner and expansion code.

Use `///` and `//!` for an API's purpose, motivation, usage, and invariants.
Use `//` for algorithm steps, optimization rationale, and non-obvious behavior
within an implementation. Internal API documentation can explain the design
needed to use that API; comments beside its implementation explain local
mechanics. Omit obvious optimizations from both.

## Introductions

Introduce each subject in this order:

1. Briefly state what it does.
2. Explain the problem it addresses.
3. Explain the solution to that problem.
4. Describe how this API realizes the solution more specifically.

Scale the explanation to the subject. A simple operation may need only one or
two sentences to cover this sequence; a complicated API may need paragraphs and
an example. Do not manufacture a problem or add filler to satisfy the sequence.
The final step in public documentation describes the API's operations and
behavior. Implementation mechanics belong in documentation for internal APIs
and in code comments.

For substantial module documentation, separate conceptual grounding into a
"Background" section and architectural choices and trade-offs into a "Design"
section. Keep both within the module's audience boundary. Explain why a design
choice addresses the problem, including the drawback of a naive approach when
that motivates the choice. Avoid recounting alternatives that add no useful
context.

When a module organizes several submodules, enumerate those available to its
audience with linked names and one-line summaries. Make dependencies on other
APIs explicit through links. Connect mathematical constructs to the concrete
API items or, for internal documentation, code paths that implement them.

## Signatures and semantics

Avoid explaining a straightforward signature. Argument and return types, trait
bounds, and guarantees already evident from the type system usually need no
prose restatement. Document the purpose of the operation and the semantics that
the signature leaves unstated.

For example, an addition-chain trait needs to explain associativity and the
relationship between doubling and addition. Repeating that its supertrait is
`Clone` contributes little when the declaration already says so.

Use judgment for complicated interfaces. Explain relationships between types
or show an example when that helps readers understand how to use the API.
Macro input syntax may also need explanation because it has no ordinary
function signature to communicate its accepted forms.

Document intended behavior. Incidental capabilities of the current
implementation do not belong in the contract. Avoid tables that merely
reformat declarations already visible in the code.

State preconditions directly. For example, "Must be smaller than `T`" is
sufficient; do not append generic warnings that violations may cause panics
or incorrect behavior. Document defined error and panic behavior when it is
part of the API. For `unsafe` items, include a `# Safety` section that states
the caller's obligations and explicitly warns about undefined behavior from
violating the safety requirements.

## Cross-references

Document a concept where it is defined and link to that explanation from its
uses. Each concept has one authoritative explanation. In Rustdoc, use intra-doc
links such as ``[`AdditionChain`]`` or ``[`AdditionChain::double`]`` when those
items are in scope. Use explicit link targets when the displayed name and the
resolvable path differ. In Markdown guides, use relative links to repository
documentation and source files.

Keep enough local context to explain why the reference matters. For example,
macro documentation can identify and link its support trait without repeating
the trait's method signatures and laws. Verify that links resolve from the
place where readers encounter the documentation, including public re-exports.

Link consistently within a doc block. Once an item is linked, link each prose
reference to it in that block. Put reference definitions such as
``[`AdditionChain`]: crate::addchain::AdditionChain`` at the end of the block.

Never refer to line numbers in documentation or code comments. This includes
line ranges, line-number URL fragments, and file references with line-number
suffixes. They become stale as code changes. Link to named items, modules,
files, or section anchors instead.

## Voice

The technical voice is **intelligent but with nothing to prove**.

Use direct sentences, concrete descriptions, and technical terms that serve
the explanation. Explain difficult ideas patiently without displaying
expertise for its own sake. Remove rhetorical flourishes, filler, promotional
language, and commentary about how clever or elegant the implementation is.
Each sentence should help the reader understand the problem, use the API, or
maintain the internal code being documented.

Write doc comments as complete sentences with proper punctuation. Describe
functions in the third-person singular: "Returns the sum." Start type
descriptions with an article: "A wrapper that..." Prefer relative clauses,
such as "A type that computes...", over "A type for computing...".

## Rustdoc and comment formatting

- Start each `///` block with a brief one-line summary. Add a blank doc-comment
  line before details when details are needed.
- Wrap doc prose at approximately 80 characters, excluding the `///` or `//!`
  prefix. Code blocks and display math may exceed that width.
- Use `#` for top-level module headings and `###` for subsections. Skip `##`,
  which is too visually similar to `#`.
- Always backtick code identifiers in prose, including headings. For example,
  write ``### The `ONE` Wire``.
- Separate adjacent documented struct fields with a blank source line between
  one field's declaration and the next field's doc block.
- Leave a blank line before a `//` comment unless it starts a block.

## Mathematics

Escape underscores in LaTeX subscripts to prevent Markdown interpretation:
write `$\mathbf{u}\_{i,j}$`. Put display math delimited by `$$` on separate
lines, with its delimiters and content separate from prose.

Avoid Unicode math symbols in `//` comments. When an explanation needs rendered
math, prefer smaller functions with doc comments that can carry that
explanation. Use KaTeX notation where the documentation renderer supports it.

For polynomial evaluation APIs, check fixed and free variables against the
signature. Uppercase variables such as `X` and `Y` denote polynomial variables;
lowercase variables such as `x` and `y` denote fixed evaluation points. Method
names list the fixed variables: `y()` fixes `y` and returns a polynomial in
`X`, while `xy()` fixes both. "Restricted to X" in documentation for `y()`
names the free variable of the result and is correct under this convention.
Do not flag it as a mismatch merely because the method is named `y()`.

## Reviewing changes

Read changed documentation alongside the API or code it describes and the
definitions it references. Check that:

- The content serves its audience wherever it is exposed.
- The introduction establishes purpose, problem, solution, and the API's role
  with detail appropriate to its complexity.
- Each explanation adds meaning beyond a straightforward signature.
- Public documentation contains only the intended public contract.
- Shared concepts use working cross-references instead of duplicated accounts.
- No reference depends on a line number.
- The prose is accurate, direct, and free of filler.
- Sentence forms, links, headings, spacing, and math follow the rules above.
