# Dispatch System Design Ethos

*Captured during the dispatch refactor conversation, February 2026.*

---

**Design from semantics outward.** Every decision starts with "what does this concept actually mean?" — not "what's the easiest way to implement it." Tags are coordinates, coordinates have one value per dimension, therefore unique types is a requirement, not an optimization. Axes are unordered sets of dimensions, therefore positional encoding is an implementation detail to be hidden, not a feature to be exposed. We're willing to spend significant engineering effort to make the implementation match the concept, because mismatches between semantics and representation are where bugs and confusion breed.

**Be deeply skeptical of accidental ordering.** This runs through the entire design — tag ordering, axes ordering, tuple argument ordering, function argument ordering. Any place where the system imposes an order that the user didn't ask for is design debt. Einstein notation is evidence: smart people have known for over a century that forced ordering is a notational liability. The system should be honest about what's ordered (operands, which have physical meaning in sequence) and what isn't (dispatch coordinates, which are a bag of independent selections).

**Value call-site readability above implementation elegance.** The `dispatch_set{...}` curly-brace insight is characteristic: we're willing to build internal machinery (type-directed matching, compile-time sorting) specifically to make the call site parse correctly *at a glance by someone who doesn't know the framework*. Clever internal design that produces confusing call sites is worse than no design at all. The implementation can be sophisticated as long as the interface looks obvious.

**Strong aversion to global registries and coordination requirements.** The reluctance around central registration files, the search for something co-located and self-documenting — this reflects a belief that a design should be locally comprehensible. Adding a new axis type shouldn't require knowing about a central file somewhere. We accept the cost of a trait specialization only when it's co-located with the type it describes and self-evidently necessary.

**Test ideas by adversarial self-debate.** Propose and then dismantle approaches from past experience. Ask whether complexity is justified by concrete examples. Don't be attached to ideas — be attached to finding the right answer, and use past mistakes as evidence.

**Think in terms of separation of concerns — specifically the MLIR-influenced split between configuration and computation.** Attributes (dispatch coordinates) are identified by type, unordered, and resolved before invocation. Operands are positional and passed to the resolved function. Select and invoke are separate. This comes from seeing real code where the two are tangled and recognizing that the tangle prevents useful patterns (fallback logic, device guards, error handling between selection and invocation).

**Prefer functional purity as a default.** Single-input/single-output, no void functions, referential transparency. CUDA forces impurity at the kernel boundary, but hide that immediately and present a clean functional interface above it. This isn't dogma — it's pragmatism about composability.

**No macros.** Stated as a hard constraint. Rather build more complex template machinery than introduce a preprocessor dependency. Macros are non-local, non-composable, and invisible to the type system. They violate every principle we care about.

**The overall ethos: make the type system encode the semantics, hide implementation accidents, and optimize for the person reading the code at the call site three years from now.** This is infrastructure, and infrastructure outlives its authors. The extra engineering cost of getting the abstractions right is amortized over every future user and every future maintainer — including future us.
