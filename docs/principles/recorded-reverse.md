# Recorded reverse mode

Reverse mode is a transform of the spec, not an engine procedure
that forgets the math.

## Why

A reverse scan over payload buffers computes gradients and throws
the computation away. There is nothing to inspect, compile, emit,
or differentiate again. Training then needs a special memory
posture — retain everything the scan will reread — and higher-order
gradients need a second feature.

Worse, a transform that *re-states* the derivative rules forks
them. The fork drifts: the oracle and the compiler silently
disagree, and the disagreement is not one assert away from proof.

## The idea

Derivative knowledge lives in exactly one place, written against
the recordable vocabulary, not against numbers. That body has two
readings. One computes: the engine reverse scan is the oracle.
The other records: the same scan, at construction time, with
handles in place of buffers, appends ordinary nodes.

The rules cannot tell which reading they are under. Interpretation
and transformation are two payloads of one rule. A rule changed in
one world cannot diverge in the other, because there is only one
rule.

The gradient subgraph is just more spec. It is readable, prunable,
plannable, emittable, and — because it is made of ordinary
differentiable nodes — a valid input to the transform itself.
Higher-order gradients are not a feature to build; they are the
absence of a wall.

Compiled training is then an ordinary forward plan over that extra
tape. Fusion and liveness apply to the chain rule itself. The
engine reverse scan remains the oracle the recorded form is proven
against, bit for bit. A request that merely *retains* buffers for
that scan is a memory posture, not a second compiler.

The instruction set must close under adjoints: every expansion a
rule speaks must itself have a rule. That closure is the load-bearing
theorem, not an implementation trick.

## Consequences

- Changing a derivative rule changes the oracle and the recorded
  form together, or it is not a rule change.
- A compiled plan over recorded adjoints reproduces the engine
  reverse scan bitwise: same seed, same accumulation, same
  ancestor mask.
- Training through a forward plan over extra spec is the compiled
  path. The engine scan is how you prove it, not how you ship it.
- Differentiating a recorded gradient is ordinary recording. There
  is no higher-order mode.
- An instruction that a rule speaks, and that no composition of
  the rest reproduces in bits, belongs in the vocabulary. Accuracy
  lives there; speed lives in the catalog. See
  [What earns an instruction](vocabulary.md).

## Not this

- A derivative DSL with two interpreters. The vocabulary already
  is the language.
- Treating the engine reverse scan as the compiled training path.
- Adding retention or rematerialization as a third compiler.
- Silently swapping a stable primitive for a composition that
  does not preserve bits, then calling the result a rewrite.

## Spelled today

`Tape::differentiate` / `Tape::vjp` append nodes and answer
`Adjoints`. `Run::backward` is the oracle. `Trace` is the
recording payload of `Recordable`; `Operation` is the one rule
body, and `Opcode::express` / `Opcode::vjp` are its public
surface — how a new AD mode plugs in as a payload, proven by the
forward-mode example. `Entry::backward` is the retain-buffers
posture. `Run::recorded_gradients` is the bridge from recorded
adjoints to `Parameters::step`. This section may rot; the rest
must not.

See [AD as a named reading](../openings/ad-as-a-reading.md) for
what this decision opened.
