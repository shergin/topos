# AD as a named reading

You recorded a loss. You want derivatives. Automatic
differentiation walks the graph and applies the chain rule.
It does not rewrite the math by hand, and it does not poke
the inputs with a small `ε`.

Two usual walks. Reverse mode: one output, every input. That
is training. Forward mode: one input nudge, every output.
That is cheap when there are few inputs, and it is how you
build second derivatives.

Most stacks treat those walks as engine loops. Here they are
two ways of reading the same graph.

## The decision

The tape is written once. After that you only read it: run
it, differentiate it, schedule it, print it.

The local derivative of each operation lives in one place,
written against the operations a tape can record, not against
numbers. That body already has two readings. One computes.
The other records the same steps as ordinary nodes. The rules
cannot tell which reading they are under.

That is [recorded reverse
mode](../principles/recorded-reverse.md). It was taken so
training could compile.

## What it opened

The gradient graph is just more graph. You can print it,
compile it, emit it, and differentiate it again. A second
derivative is not a new feature. It is reverse mode of a
gradient, with an explicit seed.

A new AD mode is then the same kind of thing, not a new
engine. Walk the spec. Speak the same operations. Forward
mode is the exhibit. You do not write a new derivative for
every opcode. You wrap each value as a pair — the value, and
how it moves if an input nudges — and ordinary multiply
already is the product rule. Replay the spec over those
pairs. The directional derivative falls out.

Reverse-mode knowledge stays where it is. Forward-mode
knowledge is that pair arithmetic. The slow, obvious run is
still the check. A new reading has to match it, bit for bit.
The idea does not need a fork.

## Spelled today

You can print every node (`Opcode`, `Node`, `describe`).
`Opcode::express` runs one instruction over any payload that
speaks the recordable operations. `Opcode::vjp` is the public
name of the reverse-mode rule. Over tensors, a walk is the
interpreter. Over `Trace`, it records again.
`Tape::differentiate` / `Tape::vjp` are reverse mode as extra
graph. `Run::backward` is the check.

Forward mode lives in `examples/forward_mode.rs`: a `Dual`
payload, not a crate type. Computing the pairs and recording
the pairs match each other, and both match reverse mode.

This section may rot; the rest must not.

## Not this

- A second list of derivative rules for forward mode. The
  pair arithmetic is the forward-mode body.
- Dual numbers as a built-in. The opening is the seam, not
  a new type in the crate.
- Treating `Entry::backward()` as a compiler. It keeps
  buffers so the engine scan can still run. That is a
  memory choice.
- Letting a dependent register a new instruction. New
  primitives land in the crate, under [what earns an
  instruction](../principles/vocabulary.md).

See [Vision](../vision.md) for named readings, [Recorded reverse
mode](../principles/recorded-reverse.md) for the one rule body,
and [The element is the seam](../principles/element.md) for the
other plug — a new number, not a new walk.
