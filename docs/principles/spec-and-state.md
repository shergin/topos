# Spec and state

The architecture is an immutable value. The weights are yours.
Training does not version the graph.

## Why

When live weights live inside the graph, a training step must
produce a new graph. That makes update a graph operation, which
creates generations, which creates the questions "which generation
does this name bind to" and "who may extend the recording now."
Those questions are an identity protocol. Every branch, tip,
witness, and authorship posture this crate ever needed was the
cost of that glue.

If the recorded graph and the parameter payloads are one public
object, parallel what-ifs fight over who may write, and reopening
the spec is a race rather than an ownership move.

## The idea

Split them. The spec is structure, shapes, record-site initials,
and input defaults — the whole architecture, runnable standalone,
with no live weights and no lock. The state is a caller-owned
table of payloads, born from those initials or from a checkpoint,
passed into every run and every plan.

A training step is then a pure data transform of the table. No
new spec, no generation, no lead to pass. The graph never changes
when you train, so nothing about the graph needs to be
re-identified afterwards. Clone of state is honest and costs what
the weights cost; that is the whole price of a what-if. Optimizer
moments are more tables of the same shape.

The spec's history is linear by ownership. Sealing consumes the
recording phase; reopening consumes the sealed spec. Two
divergent futures of one prefix cannot be constructed. Sharing
for concurrent runs is a borrow or a counted reference, and
neither can be consumed, so sharing and extending exclude each
other without a protocol.

This is the functional-core shape. Ownership makes the half that
is usually a convention into a compile error.

## Consequences

- A plan derived from the spec holds no state. It survives every
  training step and every reopen of a prefix.
- Concurrent what-ifs copy the table, not the right to keep
  appending.
- The sealed spec is not cloned. A second sealed copy could be
  reopened into a divergent future; that is exactly what must
  stay unrepresentable.
- Construction is the only phase with interior mutability. After
  the seal, nothing writes the graph.
- Facade state — moments, velocities, running estimates — lives
  in caller-held tables, never in the graph. The engine obeys
  the same rule it demands of facades.

## Not this

- Generations of a network, and the identity machinery they
  summon.
- Update as a graph operation.
- An authorship posture that arbitrates who may record.
- Forking the spec to try a learning rate.
- Putting optimizer state in the graph so a step has somewhere
  to write.

See [Names](names.md) for why a node name is not also a writer
protocol, and [Vision](../vision.md) for the tape as spec.

## Spelled today

`Network` is the sealed spec and is deliberately not `Clone`.
`Parameters` is the caller-owned table; `Field` is the
node-aligned analogue. `Tape::into_network` / `Network::into_tape`
are the consuming pair. `Parameters::step` mints the next state;
`Parameters::carried` keeps payloads across a reopen. This
section may rot; the rest must not.
