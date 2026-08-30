# Dropout as a feed

Dropout looks like it needs a random number generator in
the graph, and a switch that turns it off at inference.
Either one would make two identical runs disagree.
Neither is required.

Randomness stays on the host. Inference is the absence of
a feed.

## The decision

No `rand`, no clocks. Two identical runs cannot differ, so
a rerun is a checksum. Learning rates and keep probabilities
are chosen at the call site. Extra state lives in tables
you hold, never inside the graph.

## What it opened

The layer multiplies by a mask input. The default mask is
all ones, so a run that does not feed a mask is the
identity. That is inference.

Training draws the mask on the host, from a seeded factory,
and feeds it like a batch. Each element is zero (drop) or
`1 / keep` (keep, already scaled). The keep probability is
chosen where the mask is drawn, not stored on the layer.

The mask is data. It has no gradient of its own. Backward
is `gradient * mask`, which the chain rule already has.
Dropout did not need its own instruction.

An emitted training step gains one extra argument — the
mask — instead of a mode or an in-graph generator. Seeded
replay still matches, dropout included, because the bits
that change per step walked in through the feed table, the
same way a batch does.

We refused an in-graph generator so replay would stay
honest. Mask-fed dropout is what that made easy, once
inputs were already overlays on recorded defaults.

## Spelled today

`Dropout` holds the mask input's name. `init::dropout`
draws the mask. Unfed is inference. A transformer trains
by feeding masks on residual writes and samples by not
feeding them.

This section may rot; the rest must not.

## Not this

- A random-number opcode, or a generator inside the graph.
- A train/eval flag on the layer. Inference is the
  absence of a feed, like every other input default.
- A fused dropout instruction. The multiply is the formula.

See [Spec and state](../principles/spec-and-state.md) for
why run state is not graph state, and [Several exports, one
spec](several-exports.md) for feeding a training function
and not feeding a sampling one.
