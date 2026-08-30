# Fusion raises

A typical compiler rewrites the program until it can run.
This one does not. The tape stays the spec. A faster form
is an offer a run may take. It never edits the graph.

That sounds like a refusal to go fast. It is why another
compiler still sees a convolution.

## The decision

Write convolution as ordinary ops: pad, unfold, permute,
reshape, matmul. A fused kernel has to earn its place —
wrong float math, or real cost. Accuracy lives in the
instruction set. Speed lives in a catalog of matches. See
[what earns an instruction](../principles/vocabulary.md).

The matcher looks at the graph, not at who wrote it. A
hand-rolled chain and a library call that recorded the
same ops look the same. Matches are gathered once. Each
consumer picks the ones it can use. A region nobody picks
just runs the original ops. A value you asked to observe,
inside a match, blocks the match — unless the match names it
as a result and writes it back, which is how observing
batch-norm's statistics survives the fusion.

## What it opened

The same match has two jobs. At home it calls a fused
kernel. When we print the plan for another compiler, it
becomes that compiler's named op.

The im2col-and-multiply chain becomes
`stablehlo.convolution`. A max-pool window becomes
`reduce_window`. The batch-norm formulas become
`batch_norm_training` and `batch_norm_inference`.

It looks backwards: the library knew "convolution,"
recorded the pieces, and the matcher finds convolution
again. That is how a small instruction set still hands
an industrial backend the op it is good at. The other
path is a new instruction for every fused form, each
with its own backward rule.

The tape is still the spec. A raise, like a fusion, is
an offer. Observing an unnamed piece of the chain refuses
both; a named result is kept and written back instead.

## Spelled today

`Plan::patterns` is the matches this plan took.
`PatternKind` names the ones we know: window-times-multiply,
window-reduce, the two batch-norms. Home fusion calls a
kernel. `Plan::emit_stablehlo` writes the named op. Same
match. `describe` prints the choice.

This section may rot; the rest must not.

## Not this

- Rewriting the tape into fused opcodes. Speed is not a
  new instruction.
- A raise that hides a value you asked to observe.
- A `Convolution` instruction because a layer has that
  name. Convolution is a formula.

See [Vision](../vision.md) for fusion as an offer, [What earns
an instruction](../principles/vocabulary.md) for why the
fused form is not an opcode, and [Several exports, one
spec](several-exports.md) for the read-list the matcher
runs on.
