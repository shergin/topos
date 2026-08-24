# The element is the seam

The graph is always tensors. A scalar is rank 0. The open plug is
a number, not a second engine.

## Why

A stack that grew up teaching on scalars is tempted to keep two
graph kinds: a scalar engine for the page, a tensor engine for
the models, and a trait wide enough that both inhabit it. The
scalar impls then spend their lives asserting rank, ignoring
shape arguments, and reimplementing `unfold` as a no-op. The
core gets harder to read than the math requires.

The named seam people should plug is a new number — a research
float, an interval, a dual — not a reimplementation of windowing.
If the engine is generic over "payloads that might be tensors,"
every new number pays the tensor tax again.

## The idea

One graph kind. Every node is a tensor. Rank 0 is a scalar: the
shape column already says so, and the payload does not need a
second sort. Recording can still *look* scalar — a rank-0
literal, arithmetic that reads like a line — because a scalar is
a tensor, not because the engine is.

The type every public phase is generic over is the *element*:
arithmetic, the identities, the accumulator, the elementary maps,
and the optional hooks a backend may offer. Implement those on a
number and the tensor machinery, the derivative rules, and the
engine come along. A new element never reimplements `unfold`.

The recordable vocabulary is not a payload kind. It is the opcode
set, with exactly two readings: compute over tensor buffers, or
record the same operations as nodes. Forward math is inherent to
tensors; the rules never need it. That cut is what lets a new
element inherit reverse mode without writing a rule.

Backends, emission, and notebooks already speak an element
contract — precision, literals, displays. The seam is that
contract, named once.

## Consequences

- There is no scalar graph. Teaching examples are rank-0 tensors.
- Plugging the seam is implementing a number. It is not an
  engine fork and not a new opcode.
- Shape lives on the node, inferred at record time. The element
  does not carry a second shape.
- Transcendentals and comparisons that the maps need live on the
  element. Windowing, gathers, and folds do not.
- An element that does not hook a backend runs on the
  interpreter. That is the default, not a failure.
- Emission and display are further contracts on the same number,
  not a second payload axis.

## Not this

- A parallel scalar IR "for pedagogy."
- Making the recordable vocabulary a thing a new float must
  implement method-by-method.
- Fused executors on the element. Those are plan-tier kernel
  faces, not arithmetic.
- Treating layout, storage, or rank as part of the number.

See [Vision](../vision.md) for named seams, and [Recorded reverse
mode](recorded-reverse.md) for the two readings of the recordable
vocabulary.

## Spelled today

`Tape<E>` / `Network<E>` for `E: Element`. `Element` is
`Differentiable` plus `Elementary`. Built-ins: `f32`, `f64`,
`Bf16`. `Tensorial` is the recordable vocabulary (`Tensor`
computes, `Trace` records). `Emittable` and `Sample` are further
element contracts. `reference` is how an out-of-tree element
grades a kernel. This section may rot; the rest must not.
