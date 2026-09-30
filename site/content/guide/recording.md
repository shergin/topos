# Recording

## The tape

A `Tape` is the construction phase: an append-only list of operations, the Wengert list of 1964. Expressions record through `Value` proxies — thin, `Copy`, borrowing the tape — and every operator appends one node. The tape is the only phase with interior mutability; after the seal nothing writes the graph.

```rust
use topos::Tape;

let tape: Tape<f64> = Tape::new();
let constant = tape.leaf(3.0);
let weight = tape.parameter(0.5);
let sample = tape.input(1.0);
let activation = (weight * sample + constant).tanh();
```

Three kinds of source, and they are the only nodes with no operands:

- a **leaf** is a constant supplied at recording time;
- a **parameter** is trainable — the node holds a slot, and live payloads live in the caller's `Parameters`;
- an **input** is fed per run; an unfed input uses the recorded default.

{{figure rec_sources}}

The element type is the tape's type parameter: `Tape<f64>`, `Tape<f32>`, `Tape<Bf16>`, or [a number of your own](../elements/). A bare `0.5` is a rank-0 tensor; the graph is always tensors, and a scalar is the case with no axes.

## Shapes, at the line that records them

Every node's shape is inferred when it is recorded and printed on its line. A matrix product checks its inner extents; an elementwise operation requires equal shapes; a reduction says which axis it removed. Nothing broadcasts by accident: spreading a bias over a batch is an explicit `broadcast_along`, and the spec shows the axis and extent it inserted.

{{figure rec_shapes}}

A shape that does not fit panics at the expression that records it, with both shapes in the message. There is no deferred error, no later pass that discovers the problem: the line of Rust that wrote the bad node is the line in the backtrace.

{{figure rec_mismatch}}

## Values are copies

A `Value` is never consumed by an operator, so one value can feed any number of expressions. The graph records every read; a picture of it shows a node with several outgoing edges, and the reverse scan will later sum the contributions that flow back along them.

{{figure rec_shared}}

The arithmetic operators, `matmul`, the maps (`tanh`, `exp`, `ln`, `sqrt`, `sin`, `cos`, `erf`, and the rest), the shape operations (`broadcast`, `reshape`, `permute`, `narrow`, `pad`, `unfold`, `fold`), `gather` and `scatter`, and the two fused log-domain primitives `log_softmax` and `logsumexp` are the closed vocabulary. Everything else on a `Value` — `relu`, `softmax`, `transpose`, `mean_along`, `gelu` — is a composition of those, and the dump shows the composition. [What earns an instruction](../../principles/vocabulary/) is the rule that keeps the set closed.

## Names leave as symbols

A `Value` borrows the tape and dies with it. To speak about a node after the seal — read it from a run, step it in a table, differentiate with respect to it — take its `Symbol`: a detached, copyable name that is nothing more than a family token and a position. Nodes never move, so a symbol taken before a reopen still names the same node afterwards.

{{figure rec_symbols}}

`Tape::record` folds the two steps together: the closure records, its return value is the set of values that should leave as symbols, and `detach` converts them — a single value, an array, a tuple, a `Vec`, or an `Adjoints`.

```rust
use topos::{Detach, Tape};

let (network, [w, loss]) = Tape::record(|tape| {
    let w = tape.parameter(1.0_f64);
    [w, w * w].detach()
});
```

What construction detaches is the vocabulary later phases may mention. It is not a run's keep-set; [an entry](../plans/) declares that separately. You may name a sampling head you never compute on the training entry.

## Sealing and reopening

`into_network` consumes the tape and answers the sealed spec: structure, shapes, initial payloads, input defaults — the whole architecture, runnable standalone, with no live weights and no lock. `Network` is deliberately not `Clone`; share it with `&Network` or `Arc<Network>`. `into_tape` consumes the network and reopens it. Both consume, so the spec's history is linear by ownership: two divergent futures of one prefix cannot be constructed.

{{figure rec_reopen}}

Reopening never moves a node, so plans and runs derived from the old prefix stay valid, and `Parameters::carried` keeps trained payloads across the reopen while filling the new slots from their initials. This is how the notebook grows a graph cell by cell, and how the language-model examples record a decode step beside a full-context one.

## Reading the dump

`describe` on a tape, a network, or a plan prints the same column format: index, opcode, operands, parameters, shape — and, on a plan, liveness. The plates on this site are that text, with every operand reference linked: hover a line to see what it reads and who reads it. Node numbers are stable for the life of the graph, so a number on one page is the same node on another.

The dump is also executable IR. `Network::nodes` walks it as `Node` values with a public `Opcode`, and `Opcode::express` runs one instruction over any payload that speaks the recordable vocabulary. [The walkthrough](../../walkthrough/#6-the-spec-is-executable) replays a whole spec by hand that way.

## Where next

- [Differentiation](../differentiation/) — the reverse scan and the recorded transform.
- [Entries and plans](../plans/) — what a run may read, and the schedule that licenses.
- [Names](../../principles/names/) — why a symbol is thin, and what it is not.
