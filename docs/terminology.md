# Terminology

The public vocabulary. Each entry is the literature meaning and
where it lives here. Design arguments live in
[vision](vision.md) and [principles](principles/). Decisions and
what they opened live in [openings](openings/). When a
concept is added, renamed, or changes meaning, update this in the
same change.

Two rustdoc maps, nothing moves:

- **Model** (`topos::model`) — record, train, checkpoint.
- **Compiler** (`topos::compiler`) — inspect, lower, emit, extend.

## Recording

**Tape.** The construction phase: an append-only Wengert list.
Expressions record through `Value` proxies; shapes are checked at
the recording line. `Tape::into_network` seals it;
`Network::into_tape` reopens. Both consume, so history is linear
by ownership.

**Value.** The operand of recording: a `Copy` proxy that borrows
the tape. Operators do not consume it. It cannot outlive the
seal; take a `Symbol` first.

**Symbol.** A detached name: family plus position. The currency
of every phase after recording. Reads go through
`Parameters::of`, `Run::of`, `Field::of`.

**Detach.** How names leave construction: `[w, x, loss].detach()`.
`Tape::record` detaches and seals. What construction detaches is
the vocabulary later phases may mention, not a run's keep-set —
see [Observability](principles/observability.md).

**Network.** The sealed spec: structure, shapes, initials, input
defaults. No live weights, no lock, not `Clone`. Shared with
`&Network` or `Arc<Network>`.

**Parameters.** Caller-owned weights, and any other table aligned
to the same slots (gradients, moments). Born from
`Network::parameters` or a checkpoint. `step` is a pure data
transform; training never touches the network. See [Spec and
state](principles/spec-and-state.md).

**Shape.** Extent along every axis; a scalar is rank 0. Inferred
at record time; mismatches panic there. Once recorded, shapes do
not change.

**Node / Opcode.** The printable IR. `Opcode` is the closed,
payload-free instruction; `Node` is one recorded entry (symbol,
opcode, operands, shape). `describe` on tape, network, and plan
prints one line per node.

**Leaf, parameter, input.** Three source kinds. A leaf is a
constant. A parameter is trainable; the node holds a slot, live
payloads live in `Parameters`. An input is fed per run; unfed
inputs use the recorded default.

## Running

**Entry.** A function exported from a network: roots, observes,
memory posture, numerics. `network.entry([loss]).interpret(...)`
runs the interpreter over that closure under the entry's
numerics — the oracle under `Exact`; `Network::forward` is the
always-exact whole-spec oracle. `network.entry([loss]).lower()`
is the plan. One spec may export several entries that share
weights.

**Run.** One forward execution's payloads. `Network::forward`
evaluates the whole spec; an entry's `interpret` evaluates only
the declared ancestors. `Run::of` panics on a value this run did
not compute. `Run::backward` is the engine reverse scan.

**Plan.** A derived schedule: what to run, what to keep, what to
free, which patterns to elect. It freezes a prefix of the spec
and takes `Parameters` per call, so it holds no state.
`Plan::forward` matches the interpreter bit for bit wherever
only exact kernels serve. `Plan::emit_stablehlo` writes the
plan as text.

**Pattern.** A structural match over frozen columns, not a tape
rewrite. Discovery pools candidates; each consumer elects.
Fusion and emission are offers. A declared interior inside a
match is a barrier.

**Field / Gradients.** A payload per node of one family.
`Gradients` is that buffer in the role a backward sweep
produces. `Field::parameters` projects onto the parameter
slots for training.

## Differentiation

**Autodiff.** Exact derivatives by the chain rule on primitive
operations. Distinct from finite differences and from symbolic
rewriting. Topos is reverse mode over tensors (a scalar is rank
0).

**Reverse mode.** One output, all inputs, about one forward of
cost. `Run::backward` computes it; `Tape::differentiate`
records it as ordinary nodes. See [Recorded reverse
mode](principles/recorded-reverse.md).

**Gradient.** Partials of one chosen scalar target. There is no
target-free gradient of a network.

**Accumulation.** A value with several consumers sums the
incoming cotangents. Stated once in the engine: rules return
cotangents, the scan adds them.

**Seed (cotangent).** The value planted at the target. A plain
gradient uses ones. `Run::backward` requires a rank-0 target;
reduce with `sum` first. `Tape::vjp` takes an explicit seed of
any matching shape.

**VJP.** `Jᵀ seed`. `Tape::vjp` is the recorded scan;
`differentiate` is that scan with a ones seed at a scalar loss.
A seed that is itself a gradient records a Hessian-vector
product.

**Adjoints.** The transform's product: the target and one
`(wrt, gradient)` pair per entry, in `wrt` order.
`adjoints.roots()` is the training root list.

**Trace.** The payload that records instead of computing. Rules
are written against `Recordable`; `Tensor` computes, `Trace`
appends nodes. One rule body, two readings.

**Replay.** Walking a spec's nodes and expressing each opcode
over its operands' results (`Opcode::express`): over `Tensor` it
is the interpreter, over `Trace` it re-records, and any other
`Recordable` is a new interpretation of the same spec.
`Opcode::vjp` is the public name of the rule body, so a scan can
be written outside the crate with the engine as its oracle.

**JVP.** `J seed`: the directional derivative forward mode
computes. Not a library feature — dual arithmetic over a
`Recordable` replayed through the spec; `examples/forward_mode.rs`
is the worked form, graded bitwise against reverse mode.

**Operation.** A primitive's forward (inherent on `Tensor`) and
backward (the `Operation` trait over `Recordable`). Rules never
see the tape or the gradient buffer.

## Payload

**Element.** The number that fills a tensor: the open seam.
Implement `Differentiable` and `Elementary`, then `Element`.
Built-ins: `f32`, `f64`, `Bf16`. A new element does not
reimplement windowing. See [The element is the
seam](principles/element.md).

**Tensor.** The graph's only payload kind. Rank 0 is a scalar.
Immutable; views and broadcasts share the buffer.

**Map.** Unary elementwise transcendentals as one instruction
kind, carrying a `MapOperation` (`Exp`, `Ln`, `Tanh`, …). Binary
elementwise ops (`Maximum`, `Step`, `Powf`) are not maps.

**Recordable.** The recordable vocabulary: everything a
derivative rule may call, and nothing else. Every method on it
corresponds to something a tape can record, which is what lets
both `Tensor` and `Trace` implement it without a panicking
member.

**Bf16.** Brain float 16. Arithmetic in `f32`, round to
nearest-even. Accumulations stay in `f32` until the final total.

## Acceleration

**Backend.** An implementer of named formulas: hardware kernels,
the crate's fused kernels, and StableHLO emission. Opt-in cargo
features; declines fall to the interpreter.

**Numerics.** `Fast` (default) may reorder float math. `Exact`
is interpreter bits, in every build. A labeled choice on the
entry, not a feature flag.

See [Acceleration](acceleration.md).

## Facades

A facade composes through the public surface alone. A
hand-rolled equivalent behaves identically.

**Module.** A named recording function: `express(input)` records
on the input's tape and returns the output. It holds `Symbol`s,
never payloads.

**Sequential.** Stages in order, behind `Module`.

**Optimizer.** An open trait: gradients and parameters in,
next parameters out. State is more `Parameters` tables. `Sgd`,
`Adam`, `AdamW`. Learning rate is a per-step argument.

**Activation.** `Tanh` and `Relu`. Anything else is composition
on the public surface (GELU in the GPT-2 example).

**init.** Seeded shape-to-payload closures. No global generator.

**checkpoint.** Snapshot and restore of `Parameters`. Positional
or by structured path. Pure state; no graph is touched.

Layers (`Linear`, `Conv2d`, norms, dropout, losses) are the same
idea: formulas over the public ops, state in the caller's
tables.

## Further reading

- Wengert (1964) — the original tape.
- Griewank and Walther, *Evaluating Derivatives* (2008).
- Baydin et al., "Automatic Differentiation in Machine Learning"
  (2018).
- Karpathy, [micrograd](https://github.com/karpathy/micrograd).
