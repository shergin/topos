# Differentiation

## Three ways to a gradient

Topos is reverse mode over tensors: one scalar target, every input, about one forward of cost. There is no target-free gradient of a network; you always say which number you mean.

The three roads, in the order the crate recommends them:

1. **`Tape::differentiate`** records the chain rule as ordinary nodes and answers `Adjoints`: the target and one `(wrt, gradient)` pair per name. The derivative becomes spec — lower a forward-only entry over `adjoints.roots()`, and fusion and liveness apply to the chain rule itself. `Run::recorded_gradients` bridges the result to `Parameters::step`. This is compiled training.
2. **`Run::backward`** is the interpreter applying the same rules without recording: the engine reverse scan. It is the oracle the transform is proven against, bitwise, shipped forever — and the quick road in a loop.
3. **`Entry::backward()`** is neither: a memory posture that retains what the engine scan reads, so a plan that did not record its derivative can still answer `backward`.

## The engine scan

`Run::backward` takes a rank-0 target and walks the run's values backward. At every node it applies the opcode's derivative rule to the cotangent that arrived, producing one cotangent per operand; a value with several readers sums what comes back.

{{figure diff_engine}}

The rules return cotangents; the scan adds them. That accumulation is stated once, in the engine, and never in a rule — which is why `a` above gets an honest zero from two contributions that cancel, rather than one of them.

## The same rules, recording

Derivative knowledge lives in exactly one place, written against the *recordable vocabulary* — the operations a tape can record — not against numbers. That body has two readings. Over `Tensor` it computes: that is the engine scan. Over `Trace` it records the same steps as ordinary nodes: that is `differentiate`. The rules cannot tell which reading they are under.

{{figure diff_recorded}}

The appended lines are readable spec. Each `Leaf` then `Add` pair is a gradient started at zero and one consumer's share added; `error` has two readers, so its gradient collects two shares. The picture makes the pattern visible.

{{figure diff_accumulate}}

Because the gradient is spec, everything downstream applies to it unchanged: dead-node elimination drops the parts an entry does not declare, liveness frees what the chain rule no longer needs, patterns fuse across the forward and backward halves, and emission writes `(parameters, batch) -> (loss, gradients...)` as one function. [Recorded reverse mode](../../principles/recorded-reverse/) is the argument for building it this way.

## One rule body, proven

Two readings of one rule cannot drift, but a claim like that is one assert away from proof, so the crate asserts it. The engine scan and the recorded gradient must agree bit for bit — same seed, same accumulation order, same ancestor mask.

{{figure diff_equal}}

## Seeds and vector-Jacobian products

A plain gradient plants a one at a rank-0 target. `Tape::vjp` takes any target and an explicit seed of the same shape: the vector-Jacobian product `Jᵀ seed`. With a one-hot seed it picks a row of the Jacobian; with a gradient as the seed it records a second derivative.

{{figure diff_vjp}}

## Second derivatives

The recorded gradient is made of ordinary differentiable nodes, so it is a valid input to the transform itself. A Hessian-vector product is `vjp` over a recorded gradient, seeded with a direction. There is no higher-order mode to switch on; it is the absence of a wall.

{{figure diff_hessian}}

[`examples/forward_mode.rs`](../../examples/#forward_mode) computes the same product the other way round — forward-over-reverse — and asserts the two agree bit for bit at dyadic values.

## The backward pass as a curve

A `Field` holds one payload per node: far too much to print and exactly the right amount to plot. The Euclidean norm of each gradient, along the tape, is the shape of a backward pass; a vanishing or exploding region is a shape, not a number to hunt for. The [notebook card](../notebooks/) for a field draws exactly this.

{{figure diff_norms}}

## Forward mode, from outside

Forward mode — one input nudge, every output — is not a library feature. It is dual arithmetic over a `Recordable` payload, replayed through the spec with `Opcode::express`: wrap every value as a pair of value and tangent, and ordinary multiplication already is the product rule. Over `Dual<Tensor>` the walk computes a directional derivative eagerly; over `Dual<Trace>` it records the tangent computation as more spec. Both are graded, bitwise, against the reverse scan.

That is the seam a new AD mode plugs into: not a second list of derivative rules, not a fork of the engine. [`forward_mode.rs`](../../examples/#forward_mode) is the exhibit, [`dual.rs`](../../examples/#dual) does the same by plugging the [element seam](../elements/) instead, and [AD as a named reading](../../openings/ad-as-a-reading/) tells what the decision opened.

## Where next

- [Training](../training/) — from gradients to steps, with the state in your hands.
- [Entries and plans](../plans/) — compiled training as a forward-only plan over the adjoints.
