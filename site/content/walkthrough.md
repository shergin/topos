# The stack, read six ways

Topos is an autodiff compiler, and both halves of that phrase show up on the smallest graph that still has something to say: a matrix product, a `tanh`, and a loss. The loss is one number that says how wrong the output is; training pushes it down, and the gradient — how much the loss moves when each weight moves — is what tells training which way to push.

Recording writes the graph down once, as a list of operations. That list is the tape, and once sealed it is the spec: the one definition of what the program means. Everything after that is a different way of reading the same list, and this page prints each reading in turn. Nothing is rewritten between them; the spec never changes. The whole chapter is also one program, [`examples/walkthrough.rs`](../examples/#walkthrough): `cargo run --example walkthrough` prints every plate below.

```rust
use topos::{Tape, Tensor};

let tape: Tape<f32> = Tape::new();
let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4]));
let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
let hidden = input.matmul(weights).tanh();
let loss = (hidden * hidden).sum();
```

`weights` is a parameter, the thing training will change; `input` is fed per run and carries a default, which this page uses throughout.

## 1. The spec

One line per recorded operation: its number, its opcode, the numbers of the operands it reads, and its shape. Shapes were checked as each line was written; a `[1, 2]` by `[3, 3]` product would have panicked at the line that tried to record it, with both shapes in the message.

{{figure walk_spec}}

{{figure walk_graph nocode}}

## 2. The derivative, as more spec

A `Value` borrows the tape and cannot outlive it. A `Symbol` is the detached name every later phase speaks, and `differentiate` takes symbols (or values, which convert): the target, and the names to differentiate with respect to.

```rust
let adjoints = tape.differentiate(loss, [weights]);
```

The chain rule is applied by the same rules the interpreter's reverse scan uses; here they record instead of compute. The gradient becomes ordinary nodes: printable, schedulable, emittable, and differentiable again.

{{figure walk_derivative}}

How to read the appended lines:

- a Leaf holding one is the seed, spread by `Broadcast` over the shape the `Sum` reduced;
- `hidden * hidden` gets the product rule once per operand — the two `Mul 9, 3` lines;
- `1 - hidden * hidden` is the derivative of tanh — the `Leaf`, `Mul 3, 3`, `Sub`;
- the matmul rule pushes the result back as two transposed products, one per operand: `Permute` then `MatMul`, twice;
- each Leaf-then-Add pair starts a gradient at zero and adds one consumer's share. That is how a value used twice collects both contributions, and it is stated once, in the engine, not in each rule.

{{figure walk_derivative_graph nocode}}

Sealing turns the tape into an immutable `Network`. From here on nothing writes the graph: training only replaces the caller-owned `Parameters`, which start from the recorded initial values.

```rust
let (hidden, loss) = (hidden.symbol(), loss.symbol());
let network = tape.into_network();
let parameters = network.parameters();
```

## 3. The plan

An entry declares what a run may read: its roots — the loss and the gradient, in that order, which is what `adjoints.roots()` lists — plus any interior we also want to look at. Reading an undeclared interior panics. That declaration is the license the scheduler works under: the declared must survive; the undeclared may be skipped, freed, or fused.

```rust
let plan = network.entry(adjoints.roots()).observe([hidden]).lower();
println!("{}", plan.describe());
```

Each line of the plan also says what happens to the value once computed: `kept` for reading, or `freed` after its last use. The last lines are the plan's own memory accounting: the most elements alive at once, against the total if nothing were ever freed.

{{figure walk_plan}}

Four nodes are missing: 21, 22, 25, 26. The rules also produced the gradient with respect to the input; nobody declared it, so it is never computed. Fusion is an offer the catalog makes for shapes it recognizes — windows, batch-norm — and a dense layer this small runs as primitives, so no patterns were elected here. The [plans page](../guide/plans/) has a convolution that does elect two.

{{figure walk_skipped nocode}}

{{figure walk_live}}

## 4. The plan as StableHLO

Emission is `describe` in another syntax: the same schedule, written as text an industrial compiler accepts. Nothing XLA-shaped lives in the crate; the text is the boundary.

```rust
println!("{}", plan.emit_stablehlo().expect("every operation in this graph lowers"));
```

{{figure walk_hlo}}

The function takes the parameter and the input, in recording order, and returns exactly what the entry declared, in that order: the loss, the gradient, then `hidden`. The four skipped nodes are not in the text either.

## 5. Two runs that must agree

The interpreter over the whole spec is the oracle. Its bits are the same in every build, on every platform, and they define what the answer is. Its reverse scan applies the derivative rules directly, without recording.

```rust
let oracle = network.forward(&parameters, []);
let engine_gradient = oracle.backward(loss).parameters(&parameters);
```

A plan under `Exact` must reproduce those bits. Under `Fast`, the default, a backend feature may serve large products with sums in a different order; that difference is a labeled choice on the entry, never a silent effect of a feature flag.

```rust
let declare = || network.entry(adjoints.roots()).observe([hidden]);
let exact_run = declare().numerics(Numerics::Exact).lower().forward(&parameters, []);
let fast_run = declare().lower().forward(&parameters, []);
let recorded_gradient = exact_run.recorded_gradients(&adjoints);
```

{{figure walk_runs}}

A few steps of gradient descent close the loop. The plan was lowered once and holds no state: each step feeds the current parameters, reads the recorded gradient, and mints the next parameters. The network never changes, and a clone of the state is all a what-if costs. Nothing broadcasts by accident, so the scalar learning rate is spread over each gradient explicitly.

{{figure walk_descent}}

## 6. The spec is executable

The printed lines are an instruction list, and walking them is the interpreter: sources take their stored payloads, computed nodes express their opcode over operands computed earlier. `Opcode::express` is public, so the walk can be written outside the crate — and it runs under the exact posture, as the whole-spec oracle does by construction.

{{figure walk_replay}}

That walk is the seam a new AD mode plugs into. Replace `Tensor` with a payload that carries a value and a tangent, and the same loop computes a directional derivative; [`examples/forward_mode.rs`](../examples/#forward_mode) does exactly that and grades it, bit for bit, against the reverse scan. [AD as a named reading](../openings/ad-as-a-reading/) is the story.

## Run it

```sh
cargo run --example walkthrough
```

Then change the graph — a second layer, a `log_softmax`, a bigger batch — and read the six plates again. Or do the same in the [playground](../playground/) without leaving this site.
