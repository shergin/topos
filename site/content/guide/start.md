# Getting started

## Install

```sh
cargo add topos
```

Rust 1.88 or newer. The default build has `#![forbid(unsafe_code)]` and a handful of small, boring dependencies: `cow_vec`, `libm`, `smallvec`, `static_assertions`, `thiserror`. Everything the crate exists to do — the tape, the derivative rules, the tensor math, the kernels, the optimizers — is its own code. The backends (`accelerate`, `simd`, `metal`, `cuda`) and the notebook cards (`evcxr`) are opt-in features, named later on this page.

Two rustdoc maps cover the surface, and nothing moves between them: [`topos::model`](https://docs.rs/topos/latest/topos/model/index.html) to record, train, and checkpoint; [`topos::compiler`](https://docs.rs/topos/latest/topos/compiler/index.html) to inspect, lower, emit, and extend. `use topos::Tape` works either way.

## The first tape

A tape is a notebook of operations. Every expression over a `Value` appends one line: its opcode, the operands it reads, and its shape, checked at that line.

```rust
use topos::Tape;

let tape: Tape<f64> = Tape::new();
let w = tape.parameter(0.0);
let x = tape.input(0.0);
let y = tape.input(0.0);
let error = w * x - y;
let loss = error * error;
println!("{}", tape.describe());
```

{{figure start_spec}}

Read the columns left to right: the node's index, its opcode, the indices it reads, its shape. A rank-0 shape prints as `[]`: a scalar is a tensor with no axes, and the graph is always tensors. `error` is node 4, and node 5 reads it twice — the square is `Mul 4, 4`. The three sources have no operands: a parameter is trainable, an input is fed per run and carries a default, and a leaf (not used here) is a constant.

{{figure start_graph}}

## A gradient

`Tape::record` is the shorthand for the block above: the closure records, its return value names what leaves the tape, and the seal happens on the way out. What comes back is a `Network` — the sealed spec — and the detached names, called symbols.

```rust
use topos::{Detach, Tape};

let (network, [w, x, y, loss]) = Tape::record(|tape| {
    let w = tape.parameter(0.5_f64);
    let x = tape.input(2.0);
    let y = tape.input(4.0);
    let error = w * x - y;
    [w, x, y, error * error].detach()
});
let run = network.forward(&network.parameters(), []);
let gradients = run.backward(loss);
println!("{}", gradients.of(w));
```

`forward` evaluates every node and hands back a `Run` that owns the values. `backward` is the engine reverse scan: from one rank-0 target, back through every node, applying each opcode's derivative rule and summing the contributions of a value with several readers. It answers a `Field` — one payload per node — read through symbols.

{{figure start_gradient}}

## A training step

Training never touches the network. `network.parameters()` materializes the recorded initial payloads as a `Parameters` table; `step` is a pure data transform of that table, with the gradients projected onto the same slots. The loop below is the crate's README, and the whole model.

```rust
use topos::{Detach, Tape, Tensor};

let (network, [w, x, y, loss]) = Tape::record(|tape| {
    let w = tape.parameter(0.0_f64);
    let x = tape.input(0.0);
    let y = tape.input(0.0);
    let error = w * x - y;
    [w, x, y, error * error].detach()
});
let mut parameters = network.parameters();

let samples = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
for step in 0..100 {
    let (sample_x, sample_y) = samples[step % samples.len()];
    let run = network.forward(&parameters, [(x, sample_x.into()), (y, sample_y.into())]);
    let gradients = run.backward(loss).parameters(&parameters);
    parameters = parameters.step(&gradients, |w, g| {
        w.clone() - g.clone() * Tensor::from(0.02)
    });
}
assert!((parameters.of(w).scalar() - 2.0).abs() < 1e-6);
```

The second argument of `forward` is the feeds: this run's values for the inputs, overlaying the recorded defaults without touching the spec. Any number of threads can forward one shared network on different feeds, or different parameters, at once.

{{figure start_train}}

## The other readings

Everything after the seal is a different way of reading the same six lines. An *entry* declares what a run may read — here just the loss — and lowering it gives a *plan*: the same operations, scheduled, with what happens to each value once computed.

```rust
let plan = network.entry([loss]).lower();
println!("{}", plan.describe());
```

{{figure start_plan}}

And the plan writes itself as text for an industrial compiler. Nothing XLA-shaped lives in the crate; the text is the boundary.

```rust
println!("{}", plan.emit_stablehlo().expect("every operation lowers"));
```

{{figure emit_chain}}

Both readings, and four more, are walked in order on [the next page](../../walkthrough/) over a graph with a matrix product in it. Or skip ahead and [type your own](../../playground/).

## What is in the box

| you want to | reach for |
|---|---|
| record a graph | `Tape`, `Value`, `Tape::record`, `Detach` |
| name things after the seal | `Symbol`, `Value::symbol`, `Tape::resolve` |
| run it | `Network::forward` (the exact oracle), `BoundEntry::interpret` |
| a gradient | `Run::backward` (engine scan), `Tape::differentiate` (as more spec), `Tape::vjp` (any seed) |
| train | `Parameters::step`, `Run::recorded_gradients`, `Sgd`, `Adam`, `AdamW` |
| layers | `Linear`, `Mlp`, `Conv2d`, `max_pool`, `BatchNorm`, `LayerNorm`, `RmsNorm`, `Dropout`, `Embedding`, `scaled_dot_product`, `cross_entropy`, `init` |
| compile | `Network::entry` → `Entry` → `lower` → `Plan`, `Numerics::{Fast, Exact}` |
| emit | `Plan::emit_stablehlo` |
| inspect | `describe` on tape, network, and plan; `Network::nodes`, `Node`, `Opcode` |
| bring a number | `Element` = `Differentiable` + `Elementary`; `Bf16` is built in |

The features:

| feature | adds |
|---|---|
| `simd` | portable CPU microkernels for dense products, every platform — [acceleration](../acceleration/) |
| `accelerate` | Apple's AMX/SME through BLAS, vForce maps, vDSP batch-norm (macOS) |
| `metal` | the crate's own GPU kernels for large `f32` products and maps (macOS) |
| `cuda` | cuBLAS through `dlopen`, so a missing toolkit does not fail the build (Linux) |
| `evcxr` | rich cards in an Evcxr notebook — [notebooks](../notebooks/) |

Features change speed, not which programs you can write. Whatever a backend declines, the interpreter runs.

## Where next

- [The stack, read six ways](../../walkthrough/) — one graph with a matrix product, every reading in order.
- [Recording](../recording/) — shapes at record time, values and symbols, sealing and reopening.
- [The gallery](../../examples/) — every example with its output, from a scalar chain to a transformer.
- [The playground](../../playground/) — your graph, in the browser, read live.
