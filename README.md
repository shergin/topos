# topos

An autodiff compiler stack in Rust. Record a graph, inspect
it, differentiate it, compile it, emit it. The spec is an immutable
value and the state is yours. The network never moves; a training
step is a pure data transform of caller-owned parameters.

The interpreter is the executable spec; everything faster must
match it, bit for bit by default. The stack stays small enough to
read. Every result is provable.

A typical compiler rewrites the program until it can run. Topos
does not. The tape is written once; everything after it is a named
way of reading the same spec. Fusion is an offer, not a rewrite.
Emission writes the plan as text for an industrial backend — a
sibling of `describe`, not a second compiler.

The design is in [`docs/vision.md`](docs/vision.md). Constraints
it assumes live in [`docs/principles/`](docs/principles/).

## Record and train

```rust
use topos::{Keep, Tape, Tensor};

let (network, [w, x, y, loss]) = Tape::record(|tape| {
    let w = tape.parameter(0.0_f64);
    let x = tape.input(0.0);
    let y = tape.input(0.0);
    let error = w * x - y;
    [w, x, y, error * error].keep()
});
let mut parameters = network.parameters();

let samples = [(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
for step in 0..100 {
    let (sample_x, sample_y) = samples[step % samples.len()];
    let run = network.forward(
        &parameters,
        [(x, sample_x.into()), (y, sample_y.into())],
    );
    let gradients = run.backward(loss).parameters(&parameters);
    parameters = parameters.step(&gradients, |w, g| {
        w.clone() - g.clone() * Tensor::from(0.02)
    });
}

assert!((parameters.of(w).scalar() - 2.0).abs() < 1e-6);
```

The graph is recorded once. Every step feeds a sample and steps
the parameters. Training never touches the network.

## Inspect, compile, emit

```rust
use topos::{Keep, Tape};

let (network, [loss]) = Tape::record(|tape| {
    let w = tape.parameter(1.0_f64);
    [w * w].keep()
});

println!("{}", network.describe());
let plan = network.entry([loss]).lower();
println!("{}", plan.emit_stablehlo().expect("every operation lowers"));
```

Two rustdoc maps, nothing moves:
[`topos::model`](https://docs.rs/topos/latest/topos/model/index.html)
to train,
[`topos::compiler`](https://docs.rs/topos/latest/topos/compiler/index.html)
to inspect and emit.

## As a dependency

```sh
cargo add topos
```

Write networks, losses, optimizers, and element types against the
public surface. A hand-rolled layer behaves identically to a
facade; a custom optimizer has the same standing as Adam. The
opcode set, fusions, and backends stay in the crate — the core is
closed on purpose.

Opt-in backends (`accelerate`, `simd`, `metal`, `cuda`) are
documented in [`docs/acceleration.md`](docs/acceleration.md).
Vocabulary: [`docs/terminology.md`](docs/terminology.md).
Notebooks: [`docs/notebooks.md`](docs/notebooks.md). API:
[docs.rs/topos](https://docs.rs/topos).

## Examples

[`examples/`](examples/) runs from a scalar chain to a transformer:

- [`gradient_descent`](examples/gradient_descent.rs) — one spec, many states
- [`makemore/`](examples/makemore/) — Karpathy's classroom, through
  facades, compiled plans, and StableHLO
- [`mnist/`](examples/mnist/) and [`cifar10/`](examples/cifar10/)
- [`gpt2/`](examples/gpt2/) — 124M, recorded from the public op surface
- [`llama/`](examples/llama/)

## The name

A topos is a place — here, one where the whole compiler stack
stays in view.

## License

Licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE), at your option.
