# Notebooks

Topos runs in [Evcxr](https://github.com/evcxr/evcxr), the Rust
Jupyter kernel and REPL. There is no wrapper API. The `evcxr`
feature adds rich cell output; everything else is the ordinary
crate.

```sh
cargo install --locked evcxr_jupyter
evcxr_jupyter --install
```

First cell:

```rust
:dep topos = { version = "0.12", features = ["evcxr"] }
use topos::*;
```

A `~/.config/evcxr/init.evcxr` saves repeating that. Turn the
compile cache on — every cell is a real crate:

```
:cache 500
:dep topos = { version = "0.12", features = ["evcxr"] }
```

## Two rules

Evcxr compiles each cell as its own crate and keeps the variables
between them.

**A persisted variable cannot borrow another one.** A `Value`
borrows the tape, so it dies with the cell. `Symbol` is the
cross-cell name. End a recording cell with `.symbol()` and come
back through `Tape::resolve`. See [Names](principles/names.md).

```rust
let tape: Tape<f64> = Tape::new();
let w: Symbol = tape.parameter(0.0).symbol();
let x: Symbol = tape.input(0.0).symbol();
let y: Symbol = tape.input(0.0).symbol();
```

```rust
let loss: Symbol = {
    let (w, x, y) = (tape.resolve(w), tape.resolve(x), tape.resolve(y));
    ((w * x - y) * (w * x - y)).symbol()
};
```

**A persisted variable needs an explicit type.** Evcxr infers a
cell from that cell alone. Annotate every binding that must
survive. This is Evcxr, not topos — it cannot infer
`let v = vec![1.0_f64];` either.

Give each binding its own `let`. A destructured tuple does not
survive the cell, even with a type on the tuple: Evcxr cannot
name a type that came from a dependency.

## Train

`into_network` consumes the tape. Evcxr tracks the move: `tape`
ends where `network` begins. Training is a data loop over
caller-owned parameters. See [Spec and
state](principles/spec-and-state.md).

```rust
let network: Network<f64> = tape.into_network();
let mut parameters: Parameters<f64> = network.parameters();
```

```rust
let samples: Vec<(f64, f64)> = vec![(1.0, 2.0), (2.0, 4.0), (3.0, 6.0)];
for step in 0..300 {
    let (sx, sy) = samples[step % 3];
    let gradients = network
        .forward(&parameters, [(x, sx.into()), (y, sy.into())])
        .backward(loss)
        .parameters(&parameters);
    parameters = parameters.step(&gradients, |p, g| {
        p.clone() - g.clone() * Tensor::from(0.02)
    });
}
```

```rust
parameters.of(w)             // trained
network.parameters().of(w)   // initials, materialized fresh
```

To record more, `into_tape` consumes the network. Symbols still
resolve. `Parameters::carried` keeps trained slots and fills new
ones from their initials.

```rust
let tape: Tape<f64> = network.into_tape();
let cube: Symbol = { let w = tape.resolve(w); (w * w * w).symbol() };
let network: Network<f64> = tape.into_network();
let parameters: Parameters<f64> = parameters.carried(&network);
```

## Cell output

With `evcxr`, the last expression in a cell draws itself instead
of dumping `Debug`:

| Ends with | Shows |
|---|---|
| `Value` / `Tensor` | shape, extremes, payload — table when small, chart when large |
| `Tape` / `Network` | the spec dump (`describe`); long dumps elide the middle |
| `Parameters` | each slot's shape, and the value when it is a scalar |
| `Plan` | the schedule, and live volume along it |
| `Field` | one Euclidean norm per node, along the tape |
| `Run` | the same profile for a forward pass |
| `Adjoints` | each `wrt → gradient` pair |
| `Entry` | roots, observes, memory posture, numerics |
| `Symbol` | what it is, and how to read through it |

Each card is a `to_html` string plus `evcxr_display`, so output
is ordinary `cargo test`, and the terminal REPL gets
`text/plain`. Charts use [malevich](https://crates.io/crates/malevich).

## Rough edges

- A shape mistake panics the cell. The session keeps its
  variables; only that cell's work is lost.
- A cell is a real compile. A small one is a fraction of a
  second; generics can take one to two. The cache helps. It is
  not Python-snappy.
