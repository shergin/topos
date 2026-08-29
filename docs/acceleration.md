# Acceleration

Backends are opt-in cargo features. Turn one on and dispatch
happens inside the payload: nothing to call, nothing to configure.
Whatever a backend declines, the interpreter still runs. Features
change speed, not which programs you can write.

The interpreter's bits stay the spec. A faster kernel that reorders
float math is a labeled `Fast` run, never a silent effect of a
feature flag.

The tables are a snapshot from one Apple M1 Pro. They show
order-of-magnitude and which rung wins, not a promise for your
chip. Rerun on yours: `cargo bench`, the `throughput` example,
and the scripts in [`tools/`](../tools/).

## Turn it on

```sh
cargo run --release --example throughput
cargo run --release --features simd --example throughput
cargo run --release --features accelerate --example throughput
cargo run --release --features accelerate,metal --example throughput
```

There is nothing to detect. Optional loud mode, if you would
rather fail than fall back:

```rust
topos::Backend::Metal.status().expect("metal backend unavailable");
```

For `metal`, `status` also warms the kernels so the first large
product does not pay compilation.

## Which feature

| feature | speeds | where |
|---|---|---|
| default | auto-vectorized slices | everywhere |
| `simd` | dense `f32`/`f64` products | everywhere (the Linux story) |
| `accelerate` | products, maps, batch-norm | macOS; a stub elsewhere |
| `metal` | large `f32` products and maps | macOS; a stub elsewhere |
| `cuda` | large `f32`/`f64` products | Linux; experimental |

**Default.** Pure Rust. Dense products on a slice path shaped for
the auto-vectorizer; bit-identical to the naive loop.

**`simd`.** Portable CPU microkernels (`matrixmultiply`),
single-threaded, AVX-512 / AVX2 / NEON at run time. Transposed
and narrowed views run at full speed; broadcast operands decline.

**`accelerate`.** Apple's AMX/SME via BLAS, plus vForce maps and
vDSP batch-norm. Zero extra crates. On Apple Silicon it is the
usual winner for products.

**`metal`.** The crate's own GPU kernels, `f32` only. Products
often still lose to AMX, so Metal serves what BLAS declines and
leads maps once buffers are large.

**`cuda`.** cuBLAS, libraries loaded at run time so a missing
toolkit does not fail the build. Copy-bound; not yet measured on
NVIDIA hardware.

Enabling a feature is the whole activation. Selection is per
build and per task (size, precision, layout), never per call
site. `f64` never reaches Metal.

## Measured

Matrix products, 512- to 2048-square; maps are `f32 tanh` at
2M–8M elements.

| build | products | maps |
|---|---|---|
| default | 26 GFLOP/s `f32`, 13 `f64` | 0.4 Gelem/s |
| `simd` | 96 GFLOP/s `f32`, 47 `f64` | 0.4 Gelem/s |
| `accelerate` | 1.6 TFLOP/s `f32`, 550 GFLOP/s `f64` | 1.2 Gelem/s |
| `metal` | 1.4 TFLOP/s `f32` at large sizes; no `f64` | 2.2–2.7 Gelem/s above 128k |

The `simd` row is NEON on that M1; on x86 it is AVX-512 or AVX2
and should be the same order. The naive strided loop is 0.4
GFLOP/s there: the correctness anchor and the fallback for exotic
layouts.

Whole training steps, same snapshot (batch 64, `f32`):

| build | `mnist` ms/step | `cifar10` ms/step |
|---|---|---|
| default | 106.8 | 391.5 |
| `accelerate` | 82.0 (−23%) | 261.9 (−33%) |
| `metal` | 107.7 (0%) | 313.3 (−20%) |

The wins track how much of the step is a product, not the ladder
ratios: these convolutions are skinny, and window fills still
run the built-in paths. Metal's flat MNIST row is the threshold
working — every product sits below the bar, so the run matches
the default bits. Accuracies differ across `Fast` builds because
each backend sums in its own order; each trajectory is valid.

Broadcasts are views, not copies: a stride-0 window over the
source. The default build already runs them at slice speed
(bias-add 7.6 Gelem/s against 0.17 for a naive walk). The graph
records broadcasts explicitly; nothing broadcasts by accident.

## Fast and exact

Results are a function of the payloads, the compiled features,
the numerics posture, and the machine. Two identical runs of one
binary never disagree.

`Network::forward` — whole-spec evaluation, the proving road — is
exact by construction: its bits are the same in every build, on
every platform. What `Fast` (an entry's default) gives up is
bit-identity with that oracle: hardware reassociates sums.
`Exact` restores those bits in the same process — a labeled
choice, not a feature flag.

```rust
let exact = network.entry([loss]).numerics(Numerics::Exact).lower();
let fast = network.entry([loss]).lower();
```

A dependency that enables `topos/accelerate` enables it for the
whole binary. An `Exact` entry still computes interpreter bits.

Which backend actually served is run-time data: wrap any region in
`Backend::tallied` and read the per-formula tally — coverage
declares *may*, the tally reports what *did*. The `throughput`
example prints one for a full training step.

## Serving

A compiled plan is already a closed, statically shaped tensor
function. `Plan::emit_stablehlo` writes it as text; an industrial
runtime takes it from there. Nothing XLA-shaped links in-crate.
See [Vision](vision.md).

Compile cost amortized, same snapshot:

| per run | topos (`accelerate`) | XLA-CPU | XLA GPU | interpreter |
|---|---|---|---|---|
| batch-8 CNN forward | 2.6 ms | 0.24 ms | 5.4 ms | 3.0 s |
| 256-token attention | 0.46 ms | 0.37 ms | 0.65 ms | 6.3 s |

The CNN is the argument: the same tape, eleven times faster, by
handing the plan to XLA. Attention is already within a quarter of
XLA because both ride BLAS. The interpreter is a spec, not a
runner. Modules are parsed and checked against the plan whenever
a JAX toolchain is present (`TOPOS_STABLEHLO_VALIDATOR`,
`TOPOS_STABLEHLO_EVALUATOR`).

## For a new element

Override `Elementary::gemm` and `Elementary::map` to route a
number type onto kernels; the defaults compute on the built-in
paths. The engine never sees this. See [The element is the
seam](principles/element.md).
