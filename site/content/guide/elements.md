# The element seam

## One graph kind

The graph is always tensors, and a scalar is a tensor of rank 0. There is no scalar engine for the page and a tensor engine for the models; teaching examples are rank-0 tensors, and recording can *look* scalar because a scalar is a tensor, not because the engine is.

What every public phase is generic over is the **element**: the number that fills a tensor. `Tape<E>`, `Network<E>`, `Parameters<E>`, `Plan<E>` — all for `E: Element`. Implement arithmetic, the identities, an accumulator, and the elementary maps on a number, and the tensor machinery, the derivative rules, the plans, the emission, and the notebook come along. A new element never reimplements `unfold`.

## The built-ins

`f32`, `f64`, and `Bf16` — brain float 16, arithmetic in `f32` with round-to-nearest-even, accumulations kept in `f32` until the final total. One recording function, generic over the element, produces the same spec text over each; only the run's bits differ, and each element's bits are its own.

{{figure elem_precision}}

The language-model examples use exactly this genericity: the GPT-2 module tree is one type argument away from its `bf16` engine, with the checkpoint converted at the precision boundary. Half the memory, its own coherent text — a different model by rounding, not a noisy copy.

## Bringing a number

`Element` is `Differentiable` plus `Elementary`. The first is arithmetic, the identities, and the accumulator: `promote` and `demote` between the element and the type inner products sum in, `zero`, `one`, and the counted integers a facade may mint. The second is the maps the vocabulary needs — `exp`, `ln`, `sqrt`, `tanh`, `sin`, `cos`, `log1p`, `expm1`, `erf` and its derivative, `powf`, `maximum`, `step` — plus optional hooks a backend may offer, `gemm` and `map`, whose defaults compute on the built-in paths.

```rust
use topos::{Differentiable, Element, Elementary};

#[derive(Clone, Copy, Debug, PartialEq)]
struct Dual { primal: f64, tangent: f64 }

impl Differentiable for Dual { /* arithmetic, identities, the accumulator */ }
impl Elementary for Dual { /* every map as the chain rule on the primal's own kernel */ }
impl Element for Dual {}

let tape: Tape<Dual> = Tape::new();   // and the whole stack follows
```

Two examples plug the seam from outside the crate:

- [`element_seam.rs`](../../examples/#element_seam) wraps `f64` in a type that counts how often the backend chain offers it a matrix product through `gemm` and answers with the published reference kernel — proving the hook is consulted, and that the README's training loop runs on the new number without one extra line.
- [`dual.rs`](../../examples/#dual) makes the number *mean* something new: a dual carries a value and a tangent, ordinary multiplication is the product rule, and the unchanged interpreter computes a directional derivative in every payload slot. It is graded as a triangle at dyadic values: the dual tangent, the engine reverse scan, and the recorded gradient agree bit for bit.

## Grading a number

`Numerics::exactly` pins the built-in paths to the reference bits, and the [`reference`](https://docs.rs/topos/latest/topos/reference/index.html) module publishes the bitwise kernels — `reference::multiply` and its siblings — an out-of-tree element grades its hooks against. An element that offers no hooks runs on the interpreter; that is the default, not a failure.

`Emittable` and `Sample` are further contracts on the same number: how it prints into StableHLO, and how the seeded initializers draw it. Duals deliberately implement neither — a dual GEMM kernel is a research question of its own — and everything else still works.

## Not this

- A parallel scalar IR for pedagogy.
- Making the recordable vocabulary a thing a new float must implement method by method.
- Fused executors on the element. Those are plan-tier kernel faces, not arithmetic.
- Treating layout, storage, or rank as part of the number.

[The element is the seam](../../principles/element/) is the principle in full.
