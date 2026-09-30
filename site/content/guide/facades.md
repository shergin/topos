# The neural tier

## The rule

Every facade — every layer, loss, optimizer, initializer — composes through the public operation surface alone. No privileged engine access, no hidden opcode. Two things follow. A facade's internal spelling can change without breaking anyone, because nobody could depend on it. And a hand-rolled equivalent behaves identically, so each facade stands as proof that the primitives suffice.

The dump is the proof. `Linear::express` records a product, an explicit broadcast of the bias over the batch axis, and an add:

{{figure facade_linear}}

Write the same four lines by hand and the spec is the same text, asserted:

{{figure facade_hand_rolled}}

## The shapes a facade takes

Pick the facade's shape by what it is:

- a **stateless closed set of alternatives** is a plain `Copy` enum with an `express` method — `Activation::{Tanh, Relu}`;
- a **stateful or user-extensible strategy** is an open, object-safe trait with the built-ins as ordinary implementations — `Optimizer`, so a custom optimizer has the same standing as `Adam`;
- an **operand-asymmetric formula** with no natural `self` is a free function — `cross_entropy`, `conv2d`, `max_pool`, `scaled_dot_product`;
- a **factory** returns `impl FnMut` closures with explicit seeds — `init::{uniform, normal, xavier, kaiming, dropout}`.

Layers with parameters are `Module`s: a named recording function whose `express(input)` records on the input's tape, holding `Symbol`s and never payloads. `Sequential` chains modules; `visit` walks parameters under structured paths for [checkpoints](../training/#checkpoints).

## Layers

`Mlp::new(&tape, &[2, 4, 1], Activation::Tanh, init::xavier(7))` allocates two `Linear` stages and records them with the activation between. The picture is the graph the facade recorded, and nothing more.

{{figure facade_mlp}}

Initialization and hyperparameters are caller-owned and visible at the call site. A facade never chooses a distribution, a learning rate, or a float constant; its internal constants are integer counted ratios only.

## Losses

A loss is a composition that records a rank-0 objective. `cross_entropy` is the mean negative log-likelihood in the expanded, stable spelling: the fused `LogSumExp` is finite for every finite logit, and no term ever multiplies a zero target by an infinite log-probability. Targets travel as a one-hot `Tensor::selection`, so one recorded graph serves any batch of labels.

{{figure facade_cross_entropy}}

## Dropout is a feed

Dropout looks like it needs a random generator in the graph and a switch that turns it off at inference. Neither is required. The layer multiplies by a mask input whose default is all ones, so an unfed run is the identity: that is inference. Training draws the mask on the host from a seeded factory and feeds it like a batch. Seeded replay stays exact, gradients stay `gradient * mask`, and an emitted training step gains one dynamic argument instead of a mode.

{{figure facade_dropout}}

## Attention is a formula

`scaled_dot_product(query, key, value, mask, scale)` records one head, rank 2: `softmax((q @ kᵀ) * scale + mask) @ v`. The mask and the scale are caller-supplied values — `causal_mask(extent, fill)` mints the standard triangular payload with a fill you choose, because a custom element may not have an infinity. Head loops, projections, rotary embeddings, grouped queries, and caches are architecture and stay with the caller, since fused-QKV and separate-projection checkpoints split heads differently.

{{figure facade_attention}}

## Convolution is a formula

`conv2d` is pad, unfold along each spatial axis, permute, reshape, one matrix product, and the bias spread over every position. There is no `Convolution` instruction: the composition is exact, and a fused kernel earns its place only through the [pattern catalog](../plans/#patterns-are-offers), which finds this idiom whether a facade or a hand wrote it.

{{figure facade_conv}}

## Optimizers and the rest

`Sgd`, `Adam`, and `AdamW` implement `Optimizer`; moments are parameter-aligned tables that initialize from the first step's gradients. `BatchNorm` records the training normalization by batch statistics and, through `express_with`, the inference twin normalized by running estimates fed per run — the estimates live in the caller's loop as plain payloads. `LayerNorm` and `RmsNorm` are the transformer norms; `Embedding` is a `Module` holding one `[vocab, dim]` table whose `express` is the gather over a one-hot selection. [Training](../training/#optimizers-are-instruments) compares two optimizers on one graph.

## Where next

- [The element seam](../elements/) — the other kind of extension: a number, not a layer.
- [The gallery](../../examples/#makemore) — the makemore series, where each act adds one facade and trains identically to the hand-rolled act before it.
