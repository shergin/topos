# Emission

## The plan as text

A compiled plan is already a closed, statically shaped tensor function: parameters and inputs in, the declared results out. `Plan::emit_stablehlo` writes it as text — a StableHLO module an industrial runtime such as XLA compiles and runs. Nothing XLA-shaped links into the crate; the text is the boundary, and emission is a sibling of `describe`, not a second compiler.

{{figure emit_chain}}

The function's arguments are the sources in recording order, parameters first. Its body is the schedule: one StableHLO op per scheduled node, named `%vN` after the node's index, so a line of the dump and a line of the module are the same node. Constants become `stablehlo.constant`; a `Sum` becomes a `reduce` with an explicit zero; a `Broadcast` becomes `broadcast_in_dim`.

## A training step as one function

Because the gradient is recorded as spec, a training step emits as one function: `(parameters, batch) -> (loss, gradients...)`. The host keeps the update loop as plain payload arithmetic and sends the parameters with every batch. [`makemore_mlp_emitted.rs`](../../examples/#makemore_mlp_emitted) trains that way through a resident XLA process and grades the losses against the in-crate plan on the page.

{{figure walk_hlo nocode}}

## Result order is declared

The plan's results are the entry's roots then its observes, in the order they were declared, deduplicated to the first occurrence. Emission does not invent an output convention; it returns exactly that list, and `Plan::results` is the same list as symbols.

{{figure emit_results}}

## Raises

The catalog's matches have a second job. At home an elected pattern calls a fused kernel; abroad it becomes the other compiler's named op. The im2col-and-multiply chain a convolution records becomes `stablehlo.convolution`; a max-pool window becomes `reduce_window`; the batch-norm formulas become `batch_norm_training` and `batch_norm_inference`. That is how a small instruction set still hands an industrial backend the op it is good at, without a `Convolution` instruction and its own backward rule.

{{figure emit_conv}}

## Compositions, written out

What did not earn an instruction is emitted as the composition it is. One attention head is a product, a scale, an additive mask, the softmax spelled from the stable log-softmax, and a product with the values; the module says so, and the receiving compiler is free to fuse it back.

{{figure emit_attention}}

## Conformance

The slow run still checks the other side of the boundary. When `TOPOS_STABLEHLO_VALIDATOR` and `TOPOS_STABLEHLO_EVALUATOR` point at a toolchain — [`tools/validate-stablehlo.py`](https://github.com/shergin/topos/blob/master/tools/validate-stablehlo.py) and [`tools/eval-stablehlo.py`](https://github.com/shergin/topos/blob/master/tools/eval-stablehlo.py), which use the MLIR bindings a `jax` install bundles — every emitted module in the test suite is parsed by an external parser and executed by the StableHLO reference interpreter, and its results are checked against the plan's own within an envelope. Where float physics forbids bit-identity the check is a tolerance, and a miss is reported, not skipped. CI runs this job.

On a language model, the in-crate run, the in-crate plan, and XLA on CPU produce the same text. The same module through a vendor GPU plugin ran faster and was wrong — and the disagreement was provable because three independent implementations agreed with each other and it did not. [GPT-2 on topos](../../examples/gpt2/) tells it in full.

## Where next

- [Fusion raises](../../openings/fusion-raises/) and [the oracle ships](../../openings/the-oracle-ships/) — the two decisions behind this page.
- [Acceleration](../acceleration/) — the measured serving numbers, in-crate against XLA.
