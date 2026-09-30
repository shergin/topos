# Training

## Spec and state

The architecture is an immutable value. The weights are yours. Training does not version the graph.

When live weights live inside the graph, a training step has to produce a new graph, and then every name has to say which generation it binds to. Topos splits the two: the `Network` is structure, shapes, record-site initials, and input defaults — runnable standalone, with no live weights and no lock; the `Parameters` are a caller-owned table of payloads, born from those initials or from a checkpoint, passed into every run and every plan.

A training step is then a pure data transform of the table. No new spec, no generation. Cloning the state is honest and costs what the weights cost; that is the whole price of a what-if. Optimizer moments are more tables of the same shape.

{{figure train_state}}

## The loop

```rust
let mut parameters = network.parameters();
for step in 0..steps {
    let run = network.forward(&parameters, feeds(step));
    let gradients = run.backward(loss).parameters(&parameters);
    parameters = parameters.step(&gradients, |parameter, gradient| {
        parameter.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
    });
}
```

`Field::parameters` projects a backward pass onto the parameter slots, so the update is O(parameters) and never touches a graph-sized buffer. `step` calls the rule once per slot, in recording order, and answers a new table; `step_each` also hands the rule each parameter's symbol, for per-parameter policy. The learning rate is a per-step argument spread over each gradient explicitly, because nothing broadcasts by accident.

For compiled training, record the derivative and lower a forward-only plan over the adjoints' roots; then each step is one `plan.forward` and a `recorded_gradients` read, and no backward pass executes at all. [The walkthrough](../../walkthrough/#5-two-runs-that-must-agree) does this on its graph.

## Learning rates are clones

One spec, several states. Each learning rate below descends from a clone of the same initial table; nothing about the network is copied.

{{figure train_rates}}

Parallel what-ifs work the same way: clone the table per thread, share the network by reference. The sealed spec has no lock to fight over, and feeds are run state, so [`gradient_descent.rs`](../../examples/#gradient_descent) trains three states on three threads over one network, and [`makemore_mlp_parallel.rs`](../../examples/#makemore_mlp_parallel) shards a minibatch across threads and sums the shard gradients as tables.

## Optimizers are instruments

An `Optimizer` is an open trait: gradients and parameters in, next parameters out, with whatever state it needs held as more parameter-aligned tables. `Sgd` is stateless; `Adam` and `AdamW` carry moments. A custom optimizer is an ordinary implementation with the same standing as the built-ins, and a comparison loop can iterate them side by side.

{{figure train_optimizers}}

Hyperparameters are caller-owned and visible at the call site: `Adam::new` takes its betas and epsilon as payloads you write, and the learning rate stays an argument of every step. A facade never chooses a float constant.

## Feeds are the data

Input shapes are fixed at record time, and the batch arrives as a feed. A graph recorded over a `[2, 2]` input trains on any number of two-sample minibatches without ever growing the tape — and rasterizes its learned surface through the very same two-row expression afterwards, two grid cells per run.

{{figure train_xor}}

## Twins

When the shape you train on is not the shape you want to read, record both expressions of the same parameters on one tape: a batch-shaped one for training and a grid-shaped or single-row twin for reading. The twin shares every parameter symbol, so one forward with the trained table answers it, and an entry that declares only the training root never schedules the twin.

{{figure train_regression}}

{{figure train_moons}}

The language models in the [gallery](../../examples/#language-models) use the same idiom at scale: a full-context expression and a one-token decode step, on one tape, sharing 124M parameters.

## Checkpoints

A checkpoint is pure state: `snapshot` and `restore` copy a module's parameter payloads in visit order, and `named_snapshot` and `named_restore` do it under structured paths — the form that survives code evolution and maps onto a foreign layout. No graph is touched either way. Loading GPT-2's released weights is one `named_restore` over the paths the model tree announces itself.

{{figure train_checkpoint}}

## Where next

- [Entries and plans](../plans/) — compiled training, and what a plan keeps.
- [The neural tier](../facades/) — the layers a training loop is built from.
- [Spec and state](../../principles/spec-and-state/) — the argument, and [recurrence as feeds](../../openings/recurrence-as-feeds/), what it opened.
