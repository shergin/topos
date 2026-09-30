# topos

<section class="hero">

<div class="hero-text">

<h1>An autodiff compiler stack, small enough to read.</h1>

<p class="lede">Record a graph once. Then read it: run it, differentiate it, schedule it, emit it. The tape is the spec and nothing rewrites it. The weights are yours. Every faster reading has to match the slow, obvious one, bit for bit, and a claim is one assert away from proof.</p>

```sh
cargo add topos
```

<p><a class="cta" href="guide/start/">Start here</a> <a class="cta accent" href="walkthrough/">Read one graph six ways</a> <a class="cta" href="playground/">Open the playground</a></p>

</div>

{{figure hero nocode | A matrix product, a tanh, and a loss, with the derivative appended: the lighter nodes are the chain rule, recorded as more of the same graph. Hover a node to follow its edges.}}

</section>

<section class="manifesto">

<div>

<h3>The tape is the spec</h3>

<p>A typical compiler rewrites the program until it can run. This one does not. Recording writes a list of operations, one line each, with every shape inferred at the line that records it. Sealing freezes it. Everything after — gradients, schedules, fusions, backends, emitted text — is a named way of reading that same list. You can print each reading. None of it is magic.</p>

</div>

<div>

<h3>The interpreter's bits are the truth</h3>

<p>The plain interpreter is the executable spec. Its bits are the same in every build, on every platform. A compiled plan under <code>Exact</code> must reproduce them; a backend that cannot is declined, not trusted. Reordering float math is a labeled choice on the entry, never a silent effect of a feature flag. Two identical runs cannot differ, so a rerun is a checksum.</p>

</div>

<div>

<h3>Built for learning and research</h3>

<p>Learners and teachers first: the spec, its gradient, the schedule, and the emitted text are readings of one graph, all printable. Systems researchers second: a new element type, AD mode, fusion, backend, or emission target plugs in at a named seam and is graded against the interpreter, without forking the crate. The core stays closed and simple on purpose.</p>

</div>

</section>

## Six lines, then everything else is a reading

The whole library fits in one loop. Record the graph in a closure; the return value is the set of names that leave the tape. Feed a sample, ask for the gradient of one scalar, step the parameters. The network never changes.

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
    parameters = parameters.step(&gradients, |w, g| w.clone() - g.clone() * Tensor::from(0.02));
}
assert!((parameters.of(w).scalar() - 2.0).abs() < 1e-6);
```

<div class="two-up">

{{figure start_spec nocode | What that closure recorded, printed by `describe`: the spec. Hover a line.}}

{{figure start_train nocode | What the loop did to the one table it holds. The graph is the same six lines before and after.}}

</div>

## One spec, a short list of interpretations

This table is the compiler. It is small enough to read because there is nothing else, and every row is on this site with a plate.

<table class="stack-table">
<thead><tr><th>reading</th><th>what answers it</th><th>and prints</th></tr></thead>
<tbody>
<tr><td>spec</td><td><code>Tape</code>, <code>Network</code></td><td>one line per node — <a href="guide/recording/">recording</a></td></tr>
<tr><td>value</td><td><code>Network::forward</code>, <code>BoundEntry::interpret</code></td><td>the exact oracle over the whole spec, or over a declared closure</td></tr>
<tr><td>cotangent</td><td><code>Run::backward</code></td><td>the engine reverse scan — <a href="guide/differentiation/">differentiation</a></td></tr>
<tr><td>trace</td><td><code>Tape::differentiate</code></td><td>the same rules recording themselves as more spec</td></tr>
<tr><td>schedule</td><td><code>BoundEntry::lower</code></td><td>a <code>Plan</code>: keep-set, liveness, elections — <a href="guide/plans/">entries and plans</a></td></tr>
<tr><td>catalog</td><td><code>Plan::patterns</code></td><td>the offers a run elected, as data, never as rewrites</td></tr>
<tr><td>text</td><td><code>Plan::emit_stablehlo</code></td><td>the plan for an industrial compiler — <a href="guide/emission/">emission</a></td></tr>
</tbody>
</table>

## What the readings look like

<div class="tiles">

<div class="tile">

{{figure walk_derivative nocode}}

<h3>The gradient is more spec</h3>

<p>Derivative knowledge lives in exactly one rule body. One reading of it computes; the other records the same steps as ordinary nodes, below a divider. The gradient graph is printable, plannable, emittable, and — being ordinary differentiable nodes — differentiable again. Second derivatives are the absence of a wall.</p>

</div>

<div class="tile">

{{figure plan_observe nocode}}

<h3>Say what you want to read</h3>

<p>An entry names its roots and the interiors it will also read. That declaration is the license every optimization runs on: the declared must survive, the rest may be freed, skipped, or fused. Add one name and a line turns from <code>freed</code> to <code>kept</code>. Nothing else moves.</p>

</div>

<div class="tile">

{{figure emit_conv nocode}}

<h3>Fusion raises</h3>

<p>A convolution is recorded as pad, unfold, permute, reshape, and one matrix product. The catalog finds that idiom again and, at home, calls a fused kernel; abroad, it writes <code>stablehlo.convolution</code>. Speed lives in the catalog, not in a new opcode with its own backward rule.</p>

</div>

<div class="tile">

{{figure train_moons nocode}}

<h3>The state is yours</h3>

<p>Training is a pure data transform of a caller-owned table. A what-if costs one clone. A decision surface is a grid-shaped twin of the training expression, recorded once beside it and fed the raster. The <a href="examples/">gallery</a> runs this from a bigram table to GPT-2.</p>

</div>

<div class="tile">

{{figure elem_precision nocode}}

<h3>The element is the seam</h3>

<p>The graph is always tensors; a scalar is rank 0. The open plug is a number: implement two traits on your own float — a dual, an interval, a research format — and the tensor machinery, the derivative rules, the plans, and the notebook come along. <code>Bf16</code> is built in. Forward mode was written outside the crate this way.</p>

</div>

<div class="tile">

{{figure card_plan nocode}}

<h3>It draws itself</h3>

<p>In an Evcxr notebook, the last expression of a cell is a card: a spec dump, a schedule with its live volume, a tensor as a shaded table, a backward pass as a curve of norms. The cards are ordinary <code>to_html</code> strings, and this page embeds them as emitted.</p>

</div>

</div>

## Where to go

<div class="map">

<section>
<h3>Start here</h3>
<ul>
<li><a href="guide/start/">Getting started</a><span>Install, the first tape, a gradient, a training step, and the readings of one small graph.</span></li>
<li><a href="walkthrough/">The stack, read six ways</a><span>The flagship chapter: one graph as spec, derivative, plan, StableHLO, two runs that must agree, and a walk by hand.</span></li>
<li><a href="playground/">Playground</a><span>Write a small graph in the browser; every reading remakes itself as you type.</span></li>
</ul>
</section>

<section>
<h3>The guide</h3>
<ul>
<li><a href="guide/recording/">Recording</a><span>Tapes, values, symbols, shapes at record time, sealing and reopening.</span></li>
<li><a href="guide/differentiation/">Differentiation</a><span>Engine scan, recorded reverse mode, seeds, second derivatives, forward mode from outside.</span></li>
<li><a href="guide/training/">Training</a><span>Spec and state, learning rates as clones, optimizers, twins, checkpoints.</span></li>
<li><a href="guide/plans/">Entries and plans</a><span>Observability, liveness, skipped nodes, patterns, Exact and Fast.</span></li>
<li><a href="guide/emission/">Emission</a><span>The plan as StableHLO, raises, result order, conformance.</span></li>
<li><a href="guide/facades/">The neural tier</a><span>Every layer is a spelling over the public operations, and the dump proves it.</span></li>
<li><a href="guide/elements/">The element seam</a><span>Bring your own number.</span></li>
<li><a href="guide/acceleration/">Acceleration</a><span>Opt-in backends, the measured ladder, and the tally.</span></li>
<li><a href="guide/notebooks/">Notebooks</a><span>Evcxr, two rules, and the cards.</span></li>
</ul>
</section>

<section>
<h3>Examples</h3>
<ul>
<li><a href="examples/">The gallery</a><span>Every example with what it printed and its source: a scalar chain, moons, makemore act by act, MNIST, CIFAR-10, GPT-2, Llama.</span></li>
<li><a href="examples/gpt2/">GPT-2 on topos</a><span>Four engines, one text, and a GPU plugin caught being wrong.</span></li>
<li><a href="examples/llama/">Llama on topos</a><span>TinyLlama and Llama 2 7B from one module tree.</span></li>
</ul>
</section>

<section>
<h3>Why it is shaped this way</h3>
<ul>
<li><a href="principles/">Vision</a><span>Three commitments and five rules.</span></li>
<li><a href="principles/recorded-reverse/">Recorded reverse mode</a><span>One rule body, two readings.</span></li>
<li><a href="principles/observability/">Observability is a license</a><span>What you may read is declared.</span></li>
<li><a href="openings/">Openings</a><span>Six decisions and what each one made possible.</span></li>
</ul>
</section>

<section>
<h3>Reference</h3>
<ul>
<li><a href="concepts/">Terminology</a><span>The vocabulary contract, illustrated.</span></li>
<li><a href="https://docs.rs/topos">API reference</a><span>Two maps: <code>topos::model</code> to train, <code>topos::compiler</code> to inspect and emit.</span></li>
<li><a href="changelog/">Changelog</a><span>Every release, written for a person.</span></li>
</ul>
</section>

</div>

## The name

A topos is a place. Here it is the one where the whole compiler stack stays in view: the spec, its gradient, the schedule, and the emitted text are readings of one graph, and every claim is one assert from proof.

{{figure walk_live nocode | The plan's own memory accounting, drawn from `Plan::live_series`: elements alive after each scheduled node of the walkthrough's training step.}}
