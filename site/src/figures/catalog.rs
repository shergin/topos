//! The list of figures. Each name is a file under `site/figures/`; the
//! block in that file both runs and is shown. `show` and `Graph` are
//! in scope for every block, so a figure ends with the reading it
//! hands the page.

#[allow(unused_imports)]
use topos_plates::graph::Graph;

#[allow(unused_imports)]
use crate::show;

use super::{Figure, card};

pub fn all() -> Vec<Figure> {
    let mut figures = Vec::new();
    figures.extend(start());
    figures.extend(walkthrough());
    figures.extend(recording());
    figures.extend(differentiation());
    figures.extend(training());
    figures.extend(plans());
    figures.extend(emission());
    figures.extend(facades());
    figures.extend(elements());
    figures.extend(acceleration());
    figures.extend(cards());
    figures.extend(openings());
    figures
}

fn start() -> Vec<Figure> {
    catalog![
        "hero" => ("A matrix product, a tanh, and a loss, with the derivative `differentiate` appended: the lighter nodes are the chain rule, recorded as more of the same graph.", card(72, 16)),
        "start_spec" => ("The README's graph as the spec: one line per recorded operation, with its index, its opcode, the indices of the operands it reads, and its shape. Hover a line to see what it reads and who reads it.", card(72, 16)),
        "start_graph" => ("The same six lines as a picture. `error` is read twice, by both sides of the product, and the picture shows it.", card(72, 16)),
        "start_gradient" => ("The engine reverse scan, on one sample: three gradients from one `backward`, each the chain rule the textbook writes.", card(72, 16)),
        "start_train" => ("A hundred steps of the README loop. The graph never changed; the table of parameters did, and `w` walked to two.", card(72, 16)),
        "start_plan" => ("The same graph, scheduled for one root. Every value is freed after its last reader; only the loss is kept.", card(72, 16)),
    ]
}

fn walkthrough() -> Vec<Figure> {
    catalog![
        "walk_spec" => ("The spec: six recorded lines. Shapes were checked as each was written; a mismatch panics at the expression that records it.", card(72, 16)),
        "walk_graph" => ("The spec as a picture: sources at the top, the loss at the bottom, and `hidden` read twice by the square.", card(72, 16)),
        "walk_derivative" => ("`differentiate` appended twenty-three nodes to the same tape. Nothing above the divider changed; the last line is d loss / d weights.", card(72, 16)),
        "walk_derivative_graph" => ("The gradient subgraph drawn over the spec: each Leaf-then-Add pair starts a gradient at zero and adds one consumer's share, which is how a value read twice collects both.", card(72, 16)),
        "walk_plan" => ("The plan: the same operations, scheduled, with what happens to each value once computed. The underlined lines are the entry's results. Four nodes are not scheduled at all.", card(72, 16)),
        "walk_skipped" => ("What the plan skipped, drawn: the rules also produced the gradient with respect to the input, and nobody declared it, so it is never computed.", card(72, 16)),
        "walk_live" => ("The plan's own memory accounting, as a curve: elements alive after each scheduled node. The peak is what a run needs; the retain-all total is what it would need if nothing were ever freed.", card(72, 14)),
        "walk_hlo" => ("The plan as StableHLO: the same schedule, written as text an industrial compiler accepts. The function takes the parameter and the input and returns exactly what the entry declared, in that order.", card(72, 16)),
        "walk_runs" => ("Two runs that must agree. The interpreter's bits define the answer; an `Exact` plan reproduces them; the recorded gradient reproduces the engine scan. All of it asserted.", card(72, 16)),
        "walk_descent" => ("Forty steps of gradient descent through the compiled plan, which was lowered once and holds no state: each step feeds the current parameters and mints the next.", card(72, 14)),
        "walk_replay" => ("The printed lines are executable. Walking them by hand with `Opcode::express`, outside the crate, reproduces the interpreter's loss bit for bit.", card(72, 16)),
    ]
}

fn recording() -> Vec<Figure> {
    catalog![
        "rec_sources" => ("The three source kinds and one computed node. A leaf is a constant, a parameter is trainable, an input is fed per run; none has operands.", card(72, 16)),
        "rec_shapes" => ("Shapes are inferred at the line that records them and printed on every node. Nothing broadcasts by accident: the bias is spread over the batch axis by an explicit `broadcast_along`, and the attribute says so.", card(72, 16)),
        "rec_shared" => ("A value is `Copy` and never consumed, so `a` and `c` each feed two expressions. The picture shows every read.", card(72, 16)),
        "rec_mismatch" => ("What a shape mistake does: the expression that records it panics, with the shapes in the message, and the tape keeps what was recorded before it.", card(72, 16)),
        "rec_reopen" => ("Sealing consumes the tape; reopening consumes the network. Symbols taken before the seal still resolve afterwards, and the new nodes append below the old ones.", card(72, 16)),
        "rec_symbols" => ("A `Value` is the proxy of recording; a `Symbol` is the name every phase after it speaks. Parameters and runs are read through symbols.", card(72, 16)),
    ]
}

fn differentiation() -> Vec<Figure> {
    catalog![
        "diff_engine" => ("The engine reverse scan on the chain example: one `backward`, every gradient. `a` feeds two subexpressions whose contributions cancel exactly, hence its zero.", card(72, 16)),
        "diff_recorded" => ("The same rules, recording instead of computing: the gradient of the README's loss with respect to `w`, appended as ordinary nodes below the divider.", card(72, 16)),
        "diff_accumulate" => ("Accumulation, drawn. `error` has two consumers, so its gradient is a Leaf holding zero plus one Add per consumer's share. The rules return cotangents; the scan adds them.", card(72, 16)),
        "diff_vjp" => ("A vector-Jacobian product with an explicit seed on a vector target: `vjp` plants the seed instead of the ones a plain gradient uses, and picks out one row of the Jacobian.", card(72, 16)),
        "diff_hessian" => ("A second derivative is not a feature. The recorded gradient is more spec, so `vjp` walks it like anything else, and the Hessian-vector product is one more root.", card(72, 16)),
        "diff_norms" => ("A backward pass as one curve: the Euclidean norm of each node's gradient, along the tape of a small MLP. The notebook card draws exactly this.", card(72, 14)),
        "diff_equal" => ("One rule body, two readings, asserted equal bit for bit: the engine scan and the run of the recorded gradient.", card(72, 16)),
    ]
}

fn training() -> Vec<Figure> {
    catalog![
        "train_rates" => ("Three learning rates on one spec, each descending from a clone of the same initial state. On a log scale a constant convergence rate is a straight line.", card(72, 16)),
        "train_state" => ("Training does not version the graph. After a hundred steps the network prints the same lines and still answers the initial parameters; the loop's own table is what moved.", card(72, 16)),
        "train_optimizers" => ("Plain SGD against Adam on one small regression, from the same initial state and the same recorded gradients. The optimizer is a caller-owned instrument the loop hands the gradients to.", card(72, 16)),
        "train_regression" => ("A tanh MLP fit to a noisy sine. The fit line is the trained expression itself, fed an even grid in place of the training samples.", card(72, 16)),
        "train_xor" => ("XOR learned by a `[2, 4, 1]` perceptron fed two samples per run, and the surface it learned, rasterized through the very same two-row expression training used.", card(64, 14)),
        "train_moons" => ("The two moons, separated: the decision surface of a `[2, 16, 16, 1]` MLP after two thousand full-batch steps, drawn by a grid-shaped twin of the training expression.", card(72, 16)),
        "train_checkpoint" => ("A checkpoint is pure state. `named_snapshot` walks a module's parameters under structured paths; `restore` builds a new table and touches no graph.", card(72, 16)),
    ]
}

fn plans() -> Vec<Figure> {
    catalog![
        "plan_forward" => ("A forward-only plan over one root. Every intermediate is freed after its last reader, the loss is kept, and the summary counts the peak.", card(72, 16)),
        "plan_observe" => ("Observability is declared. Lowering the same graph with `hidden` observed turns one line from `freed` to `kept`; nothing else changes.", card(72, 16)),
        "plan_backward" => ("The engine-backward posture: buffers are retained so `Run::backward` can still scan them. The column says what the analysis *could* release, and the summary reports the retain-all total a run actually holds.", card(72, 16)),
        "plan_live" => ("Live volume along the walkthrough's training plan: the loss, its gradient, and one observed interior. The curve climbs while the forward values are still needed and falls as the chain rule frees them.", card(72, 14)),
        "plan_patterns" => ("A convolution and a max-pool, planned. The catalog found the windowed product and the window reduction it knows, elected them, and the interiors of each match print as `fused`.", card(72, 16)),
        "plan_exact_fast" => ("Three losses from one graph: the interpreter, an `Exact` plan, and the default `Fast` plan. The first two are asserted equal in every build; the third differs only where a backend served and reordered a sum.", card(72, 16)),
    ]
}

fn emission() -> Vec<Figure> {
    catalog![
        "emit_chain" => ("The README's graph as a StableHLO module: a function of the parameter and the inputs, returning the loss. Emission is `describe` in another syntax.", card(72, 16)),
        "emit_conv" => ("A convolution recorded as pad, unfold, permute, reshape, and a matrix product, emitted as `stablehlo.convolution`; a max-pool as `reduce_window`. The same match that fuses at home raises abroad.", card(72, 16)),
        "emit_attention" => ("One attention head as StableHLO: the composition the facade recorded, written out for another compiler. The stable softmax is the only fused core in it.", card(72, 16)),
        "emit_results" => ("Result order is declared. The plan's results are the entry's roots then its observes, in that order, and the emitted function returns exactly that list.", card(72, 16)),
    ]
}

fn facades() -> Vec<Figure> {
    catalog![
        "facade_linear" => ("`Linear::express` records four nodes over the public surface: a product, an explicit broadcast of the bias over the batch axis, and an add. There is no privileged engine access to hide behind.", card(72, 16)),
        "facade_hand_rolled" => ("The facade and a hand-rolled affine layer record the same spec, line for line, asserted equal. A facade is packaging, not different math.", card(72, 16)),
        "facade_mlp" => ("A `[2, 4, 1]` perceptron with a tanh hidden stage, as the graph the facade recorded.", card(72, 16)),
        "facade_cross_entropy" => ("Cross-entropy is a composition: the fused `LogSumExp` is the only primitive that earned a seat, for stability; everything around it is plain sums and products.", card(72, 16)),
        "facade_dropout" => ("Dropout is a multiply by a mask input whose default is all ones. Unfed, the expression is the identity; that is inference. Fed a seeded mask, it drops and rescales.", card(72, 16)),
        "facade_attention" => ("One head of scaled dot-product attention as the spec: a product, a scale, an additive causal mask, a softmax composed from the stable log-softmax, and a product with the values.", card(72, 16)),
        "facade_conv" => ("A convolution is a formula: pad, unfold along each spatial axis, permute, reshape, one matrix product, and the bias spread over every position. No `Convolution` instruction exists.", card(72, 16)),
    ]
}

fn elements() -> Vec<Figure> {
    catalog![
        "elem_precision" => ("One recording over three elements. The spec is the same text over each; the run's bits are each number's own.", card(72, 16)),
    ]
}

fn acceleration() -> Vec<Figure> {
    catalog![
        "accel_status" => ("Every backend's answer in the build that made this page.", card(72, 16)),
        "accel_products" => ("Dense `f32` products, in GFLOP/s, from the measured table on this page: one Apple M1 Pro, 512- to 2048-square.", card(72, 12)),
        "accel_tally" => ("Coverage declares what a backend *may* serve; the tally reports what *did*. Here every formula fell to the reference paths, because no feature is on.", card(72, 16)),
    ]
}

fn cards() -> Vec<Figure> {
    catalog![
        "card_network" => ("The network card.", card(72, 16)),
        "card_plan" => ("The plan card.", card(72, 16)),
        "card_tensor" => ("The tensor card.", card(72, 16)),
        "card_field" => ("The field card.", card(72, 16)),
        "card_parameters" => ("The parameters card.", card(72, 16)),
        "card_adjoints" => ("The adjoints card.", card(72, 16)),
        "card_entry" => ("The entry card.", card(72, 16)),
    ]
}

fn openings() -> Vec<Figure> {
    catalog![
        "open_exports" => ("Two entries over one sealed network. The training entry schedules the loss and never the sampling head; the sampling entry schedules the softmax and never the loss. Same weights, different closures.", card(72, 16)),
    ]
}
