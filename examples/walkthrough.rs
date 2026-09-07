//! The whole stack on one graph, read five ways.
//!
//! Topos is an autodiff compiler, and both halves of that phrase
//! show up here on the smallest graph that still has something to
//! say: a matrix product, a `tanh`, and a loss. The loss is one
//! number that says how wrong the output is; training pushes it
//! down, and the gradient (how much the loss moves when each weight
//! moves) is what tells training which way to push.
//!
//! Recording writes the graph down once, as a list of operations.
//! That list is the tape, and once sealed it is the spec: the one
//! definition of what the program means. Everything after that is
//! a different way of reading the same list, and this file prints
//! each reading in turn:
//!
//! 1. The spec: one line per operation, in the order written.
//! 2. The derivative: `differentiate` appends the chain rule to
//!    the same list, so a gradient is just more spec.
//! 3. The plan: a schedule of what to run, what to keep, and what
//!    to free, derived from the spec and holding no state.
//! 4. StableHLO: the plan written out as text that an industrial
//!    compiler such as XLA accepts.
//! 5. The runs: the interpreter, whose bits define the answer, and
//!    the compiled plan, which must reproduce them.
//!
//! Nothing is rewritten between the readings; the spec never
//! changes. A closing section walks its lines by hand, to show that
//! the printed IR is executable and not decorative.
//!
//! Run with: `cargo run --example walkthrough`

use topos::{Numerics, Tape, Tensor};

fn main() {
    // A tape is a notebook of operations: every expression over a
    // `Value` appends one line. `weights` is a parameter, the thing
    // training will change; `input` is fed per run and carries a
    // default, which this file uses throughout.
    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.25, 0.4]));
    let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
    let hidden = input.matmul(weights).tanh();
    let loss = (hidden * hidden).sum();

    banner("1. the spec");
    println!("One line per recorded operation: its number, its opcode, the numbers of");
    println!("the operands it reads, and its shape. Shapes were checked as each line");
    println!("was written; a mismatch panics at the expression that records it.\n");
    print!("{}", tape.describe());
    let spec_nodes = tape.len();

    // A `Value` borrows the tape and cannot outlive it. A `Symbol`
    // is the detached name every later phase speaks.
    let (weights, hidden, loss) = (weights.symbol(), hidden.symbol(), loss.symbol());

    banner("2. the derivative, as more spec");
    // The chain rule is applied by the same rules the interpreter's
    // reverse scan uses; here they record instead of compute. The
    // gradient becomes ordinary nodes: printable, schedulable,
    // emittable, and differentiable again.
    let adjoints = tape.differentiate(loss, [weights]);
    let gradient = adjoints.of(weights);
    println!(
        "`differentiate` appended {} nodes to the same tape; nothing above them changed.",
        tape.len() - spec_nodes
    );
    println!("The last one is d loss / d weights.\n");
    for line in tape.describe().lines().skip(spec_nodes) {
        println!("{line}");
    }
    println!("\nHow to read them:");
    println!("  - a Leaf holding one is the seed, spread over the shape the sum reduced");
    println!("  - `hidden * hidden` gets the product rule once per operand");
    println!("  - `1 - hidden * hidden` is the derivative of tanh");
    println!("  - the matmul rule pushes the result back as two transposed products,");
    println!("    one per operand");
    println!("  - each Leaf-then-Add pair starts a gradient at zero and adds one");
    println!("    consumer's share; that is how a value used twice collects both");

    // Sealing turns the tape into an immutable `Network`. From here
    // on nothing writes the graph: training only replaces the
    // caller-owned `Parameters`, which start from the recorded
    // initial values.
    let network = tape.into_network();
    let parameters = network.parameters();

    banner("3. the plan");
    // An entry declares what a run may read: its roots (the loss and
    // the gradient, in that order) plus any interior we also want to
    // look at. Reading an undeclared interior panics. That
    // declaration is the license the scheduler works under: the
    // declared must survive; the undeclared may be skipped, freed,
    // or fused.
    let declare = || network.entry(adjoints.roots()).observe([hidden]);
    let plan = declare().lower();
    println!("The same operations, scheduled. Each line also says what happens to the");
    println!("value once computed: kept for reading, or freed after its last use. The");
    println!("last line is the plan's own memory accounting: the most elements alive at");
    println!("once, against the total if nothing were ever freed.\n");
    print!("{}", plan.describe());
    let scheduled: Vec<usize> = plan.nodes().map(|node| node.symbol().index()).collect();
    let skipped: Vec<String> = network
        .nodes()
        .filter(|node| !scheduled.contains(&node.symbol().index()))
        .map(|node| node.symbol().index().to_string())
        .collect();
    if skipped.is_empty() {
        println!("\nEvery node is scheduled.");
    } else {
        println!(
            "\nNot scheduled: nodes {}. The rules also produced the gradient with",
            skipped.join(", ")
        );
        println!("respect to the input; nobody declared it, so it is never computed.");
    }
    let patterns = plan.patterns();
    if patterns.is_empty() {
        println!("Elected patterns: none. Fusion is an offer the catalog makes for shapes it");
        println!("recognizes (windows, batch-norm); a dense layer this small runs as primitives.");
    } else {
        println!("Elected patterns:");
        for pattern in &patterns {
            println!(
                "  {:?} rooted at node {}",
                pattern.kind(),
                pattern.root().index()
            );
        }
    }

    banner("4. the plan as StableHLO");
    // Emission is `describe` in another syntax: the same schedule,
    // written as text an industrial compiler accepts. Nothing
    // XLA-shaped lives in the crate; the text is the boundary.
    let module = plan
        .emit_stablehlo()
        .expect("every operation in this graph lowers");
    print!("{module}");
    println!("\nThe function takes the parameter and the input, in recording order, and");
    println!("returns exactly what the entry declared, in that order: the loss, the");
    println!("gradient, then `hidden`.");
    assert_eq!(plan.results(), &[loss, gradient, hidden][..]);
    assert!(module.contains("stablehlo.dot_general"));
    assert!(module.contains("stablehlo.tanh"));

    banner("5. two runs that must agree");
    // The interpreter over the whole spec is the oracle. Its bits are
    // the same in every build, on every platform, and they define
    // what the answer is. Its reverse scan applies the derivative
    // rules directly, without recording.
    let oracle = network.forward(&parameters, []);
    let engine_gradient = oracle.backward(loss).parameters(&parameters);

    // A plan under `Exact` must reproduce those bits. Under `Fast`,
    // the default, a backend feature may serve large products with
    // sums in a different order; that difference is a labeled
    // choice on the entry, never a silent effect of a feature flag.
    let exact = declare().numerics(Numerics::Exact).lower();
    let exact_run = exact.forward(&parameters, []);
    let fast_run = plan.forward(&parameters, []);
    let recorded_gradient = exact_run.recorded_gradients(&adjoints);

    println!("loss:");
    println!("  interpreter        {}", oracle.of(loss));
    println!("  exact plan         {}", exact_run.of(loss));
    println!("  fast plan          {}", fast_run.of(loss));
    println!("d loss / d weights:");
    println!("  engine scan        {}", engine_gradient.of(weights));
    println!("  recorded gradient  {}", recorded_gradient.of(weights));
    println!("hidden, declared readable: {}", exact_run.of(hidden));

    assert!(bits_equal(oracle.of(loss), exact_run.of(loss)));
    assert!(bits_equal(
        engine_gradient.of(weights),
        recorded_gradient.of(weights)
    ));
    println!("\nexact plan == interpreter, bit for bit: asserted");
    println!("recorded gradient == engine scan, bit for bit: asserted");
    if bits_equal(fast_run.of(loss), exact_run.of(loss)) {
        println!(
            "fast plan == exact plan: yes; no backend claimed any of this graph in this build"
        );
    } else {
        println!("fast plan == exact plan: no; a backend served part of this graph under Fast,");
        println!("and Exact is how you ask for the reference bits back");
    }

    // A few steps of gradient descent close the loop. The plan was
    // compiled once and holds no state: each step feeds the current
    // parameters, reads the recorded gradient, and mints the next
    // parameters. The network never changes, and a clone of the
    // state is all a what-if costs. Nothing broadcasts by accident,
    // so the scalar learning rate is spread over each gradient
    // explicitly.
    let learning_rate = Tensor::from(0.05_f32);
    let mut current = parameters.clone();
    let mut losses = Vec::new();
    for _ in 0..5 {
        let run = plan.forward(&current, []);
        losses.push(run.of(loss).scalar());
        let gradients = run.recorded_gradients(&adjoints);
        current = current.step(&gradients, |weight, gradient| {
            weight.clone() - gradient.clone() * learning_rate.broadcast_like(gradient)
        });
    }
    losses.push(plan.forward(&current, []).of(loss).scalar());
    let trajectory: Vec<String> = losses.iter().map(|value| format!("{value:.4}")).collect();
    println!("\nloss before training, then after each of five steps of size 0.05:");
    println!("  {}", trajectory.join(", "));
    assert!(losses[losses.len() - 1] < losses[0]);

    banner("6. the spec is executable");
    // The printed lines are an instruction list, and walking them is
    // the interpreter: sources take their stored payloads, computed
    // nodes express their opcode over operands computed earlier. The
    // walk runs under the exact posture, as the whole-spec oracle
    // does by construction.
    let values = Numerics::exactly(|| {
        let mut values: Vec<Tensor<f32>> = Vec::new();
        for node in network.nodes() {
            let value = if node.is_source() {
                network
                    .payload(node.symbol())
                    .expect("sources hold payloads")
                    .clone()
            } else {
                let operands: Vec<&Tensor<f32>> = node
                    .operands()
                    .iter()
                    .map(|symbol| &values[symbol.index()])
                    .collect();
                node.opcode().express(&operands)
            };
            values.push(value);
        }
        values
    });
    assert!(bits_equal(oracle.of(loss), &values[loss.index()]));
    println!(
        "Walking all {} lines by hand, outside the crate, reproduces the interpreter's",
        values.len()
    );
    println!("loss {} bit for bit.", values[loss.index()]);
}

/// Prints a section heading.
fn banner(title: &str) {
    println!("\n== {title} ==\n");
}

/// Compares two tensors bit for bit, the only equality the oracle
/// promises: same shape, same bits in every element.
fn bits_equal(left: &Tensor<f32>, right: &Tensor<f32>) -> bool {
    left.shape() == right.shape()
        && left
            .to_vec()
            .iter()
            .zip(right.to_vec())
            .all(|(left, right)| left.to_bits() == right.to_bits())
}
