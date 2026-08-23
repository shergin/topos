//! The oracle walkthrough through the public surface alone: record,
//! seal, describe, differentiate, lower under both numerics postures,
//! and emit — every call through `topos::` re-exports, which is the
//! vantage from which a seam closure or a missing export is visible.

use topos::{Entry, Numerics, Opcode, Tape, Tensor};

#[test]
fn the_stack_walks_end_to_end_through_the_public_surface() {
    // Record: a tiny dense layer with a scalar loss.
    let tape: Tape<f32> = Tape::new();
    let weights = tape.parameter(Tensor::new([2, 2], [0.5_f32, -0.25, 0.75, 0.1]));
    let input = tape.input(Tensor::new([1, 2], [1.0_f32, 2.0]));
    let loss = (input.matmul(weights).tanh() * input.matmul(weights).tanh()).sum();
    let (weights_symbol, input_symbol, loss) = (weights.symbol(), input.symbol(), loss.symbol());

    // Differentiate: the chain rule as recorded nodes, before the seal.
    let adjoints = tape.differentiate(loss, [weights_symbol]);

    // Seal, and read the spec as IR.
    let network = tape.into_network();
    let described = network.describe();
    assert!(described.contains("MatMul"));
    assert!(described.contains("Tanh"));
    let loss_node = network.node(loss);
    assert_eq!(*loss_node.opcode(), Opcode::Sum);
    assert_eq!(
        network.payload(input_symbol),
        Some(&Tensor::new([1, 2], [1.0_f32, 2.0]))
    );

    // Run the oracle and project the gradients both ways.
    let parameters = network.parameters();
    let run = network.forward(&parameters, []);
    let oracle = run.backward(loss).parameters(&parameters);

    // Lower under both postures; the plans must answer the oracle's
    // bits, and their describes must be spec lines plus liveness.
    for numerics in [Numerics::Exact, Numerics::Fast] {
        let plan = network.compile(
            Entry::roots(adjoints.roots())
                .observe([loss])
                .numerics(numerics),
        );
        let planned = plan.forward(&parameters, []);
        assert_eq!(planned.of(loss).scalar(), run.of(loss).scalar());
        let recorded = planned.recorded_gradients(&adjoints);
        assert_eq!(
            recorded.of(weights_symbol).to_vec(),
            oracle.of(weights_symbol).to_vec()
        );
        for node in plan.nodes() {
            assert!(
                described.contains(node.to_string().trim_end()),
                "plan schedules a node the spec does not print: {}",
                node.name()
            );
        }
    }

    // Emit: the declared results, in declared order.
    let plan = network.compile(Entry::roots(adjoints.roots()).observe([loss]));
    let module = plan
        .emit_stablehlo()
        .expect("every current operation lowers");
    assert!(module.contains("func.func @main"));
    assert!(module.contains("stablehlo.dot_general"));
    // `adjoints.roots()` is target-then-gradients, and the duplicate
    // `observe` of the loss dedupes to its first occurrence: the loss
    // leads, the one gradient follows.
    let results = plan.results();
    assert_eq!(results.first(), Some(&loss));
    assert_eq!(results.len(), 2);
}
